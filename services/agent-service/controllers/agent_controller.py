import logging
import json
import numpy as np
import redis
from datetime import datetime
from fastapi import HTTPException
from config.db import get_db
from config.settings import settings
from models.agent import QueueItem, SearchResultImage
from utils.chroma_client import search_image_vectors, index_image_vector

logger = logging.getLogger(__name__)

_redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

def _publish_queue_event(user_id: str, event: str, payload: dict = None):
    """Publish a queue state-change event to the user-specific Redis pub/sub channel."""
    try:
        data = {"event": event, **(payload or {})}
        _redis_client.publish(f"queue_events:{user_id}", json.dumps(data))
    except Exception as e:
        logger.error(f"Failed to publish queue event: {e}")

def deserialize_embedding(data) -> np.ndarray:
    r"""
    Decodes the bytea format returned by Supabase client (usually a hex string beginning with \x)
    back into a 128-D numpy float32 vector.
    """

    try:
        if isinstance(data, str):
            if data.startswith("\\x"):
                hex_data = data[2:]
            else:
                hex_data = data
            byte_data = bytes.fromhex(hex_data)
        elif isinstance(data, bytes):
            byte_data = data
        else:
            byte_data = bytes(data)
            
        # Handle backward compatibility: if byte_data has 1024 bytes, it is a hex string stored as ASCII text.
        if len(byte_data) == 1024:
            try:
                byte_data = bytes.fromhex(byte_data.decode('ascii'))
            except Exception as double_err:
                logger.error(f"Failed to double-decode hex ASCII embedding: {double_err}")

        return np.frombuffer(byte_data, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to deserialize embedding: {e}")
        # Return a zero vector fallback to prevent crashes
        return np.zeros(128, dtype=np.float32)

class AgentController:
    @staticmethod
    async def get_queue(user_id: str) -> list[QueueItem]:
        db = await get_db()
        
        # Select pending queue items including parent image public URL
        res = await db.table("face_queue").select(
            "face_id, temporary_name, faces(face_thumbnail_url, images(public_url))"
        ).eq("user_id", user_id).eq("status", "pending").order("created_at").execute()
        
        queue_items = []
        for row in res.data:
            faces = row.get("faces")
            face_thumbnail_url = faces.get("face_thumbnail_url") if faces else ""
            
            images = faces.get("images") if faces else None
            image_url = ""
            if images:
                if isinstance(images, list) and len(images) > 0:
                    image_url = images[0].get("public_url", "")
                elif isinstance(images, dict):
                    image_url = images.get("public_url", "")
                    
            queue_items.append(
                QueueItem(
                    face_id=row["face_id"],
                    temporary_name=row["temporary_name"],
                    face_thumbnail_url=face_thumbnail_url,
                    image_url=image_url
                )
            )
        return queue_items


    @staticmethod
    async def resolve_face(user_id: str, face_id: str, label_name: str, relationship: str = None) -> dict:
        label_name_cleaned = label_name.strip()
        if not label_name_cleaned:
            raise HTTPException(status_code=400, detail="Label name cannot be empty")
            
        relationship_cleaned = relationship.strip() if relationship else None
        if relationship_cleaned == "":
            relationship_cleaned = None

        db = await get_db()

        # 1. Retrieve or create face_labels record
        lbl_res = await db.table("face_labels").select("id, relationship").eq("user_id", user_id).eq("name", label_name_cleaned).execute()
        if lbl_res.data:
            label_id = lbl_res.data[0]["id"]
            existing_rel = lbl_res.data[0].get("relationship")
            # If relationship is passed and differs from existing, update it
            if relationship_cleaned and relationship_cleaned != existing_rel:
                await db.table("face_labels").update({"relationship": relationship_cleaned}).eq("id", label_id).execute()
        else:
            ins_payload = {
                "user_id": user_id,
                "name": label_name_cleaned
            }
            if relationship_cleaned:
                ins_payload["relationship"] = relationship_cleaned
                
            ins_lbl = await db.table("face_labels").insert(ins_payload).execute()
            if not ins_lbl.data:
                raise HTTPException(status_code=500, detail="Failed to create face label")
            label_id = ins_lbl.data[0]["id"]

        # 2. Get the target face embedding
        face_res = await db.table("faces").select("embedding_vector, image_id").eq("id", face_id).eq("user_id", user_id).execute()
        if not face_res.data:
            raise HTTPException(status_code=404, detail="Face record not found")
        
        row = face_res.data[0]
        target_embedding = deserialize_embedding(row["embedding_vector"])
        images_to_update = {row["image_id"]}

        # 3. Associate label to target face and mark resolved
        await db.table("faces").update({"label_id": label_id}).eq("id", face_id).execute()
        await db.table("face_queue").update({"status": "resolved"}).eq("face_id", face_id).execute()
        
        # Publish resolved event so the SSE stream removes this face from the UI
        _publish_queue_event(user_id, "face_resolved", {"face_id": face_id})

        # 4. Auto-Propagation (Clustering)
        # Find all other pending faces for this user
        pending_res = await db.table("faces").select(
            "id, embedding_vector, image_id, face_queue(status)"
        ).eq("user_id", user_id).is_("label_id", "null").execute()
        
        propagated_count = 0
        for f in pending_res.data:
            # Verify if this face is pending in the queue
            fq = f.get("face_queue")
            # In postgrest joins, single relations are objects. Double check format
            fq_status = fq.get("status") if isinstance(fq, dict) else (fq[0].get("status") if isinstance(fq, list) and fq else None)
            
            if fq_status == "pending":
                pending_face_id = f["id"]
                pending_img_id = f["image_id"]
                pending_emb = deserialize_embedding(f["embedding_vector"])
                
                # Compute distance
                dist = np.linalg.norm(target_embedding - pending_emb)
                if dist < 0.45: # Propagation similarity threshold
                    await db.table("faces").update({"label_id": label_id}).eq("id", pending_face_id).execute()
                    await db.table("face_queue").update({"status": "resolved"}).eq("face_id", pending_face_id).execute()
                    images_to_update.add(pending_img_id)
                    propagated_count += 1

        # 5. ChromaDB vector metadata sync
        for img_id in images_to_update:
            # Fetch description
            img_res = await db.table("images").select("scene_description, tags, storage_path, mime_type").eq("id", img_id).execute()
            desc = img_res.data[0]["scene_description"] if img_res.data else ""
            tags = img_res.data[0].get("tags") if img_res.data else {}
            storage_path = img_res.data[0]["storage_path"] if img_res.data else ""
            mime_type = img_res.data[0]["mime_type"] if img_res.data else "image/jpeg"
            
            # Download image bytes for Gemini re-analysis
            image_bytes = None
            if storage_path:
                try:
                    image_bytes = await db.storage.from_("Memora ai").download(storage_path)
                except Exception as dl_err:
                    logger.error(f"Failed to download image {img_id} for re-analysis: {dl_err}")
            
            # Fetch all labeled faces for this image
            labeled_res = await db.table("faces").select(
                "face_labels(name, relationship)"
            ).eq("image_id", img_id).not_.is_("label_id", "null").execute()
            
            labeled_list = []
            for fd in labeled_res.data:
                fl = fd.get("face_labels")
                if fl:
                    if isinstance(fl, dict):
                        labeled_list.append({
                            "name": fl.get("name"),
                            "relationship": fl.get("relationship")
                        })
                    elif isinstance(fl, list) and fl:
                        labeled_list.append({
                            "name": fl[0].get("name"),
                            "relationship": fl[0].get("relationship")
                        })
            
            # Re-run Gemini Vision with identity context
            if image_bytes:
                try:
                    from utils.gcp_vertex import generate_scene_description
                    analysis = generate_scene_description(image_bytes, mime_type, labeled_list)
                    desc = analysis.get("description", desc)
                    tags = analysis.get("tags", tags)
                    
                    # Update database image record
                    await db.table("images").update({
                        "scene_description": desc,
                        "tags": tags
                    }).eq("id", img_id).execute()
                except Exception as gemini_err:
                    logger.error(f"Failed to regenerate scene description with identities: {gemini_err}")
            
            # Fetch all labels and pending queue names for this image
            faces_details = await db.table("faces").select(
                "id, face_labels(name), face_queue(temporary_name, status)"
            ).eq("image_id", img_id).execute()
            
            detected_names = []
            for fd in faces_details.data:
                # 1. Grab resolved label
                fl = fd.get("face_labels")
                fl_name = fl.get("name") if isinstance(fl, dict) else None
                if fl_name:
                    detected_names.append(fl_name)
                    continue
                
                # 2. Grab pending temporary name
                fq = fd.get("face_queue")
                fq_data = fq if isinstance(fq, dict) else (fq[0] if isinstance(fq, list) and fq else None)
                if fq_data and fq_data.get("status") == "pending":
                    detected_names.append(fq_data.get("temporary_name"))

            # Re-index in Chroma
            index_image_vector(
                image_id=img_id,
                user_id=user_id,
                scene_description=desc,
                detected_faces=detected_names,
                tags=tags
            )


        return {
            "status": "success",
            "message": f"Resolved face. Automatically tagged {propagated_count} other similar faces.",
            "propagated_count": propagated_count
        }

    @staticmethod
    async def search(user_id: str, query: str, history: list = None, filters: dict = None) -> dict:
        import re
        from utils.gcp_vertex import generate_chat_search_decision
        
        db = await get_db()
        query_lower = query.lower().strip()
        
        # ── Step 1: Parse mentions & DB exact matches ─────────────────────────
        mentions = re.findall(r"@([a-zA-Z0-9_]+)", query)
        matched_labels = []
        explicit_image_ids = set()
        
        if mentions:
            mentions_lower = [m.lower() for m in mentions]
            lbl_res = await db.table("face_labels").select("id, name, relationship").eq("user_id", user_id).execute()
            
            for row in (lbl_res.data or []):
                name = row.get("name", "")
                if name.lower() in mentions_lower:
                    matched_labels.append(row)
            
            if matched_labels:
                label_ids = [lbl["id"] for lbl in matched_labels]
                face_res = await db.table("faces").select("image_id").eq("user_id", user_id).in_("label_id", label_ids).execute()
                for f in (face_res.data or []):
                    explicit_image_ids.add(f["image_id"])

        # ── Step 2: Vector semantic search ────────────────────────────────────
        # Expand query for vector search by adding relationship details to improve semantic extraction
        expanded_query = query
        for lbl in matched_labels:
            name = lbl.get("name", "")
            rel = lbl.get("relationship", "")
            if rel:
                expanded_query += f" {name} {rel}"
            else:
                expanded_query += f" {name}"
                
        chroma_hits = search_image_vectors(user_id, expanded_query, limit=20, filters=filters)
        
        # ── Step 3: Reranker Pass ─────────────────────────────────────────────
        all_candidate_ids = explicit_image_ids | {hit["image_id"] for hit in chroma_hits}
        if not all_candidate_ids:
            # Check if this is a chat message (like hello) or if we really just found nothing
            relationships_context = ""
            all_lbls = await db.table("face_labels").select("name, relationship").eq("user_id", user_id).execute()
            if all_lbls.data:
                relationships_context = ", ".join([
                    f"{r['name']}: {r['relationship']}" if r.get('relationship') else r['name']
                    for r in all_lbls.data
                ])
                
            history_list = []
            if history:
                for msg in history:
                    history_list.append({
                        "role": getattr(msg, "role", msg.get("role", "user") if isinstance(msg, dict) else "user"),
                        "text": getattr(msg, "text", msg.get("text", "") if isinstance(msg, dict) else "")
                    })
            
            gemini_res = generate_chat_search_decision(
                query=query,
                candidates=[],
                history=history_list,
                relationships=relationships_context
            )
            return {
                "response_text": gemini_res.get("response_text", "I couldn't find any photos in your gallery. Can you explain more?"),
                "images": []
            }
            
        res = await db.table("images").select("id, public_url, scene_description, tags, created_at") \
            .eq("user_id", user_id).eq("is_deleted", False).in_("id", list(all_candidate_ids)).execute()
            
        # Fetch all faces for these images to know their names and labels
        faces_res = await db.table("faces").select("image_id, face_labels(name, relationship), face_queue(temporary_name, status)").in_("image_id", list(all_candidate_ids)).execute()
        
        # Group faces by image_id
        image_faces = {}
        for f in (faces_res.data or []):
            img_id = f.get("image_id")
            if not img_id:
                continue
            if img_id not in image_faces:
                image_faces[img_id] = []
                
            fl = f.get("face_labels")
            if fl:
                fl_data = fl if isinstance(fl, dict) else (fl[0] if isinstance(fl, list) and fl else None)
                if fl_data:
                    name = fl_data.get("name")
                    if name:
                        image_faces[img_id].append(name)
                        continue
            fq = f.get("face_queue")
            fq_data = fq if isinstance(fq, dict) else (fq[0] if isinstance(fq, list) and fq else None)
            if fq_data and fq_data.get("status") == "pending":
                temp_name = fq_data.get("temporary_name")
                if temp_name:
                    image_faces[img_id].append(temp_name)

        # Map hit score from Chroma
        chroma_score_map = {hit["image_id"]: hit["score"] for hit in chroma_hits}
        
        candidates_to_rank = []
        for r in (res.data or []):
            img_id = r["id"]
            vector_score = chroma_score_map.get(img_id, 0.40) # baseline for database-only hits
            
            # Count how many of the `@` mentioned people are explicitly in this image
            detected_in_img = image_faces.get(img_id, [])
            mentions_matched = 0
            for lbl in matched_labels:
                if lbl["name"] in detected_in_img:
                    mentions_matched += 1
            
            # Boost score: +0.35 weight per matching mention
            rerank_score = vector_score + (0.35 * mentions_matched)
            
            # Add minor boosts for tags
            tag_matches = 0
            tags_dict = r.get("tags") or {}
            for key, val in tags_dict.items():
                if val:
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, str) and item.lower() in query_lower:
                                tag_matches += 1
                    elif isinstance(val, str) and val.lower() in query_lower:
                        tag_matches += 1
            
            rerank_score += (0.05 * tag_matches)
            
            candidates_to_rank.append({
                "id": img_id,
                "public_url": r["public_url"],
                "scene_description": r.get("scene_description") or "",
                "created_at": r["created_at"],
                "tags": tags_dict,
                "detected_faces": detected_in_img,
                "rerank_score": rerank_score,
                "mentions_matched": mentions_matched
            })
            
        # Sort candidates by rerank score descending
        candidates_to_rank.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # ── Step 4: Gemini Decision & Response ────────────────────────────────
        # Select top 10 candidates for Gemini reasoning
        top_candidates = candidates_to_rank[:10]
        gemini_candidates = []
        for c in top_candidates:
            gemini_candidates.append({
                "id": str(c["id"]),
                "description": c["scene_description"],
                "detected_faces": c["detected_faces"],
                "tags": c["tags"]
            })
            
        relationships_context = ", ".join([
            f"{lbl['name']} (who is the user's {lbl['relationship']})" if lbl.get('relationship') else lbl['name']
            for lbl in matched_labels
        ]) if matched_labels else ""
        
        if not relationships_context:
            all_lbls = await db.table("face_labels").select("name, relationship").eq("user_id", user_id).execute()
            if all_lbls.data:
                relationships_context = ", ".join([
                    f"{r['name']}: {r['relationship']}" if r.get('relationship') else r['name']
                    for r in all_lbls.data
                ])
                
        history_list = []
        if history:
            for msg in history:
                history_list.append({
                    "role": getattr(msg, "role", msg.get("role", "user") if isinstance(msg, dict) else "user"),
                    "text": getattr(msg, "text", msg.get("text", "") if isinstance(msg, dict) else "")
                })
                
        gemini_res = generate_chat_search_decision(
            query=query,
            candidates=gemini_candidates,
            history=history_list,
            relationships=relationships_context
        )
        
        ai_response_text = gemini_res.get("response_text", "")
        appropriate_ids = gemini_res.get("appropriate_image_ids", [])
        appropriate_set = {str(uid) for uid in appropriate_ids}
        
        selected_candidates = []
        for c in candidates_to_rank:
            if str(c["id"]) in appropriate_set:
                created_at_dt = datetime.fromisoformat(c["created_at"].replace("Z", "+00:00"))
                selected_candidates.append(
                    SearchResultImage(
                        id=c["id"],
                        public_url=c["public_url"],
                        scene_description=c["scene_description"],
                        created_at=created_at_dt,
                        score=c["rerank_score"],
                        detected_faces=c["detected_faces"],
                        tags=c["tags"]
                    )
                )
                
        return {
            "response_text": ai_response_text or f"I found {len(selected_candidates)} matching photos.",
            "images": selected_candidates
        }


    @staticmethod
    async def get_labeled_people(user_id: str) -> list[dict]:
        db = await get_db()
        
        # 1. Fetch all face labels for the user
        lbl_res = await db.table("face_labels").select("id, name, relationship").eq("user_id", user_id).order("name").execute()
        if not lbl_res.data:
            return []
            
        people = []
        for row in lbl_res.data:
            label_id = row["id"]
            
            # 2. Count faces identified under this label, and pick one thumbnail to use as avatar
            faces_res = await db.table("faces").select("face_thumbnail_url").eq("label_id", label_id).eq("user_id", user_id).execute()
            
            face_count = len(faces_res.data)
            avatar_url = ""
            if faces_res.data:
                # Find the first face that has a valid thumbnail url
                for face in faces_res.data:
                    if face.get("face_thumbnail_url"):
                        avatar_url = face["face_thumbnail_url"]
                        break
                        
            people.append({
                "id": str(label_id),
                "name": row["name"],
                "relationship": row.get("relationship") or "",
                "face_count": face_count,
                "avatar_url": avatar_url
            })
            
        return people

    @staticmethod
    async def update_relationship(user_id: str, label_id: str, relationship: str) -> dict:
        relationship_cleaned = relationship.strip() if relationship else None
        if relationship_cleaned == "":
            relationship_cleaned = None

        db = await get_db()
        
        # Update relationship in face_labels table
        await db.table("face_labels").update({
            "relationship": relationship_cleaned
        }).eq("id", label_id).eq("user_id", user_id).execute()
        
        # Get all image IDs that contain this labeled face
        faces_res = await db.table("faces").select("image_id").eq("label_id", label_id).eq("user_id", user_id).execute()
        images_to_update = {row["image_id"] for row in faces_res.data}
        
        for img_id in images_to_update:
            img_res = await db.table("images").select("scene_description, tags, storage_path, mime_type").eq("id", img_id).execute()
            desc = img_res.data[0]["scene_description"] if img_res.data else ""
            tags = img_res.data[0].get("tags") if img_res.data else {}
            storage_path = img_res.data[0]["storage_path"] if img_res.data else ""
            mime_type = img_res.data[0]["mime_type"] if img_res.data else "image/jpeg"
            
            image_bytes = None
            if storage_path:
                try:
                    image_bytes = await db.storage.from_("Memora ai").download(storage_path)
                except Exception as dl_err:
                    logger.error(f"Failed to download image {img_id} for re-analysis: {dl_err}")
            
            labeled_res = await db.table("faces").select(
                "face_labels(name, relationship)"
            ).eq("image_id", img_id).not_.is_("label_id", "null").execute()
            
            labeled_list = []
            for fd in labeled_res.data:
                fl = fd.get("face_labels")
                if fl:
                    if isinstance(fl, dict):
                        labeled_list.append({
                            "name": fl.get("name"),
                            "relationship": fl.get("relationship")
                        })
                    elif isinstance(fl, list) and fl:
                        labeled_list.append({
                            "name": fl[0].get("name"),
                            "relationship": fl[0].get("relationship")
                        })
            
            if image_bytes:
                try:
                    from utils.gcp_vertex import generate_scene_description
                    analysis = generate_scene_description(image_bytes, mime_type, labeled_list)
                    desc = analysis.get("description", desc)
                    tags = analysis.get("tags", tags)
                    
                    await db.table("images").update({
                        "scene_description": desc,
                        "tags": tags
                    }).eq("id", img_id).execute()
                except Exception as gemini_err:
                    logger.error(f"Failed to regenerate scene description with identities: {gemini_err}")
            
            # Fetch all labels and pending queue names for this image
            faces_details = await db.table("faces").select(
                "id, face_labels(name), face_queue(temporary_name, status)"
            ).eq("image_id", img_id).execute()
            
            detected_names = []
            for fd in faces_details.data:
                fl = fd.get("face_labels")
                fl_name = fl.get("name") if isinstance(fl, dict) else None
                if fl_name:
                    detected_names.append(fl_name)
                    continue
                fq = fd.get("face_queue")
                fq_data = fq if isinstance(fq, dict) else (fq[0] if isinstance(fq, list) and fq else None)
                if fq_data and fq_data.get("status") == "pending":
                    detected_names.append(fq_data.get("temporary_name"))
            
            # Re-index in Chroma
            index_image_vector(
                image_id=img_id,
                user_id=user_id,
                scene_description=desc,
                detected_faces=detected_names,
                tags=tags
            )
        
        return {"status": "success", "message": "Relationship updated successfully"}
