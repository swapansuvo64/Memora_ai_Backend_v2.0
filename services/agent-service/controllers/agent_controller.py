import logging
import json
import numpy as np
import redis
from datetime import datetime, timedelta
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

def parse_temporal_query(query: str) -> tuple[datetime | None, datetime | None, str]:
    import re
    now = datetime.utcnow() # Use UTC since database stores in UTC
    query_lower = f" {query.lower().strip()} "
    
    # Standardize common typos and phrases
    query_lower = re.sub(r'\byester\s+day\b', 'yesterday', query_lower)
    query_lower = re.sub(r'\btodays\b', 'today', query_lower)
    query_lower = re.sub(r'\bmy\s+self\b', 'myself', query_lower)
    
    start_date = None
    end_date = None
    clean_query = query.strip()
    
    # 1. Matches N days/weeks/months/years (with typo yera/year)
    # e.g., "2 days later", "20 days later", "2 month", "5 yera back", "3 years ago"
    days_match = re.search(r'\b(\d+)\s+days?\s*(ago|back|later|after)?\b', query_lower)
    weeks_match = re.search(r'\b(\d+)\s+weeks?\s*(ago|back|later|after)?\b', query_lower)
    months_match = re.search(r'\b(\d+)\s+months?\s*(ago|back|later|after)?\b', query_lower)
    years_match = re.search(r'\b(\d+)\s+(yeras?|years?)\s*(ago|back|later|after)?\b', query_lower)
    
    if days_match:
        val = int(days_match.group(1))
        direction = days_match.group(2)
        if direction in ["later", "after"]:
            target_day = now + timedelta(days=val)
        else:
            target_day = now - timedelta(days=val)
        start_date = datetime(target_day.year, target_day.month, target_day.day, 0, 0, 0)
        end_date = datetime(target_day.year, target_day.month, target_day.day, 23, 59, 59)
        clean_query = re.sub(r'\b\d+\s+days?\s*(ago|back|later|after)?\b', '', query_lower)
        
    elif weeks_match:
        val = int(weeks_match.group(1))
        direction = weeks_match.group(2)
        if direction in ["later", "after"]:
            target_week = now + timedelta(weeks=val)
        else:
            target_week = now - timedelta(weeks=val)
        start_date = target_week - timedelta(days=target_week.weekday())
        start_date = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0)
        end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
        clean_query = re.sub(r'\b\d+\s+weeks?\s*(ago|back|later|after)?\b', '', query_lower)
        
    elif months_match:
        val = int(months_match.group(1))
        direction = months_match.group(2)
        if direction in ["later", "after"]:
            # Future month calculation
            year = now.year
            month = now.month + val
            while month > 12:
                month -= 12
                year += 1
        else:
            # Past month calculation
            year = now.year
            month = now.month - val
            while month <= 0:
                month += 12
                year -= 1
        start_date = datetime(year, month, 1, 0, 0, 0)
        if month == 12:
            end_date = datetime(year, 12, 31, 23, 59, 59)
        else:
            end_date = datetime(year, month + 1, 1, 0, 0, 0) - timedelta(seconds=1)
        clean_query = re.sub(r'\b\d+\s+months?\s*(ago|back|later|after)?\b', '', query_lower)
        
    elif years_match:
        val = int(years_match.group(1))
        direction = years_match.group(3)
        if direction in ["later", "after"]:
            year = now.year + val
        else:
            year = now.year - val
        start_date = datetime(year, 1, 1, 0, 0, 0)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        clean_query = re.sub(r'\b\d+\s+(yeras?|years?)\s*(ago|back|later|after)?\b', '', query_lower)
        
    # 2. Match single/relative keywords
    elif " yesterday " in query_lower:
        yesterday = now - timedelta(days=1)
        start_date = datetime(yesterday.year, yesterday.month, yesterday.day, 0, 0, 0)
        end_date = datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, 59)
        clean_query = query_lower.replace(" yesterday ", " ")
        
    elif " today " in query_lower:
        start_date = datetime(now.year, now.month, now.day, 0, 0, 0)
        end_date = datetime(now.year, now.month, now.day, 23, 59, 59)
        clean_query = query_lower.replace(" today ", " ")
        
    elif " last week " in query_lower:
        last_week = now - timedelta(weeks=1)
        start_date = last_week - timedelta(days=last_week.weekday())
        start_date = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0)
        end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
        clean_query = query_lower.replace(" last week ", " ")
        
    elif " last month " in query_lower:
        year = now.year
        month = now.month - 1
        if month <= 0:
            month = 12
            year -= 1
        start_date = datetime(year, month, 1, 0, 0, 0)
        if month == 12:
            end_date = datetime(year, 12, 31, 23, 59, 59)
        else:
            end_date = datetime(year, month + 1, 1, 0, 0, 0) - timedelta(seconds=1)
        clean_query = query_lower.replace(" last month ", " ")
        
    elif " last year " in query_lower:
        year = now.year - 1
        start_date = datetime(year, 1, 1, 0, 0, 0)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        clean_query = query_lower.replace(" last year ", " ")

    # 3. Match specific 4-digit years (e.g. 2024, 2025, 2023)
    year_match = re.search(r'\b(20\d{2}|19\d{2})\b', query_lower)
    if year_match and not start_date:
        year = int(year_match.group(1))
        start_date = datetime(year, 1, 1, 0, 0, 0)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        clean_query = re.sub(r'\b(20\d{2}|19\d{2})\b', '', query_lower)
        
    clean_query = re.sub(r'\s+', ' ', clean_query).strip()
    return start_date, end_date, clean_query

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
    async def get_queue(user_id: str, page: int = None, limit: int = None) -> list[QueueItem]:
        db = await get_db()
        
        # Select pending queue items including parent image public URL
        query = db.table("face_queue").select(
            "face_id, temporary_name, faces(face_thumbnail_url, images(public_url))"
        ).eq("user_id", user_id).eq("status", "pending").order("created_at", desc=True)
        
        if page is not None and limit is not None:
            start = (page - 1) * limit
            end = start + limit - 1
            query = query.range(start, end)
            
        res = await query.execute()
        
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
        embedding_raw = row.get("embedding_vector")
        is_embedding_available = embedding_raw is not None
        images_to_update = {row["image_id"]}

        # 3. Associate label to target face and mark resolved
        await db.table("faces").update({"label_id": label_id}).eq("id", face_id).execute()
        await db.table("face_queue").update({"status": "resolved"}).eq("face_id", face_id).execute()
        
        # Publish resolved event so the SSE stream removes this face from the UI
        _publish_queue_event(user_id, "face_resolved", {"face_id": face_id})

        # 4. Auto-Propagation (Clustering)
        propagated_count = 0
        if is_embedding_available:
            target_embedding = deserialize_embedding(embedding_raw)
            # Find all other pending faces for this user
            pending_res = await db.table("faces").select(
                "id, embedding_vector, image_id, face_queue(status)"
            ).eq("user_id", user_id).is_("label_id", "null").execute()
            
            for f in pending_res.data:
                # Verify if this face is pending in the queue
                fq = f.get("face_queue")
                # In postgrest joins, single relations are objects. Double check format
                fq_status = fq.get("status") if isinstance(fq, dict) else (fq[0].get("status") if isinstance(fq, list) and fq else None)
                
                pending_emb_raw = f.get("embedding_vector")
                if fq_status == "pending" and pending_emb_raw is not None:
                    pending_face_id = f["id"]
                    pending_img_id = f["image_id"]
                    pending_emb = deserialize_embedding(pending_emb_raw)
                    
                    # Compute distance
                    dist = np.linalg.norm(target_embedding - pending_emb)
                    if dist < 0.45: # Propagation similarity threshold
                        await db.table("faces").update({"label_id": label_id}).eq("id", pending_face_id).execute()
                        await db.table("face_queue").update({"status": "resolved"}).eq("face_id", pending_face_id).execute()
                        images_to_update.add(pending_img_id)
                        propagated_count += 1

        # 5. ChromaDB vector metadata sync
        for img_id in images_to_update:
            # Fetch description and other metadata
            img_res = await db.table("images").select("scene_description, tags, storage_path, mime_type, category, document_details, landscape_details, custom_tags").eq("id", img_id).execute()
            
            desc = ""
            tags = {}
            storage_path = ""
            mime_type = "image/jpeg"
            category = "other"
            document_details = None
            landscape_details = None
            custom_tags = []
            
            if img_res.data:
                row_img = img_res.data[0]
                desc = row_img.get("scene_description") or ""
                tags = row_img.get("tags") or {}
                storage_path = row_img.get("storage_path") or ""
                mime_type = row_img.get("mime_type") or "image/jpeg"
                category = row_img.get("category") or "other"
                document_details = row_img.get("document_details")
                landscape_details = row_img.get("landscape_details")
                custom_tags = row_img.get("custom_tags") or []
            
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
                    category = analysis.get("category", category)
                    document_details = analysis.get("document_details", document_details)
                    landscape_details = analysis.get("landscape_details", landscape_details)
                    
                    # Update database image record
                    await db.table("images").update({
                        "scene_description": desc,
                        "tags": tags,
                        "category": category,
                        "document_details": document_details,
                        "landscape_details": landscape_details
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
                tags=tags,
                category=category,
                document_details=document_details,
                landscape_details=landscape_details,
                custom_tags=custom_tags
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
        
        # Parse temporal queries (e.g. yesterday, N days later, last month, etc.)
        start_date, end_date, clean_query = parse_temporal_query(query)
        logger.info(f"Temporal Search: query='{query}' -> clean='{clean_query}', start={start_date}, end={end_date}")
        
        query_lower = clean_query.lower().strip()
        
        # ── Step 1: Parse mentions & DB exact matches ─────────────────────────
        mentions = re.findall(r"@([a-zA-Z0-9_]+)", clean_query)
        
        # Check if the query refers to the user themselves ("me", "myself", "my self", "my photo", "photos of me")
        self_keywords = [" me ", " me?", " me.", "myself", "my self", "my photo", "photos of me"]
        refers_to_self = False
        padded_query = f" {query_lower} "
        if query_lower == "me" or any(kw in padded_query for kw in self_keywords):
            refers_to_self = True
            
        matched_labels = []
        explicit_image_ids = set()
        
        lbl_res = await db.table("face_labels").select("id, name, relationship").eq("user_id", user_id).execute()
        
        for row in (lbl_res.data or []):
            name = row.get("name", "")
            rel = row.get("relationship", "")
            rel_clean = rel.lower().strip() if rel else ""
            
            is_match = False
            # Check if mentioned explicitly via @name
            if mentions and name.lower() in [m.lower() for m in mentions]:
                is_match = True
            # Check if refers to self and relationship matches self-relationship keyword
            elif refers_to_self and rel_clean in ["my self", "myself", "me", "self"]:
                is_match = True
            # Check if refers to relationship (e.g. "my wife", "brother", "husband")
            elif rel_clean:
                # Match "my {relationship}", "{relationship}" as word, or plural "{relationship}s"
                if f"my {rel_clean}" in query_lower or f" {rel_clean} " in padded_query or f" {rel_clean}s " in padded_query:
                    is_match = True
                
            if is_match:
                matched_labels.append(row)
            
        if matched_labels:
            label_ids = [lbl["id"] for lbl in matched_labels]
            face_res = await db.table("faces").select("image_id").eq("user_id", user_id).in_("label_id", label_ids).execute()
            for f in (face_res.data or []):
                explicit_image_ids.add(f["image_id"])

        # ── Step 2: Vector semantic search ────────────────────────────────────
        chroma_hits = []
        # Expand query for vector search by adding relationship details to improve semantic extraction
        expanded_query = clean_query
        for lbl in matched_labels:
            name = lbl.get("name", "")
            rel = lbl.get("relationship", "")
            if rel:
                expanded_query += f" {name} {rel}"
            else:
                expanded_query += f" {name}"
                
        expanded_query = expanded_query.strip()
        if expanded_query:
            chroma_hits = search_image_vectors(user_id, expanded_query, limit=20, filters=filters)
        
        # ── Step 3: Reranker Pass ─────────────────────────────────────────────
        all_candidate_ids = explicit_image_ids | {hit["image_id"] for hit in chroma_hits}
        
        # Fallback: if no candidates match via vector search but we have a date filter, retrieve all user images to filter them
        if not all_candidate_ids and (start_date and end_date):
            res_all = await db.table("images").select("id").eq("user_id", user_id).eq("is_deleted", False).execute()
            all_candidate_ids = {r["id"] for r in (res_all.data or [])}
            
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
            
        res = await db.table("images").select("id, public_url, scene_description, tags, category, document_details, landscape_details, custom_tags, created_at") \
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
            
            # Date filter validation
            if start_date and end_date:
                # Parse created_at UTC and convert to naive datetime for matching
                created_dt = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")).replace(tzinfo=None)
                if not (start_date <= created_dt <= end_date):
                    continue
                    
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
            
            # Boost for custom manual tags (high importance)
            custom_tags_list = r.get("custom_tags") or []
            custom_tag_matches = 0
            for tag in custom_tags_list:
                if isinstance(tag, str) and tag.lower() in query_lower:
                    custom_tag_matches += 1
            
            rerank_score += (0.50 * custom_tag_matches)
            
            # Boost for category matches
            category_val = r.get("category") or "other"
            if category_val in query_lower:
                rerank_score += 0.20
                
            # Boost for document text matches
            doc_details = r.get("document_details") or {}
            if doc_details:
                for k, v in doc_details.items():
                    if isinstance(v, str) and v.lower() in query_lower:
                        rerank_score += 0.15
                        
            # Boost for landscape text matches
            land_details = r.get("landscape_details") or {}
            if land_details:
                for k, v in land_details.items():
                    if isinstance(v, str) and v.lower() in query_lower:
                        rerank_score += 0.15
            
            candidates_to_rank.append({
                "id": img_id,
                "public_url": r["public_url"],
                "scene_description": r.get("scene_description") or "",
                "created_at": r["created_at"],
                "tags": tags_dict,
                "detected_faces": detected_in_img,
                "category": category_val,
                "document_details": doc_details,
                "landscape_details": land_details,
                "custom_tags": custom_tags_list,
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
                "tags": c["tags"],
                "category": c["category"],
                "document_details": c["document_details"],
                "landscape_details": c["landscape_details"],
                "custom_tags": c["custom_tags"]
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
        
        if gemini_res.get("error"):
            fallback_images = []
            for c in candidates_to_rank[:10]:
                created_at_dt = datetime.fromisoformat(c["created_at"].replace("Z", "+00:00"))
                fallback_images.append(
                    SearchResultImage(
                        id=c["id"],
                        public_url=c["public_url"],
                        scene_description=c["scene_description"],
                        created_at=created_at_dt,
                        score=c["rerank_score"],
                        detected_faces=c["detected_faces"],
                        tags=c["tags"],
                        category=c["category"],
                        document_details=c["document_details"],
                        landscape_details=c["landscape_details"],
                        custom_tags=c["custom_tags"]
                    )
                )
            return {
                "response_text": "I found these potential matches in your gallery, but search filters couldn't be fully refined due to temporary rate limits.",
                "images": fallback_images
            }

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
                        tags=c["tags"],
                        category=c["category"],
                        document_details=c["document_details"],
                        landscape_details=c["landscape_details"],
                        custom_tags=c["custom_tags"]
                    )
                )
                
        return {
            "response_text": ai_response_text or f"I found {len(selected_candidates)} matching photos.",
            "images": selected_candidates
        }


    @staticmethod
    async def get_labeled_people(user_id: str, page: int = None, limit: int = None) -> list[dict]:
        db = await get_db()
        
        # 1. Fetch all face labels for the user
        query = db.table("face_labels").select("id, name, relationship").eq("user_id", user_id).order("name")
        if page is not None and limit is not None:
            start = (page - 1) * limit
            end = start + limit - 1
            query = query.range(start, end)
            
        lbl_res = await query.execute()
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
