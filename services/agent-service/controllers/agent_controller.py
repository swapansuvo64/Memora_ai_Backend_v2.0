import logging
import json
import numpy as np
from fastapi import HTTPException
from config.db import get_db
from models.agent import QueueItem, SearchResultImage
from utils.chroma_client import search_image_vectors, index_image_vector

logger = logging.getLogger(__name__)

def deserialize_embedding(data) -> np.ndarray:
    """
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
        return np.frombuffer(byte_data, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to deserialize embedding: {e}")
        # Return a zero vector fallback to prevent crashes
        return np.zeros(128, dtype=np.float32)

class AgentController:
    @staticmethod
    async def get_queue(user_id: str) -> list[QueueItem]:
        db = await get_db()
        
        # Select pending queue items
        res = await db.table("face_queue").select(
            "face_id, temporary_name, faces(face_thumbnail_url)"
        ).eq("user_id", user_id).eq("status", "pending").order("created_at").execute()
        
        queue_items = []
        for row in res.data:
            faces = row.get("faces")
            face_thumbnail_url = faces.get("face_thumbnail_url") if faces else ""
            queue_items.append(
                QueueItem(
                    face_id=row["face_id"],
                    temporary_name=row["temporary_name"],
                    face_thumbnail_url=face_thumbnail_url
                )
            )
        return queue_items

    @staticmethod
    async def resolve_face(user_id: str, face_id: str, label_name: str) -> dict:
        label_name_cleaned = label_name.strip()
        if not label_name_cleaned:
            raise HTTPException(status_code=400, detail="Label name cannot be empty")

        db = await get_db()

        # 1. Retrieve or create face_labels record
        lbl_res = await db.table("face_labels").select("id").eq("user_id", user_id).eq("name", label_name_cleaned).execute()
        if lbl_res.data:
            label_id = lbl_res.data[0]["id"]
        else:
            ins_lbl = await db.table("face_labels").insert({
                "user_id": user_id,
                "name": label_name_cleaned
            }).execute()
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
            img_res = await db.table("images").select("scene_description").eq("id", img_id).execute()
            desc = img_res.data[0]["scene_description"] if img_res.data else ""
            
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
                detected_faces=detected_names
            )

        return {
            "status": "success",
            "message": f"Resolved face. Automatically tagged {propagated_count} other similar faces.",
            "propagated_count": propagated_count
        }

    @staticmethod
    async def search(user_id: str, query: str) -> list[SearchResultImage]:
        # 1. Search vector database
        chroma_hits = search_image_vectors(user_id, query)
        if not chroma_hits:
            return []

        # 2. Extract image IDs and map details
        hit_ids = [hit["image_id"] for hit in chroma_hits]
        score_map = {hit["image_id"]: hit["score"] for hit in chroma_hits}
        faces_map = {hit["image_id"]: hit["detected_faces"] for hit in chroma_hits}

        if not hit_ids:
            return []

        # 3. Retrieve active images from PostgreSQL matching these IDs
        db = await get_db()
        res = await db.table("images").select("id, public_url, scene_description, created_at").eq("user_id", user_id).eq("is_deleted", False).in_("id", hit_ids).execute()
        
        db_results = []
        for r in res.data:
            img_id = r["id"]
            score = score_map.get(img_id, 0.0)
            faces = faces_map.get(img_id, [])
            created_at_dt = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
            
            db_results.append(
                SearchResultImage(
                    id=r["id"],
                    public_url=r["public_url"],
                    scene_description=r.get("scene_description"),
                    created_at=created_at_dt,
                    score=score,
                    detected_faces=faces
                )
            )

        # Sort by vector similarity score descending
        db_results.sort(key=lambda x: x.score, reverse=True)
        return db_results
