import json
import logging
import uuid
import asyncio
import numpy as np
import redis
import threading
from supabase import create_async_client, AsyncClient
from config.settings import settings
from config.db import get_db, init_db
from utils.face_detector import detect_and_crop_faces, compare_faces
from utils.aws_bedrock import generate_scene_description
from utils.chroma_client import index_image_vector, deindex_image_vector

logger = logging.getLogger(__name__)

def serialize_embedding(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()

def deserialize_embedding(data) -> np.ndarray:
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
        logger.error(f"Failed to deserialize embedding in worker: {e}")
        return np.zeros(128, dtype=np.float32)

class RedisEventWorker:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
        self.pubsub = self.redis_client.pubsub()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.pubsub.subscribe(**{
            "image_uploaded": self.handle_upload_event,
            "permanent_delete": self.handle_delete_event
        })
        self.thread = threading.Thread(target=self.run_loop, daemon=True)
        self.thread.start()
        logger.info("Background Redis Event Worker thread started")

    def run_loop(self):
        while self.running:
            try:
                message = self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    logger.info(f"Worker received event queue item: {message}")
            except Exception as e:
                logger.error(f"Worker loop error: {e}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.pubsub.unsubscribe()
        logger.info("Background Redis Event Worker thread stopped")

    def handle_upload_event(self, message):
        # Runs the async handler synchronously inside the thread event loop
        asyncio.run(self.async_handle_upload_event(message))

    def handle_delete_event(self, message):
        asyncio.run(self.async_handle_delete_event(message))

    async def async_handle_upload_event(self, message):
        try:
            data = json.loads(message["data"])
            image_id = data["image_id"]
            user_id = data["user_id"]
            logger.info(f"Processing async upload for image {image_id} (user: {user_id})")
            
            # Ensure database is initialized in this thread's event loop
            try:
                db = await get_db()
            except Exception:
                # If not initialized, initialize it here
                db = await init_db()

            # 1. Fetch image details
            res = await db.table("images").select("storage_path, mime_type").eq("id", image_id).eq("user_id", user_id).execute()
            if not res.data:
                logger.error(f"Image {image_id} metadata not found in database.")
                return
                
            storage_path = res.data[0]["storage_path"]
            mime_type = res.data[0]["mime_type"]

            # 2. Download raw image bytes from storage
            image_bytes = await db.storage.from_("Memora ai").download(storage_path)

            # 3. Process faces
            detected_faces = detect_and_crop_faces(image_bytes)
            detected_names = []
            
            # 4. Check face matches
            for index, face in enumerate(detected_faces):
                embedding = face["embedding"]
                box = face["box"]
                thumb_bytes = face["thumbnail_bytes"]
                
                # Fetch known embeddings for this user
                known_res = await db.table("faces").select("id, embedding_vector, label_id, face_labels(name)").eq("user_id", user_id).not_.is_("label_id", "null").execute()
                
                known_embeddings = []
                known_labels = []
                for kr in known_res.data:
                    known_embeddings.append(deserialize_embedding(kr["embedding_vector"]))
                    fl = kr.get("face_labels")
                    name = fl.get("name") if isinstance(fl, dict) else None
                    known_labels.append((kr["label_id"], name))
                
                # Match
                match_idx = compare_faces(known_embeddings, embedding, tolerance=0.6)
                
                face_id = str(uuid.uuid4())
                label_id = None
                
                # Upload thumbnail crop
                thumbnail_path = f"cropped_faces/{user_id}/{face_id}.jpg"
                await db.storage.from_("Memora ai").upload(
                    path=thumbnail_path,
                    file=thumb_bytes,
                    file_options={"content-type": "image/jpeg", "upsert": "true"}
                )
                face_thumbnail_url = db.storage.from_("Memora ai").get_public_url(thumbnail_path)
                
                serialized_emb = serialize_embedding(embedding)
                
                if match_idx != -1:
                    label_id, label_name = known_labels[match_idx]
                    detected_names.append(label_name)
                    logger.info(f"Matched known face: {label_name}")
                else:
                    temp_name = f"Subject_Face_{uuid.uuid4().hex[:6]}"
                    detected_names.append(temp_name)
                    logger.info(f"Unmatched face. Queueing temporary name: {temp_name}")
                
                # Save face
                await db.table("faces").insert({
                    "id": face_id,
                    "image_id": image_id,
                    "user_id": user_id,
                    "box_top": box["top"],
                    "box_right": box["right"],
                    "box_bottom": box["bottom"],
                    "box_left": box["left"],
                    "embedding_vector": serialized_emb.hex(), # insert as hex format for bytea in Supabase
                    "face_thumbnail_url": face_thumbnail_url,
                    "label_id": label_id
                }).execute()
                
                # Insert to face resolution queue if unmatched
                if match_idx == -1:
                    await db.table("face_queue").insert({
                        "face_id": face_id,
                        "user_id": user_id,
                        "temporary_name": temp_name
                    }).execute()
            
            # 5. Generate Scene Context using Claude
            scene_description = generate_scene_description(image_bytes, mime_type)
            
            # 6. Save description to DB
            await db.table("images").update({"scene_description": scene_description}).eq("id", image_id).execute()
            
            # 7. Index in ChromaDB
            index_image_vector(
                image_id=image_id,
                user_id=user_id,
                scene_description=scene_description,
                detected_faces=detected_names
            )
            logger.info(f"Successfully processed and indexed image {image_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle async upload event: {e}", exc_info=True)

    async def async_handle_delete_event(self, message):
        try:
            data = json.loads(message["data"])
            image_id = data["image_id"]
            user_id = data["user_id"]
            files_to_delete = data["files_to_delete"]
            logger.info(f"Processing async permanent delete for image {image_id} (user: {user_id})")
            
            # 1. De-index from ChromaDB
            deindex_image_vector(image_id, user_id)
            
            # 2. Delete files from storage
            try:
                db = await get_db()
            except Exception:
                db = await init_db()
                
            if files_to_delete:
                await db.storage.from_("Memora ai").remove(files_to_delete)
                logger.info(f"Deleted files from Supabase bucket: {files_to_delete}")
                
            logger.info(f"Successfully completed async de-indexing/cleanup for image {image_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle async delete event: {e}", exc_info=True)
