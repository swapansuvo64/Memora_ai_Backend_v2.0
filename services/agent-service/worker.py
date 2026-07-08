import json
import time
import logging
from datetime import datetime, timezone, timedelta

import uuid
import asyncio
import numpy as np
import redis
import redis.asyncio as aioredis
import threading
from supabase._async.client import create_client, AsyncClient
from config.settings import settings
from config.db import get_db, init_db

from utils.face_detector import detect_and_crop_faces, compare_faces
from utils.gcp_vertex import generate_scene_description
from utils.chroma_client import index_image_vector, deindex_image_vector

logger = logging.getLogger(__name__)

async def get_worker_db() -> AsyncClient:
    return await create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_KEY
    )

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
            
        # Handle backward compatibility: if byte_data has 1024 bytes, it is a hex string stored as ASCII text.
        if len(byte_data) == 1024:
            try:
                byte_data = bytes.fromhex(byte_data.decode('ascii'))
            except Exception as double_err:
                logger.error(f"Failed to double-decode hex ASCII embedding: {double_err}")

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
        self.thread = threading.Thread(target=self.run_loop, daemon=True)
        self.thread.start()
        logger.info("Background Redis Event Worker list-based queue thread started")

    def run_loop(self):
        logger.info("Background Redis Event Worker list-based queue loop starting")
        last_cleanup = 0
        while self.running:
            try:
                # 30-day auto-cleanup check every hour
                now_ts = time.time()
                if now_ts - last_cleanup > 3600:
                    last_cleanup = now_ts
                    try:
                        asyncio.run(self.cleanup_expired_deleted_images())
                    except Exception as clean_err:
                        logger.error(f"Worker auto-cleanup error: {clean_err}")

                # BRPOP takes a list of keys and blocks for timeout seconds.
                # Returns a tuple of (key, value) or None if timeout.
                result = self.redis_client.brpop(["image_uploaded", "permanent_delete", "image_custom_tags_updated"], timeout=1.0)
                if result:
                    queue_name, data_str = result
                    logger.info(f"Worker received queue item from list '{queue_name}': {data_str}")
                    # Construct message payload format compatible with original handlers
                    # The original handler expects message["data"] to contain a JSON string
                    message = {"data": data_str}
                    if queue_name == "image_uploaded":
                        self.handle_upload_event(message)
                    elif queue_name == "permanent_delete":
                        self.handle_delete_event(message)
                    elif queue_name == "image_custom_tags_updated":
                        self.handle_custom_tags_updated_event(message)
            except Exception as e:
                logger.error(f"Worker list-based queue loop error: {e}")
                time.sleep(1.0)


    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Background Redis Event Worker thread stopped")


    def handle_upload_event(self, message):
        # Runs the async handler synchronously inside the thread event loop
        asyncio.run(self.async_handle_upload_event(message))

    def handle_delete_event(self, message):
        asyncio.run(self.async_handle_delete_event(message))

    def handle_custom_tags_updated_event(self, message):
        asyncio.run(self.async_handle_custom_tags_updated_event(message))

    async def async_handle_upload_event(self, message):
        db = await get_worker_db()
        try:
            data = json.loads(message["data"])
            image_id = data["image_id"]
            user_id = data["user_id"]
            logger.info(f"Processing async upload for image {image_id} (user: {user_id})")

            # 1. Fetch image details
            res = await db.table("images").select("storage_path, mime_type").eq("id", image_id).eq("user_id", user_id).execute()
            if not res.data:
                logger.error(f"Image {image_id} metadata not found in database.")
                return
                
            storage_path = res.data[0]["storage_path"]
            mime_type = res.data[0]["mime_type"]

            # Clear any existing faces/queue entries for this image in case of reprocessing / retries
            await db.table("faces").delete().eq("image_id", image_id).eq("user_id", user_id).execute()

            # 2. Download raw image bytes from storage
            image_bytes = await db.storage.from_("Memora ai").download(storage_path)

            # Load image with PIL to extract exact dimensions (width, height)
            from PIL import Image
            import io
            img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            width, height = img_pil.size

            # 3. Detect faces and generate face crops/embeddings
            faces = detect_and_crop_faces(image_bytes)
            logger.info(f"Successfully processed image, detected {len(faces)} faces")

            # 4. Process each face
            detected_names = []
            labeled_list = []
            if faces:
                # Query all known face signatures for this user
                existing_faces_res = await db.table("faces").select("id, embedding_vector, label_id, face_labels(name, relationship)").eq("user_id", user_id).not_.is_("label_id", "null").execute()
                known_faces = existing_faces_res.data or []

                for face in faces:
                    # Search match
                    face_id = str(uuid.uuid4())
                    matched_label_id = None
                    matched_name = None
                    matched_rel = None

                    face_embedding = face["embedding"] # 128 float vector

                    # Linear search search matching signatures
                    best_match = None
                    best_dist = 0.6  # tolerance threshold (lower is better, FaceNet matches < 0.6)

                    for k_face in known_faces:
                        k_vector_bytes = k_face.get("embedding_vector")
                        if k_vector_bytes:
                            k_vector = deserialize_embedding(k_vector_bytes)
                            dist = np.linalg.norm(face_embedding - k_vector)
                            if dist < best_dist:
                                best_dist = dist
                                best_match = k_face

                    if best_match:
                        matched_label_id = best_match["label_id"]
                        fl = best_match.get("face_labels")
                        if isinstance(fl, dict):
                            matched_name = fl.get("name")
                            matched_rel = fl.get("relationship")
                        elif isinstance(fl, list) and fl:
                            matched_name = fl[0].get("name")
                            matched_rel = fl[0].get("relationship")
                            
                        if matched_name:
                            logger.info(f"Matched face with label: {matched_name} (label_id: {matched_label_id})")
                            labeled_list.append({
                                "name": matched_name,
                                "relationship": matched_rel
                            })

                    # Upload face thumbnail to Supabase Storage
                    crop_filename = f"{face_id}.jpg"
                    crop_path = f"cropped_faces/{user_id}/{crop_filename}"
                    await db.storage.from_("Memora ai").upload(crop_path, face["thumbnail_bytes"], file_options={"content-type": "image/jpeg"})
                    
                    # Fetch public url for the face
                    face_thumbnail_url = await db.storage.from_("Memora ai").get_public_url(crop_path)

                    box = face["box"]
                    if matched_label_id:
                        # Insert recognized face directly
                        await db.table("faces").insert({
                            "id": face_id,
                            "user_id": user_id,
                            "image_id": image_id,
                            "box_top": box["top"],
                            "box_right": box["right"],
                            "box_bottom": box["bottom"],
                            "box_left": box["left"],
                            "embedding_vector": f"\\x{serialize_embedding(face_embedding).hex()}",
                            "label_id": matched_label_id,
                            "face_thumbnail_url": face_thumbnail_url
                        }).execute()
                        detected_names.append(matched_name)
                    else:
                        # Unmatched face, insert with a temporary label queue entry
                        temp_name = f"Subject_Face_{uuid.uuid4().hex[:6]}"
                        logger.info(f"Unmatched face. Queueing temporary name: {temp_name}")

                        await db.table("faces").insert({
                            "id": face_id,
                            "user_id": user_id,
                            "image_id": image_id,
                            "box_top": box["top"],
                            "box_right": box["right"],
                            "box_bottom": box["bottom"],
                            "box_left": box["left"],
                            "embedding_vector": f"\\x{serialize_embedding(face_embedding).hex()}",
                            "face_thumbnail_url": face_thumbnail_url
                        }).execute()

                        await db.table("face_queue").insert({
                            "face_id": face_id,
                            "user_id": user_id,
                            "temporary_name": temp_name,
                            "status": "pending"
                        }).execute()
                        detected_names.append(temp_name)
                        
                        # Publish queue_ready event to notify the frontend via SSE
                        try:
                            rc = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
                            rc.publish(f"queue_events:{user_id}", json.dumps({
                                "event": "queue_ready",
                                "face_id": face_id,
                                "temporary_name": temp_name,
                                "face_thumbnail_url": face_thumbnail_url
                            }))
                            rc.close()
                        except Exception as pub_err:
                            logger.error(f"Failed to publish queue_ready event: {pub_err}")

            # 5. Generate AI descriptive caption/labels
            scene_desc_text = "An uploaded image."
            tags = {}
            category = "other"
            document_details = None
            landscape_details = None
            detected_humans = []
            
            try:
                analysis_result = generate_scene_description(image_bytes, mime_type, labeled_list)
                scene_desc_text = analysis_result.get("description", "An uploaded image.")
                tags = analysis_result.get("tags", {})
                category = analysis_result.get("category", "other")
                document_details = analysis_result.get("document_details")
                landscape_details = analysis_result.get("landscape_details")
                detected_humans = analysis_result.get("detected_humans") or []
            except Exception as bedrock_err:
                logger.error(f"Failed to generate description via Google Vertex: {bedrock_err}")

            # Process detected humans without clear faces (e.g. backward facing)
            if detected_humans:
                for human in detected_humans:
                    box_norm = human.get("box") # [ymin, xmin, ymax, xmax] (0-1000 scale)
                    desc_helper = human.get("description", "person")
                    
                    if not box_norm or len(box_norm) < 4:
                        continue
                    
                    # Convert normalized coordinates to pixel values
                    ymin, xmin, ymax, xmax = box_norm
                    h_top = int((ymin * height) / 1000)
                    h_left = int((xmin * width) / 1000)
                    h_bottom = int((ymax * height) / 1000)
                    h_right = int((xmax * width) / 1000)
                    
                    # Ensure coordinates are within bounds
                    h_top = max(0, min(height - 1, h_top))
                    h_left = max(0, min(width - 1, h_left))
                    h_bottom = max(0, min(height, h_bottom))
                    h_right = max(0, min(width, h_right))
                    
                    if h_bottom <= h_top or h_right <= h_left:
                        continue
                        
                    # Check overlap with standard face_locations
                    human_box = {"top": h_top, "right": h_right, "bottom": h_bottom, "left": h_left}
                    overlap_found = False
                    for face in faces:
                        f_box = face["box"]
                        f_center_y = (f_box["top"] + f_box["bottom"]) / 2
                        f_center_x = (f_box["left"] + f_box["right"]) / 2
                        if (human_box["left"] <= f_center_x <= human_box["right"] and 
                            human_box["top"] <= f_center_y <= human_box["bottom"]):
                            overlap_found = True
                            break
                            
                    if overlap_found:
                        logger.info(f"Skipping detected human '{desc_helper}' as it overlaps with an existing face")
                        continue
                        
                    # Crop this human figure as thumbnail
                    try:
                        cropped_img = img_pil.crop((h_left, h_top, h_right, h_bottom))
                        buf = io.BytesIO()
                        cropped_img.save(buf, format="JPEG", quality=90)
                        human_thumbnail_bytes = buf.getvalue()
                        
                        # Generate UUID and path
                        face_id = str(uuid.uuid4())
                        crop_filename = f"{face_id}.jpg"
                        crop_path = f"cropped_faces/{user_id}/{crop_filename}"
                        
                        # Upload to Supabase Storage
                        await db.storage.from_("Memora ai").upload(crop_path, human_thumbnail_bytes, file_options={"content-type": "image/jpeg"})
                        
                        # Fetch public url
                        face_thumbnail_url = await db.storage.from_("Memora ai").get_public_url(crop_path)
                        
                        # Insert in faces table (with embedding_vector = NULL)
                        await db.table("faces").insert({
                            "id": face_id,
                            "user_id": user_id,
                            "image_id": image_id,
                            "box_top": h_top,
                            "box_right": h_right,
                            "box_bottom": h_bottom,
                            "box_left": h_left,
                            "embedding_vector": None,
                            "label_id": None,
                            "face_thumbnail_url": face_thumbnail_url
                        }).execute()
                        
                        # Insert in face_queue
                        temp_name = f"Silhouette_{uuid.uuid4().hex[:6]}"
                        logger.info(f"Adding non-clear face to queue: {temp_name}")
                        await db.table("face_queue").insert({
                            "face_id": face_id,
                            "user_id": user_id,
                            "temporary_name": temp_name,
                            "status": "pending"
                        }).execute()
                        detected_names.append(temp_name)
                        
                        # Publish SSE event
                        try:
                            rc = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
                            rc.publish(f"queue_events:{user_id}", json.dumps({
                                "event": "queue_ready",
                                "face_id": face_id,
                                "temporary_name": temp_name,
                                "face_thumbnail_url": face_thumbnail_url
                            }))
                            rc.close()
                        except Exception as pub_err:
                            logger.error(f"Failed to publish queue_ready event: {pub_err}")
                            
                    except Exception as crop_err:
                        logger.error(f"Failed to crop and process human silhouette: {crop_err}")

            # 6. Save description, tags, category, and details to DB
            await db.table("images").update({
                "scene_description": scene_desc_text,
                "tags": tags,
                "category": category,
                "document_details": document_details,
                "landscape_details": landscape_details,
                "width": width,
                "height": height,
                "status": "ready"
            }).eq("id", image_id).execute()
            
            # 7. Index in ChromaDB
            index_image_vector(
                image_id=image_id,
                user_id=user_id,
                scene_description=scene_desc_text,
                detected_faces=detected_names,
                tags=tags,
                category=category,
                document_details=document_details,
                landscape_details=landscape_details,
                custom_tags=[]
            )
            logger.info(f"Successfully processed and indexed image {image_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle async upload event: {e}", exc_info=True)
        finally:
            await db.postgrest.session.aclose()

    async def async_handle_custom_tags_updated_event(self, message):
        db = await get_worker_db()
        try:
            data = json.loads(message["data"])
            image_id = data["image_id"]
            user_id = data["user_id"]
            logger.info(f"Processing custom tags update for image {image_id} (user: {user_id})")
            
            # Fetch all details for this image
            res = await db.table("images").select("scene_description, tags, category, document_details, landscape_details, custom_tags").eq("id", image_id).eq("user_id", user_id).execute()
            if not res.data:
                logger.error(f"Image {image_id} not found in database for re-indexing.")
                return
                
            row = res.data[0]
            desc = row.get("scene_description") or ""
            tags = row.get("tags") or {}
            category = row.get("category") or "other"
            document_details = row.get("document_details")
            landscape_details = row.get("landscape_details")
            custom_tags = row.get("custom_tags") or []
            
            # Fetch all labels and pending queue names for this image
            faces_details = await db.table("faces").select(
                "id, face_labels(name), face_queue(temporary_name, status)"
            ).eq("image_id", image_id).execute()
            
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
                    
            # Update index in ChromaDB
            index_image_vector(
                image_id=image_id,
                user_id=user_id,
                scene_description=desc,
                detected_faces=detected_names,
                tags=tags,
                category=category,
                document_details=document_details,
                landscape_details=landscape_details,
                custom_tags=custom_tags
            )
            logger.info(f"Successfully reindexed image {image_id} after custom tags update")
            
        except Exception as e:
            logger.error(f"Failed to handle custom tags update event: {e}", exc_info=True)
        finally:
            await db.postgrest.session.aclose()

    async def async_handle_delete_event(self, message):
        db = await get_worker_db()
        try:
            data = json.loads(message["data"])
            image_id = data["image_id"]
            user_id = data["user_id"]
            files_to_delete = data["files_to_delete"]
            logger.info(f"Processing async permanent delete for image {image_id} (user: {user_id})")
            
            # 1. De-index from ChromaDB
            deindex_image_vector(image_id, user_id)
            
            # 2. Delete files from storage
            if files_to_delete:
                await db.storage.from_("Memora ai").remove(files_to_delete)
                logger.info(f"Deleted files from Supabase bucket: {files_to_delete}")
                
            logger.info(f"Successfully completed async de-indexing/cleanup for image {image_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle async delete event: {e}", exc_info=True)
        finally:
            await db.postgrest.session.aclose()

    async def cleanup_expired_deleted_images(self):
        db = await get_worker_db()
        try:
            logger.info("Auto-cleanup: Checking for expired soft-deleted images...")
            
            # Calculate 30-day cutoff
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            
            # 1. Fetch images where is_deleted = True and deleted_at < cutoff
            res = await db.table("images").select("id, user_id, storage_path").eq("is_deleted", True).lt("deleted_at", cutoff).execute()
            if not res.data:
                logger.info("Auto-cleanup: No expired soft-deleted images found.")
                return
                
            found_images = res.data
            found_ids = [img["id"] for img in found_images]
            logger.info(f"Auto-cleanup: Found {len(found_ids)} expired images to purge permanently: {found_ids}")
            
            # 2. Fetch associated face thumbnails for storage deletion
            faces_res = await db.table("faces").select("image_id, face_thumbnail_url").in_("image_id", found_ids).execute()
            
            # Group face thumbnails to delete by image
            image_faces = {img_id: [] for img_id in found_ids}
            for f in faces_res.data:
                img_id = f.get("image_id")
                thumb_url = f.get("face_thumbnail_url")
                if img_id and thumb_url:
                    thumb_path = None
                    if "/public/Memora%20ai/" in thumb_url:
                        thumb_path = thumb_url.split("/public/Memora%20ai/")[1]
                    elif "/public/Memora ai/" in thumb_url:
                        thumb_path = thumb_url.split("/public/Memora ai/")[1]
                    if thumb_path and img_id in image_faces:
                        image_faces[img_id].append(thumb_path)
            
            # 3. De-index from ChromaDB and delete from Supabase storage per image
            for img in found_images:
                img_id = img["id"]
                user_id = img["user_id"]
                
                # De-index from ChromaDB
                try:
                    deindex_image_vector(img_id, user_id)
                except Exception as e:
                    logger.error(f"Auto-cleanup: Failed to deindex image {img_id} from ChromaDB: {e}")
                    
                # Delete files from Supabase Storage
                files_to_delete = [img["storage_path"]] + image_faces.get(img_id, [])
                try:
                    if files_to_delete:
                        await db.storage.from_("Memora ai").remove(files_to_delete)
                except Exception as e:
                    logger.error(f"Auto-cleanup: Failed to delete files {files_to_delete} from Supabase: {e}")
                    
            # 4. Delete DB rows (cascading deletes faces & face_queue entries)
            await db.table("images").delete().in_("id", found_ids).execute()
            logger.info(f"Auto-cleanup: Permanently purged {len(found_ids)} images from database, vector store, and storage bucket.")
            
        except Exception as e:
            logger.error(f"Auto-cleanup error: {e}", exc_info=True)
        finally:
            await db.postgrest.session.aclose()

