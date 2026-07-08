from datetime import datetime
from fastapi import HTTPException, status
from config.db import get_db
from models.gallery import ImageOut, FaceOut, ImageDetailResponse
from utils.supabase_storage import upload_image
from utils.redis_client import publish_event

class GalleryController:
    @staticmethod
    async def upload(user_id: str, file_name: str, file_bytes: bytes, content_type: str, file_size: int) -> ImageOut:
        # 1. Upload to Supabase Storage
        try:
            storage_details = await upload_image(user_id, file_name, file_bytes, content_type)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Upload to Supabase Storage failed: {e}"
            )

        storage_path = storage_details["storage_path"]
        public_url = storage_details["public_url"]

        # 2. Save metadata in DB
        db = await get_db()
        insert_res = await db.table("images").insert({
            "user_id": user_id,
            "storage_path": storage_path,
            "public_url": public_url,
            "mime_type": content_type,
            "size_bytes": file_size
        }).execute()
        
        if not insert_res.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register image metadata"
            )
            
        row = insert_res.data[0]
        
        created_at_dt = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        deleted_at_dt = datetime.fromisoformat(row["deleted_at"].replace("Z", "+00:00")) if row.get("deleted_at") else None
        
        image_out = ImageOut(
            id=row["id"],
            user_id=row["user_id"],
            storage_path=row["storage_path"],
            public_url=row["public_url"],
            mime_type=row.get("mime_type"),
            size_bytes=row.get("size_bytes"),
            width=row.get("width"),
            height=row.get("height"),
            scene_description=row.get("scene_description"),
            folder_id=row.get("folder_id"),
            tags=row.get("tags"),
            status=row.get("status"),
            category=row.get("category", "other"),
            document_details=row.get("document_details"),
            landscape_details=row.get("landscape_details"),
            custom_tags=row.get("custom_tags", []),
            is_deleted=row["is_deleted"],
            deleted_at=deleted_at_dt,
            created_at=created_at_dt
        )

        # 3. Publish event to Redis
        event_data = {
            "image_id": str(image_out.id),
            "user_id": str(image_out.user_id)
        }
        publish_event("image_uploaded", event_data)

        return image_out

    @staticmethod
    async def list_images(user_id: str, page: int = None, limit: int = None, folder_id: str = None) -> list[ImageOut]:
        db = await get_db()
        query = db.table("images").select("*").eq("user_id", user_id).eq("is_deleted", False)
        
        if folder_id is not None:
            if folder_id == "null":
                query = query.is_("folder_id", "null")
            else:
                query = query.eq("folder_id", folder_id)
                
        query = query.order("created_at", desc=True)
        
        if page is not None and limit is not None:
            start = (page - 1) * limit
            end = start + limit - 1
            query = query.range(start, end)
            
        res = await query.execute()
        
        images = []
        for row in res.data:
            created_at_dt = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            deleted_at_dt = datetime.fromisoformat(row["deleted_at"].replace("Z", "+00:00")) if row.get("deleted_at") else None
            images.append(
                ImageOut(
                    id=row["id"],
                    user_id=row["user_id"],
                    storage_path=row["storage_path"],
                    public_url=row["public_url"],
                    mime_type=row.get("mime_type"),
                    size_bytes=row.get("size_bytes"),
                    width=row.get("width"),
                    height=row.get("height"),
                    scene_description=row.get("scene_description"),
                    folder_id=row.get("folder_id"),
                    tags=row.get("tags"),
                    status=row.get("status"),
                    category=row.get("category", "other"),
                    document_details=row.get("document_details"),
                    landscape_details=row.get("landscape_details"),
                    custom_tags=row.get("custom_tags", []),
                    is_deleted=row["is_deleted"],
                    deleted_at=deleted_at_dt,
                    created_at=created_at_dt
                )
            )
        return images

    @staticmethod
    async def get_image_detail(user_id: str, image_id: str) -> ImageDetailResponse:
        db = await get_db()
        
        # 1. Fetch image
        res = await db.table("images").select("*").eq("id", image_id).eq("user_id", user_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Image not found")
            
        row = res.data[0]
        created_at_dt = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        deleted_at_dt = datetime.fromisoformat(row["deleted_at"].replace("Z", "+00:00")) if row.get("deleted_at") else None
        
        image_out = ImageOut(
            id=row["id"],
            user_id=row["user_id"],
            storage_path=row["storage_path"],
            public_url=row["public_url"],
            mime_type=row.get("mime_type"),
            size_bytes=row.get("size_bytes"),
            width=row.get("width"),
            height=row.get("height"),
            scene_description=row.get("scene_description"),
            folder_id=row.get("folder_id"),
            tags=row.get("tags"),
            status=row.get("status"),
            category=row.get("category", "other"),
            document_details=row.get("document_details"),
            landscape_details=row.get("landscape_details"),
            custom_tags=row.get("custom_tags", []),
            is_deleted=row["is_deleted"],
            deleted_at=deleted_at_dt,
            created_at=created_at_dt
        )

        # 2. Fetch faces
        faces_res = await db.table("faces").select("id, box_top, box_right, box_bottom, box_left, face_thumbnail_url, face_labels(name)").eq("image_id", image_id).execute()
        
        faces = []
        for f in faces_res.data:
            label_name = None
            if f.get("face_labels") and isinstance(f["face_labels"], dict):
                label_name = f["face_labels"].get("name")
                
            faces.append(
                FaceOut(
                    id=f["id"],
                    box_top=f["box_top"],
                    box_right=f["box_right"],
                    box_bottom=f["box_bottom"],
                    box_left=f["box_left"],
                    face_thumbnail_url=f.get("face_thumbnail_url"),
                    label_name=label_name
                )
            )

        return ImageDetailResponse(image=image_out, faces=faces)

    @staticmethod
    async def move_to_bin(user_id: str, image_id: str) -> dict:
        db = await get_db()
        now_str = datetime.utcnow().isoformat()
        res = await db.table("images").update({
            "is_deleted": True,
            "deleted_at": now_str
        }).eq("id", image_id).eq("user_id", user_id).execute()
        
        if not res.data:
            raise HTTPException(status_code=404, detail="Image not found")
        return {"status": "success", "message": "Moved image to bin"}

    @staticmethod
    async def list_bin(user_id: str, page: int = None, limit: int = None) -> list[ImageOut]:
        db = await get_db()
        query = db.table("images").select("*").eq("user_id", user_id).eq("is_deleted", True).order("deleted_at", desc=True)
        if page is not None and limit is not None:
            start = (page - 1) * limit
            end = start + limit - 1
            query = query.range(start, end)
            
        res = await query.execute()
        
        images = []
        for row in res.data:
            created_at_dt = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            deleted_at_dt = datetime.fromisoformat(row["deleted_at"].replace("Z", "+00:00")) if row.get("deleted_at") else None
            images.append(
                ImageOut(
                    id=row["id"],
                    user_id=row["user_id"],
                    storage_path=row["storage_path"],
                    public_url=row["public_url"],
                    mime_type=row.get("mime_type"),
                    size_bytes=row.get("size_bytes"),
                    width=row.get("width"),
                    height=row.get("height"),
                    scene_description=row.get("scene_description"),
                    folder_id=row.get("folder_id"),
                    tags=row.get("tags"),
                    status=row.get("status"),
                    category=row.get("category", "other"),
                    document_details=row.get("document_details"),
                    landscape_details=row.get("landscape_details"),
                    custom_tags=row.get("custom_tags", []),
                    is_deleted=row["is_deleted"],
                    deleted_at=deleted_at_dt,
                    created_at=created_at_dt
                )
            )
        return images

    @staticmethod
    async def restore_from_bin(user_id: str, image_id: str) -> dict:
        db = await get_db()
        res = await db.table("images").update({
            "is_deleted": False,
            "deleted_at": None
        }).eq("id", image_id).eq("user_id", user_id).execute()
        
        if not res.data:
            raise HTTPException(status_code=404, detail="Image not found")
        return {"status": "success", "message": "Restored image from bin"}

    @staticmethod
    async def permanent_delete(user_id: str, image_id: str) -> dict:
        db = await get_db()
        files_to_delete = []

        # 1. Fetch image path
        res = await db.table("images").select("storage_path").eq("id", image_id).eq("user_id", user_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        files_to_delete.append(res.data[0]["storage_path"])

        # 2. Fetch face thumbnails paths
        faces_res = await db.table("faces").select("face_thumbnail_url").eq("image_id", image_id).execute()
        for f in faces_res.data:
            thumb_url = f.get("face_thumbnail_url")
            if thumb_url:
                if "/public/Memora%20ai/" in thumb_url:
                    parts = thumb_url.split("/public/Memora%20ai/")
                    files_to_delete.append(parts[1])
                elif "/public/Memora ai/" in thumb_url:
                    parts = thumb_url.split("/public/Memora ai/")
                    files_to_delete.append(parts[1])

        # 3. Delete DB rows (cascading deletes face rows)
        delete_res = await db.table("images").delete().eq("id", image_id).eq("user_id", user_id).execute()
        if not delete_res.data:
            raise HTTPException(status_code=404, detail="Image not found")

        # 4. Trigger asynchronous file and vector purges in background
        event_data = {
            "image_id": image_id,
            "user_id": user_id,
            "files_to_delete": files_to_delete
        }
        publish_event("permanent_delete", event_data)

        return {"status": "success", "message": "Permanently deleted image, cleanup triggered"}

    @staticmethod
    async def move_to_bin_multiple(user_id: str, image_ids: list[str]) -> dict:
        if not image_ids:
            return {"status": "success", "message": "No images provided"}
        db = await get_db()
        now_str = datetime.utcnow().isoformat()
        res = await db.table("images").update({
            "is_deleted": True,
            "deleted_at": now_str
        }).eq("user_id", user_id).in_("id", image_ids).execute()
        return {"status": "success", "message": f"Moved {len(res.data) if res.data else 0} images to bin"}

    @staticmethod
    async def restore_from_bin_multiple(user_id: str, image_ids: list[str]) -> dict:
        if not image_ids:
            return {"status": "success", "message": "No images provided"}
        db = await get_db()
        res = await db.table("images").update({
            "is_deleted": False,
            "deleted_at": None
        }).eq("user_id", user_id).in_("id", image_ids).execute()
        return {"status": "success", "message": f"Restored {len(res.data) if res.data else 0} images from bin"}

    @staticmethod
    async def permanent_delete_multiple(user_id: str, image_ids: list[str]) -> dict:
        if not image_ids:
            return {"status": "success", "message": "No images provided"}
        db = await get_db()
        
        # 1. Fetch images to delete
        res = await db.table("images").select("id, storage_path").eq("user_id", user_id).in_("id", image_ids).execute()
        if not res.data:
            return {"status": "success", "message": "No images found"}
            
        found_images = res.data
        found_ids = [img["id"] for img in found_images]
        
        # 2. Fetch all face thumbnails associated with these images
        faces_res = await db.table("faces").select("image_id, face_thumbnail_url").eq("user_id", user_id).in_("image_id", found_ids).execute()
        
        # Map faces to their respective image_id in memory
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
                    
        # 3. Delete DB rows (cascading deletes faces & face_queue entries)
        await db.table("images").delete().eq("user_id", user_id).in_("id", found_ids).execute()
        
        # 4. Trigger async de-indexing and storage cleanup per image
        for img in found_images:
            img_id = img["id"]
            files_to_delete = [img["storage_path"]] + image_faces.get(img_id, [])
            
            event_data = {
                "image_id": img_id,
                "user_id": user_id,
                "files_to_delete": files_to_delete
            }
            publish_event("permanent_delete", event_data)
            
        return {"status": "success", "message": f"Permanently deleted {len(found_ids)} images, cleanup triggered"}


    @staticmethod
    async def list_folders(user_id: str) -> list[dict]:
        db = await get_db()
        res = await db.table("folders").select("*").eq("user_id", user_id).execute()
        return res.data if res.data else []

    @staticmethod
    async def create_folder(user_id: str, name: str, parent_folder_id: str = None) -> dict:
        db = await get_db()
        payload = {
            "user_id": user_id,
            "name": name
        }
        if parent_folder_id:
            payload["parent_folder_id"] = parent_folder_id
        res = await db.table("folders").insert(payload).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Failed to create folder")
        return res.data[0]

    @staticmethod
    async def delete_folder(user_id: str, folder_id: str) -> dict:
        db = await get_db()
        res = await db.table("folders").delete().eq("user_id", user_id).eq("id", folder_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Folder not found")
        return {"status": "success", "message": "Folder deleted"}

    @staticmethod
    async def move_image_to_folder(user_id: str, image_id: str, folder_id: str = None) -> dict:
        db = await get_db()
        # Parse 'null' string if passed from clients as null value
        f_id = None if folder_id == 'null' or not folder_id else folder_id
        res = await db.table("images").update({"folder_id": f_id}).eq("user_id", user_id).eq("id", image_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Re-index this image in vector search DB to sync the tags/folder changes
        # Trigger it in background by publishing an event
        event_data = {
            "image_id": image_id,
            "user_id": user_id
        }
        publish_event("image_uploaded", event_data)
        
        return {"status": "success", "message": "Moved image to folder successfully"}

    @staticmethod
    async def update_custom_tags(user_id: str, image_id: str, custom_tags: list[str]) -> dict:
        db = await get_db()
        res = await db.table("images").update({
            "custom_tags": custom_tags
        }).eq("id", image_id).eq("user_id", user_id).execute()
        
        if not res.data:
            raise HTTPException(status_code=404, detail="Image not found")
            
        # Re-index this image in vector search DB
        event_data = {
            "image_id": image_id,
            "user_id": user_id
        }
        publish_event("image_custom_tags_updated", event_data)
        
        return {"status": "success", "message": "Custom tags updated successfully"}

