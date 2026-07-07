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
    async def list_images(user_id: str) -> list[ImageOut]:
        db = await get_db()
        res = await db.table("images").select("*").eq("user_id", user_id).eq("is_deleted", False).order("created_at", desc=True).execute()
        
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
    async def list_bin(user_id: str) -> list[ImageOut]:
        db = await get_db()
        res = await db.table("images").select("*").eq("user_id", user_id).eq("is_deleted", True).order("deleted_at", desc=True).execute()
        
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
