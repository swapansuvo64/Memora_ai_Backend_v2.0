import logging
from config.db import get_db

logger = logging.getLogger(__name__)

async def upload_image(user_id: str, file_name: str, file_bytes: bytes, content_type: str) -> dict:
    """
    Asynchronously uploads an image file to the Supabase Storage bucket 'Memora ai'.
    """
    db = await get_db()
    bucket_name = "Memora ai"
    storage_path = f"uploads/{user_id}/{file_name}"
    
    try:
        # Perform async upload
        await db.storage.from_(bucket_name).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        
        # Get public URL (this is sync in supabase-py, doesn't need await)
        public_url = db.storage.from_(bucket_name).get_public_url(storage_path)
        
        return {
            "storage_path": storage_path,
            "public_url": public_url
        }
    except Exception as e:
        logger.error(f"Async Supabase storage upload error: {e}")
        raise e
