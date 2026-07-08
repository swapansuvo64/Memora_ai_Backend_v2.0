import logging
import re
from config.db import get_db

logger = logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    # Replace narrow non-breaking spaces (\u202f) and non-breaking spaces (\u00a0) with underscore
    cleaned = filename.replace('\u202f', '_').replace('\u00a0', '_')
    # Keep only ASCII alphanumeric, dots, underscores, dashes, and regular spaces
    cleaned = re.sub(r'[^a-zA-Z0-9._\- ]', '', cleaned)
    return cleaned.strip()

async def upload_image(user_id: str, file_name: str, file_bytes: bytes, content_type: str) -> dict:
    """
    Asynchronously uploads an image file to the Supabase Storage bucket 'Memora ai'.
    """
    db = await get_db()
    bucket_name = "Memora ai"
    clean_name = sanitize_filename(file_name)
    storage_path = f"uploads/{user_id}/{clean_name}"

    
    try:
        # Perform async upload
        await db.storage.from_(bucket_name).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        
        # Get public URL
        public_url = await db.storage.from_(bucket_name).get_public_url(storage_path)

        
        return {
            "storage_path": storage_path,
            "public_url": public_url
        }
    except Exception as e:
        logger.error(f"Async Supabase storage upload error: {e}")
        raise e
