import io
import logging
import numpy as np
from PIL import Image
import face_recognition

logger = logging.getLogger(__name__)

def detect_and_crop_faces(image_bytes: bytes) -> list[dict]:
    """
    Detects faces in an image, computes their 128-dimensional encodings, 
    and crops them to return thumbnail bytes.
    """
    detected_faces = []
    
    try:
        # Load image with PIL and convert to numpy RGB array
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_arr = np.array(img)
        
        # Detect locations and encodings
        face_locations = face_recognition.face_locations(img_arr)
        face_encodings = face_recognition.face_encodings(img_arr, face_locations)
        
        for i, (location, encoding) in enumerate(zip(face_locations, face_encodings)):
            top, right, bottom, left = location
            
            # Crop image to PIL thumbnail
            # Add small padding to crop area
            width, height = img.size
            padding = int((bottom - top) * 0.1)
            
            crop_left = max(0, left - padding)
            crop_top = max(0, top - padding)
            crop_right = min(width, right + padding)
            crop_bottom = min(height, bottom + padding)
            
            cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
            
            # Convert crop to JPEG bytes
            buf = io.BytesIO()
            cropped_img.save(buf, format="JPEG", quality=90)
            face_thumbnail_bytes = buf.getvalue()
            
            detected_faces.append({
                "box": {
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "left": left
                },
                "embedding": encoding,
                "thumbnail_bytes": face_thumbnail_bytes
            })
            
        logger.info(f"Successfully processed image, detected {len(detected_faces)} faces")
        
    except Exception as e:
        logger.error(f"Error during face detection / processing: {e}")
        
    return detected_faces

def compare_faces(known_embeddings: list[np.ndarray], face_embedding: np.ndarray, tolerance: float = 0.6) -> int:
    """
    Compares a face embedding against a list of known face embeddings.
    Returns the index of the best match, or -1 if no matches meet the tolerance threshold.
    """
    if not known_embeddings:
        return -1
        
    # Calculate euclidean distance
    distances = face_recognition.face_distance(known_embeddings, face_embedding)
    
    best_match_idx = np.argmin(distances)
    if distances[best_match_idx] <= tolerance:
        return int(best_match_idx)
        
    return -1
