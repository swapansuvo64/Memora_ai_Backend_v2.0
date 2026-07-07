from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import datetime

class ResolveRequest(BaseModel):
    face_id: UUID
    name: str

class QueueItem(BaseModel):
    face_id: UUID
    temporary_name: str
    face_thumbnail_url: str

class SearchResultImage(BaseModel):
    id: UUID
    public_url: str
    scene_description: Optional[str] = None
    created_at: datetime
    score: float
    detected_faces: List[str]
