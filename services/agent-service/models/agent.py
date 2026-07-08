from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import datetime

class ResolveRequest(BaseModel):
    face_id: UUID
    name: str
    relationship: Optional[str] = None

class QueueItem(BaseModel):
    face_id: UUID
    temporary_name: str
    face_thumbnail_url: str
    image_url: Optional[str] = None


class SearchResultImage(BaseModel):
    id: UUID
    public_url: str
    scene_description: Optional[str] = None
    created_at: datetime
    score: float
    detected_faces: List[str]
    tags: Optional[dict] = None
    category: Optional[str] = 'other'
    document_details: Optional[dict] = None
    landscape_details: Optional[dict] = None
    custom_tags: List[str] = []


class ChatMessagePayload(BaseModel):
    role: str
    text: str


class ChatSearchRequest(BaseModel):
    query: str
    history: List[ChatMessagePayload] = []
    filters: Optional[dict] = None


class ChatSearchResponse(BaseModel):
    response_text: str
    images: List[SearchResultImage]

