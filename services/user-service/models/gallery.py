from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import datetime

class ImageOut(BaseModel):
    id: UUID
    user_id: UUID
    storage_path: str
    public_url: str
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    scene_description: Optional[str] = None
    folder_id: Optional[UUID] = None
    tags: Optional[dict] = None
    status: Optional[str] = None
    category: str = 'other'
    document_details: Optional[dict] = None
    landscape_details: Optional[dict] = None
    custom_tags: List[str] = []
    is_deleted: bool
    deleted_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class FaceOut(BaseModel):
    id: UUID
    box_top: int
    box_right: int
    box_bottom: int
    box_left: int
    face_thumbnail_url: Optional[str] = None
    label_name: Optional[str] = None

class ImageDetailResponse(BaseModel):
    image: ImageOut
    faces: List[FaceOut]
