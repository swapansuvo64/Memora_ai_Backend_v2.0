from fastapi import APIRouter, Depends, File, UploadFile, status
from typing import List
from models.gallery import ImageOut, ImageDetailResponse
from controllers.gallery_controller import GalleryController
from utils.security import get_current_user_id

router = APIRouter(prefix="/api/v1/gallery", tags=["Gallery Management"])

@router.post("/upload", response_model=ImageOut, status_code=status.HTTP_201_CREATED)
async def upload_image(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)
):
    file_bytes = await file.read()
    return await GalleryController.upload(
        user_id=user_id,
        file_name=file.filename,
        file_bytes=file_bytes,
        content_type=file.content_type,
        file_size=len(file_bytes)
    )

@router.get("/images", response_model=List[ImageOut])
async def list_images(user_id: str = Depends(get_current_user_id)):
    return await GalleryController.list_images(user_id)

@router.get("/images/{image_id}", response_model=ImageDetailResponse)
async def get_image_detail(
    image_id: str,
    user_id: str = Depends(get_current_user_id)
):
    return await GalleryController.get_image_detail(user_id, image_id)

@router.post("/images/{image_id}/bin")
async def move_to_bin(
    image_id: str,
    user_id: str = Depends(get_current_user_id)
):
    return await GalleryController.move_to_bin(user_id, image_id)

@router.get("/bin", response_model=List[ImageOut])
async def list_bin(user_id: str = Depends(get_current_user_id)):
    return await GalleryController.list_bin(user_id)

@router.post("/images/{image_id}/restore")
async def restore_from_bin(
    image_id: str,
    user_id: str = Depends(get_current_user_id)
):
    return await GalleryController.restore_from_bin(user_id, image_id)

@router.delete("/images/{image_id}")
async def permanent_delete(
    image_id: str,
    user_id: str = Depends(get_current_user_id)
):
    return await GalleryController.permanent_delete(user_id, image_id)
