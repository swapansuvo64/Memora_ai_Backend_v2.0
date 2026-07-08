from fastapi import APIRouter, Depends, File, UploadFile, status
from pydantic import BaseModel
from typing import List, Optional
from models.gallery import ImageOut, ImageDetailResponse
from controllers.gallery_controller import GalleryController
from utils.security import get_current_user_id

class BulkImageAction(BaseModel):
    image_ids: List[str]

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

@router.post("/upload/multiple", response_model=List[ImageOut], status_code=status.HTTP_201_CREATED)
async def upload_multiple_images(
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user_id)
):
    uploaded_images = []
    for file in files:
        file_bytes = await file.read()
        # Upload using the controller
        img_out = await GalleryController.upload(
            user_id=user_id,
            file_name=file.filename,
            file_bytes=file_bytes,
            content_type=file.content_type,
            file_size=len(file_bytes)
        )
        uploaded_images.append(img_out)
    return uploaded_images


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

@router.post("/images/bin-multiple")
async def move_to_bin_multiple(
    action: BulkImageAction,
    user_id: str = Depends(get_current_user_id)
):
    return await GalleryController.move_to_bin_multiple(user_id, action.image_ids)

@router.post("/images/restore-multiple")
async def restore_from_bin_multiple(
    action: BulkImageAction,
    user_id: str = Depends(get_current_user_id)
):
    return await GalleryController.restore_from_bin_multiple(user_id, action.image_ids)

@router.post("/images/delete-multiple")
async def permanent_delete_multiple(
    action: BulkImageAction,
    user_id: str = Depends(get_current_user_id)
):
    return await GalleryController.permanent_delete_multiple(user_id, action.image_ids)



# FOLDER MANAGEMENT ENDPOINTS
@router.get("/folders")
async def list_folders(user_id: str = Depends(get_current_user_id)):
    return await GalleryController.list_folders(user_id)

@router.post("/folders")
async def create_folder(
    name: str,
    parent_folder_id: Optional[str] = None,
    user_id: str = Depends(get_current_user_id)
):
    return await GalleryController.create_folder(user_id, name, parent_folder_id)

@router.delete("/folders/{folder_id}")
async def delete_folder(
    folder_id: str,
    user_id: str = Depends(get_current_user_id)
):
    return await GalleryController.delete_folder(user_id, folder_id)

@router.post("/images/{image_id}/move")
async def move_image_to_folder(
    image_id: str,
    folder_id: Optional[str] = None,
    user_id: str = Depends(get_current_user_id)
):
    return await GalleryController.move_image_to_folder(user_id, image_id, folder_id)

