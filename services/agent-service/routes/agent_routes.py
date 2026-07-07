from fastapi import APIRouter, Depends, Query
from typing import List
from models.agent import QueueItem, ResolveRequest, SearchResultImage
from controllers.agent_controller import AgentController
from utils.security import get_current_user_id

router = APIRouter(prefix="/api/v1/agent", tags=["AI Agent Ops"])

@router.get("/queue", response_model=List[QueueItem])
async def get_resolution_queue(user_id: str = Depends(get_current_user_id)):
    return await AgentController.get_queue(user_id)

@router.post("/queue/resolve")
async def resolve_queue_face(
    payload: ResolveRequest,
    user_id: str = Depends(get_current_user_id)
):
    return await AgentController.resolve_face(user_id, str(payload.face_id), payload.name)

@router.get("/search", response_model=List[SearchResultImage])
async def search_gallery(
    q: str = Query(..., description="Semantic search query"),
    user_id: str = Depends(get_current_user_id)
):
    return await AgentController.search(user_id, q)
