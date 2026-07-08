from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from typing import List, Optional
from models.agent import QueueItem, ResolveRequest, SearchResultImage, ChatSearchRequest, ChatSearchResponse
from controllers.agent_controller import AgentController
from utils.security import get_current_user_id
from pydantic import BaseModel
import asyncio
import json
import redis.asyncio as aioredis
from config.settings import settings

router = APIRouter(prefix="/api/v1/agent", tags=["AI Agent Ops"])

@router.get("/queue", response_model=List[QueueItem])
async def get_resolution_queue(
    page: Optional[int] = Query(None, ge=1),
    limit: Optional[int] = Query(None, ge=1),
    user_id: str = Depends(get_current_user_id)
):
    return await AgentController.get_queue(user_id, page, limit)

@router.get("/queue/stream")
async def stream_queue_events(request: Request, user_id: str = Depends(get_current_user_id)):
    """SSE endpoint — streams real-time face queue events to the browser."""
    channel = f"queue_events:{user_id}"

    async def event_generator():
        r = aioredis.from_url(f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}")
        pubsub = r.pubsub()
        await pubsub.subscribe(channel)
        try:
            # Initial heartbeat so browser knows connection is live
            yield "event: connected\ndata: {}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    payload = json.loads(data)
                    event_type = payload.get("event", "update")
                    yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
                else:
                    # Keepalive comment every ~1s to prevent nginx/browser timeout
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.1)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
            await r.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

@router.post("/queue/resolve")
async def resolve_queue_face(
    payload: ResolveRequest,
    user_id: str = Depends(get_current_user_id)
):
    return await AgentController.resolve_face(user_id, str(payload.face_id), payload.name, payload.relationship)

class UpdateRelationshipRequest(BaseModel):
    relationship: str

@router.get("/people")
async def get_named_people(
    page: Optional[int] = Query(None, ge=1),
    limit: Optional[int] = Query(None, ge=1),
    user_id: str = Depends(get_current_user_id)
):
    return await AgentController.get_labeled_people(user_id, page, limit)

@router.put("/people/{label_id}/relationship")
async def update_person_relationship(
    label_id: str,
    payload: UpdateRelationshipRequest,
    user_id: str = Depends(get_current_user_id)
):
    return await AgentController.update_relationship(user_id, label_id, payload.relationship)

@router.post("/search", response_model=ChatSearchResponse)
async def search_gallery(
    payload: ChatSearchRequest,
    user_id: str = Depends(get_current_user_id)
):
    return await AgentController.search(
        user_id=user_id,
        query=payload.query,
        history=payload.history,
        filters=payload.filters if payload.filters else None
    )


