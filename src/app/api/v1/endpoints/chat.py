"""Chat and streaming endpoints for the crop advisory API."""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.logging import get_logger
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService, get_chat_service

router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    logger.info("Received message for session=%s", request.session_id)
    return await service.chat(request)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    logger.info("Received streaming message for session=%s", request.session_id)
    return StreamingResponse(
        service.stream_chat(request),
        media_type="text/event-stream",
    )