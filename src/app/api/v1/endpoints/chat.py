from fastapi import APIRouter, Depends

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
