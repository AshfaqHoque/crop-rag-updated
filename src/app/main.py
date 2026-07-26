from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="Crop RAG Chatbot",
    version="0.2.0",
    description="RAG chatbot for crop advisory Q&A (Bangla/English)",
)


@app.exception_handler(AppError)
async def app_error_handler(request, exc: AppError):
    logger.warning("AppError: %s", exc.message)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {
        "status": "ok",
        "chat_model": settings.chat_model,
        "embed_model": settings.embed_model,
        "env": settings.app_env,
    }


app.include_router(api_router)
