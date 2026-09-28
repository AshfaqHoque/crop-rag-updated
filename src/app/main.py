"""FastAPI application entry point for the Aunkur AI chatbot."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="Aunkur AI Chatbot",
    version="0.2.0",
    description="RAG chatbot for crop advisory Q&A (Bangla/English)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Your hosted UI domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
