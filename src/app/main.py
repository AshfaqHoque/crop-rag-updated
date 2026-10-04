"""FastAPI application entry point for the Aunkur AI chatbot."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger
from app.services.pipeline.checkpointer import close_checkpointer, init_checkpointer
from app.services.pipeline.graph import reset_chat_graph

configure_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_checkpointer()
    reset_chat_graph()
    logger.info("app startup complete")
    yield
    await close_checkpointer()
    logger.info("app shutdown complete")


app = FastAPI(
    title="Aunkur AI Chatbot",
    version="0.2.0",
    description="RAG chatbot for crop advisory Q&A (Bangla/English)",
    lifespan=lifespan,
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a clean, client-friendly 422 for bad input."""
    errors = []
    for err in exc.errors():
        loc = " → ".join(str(x) for x in err.get("loc", []) if x != "body")
        msg = err.get("msg", "Invalid value")
        errors.append(f"{loc}: {msg}" if loc else msg)

    detail = "; ".join(errors)

    # Friendly messages for the most common cases
    if any("language_type" in e for e in errors):
        detail = "language_type must be one of: bn, en, ar"
    elif any("message" in e and ("max_length" in e or "at most" in e.lower() or "ensure this value" in e.lower()) for e in errors):
        detail = "message is too long (maximum 1500 characters)"
    elif any("message" in e and ("min_length" in e or "at least" in e.lower()) for e in errors):
        detail = "message must not be empty"
    elif any("session_id" in e for e in errors):
        detail = "session_id is required and must not be blank"

    logger.warning("Validation error: %s", detail)
    return JSONResponse(
        status_code=422,
        content={"detail": detail},
    )


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {
        "status": "ok",
        "chat_model": settings.chat_model,
        "embed_model": settings.embed_model,
        "env": settings.app_env,
    }


app.include_router(api_router)
