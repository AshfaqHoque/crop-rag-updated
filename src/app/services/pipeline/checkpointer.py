"""Process-wide checkpointer. Redis uses AsyncRedisSaver (needed for astream)."""
from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

_checkpointer = None


def get_checkpointer():
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    raise RuntimeError(
        "Checkpointer not initialized. "
        "Call init_checkpointer() from FastAPI lifespan before serving."
    )


async def init_checkpointer():
    """Call once on app startup."""
    global _checkpointer

    if settings.checkpoint_backend != "redis":
        from langgraph.checkpoint.memory import MemorySaver
        logger.info("checkpointer backend=memory")
        _checkpointer = MemorySaver()
        return _checkpointer

    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    logger.info("checkpointer backend=redis url=%s", settings.redis_url)

    # Preferred API (langgraph-checkpoint-redis)
    cm = AsyncRedisSaver.from_conn_string(settings.redis_url)
    if hasattr(cm, "__aenter__"):
        saver = await cm.__aenter__()
    else:
        saver = cm

    await saver.asetup()  # creates RediSearch indexes (needs Redis Stack)
    _checkpointer = saver
    logger.info("redis checkpointer ready")
    return _checkpointer


async def close_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        return
    close = getattr(_checkpointer, "aclose", None)
    if close is not None:
        result = close()
        if hasattr(result, "__await__"):
            await result
    _checkpointer = None