"""Checkpointer factory. Redis path uses AsyncRedisSaver for astream()."""
from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

_checkpointer = None  # set once at startup (async Redis) or on first call (memory)


def get_checkpointer():
    """Return the process-wide checkpointer instance."""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    # Fallback if startup did not initialize Redis yet
    if settings.checkpoint_backend == "redis":
        raise RuntimeError(
            "Redis checkpointer not initialized. "
            "App lifespan must call init_redis_checkpointer() before serving requests."
        )

    from langgraph.checkpoint.memory import MemorySaver
    logger.info("checkpointer backend=memory")
    _checkpointer = MemorySaver()
    return _checkpointer


async def init_redis_checkpointer():
    """Create and set up AsyncRedisSaver. Call once from FastAPI lifespan."""
    global _checkpointer
    if settings.checkpoint_backend != "redis":
        from langgraph.checkpoint.memory import MemorySaver
        logger.info("checkpointer backend=memory")
        _checkpointer = MemorySaver()
        return _checkpointer

    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    logger.info("checkpointer backend=redis (async) url=%s", settings.redis_url)
    # Keep the connection open for the life of the process
    saver = AsyncRedisSaver.from_conn_string(settings.redis_url)
    # from_conn_string may return a context manager in some versions:
    # if so, enter it and keep the entered saver.
    if hasattr(saver, "__aenter__"):
        saver = await saver.__aenter__()
    await saver.asetup()
    _checkpointer = saver
    logger.info("redis checkpointer ready")
    return _checkpointer


async def close_checkpointer():
    """Optional cleanup on shutdown."""
    global _checkpointer
    if _checkpointer is None:
        return
    close = getattr(_checkpointer, "aclose", None) or getattr(_checkpointer, "close", None)
    if close is not None:
        result = close()
        if hasattr(result, "__await__"):
            await result
    _checkpointer = None