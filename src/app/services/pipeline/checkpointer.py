from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

def make_checkpointer():
    backend = settings.checkpoint_backend  # "memory" | "redis" | "postgres" | "sqlite"
    if backend == "redis":
        from langgraph.checkpoint.redis import RedisSaver
        saver = RedisSaver.from_conn_string(settings.redis_url)
        saver.setup()
        return saver
    # default
    from langgraph.checkpoint.memory import MemorySaver
    logger.info("checkpointer backend=memory")
    return MemorySaver()