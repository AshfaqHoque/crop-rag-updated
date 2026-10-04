from contextlib import contextmanager
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import RedisSaver

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

def make_checkpointer():
    backend = settings.checkpoint_backend  # "memory" | "redis" | "postgres" | "sqlite"
    if backend == "redis":
        logger.info("checkpointer backend=redis")
        # from_conn_string() is a context manager in current langgraph-checkpoint-redis
        cm = RedisSaver.from_conn_string(settings.redis_url)
        saver = cm.__enter__()   # keep the connection open for the process lifetime
        saver.setup()
        return saver
    # default
    logger.info("checkpointer backend=memory")
    return MemorySaver()