"""Semantic retrieval node."""
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pipeline.state import PipelineState
from app.services.retrieval.hybrid import SemanticRetriever

logger = get_logger(__name__)


@lru_cache
def get_semantic_retriever() -> SemanticRetriever:
    return SemanticRetriever()


def retrieve(state: PipelineState) -> PipelineState:
    query = state.get("rewritten_query") or state["raw_query"]
    chunks, mode = get_semantic_retriever().retrieve(
        query,
        crops=state.get("crops"),
        sections=state.get("sections"),
        top_k=get_settings().retrieval_top_k,
    )
    logger.info("retrieve mode=%s chunks=%d", mode, len(chunks))
    return {**state, "retrieved_chunks": chunks, "retrieval_mode": mode}
