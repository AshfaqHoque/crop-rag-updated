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
    query = state.get("current_subquery") or state.get("rewritten_query") or state["raw_query"]
    documents, _ = get_semantic_retriever().retrieve(
        query,
        crops=state.get("crops"),
        top_k=get_settings().retrieval_top_k,
    )
    logger.info("retrieve documents=%d for query='%s'", len(documents), query)
    return {"retrieved_documents": documents}
