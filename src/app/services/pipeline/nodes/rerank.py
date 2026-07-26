"""Cross-encoder reranking node."""
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pipeline.state import PipelineState
from app.services.reranking.bge import get_reranker

logger = get_logger(__name__)


def rerank(state: PipelineState) -> PipelineState:
    query = state.get("rewritten_query") or state["raw_query"]
    chunks = get_reranker().rerank(
        query,
        state.get("retrieved_chunks", []),
        top_k=get_settings().rerank_top_k,
    )
    logger.info("rerank chunks=%d", len(chunks))
    return {**state, "reranked_chunks": chunks}
