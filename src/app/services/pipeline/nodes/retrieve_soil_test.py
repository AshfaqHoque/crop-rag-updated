"""Retrieve soil test knowledge from the dedicated Chroma collection."""
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pipeline.state import PipelineState
from app.services.retrieval.hybrid import SemanticRetriever

logger = get_logger(__name__)


@lru_cache
def get_soil_test_retriever() -> SemanticRetriever:
    return SemanticRetriever(
        collection_name=get_settings().chroma_soil_test_collection,
    )


def retrieve_soil_test(state: PipelineState) -> PipelineState:
    query = state.get("rewritten_query") or state["raw_query"]
    documents, _ = get_soil_test_retriever().retrieve(
        query,
        top_k=get_settings().soil_test_retrieval_top_k,
    )
    logger.info("retrieve_soil_test documents=%d", len(documents))
    return {
        **state,
        "retrieved_documents": documents,
        "retrieval_mode": "soil_test_dense",
    }