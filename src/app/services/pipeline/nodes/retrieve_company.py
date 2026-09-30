"""Retrieve company knowledge from the dedicated Chroma collection."""
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pipeline.state import PipelineState
from app.services.retrieval.hybrid import SemanticRetriever

logger = get_logger(__name__)


@lru_cache
def get_company_retriever() -> SemanticRetriever:
    return SemanticRetriever(
        collection_name=get_settings().chroma_company_collection,
    )


def retrieve_company(state: PipelineState) -> PipelineState:
    query = state.get("rewritten_query") or state["raw_query"]
    documents, _ = get_company_retriever().retrieve(
        query,
        top_k=get_settings().company_retrieval_top_k,
    )
    logger.info("retrieve_company documents=%d", len(documents))
    return {
        **state,
        "retrieved_documents": documents,
    }