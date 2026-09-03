"""Rerank retrieved chunks with the external reranker service."""

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)


def rerank(state: PipelineState) -> PipelineState:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {**state, "reranked_chunks": []}

    query = state.get("rewritten_query") or state["raw_query"]
    response = httpx.post(
        get_settings().reranker_url,
        json={
            "query": query,
            "documents": [str(chunk.get("content", "")) for chunk in chunks],
        },
        timeout=30.0,
    )
    response.raise_for_status()

    reranked = []
    for result in response.json()["results"]:
        chunk = dict(chunks[result["index"]])
        chunk["relevance_score"] = float(result["relevance_score"])
        reranked.append(chunk)

    reranked.sort(key=lambda chunk: chunk["relevance_score"], reverse=True)
    reranked = reranked[: get_settings().rerank_top_k]

    logger.info("rerank chunks=%d", len(reranked))
    return {**state, "reranked_chunks": reranked}
