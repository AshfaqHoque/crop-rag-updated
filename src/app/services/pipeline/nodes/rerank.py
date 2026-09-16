"""Rerank retrieved chunks with the external reranker service."""

import statistics

import httpx
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

def cut_at_unusual_gap(reranked: list[Document]) -> list[Document]:
    """
    Cut reranked chunks when the largest score gap is unusually large.

    Rule:
        max_gap >= 3 * median(other_gaps)
    """

    if len(reranked) < 3:
        return reranked

    # Scores are already sorted, but keep this function independent
    reranked.sort(key=lambda document: document.metadata["relevance_score"], reverse=True)
    scores = [document.metadata["relevance_score"] for document in reranked]
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]

    # Find biggest gap
    max_gap = max(gaps)
    max_gap_index = gaps.index(max_gap)

    # Remove biggest gap before calculating normal/typical gap
    other_gaps = [gap for index, gap in enumerate(gaps) if index != max_gap_index]
    median_gap = statistics.median(other_gaps)
    threshold = median_gap * 60
    unusual = max_gap >= threshold

    logger.info("rerank gap analysis scores=%s gaps=%s max_gap=%.4f "
        "median_gap=%.4f threshold=%.4f unusual=%s",
        [round(score, 4) for score in scores],
        [round(gap, 4) for gap in gaps],
        max_gap,
        median_gap,
        threshold,
        unusual,
    )

    if unusual:
        # If gap is between index 4 and 5,
        # keep everything through index 4.
        cutoff = max_gap_index + 1
        logger.info("rerank unusual gap detected cut_after=%d",cutoff)
        return reranked[:cutoff]

    minimum_score = scores[0] - 70
    reranked = [ document for document in reranked if document.metadata["relevance_score"] >= minimum_score ]
    logger.info("rerank score threshold top_score=%.4f minimum_score=%.4f chunks_after=%d",scores[0],minimum_score,len(reranked),)

    return reranked

def rerank(state: PipelineState) -> PipelineState:
    documents = state.get("retrieved_documents", [])
    if not documents:
        return {**state, "reranked_documents": []}

    query = state.get("rewritten_query") or state["raw_query"]
    response = httpx.post(
        get_settings().reranker_url,
        json={
            "query": query,
            "documents": [document.page_content for document in documents],
        },
        timeout=30.0,
    )
    response.raise_for_status()

    reranked: list[Document] = []
    for result in response.json()["results"]:
        document = documents[result["index"]]
        metadata = dict(document.metadata)
        metadata["relevance_score"] = float(result["relevance_score"])
        reranked.append(Document(page_content=document.page_content, metadata=metadata))

    reranked.sort(key=lambda document: document.metadata["relevance_score"], reverse=True)
    reranked = reranked[: get_settings().rerank_top_k]
    reranked = cut_at_unusual_gap(reranked)

    logger.info(
        "rerank final chunks=%d scores=%s",
        len(reranked),
        [
            round(document.metadata["relevance_score"], 4)
            for document in reranked
        ],
    )

    logger.info("rerank chunks=%d", len(reranked))
    return {**state, "reranked_documents": reranked}
