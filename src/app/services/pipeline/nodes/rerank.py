"""Rerank retrieved chunks with the external reranker service."""

import statistics

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

def cut_at_unusual_gap(reranked: list[dict]) -> list[dict]:
    """
    Cut reranked chunks when the largest score gap is unusually large.

    Rule:
        max_gap >= 3 * median(other_gaps)
    """

    if len(reranked) < 3:
        return reranked

    # Scores are already sorted, but keep this function independent
    reranked.sort(key=lambda chunk: chunk["relevance_score"],reverse=True,)
    scores = [chunk["relevance_score"] for chunk in reranked]
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]

    # Find biggest gap
    max_gap = max(gaps)
    max_gap_index = gaps.index(max_gap)

    # Remove biggest gap before calculating normal/typical gap
    other_gaps = [gap for index, gap in enumerate(gaps) if index != max_gap_index]
    median_gap = statistics.median(other_gaps)
    threshold = median_gap * 3
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

    return reranked

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
    reranked = cut_at_unusual_gap(reranked)

    logger.info(
        "rerank final chunks=%d scores=%s",
        len(reranked),
        [
            round(chunk["relevance_score"], 4)
            for chunk in reranked
        ],
    )

    logger.info("rerank chunks=%d", len(reranked))
    return {**state, "reranked_chunks": reranked}
