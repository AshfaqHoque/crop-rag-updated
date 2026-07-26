"""State threaded through the LangGraph pipeline."""
from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    # input
    session_id: str
    raw_query: str
    history: list[dict[str, str]]

    # query understanding
    language: str
    intent: str
    crops: list[str]
    sections: list[str]

    # subject resolution
    rewritten_query: str
    rewrite_used_history: bool

    # retrieval/reranking
    retrieved_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    retrieval_mode: str

    # output
    answer: str
