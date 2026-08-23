"""State threaded through the LangGraph pipeline."""
from typing import Any, TypedDict
from langchain_core.messages import BaseMessage

class PipelineState(TypedDict, total=False):
    # input
    session_id: str
    raw_query: str
    history: list[BaseMessage]

    #normalize language
    normalized_query: str
    language: str

    # query understanding
    intent: str
    sections: list[str]  

    # subject resolution
    previous_query: str
    rewritten_query: str
    rewrite_used_history: bool

    # deterministic entity extraction
    crops: list[str]

    # retrieval/reranking
    retrieved_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    filtered_chunks: list[dict[str, Any]]
    retrieval_mode: str

    # output
    answer: str
