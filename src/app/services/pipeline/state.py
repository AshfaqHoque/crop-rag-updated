"""State threaded through the LangGraph pipeline."""
from typing import Annotated, Any, TypedDict
from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph import add_messages

class PipelineState(TypedDict, total=False):

    # Conversation (persisted by checkpointer via thread_id)
    messages: Annotated[list[AnyMessage], add_messages]
    
    # input
    session_id: str
    raw_query: str

    #normalize language
    normalized_query: str
    language: str

    # query understanding
    intent: str
    sections: list[str]  

    # subject resolution
    rewritten_query: str
    rewrite_used_history: bool

    # deterministic entity extraction
    crops: list[str]

    # retrieval/reranking
    retrieved_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    compressed_chunks: list[dict[str, Any]]
    retrieval_mode: str

    # output
    answer: str
