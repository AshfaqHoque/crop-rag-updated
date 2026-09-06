"""State threaded through the LangGraph pipeline."""
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
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
    retrieved_documents: list[Document]
    reranked_documents: list[Document]
    compressed_documents: list[Document]
    retrieval_mode: str

    # output
    answer: str
