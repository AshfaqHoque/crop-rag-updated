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
    language_type: str

    # query routing
    intent: str
    descriptive: bool

    # subject resolution
    rewritten_query: str

    # deterministic entity extraction
    crops: list[str]

    # retrieval/reranking
    retrieved_documents: list[Document]
    reranked_documents: list[Document]
    compressed_documents: list[Document]
    retrieval_mode: str

    # output
    answer: str

    # HITL escalation
    status: str | None
