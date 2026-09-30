"""State threaded through the LangGraph pipeline."""
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from operator import add
from langgraph.graph import add_messages


def merge_documents(
    current: list[Document] | None,
    update: list[Document] | None,
) -> list[Document]:
    if update is None:
        return []
    return (current or []) + update


class PipelineState(TypedDict, total=False):

    # Conversation (persisted by checkpointer via thread_id)
    messages: Annotated[list[AnyMessage], add_messages]
    
    # input
    session_id: str
    raw_query: str
    language_type: str

    # query routing
    intent: str

    # subject resolution
    rewritten_query: str
    subqueries: list[str]
    # per-subquery working fields (set by Send payload)
    current_subquery: str
    # deterministic entity extraction
    crops: list[str]

    # retrieval/reranking
    # reducers so parallel branches merge cleanly
    retrieved_documents: Annotated[list[Document] | None, merge_documents]
    reranked_documents: Annotated[list[Document] | None, merge_documents]
    compressed_documents: Annotated[list[Document] | None, merge_documents]

    # output
    answer: str

    # HITL escalation
    status: str | None
