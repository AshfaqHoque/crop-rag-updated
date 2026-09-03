"""Use the chat model to remove retrieved chunks unrelated to the question.

Runs one short LLM call per chunk (keep/discard) instead of one big call
judging all chunks at once. Simpler prompt, easier for a small local model
to get right consistently — trade-off is more LLM calls per request.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.schemas.extraction import ChunkRelevance
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an expert agricultural relevance checker.
Your job is to determine if a specific Chunk directly helps answer the User Query, considering the Conversation History.
The Chunk must be about the same crop, variety, disease, pest, or agricultural subject asked in the User Query. 
If it is not, you should discard it. If it is, you should keep it.
Output a true or false decision.
"""  # noqa: E501

def filter_relevant(state: PipelineState) -> PipelineState:
    chunks = state.get("reranked_chunks", [])
    if not chunks:
        return {**state, "filtered_chunks": []}

    conversation = list(state.get("messages") or [])
    history = conversation[-3:-1] if conversation else [] 
    question = state.get("normalized_query", "")

    filtered = []
    for index, chunk in enumerate(chunks):
        content = str(chunk.get("content", ""))
        current_turn_content = (
            f"Chunk to Evaluate:\n{content}\n\n"
            f"User Query: {question}"
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            *history,
            HumanMessage(content=current_turn_content)
        ]
        try:
            result = invoke_structured(ChunkRelevance, messages)
            keep = result.relevant
        except Exception: 
            logger.exception("Relevance check failed for chunk_index=%d; keeping it", index)
            keep = True

        if keep:
            filtered.append(dict(chunk))

    logger.info("filter_relevant chunks_before=%d chunks_after=%d", len(chunks), len(filtered))
    return {**state, "filtered_chunks": filtered}