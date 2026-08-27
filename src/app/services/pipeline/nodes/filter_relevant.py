"""Use the chat model to remove retrieved chunks unrelated to the question.

Runs one short LLM call per chunk (keep/discard) instead of one big call
judging all chunks at once. Simpler prompt, easier for a small local model
to get right consistently — trade-off is more LLM calls per request.
"""
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.schemas.extraction import ChunkRelevance, RelevantChunks
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a relevance checker for an agricultural question-answering system.

Given the conversation history and the current question, decide whether the single
chunk below is needed to answer the current question. A chunk that was relevant to
an earlier turn but not the current question is NOT relevant. Answer true only if the
chunk directly helps answer the current question.
"""

def _format_history(messages: list[BaseMessage]) -> str:
    # Full history for now, no truncation — passed in as-is; can cap this later
    # once it's clear how long conversations typically get.
    turns = messages[:-1] if messages else []
    # turns = turns[-_MAX_HISTORY_MESSAGES:]
    if not turns:
        return "(no prior conversation)"
    lines = []
    for message in turns:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"Assistant: {message.content}")
    return "\n".join(lines) if lines else "(no prior conversation)"

def filter_relevant(state: PipelineState) -> PipelineState:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {**state, "filtered_chunks": []}

    history_text = _format_history(state.get("messages", []))
    question = state.get("normalized_query", "")

    filtered = []
    for index, chunk in enumerate(chunks):
        content = str(chunk.get("content", ""))
        prompt = HumanMessage(
            content=(
                f"<conversation_history>\n{history_text}\n</conversation_history>\n"
                f"<current_question>\n{question}\n</current_question>\n"
                f"<chunk>\n{content}\n</chunk>"
            )
        )
        try:
            result = invoke_structured(
                ChunkRelevance,
                [SystemMessage(content=_SYSTEM_PROMPT), prompt],
                temperature=0,
            )
            keep = result.relevant
        except Exception: 
            logger.exception("Relevance check failed for chunk_index=%d; keeping it", index)
            keep = True

        if keep:
            filtered.append(dict(chunk))

    logger.info("filter_relevant chunks_before=%d chunks_after=%d", len(chunks), len(filtered))
    return {**state, "filtered_chunks": filtered}