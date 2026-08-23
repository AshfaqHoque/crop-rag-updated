"""Use the chat model to remove retrieved chunks unrelated to the question."""
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.schemas.extraction import RelevantChunks
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a strict relevance filter for an agricultural question-answering system.

Select only chunks that contain information directly useful for answering the user's
original question. The rewritten query is only an additional retrieval aid; do not
use details present only in the rewritten query when they are absent from the original
question. For list questions, keep a chunk only when it supports the requested list
or explicitly describes the requested property. Exclude related but different facts,
including varieties that do not satisfy the requested property.

Return zero-based indexes of relevant chunks only. Never add or modify chunk indexes.
"""


def filter_relevant(state: PipelineState) -> PipelineState:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {**state, "filtered_chunks": []}

    chunk_text = "\n\n".join(
        f"[{index}] {chunk.get('content', '')}" for index, chunk in enumerate(chunks)
    )
    prompt = HumanMessage(
        content=(
            f"<original_question>\n{state.get('raw_query', '')}\n</original_question>\n"
            f"<rewritten_query>\n{state.get('rewritten_query', '')}\n</rewritten_query>\n"
            f"<retrieved_chunks>\n{chunk_text}\n</retrieved_chunks>"
        )
    )
    try:
        result = invoke_structured(
            RelevantChunks,
            [SystemMessage(content=_SYSTEM_PROMPT), prompt],
            temperature=0,
        )
        valid_indexes = {
            index for index in result.relevant_indexes if 0 <= index < len(chunks)
        }
        filtered = [dict(chunk) for index, chunk in enumerate(chunks) if index in valid_indexes]
    except Exception:  # noqa: BLE001
        logger.exception("LLM relevance filter failed; returning an empty context")
        filtered = []

    logger.info("filter_relevant chunks_before=%d chunks_after=%d", len(chunks), len(filtered))
    return {**state, "filtered_chunks": filtered}