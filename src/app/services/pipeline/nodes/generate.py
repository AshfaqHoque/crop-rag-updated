"""Grounded final-answer generation node."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.llm.client import invoke_text
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_TEMPLATE = """You are a professional crop-advisory assistant for farmers in Bangladesh.
Answer in {answer_language}.

Rules:
- Give a direct answer first. Keep it concise.
- Base answers ONLY on the provided Context, speaking naturally and directly to the user as if it is your own expertise.
- If Context is missing, insufficient, or not relevant to the User Query, state that you do not have enough information. 
Do not guess or mix in unrelated information from the Context.
- Do NOT mention "Context", "retrieval", "chunks", or internal systems.
- Do NOT include citations, source numbers, or footnotes.
"""  # noqa: E501


def _answer_language(language: str) -> str:
    if language == "bn":
        return "natural Bangla"
    return "clear English"


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(No relevant agricultural information was retrieved.)"
    max_chars = get_settings().context_max_chars_per_chunk
    return "\n\n".join(str(chunk.get("content", ""))[:max_chars] for chunk in chunks)


def generate(state: PipelineState) -> PipelineState:
    conversation = list(state.get("messages") or [])
    history = conversation[-3:-1] if conversation else []
    context_chunks = (
        state.get("filtered_chunks")
        or state.get("reranked_chunks")
        or state.get("retrieved_chunks", [])
    )

    current_message = (
        f"Context:\n{_format_context(context_chunks)}\n\n"
        f"User Query: {state.get('normalized_query')}"
    )

    messages = [
        SystemMessage(
            content=_SYSTEM_TEMPLATE.format(
                answer_language=_answer_language(state.get("language", "bn"))
            )
        ),
        *history,
        HumanMessage(content=current_message),
    ]

    answer = invoke_text(messages).strip()
    logger.info("generate answer_chars=%d, length of history used=%d", len(answer), len(history))
    return {
        **state,
        "messages": [AIMessage(content=answer)],
        "answer": answer,
    }
