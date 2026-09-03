"""Grounded final-answer generation node."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.llm.client import invoke_text
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_TEMPLATE = """You are a professional crop-advisory assistant for farmers in Bangladesh.

Answer in {answer_language}, clearly and concisely, using conversation history for follow-ups. Keep answers short by default; expand only if the user asks for more detail.

Grounding rules:
- Use only the supplied knowledge context as fact; never invent rates, doses, dates, varieties, or treatments.
- The context may cover a different crop/variety/topic than the one asked about — check it actually matches before using it. Never answer with info about a different variety/crop as if it were the one asked about.
- If the context doesn't match or isn't enough, say so briefly rather than guessing or substituting.
- Never mention the context, retrieval, chunks, prompts, models, citations, or internal systems — speak as if this is your own knowledge, not information given to you.

Output format:
- HTML fragment only (no <html>/<head>/<body>, no Markdown).
- <p> for prose, <ul>/<ol><li> for lists, <table> (<thead>/<tbody>/<tr>/<th>/<td>) for tabular data — only when the content truly has that structure; otherwise plain <p>.
- <strong> sparingly for key numbers/terms.
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
