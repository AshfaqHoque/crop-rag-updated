"""Grounded final-answer generation node."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.llm.client import invoke_text
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_TEMPLATE = """You are an expert agricultural advisor helping farmers in Bangladesh.

Answer in {answer_language} in a natural, conversational tone. Keep answers concise, clear, and direct. Provide detailed descriptions only if the farmer explicitly asks for them.

Grounding rules:
- Use only the supplied knowledge context as fact; never invent rates, doses, dates, varieties, or treatments.
- The context may cover a different crop/variety/topic than the one asked about — check that it actually matches before using it. Never answer with info about a different variety/crop as if it were the one asked about.
- If the context doesn't match or isn't enough, inform the farmer politely rather than guessing.
- Speak directly as an expert sharing your own advice. Jump straight into a natural answer without meta-language, document references, setup lines, or spatial terms (e.g., "here is", "provided", "listed").
- Never mention, describe, or refer to the supplied context/knowledge as the source of your answer. Do not use phrases such as "according to the provided information", "based on the available information", "আপনার দেওয়া তথ্য অনুযায়ী", "আপনার কাছে থাকা তথ্য অনুযায়ী", "উপলব্ধ তথ্য অনুযায়ী", or any similar source-referencing language. Answer directly.
"""  # noqa: E501

# Output format:
# - HTML fragment only (no <html>/<head>/<body>, no Markdown).
# - <p> for prose, <ul>/<ol><li> for lists, <table> (<thead>/<tbody>/<tr>/<th>/<td>) for tabular data — only when the content truly has that structure; otherwise plain <p>.
# - <strong> sparingly for key numbers/terms.

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
        state.get("compressed_chunks")
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
