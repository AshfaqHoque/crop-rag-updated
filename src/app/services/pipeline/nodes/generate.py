"""Grounded final-answer generation node."""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.exceptions import LLMGenerationError
from app.core.logging import get_logger
from app.services.llm.client import invoke_text
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_TEMPLATE = """You are an expert agricultural advisor helping farmers in Bangladesh. Do not mention where the information came from.

Answer strictly and exclusively in {answer_language} in a natural, conversational tone. Do not output any foreign scripts, characters, or mixed alphabets under any circumstances. Keep answers concise, clear, and direct. Provide detailed descriptions only if the farmer explicitly asks for them.
When the answer involves a calculation (e.g. dosage, area, quantity, cost), always show the step-by-step math before giving the final result — this is not optional detail, it's part of the answer.

Grounding rules:
- Use only the supplied knowledge context as fact; never invent rates, doses, dates, varieties, or treatments.
- The context may cover a different crop/variety/topic than the one asked about — check that it actually matches before using it. Never answer with info about a different variety/crop as if it were the one asked about.
- If the context doesn't match or isn't enough, inform the farmer politely rather than guessing, then ask if they would like to talk to an agronomist.
- Speak directly as an expert sharing your own advice. Jump straight into a natural answer without meta-language, document references, setup lines, or spatial terms (e.g., "here is", "provided", "listed").
- Never mention, describe, or refer to the supplied context/knowledge as the source of your answer. Do not use phrases such as "according to the provided information", "based on the available information", "আপনার দেওয়া তথ্য অনুযায়ী", "আপনার কাছে থাকা তথ্য অনুযায়ী", "উপলব্ধ তথ্য অনুযায়ী", or any similar source-referencing language. Answer directly.
"""  # noqa: E501

# Output format:
# - HTML fragment only (no <html>/<head>/<body>, no Markdown).
# - Use HTML paragraphs and lists only when the answer truly needs structure.
# - <strong> sparingly for key numbers/terms.

def _answer_language(language_type: str) -> str:
    if language_type == "bn":
        return "natural Bangla"
    if language_type == "ar":
        return "clear Arabic"
    if language_type == "en":
        return "clear English"
    return "clear English"


def _format_context(documents: list[Document]) -> str:
    if not documents:
        return "(No relevant agricultural information was retrieved.)"
    max_chars = get_settings().context_max_chars_per_chunk
    formatted = []
    for index, document in enumerate(documents, start=1):
        chunk_id = document.metadata.get("chunk_id", "unknown")
        formatted.append(f"\n{document.page_content[:max_chars]}")
    return "\n\n".join(formatted)

def _fallback_message(language_type: str) -> str:
    if language_type == "bn":
        return (
            "এই প্রশ্নের জন্য খুব বেশি তথ্য এসেছে, তাই আমি এখন সঠিক উত্তর দিতে পারছি না। "
            "অনুগ্রহ করে প্রশ্নটি আরও নির্দিষ্ট করে আবার জিজ্ঞাসা করুন, "
            "অথবা একজন কৃষিবিদের সাথে কথা বলতে চান কিনা জানান।"
        )
    return (
        "This question retrieved too much information for me to answer accurately right now. "
        "Please make the question more specific and try again, "
        "or let me know if you would like to speak with an agronomist."
    )

def generate(state: PipelineState) -> PipelineState:
    conversation = list(state.get("messages") or [])
    history = conversation[-3:-1] if conversation else []
    context_documents = (state.get("compressed_documents") or state.get("reranked_documents") or state.get("retrieved_documents", []))
    query = (state.get("raw_query") or state.get("raw_query", ""))
    current_message = f"Your Knowledge Context:\n{_format_context(context_documents)}\n\nUser Query: {query}"

    messages = [
        SystemMessage(
            content=_SYSTEM_TEMPLATE.format(
                answer_language=_answer_language(state.get("language_type", "english"))
            )
        ),
        *history,
        HumanMessage(content=current_message),
    ]
    
    try:
        answer = invoke_text(messages).strip()
    except LLMGenerationError as e:
        err = str(e).lower()
        if any(k in err for k in ("token", "context length", "maximum context", "8192", "too long")):
            logger.warning("generate context overflow: %s", e)
            answer = _fallback_message(state.get("language_type", "english"))
        else:
            raise  # let other LLM failures bubble up as before

    logger.info("generate answer_chars=%d, length of history used=%d", len(answer), len(history))
    return {
        **state,
        "messages": [AIMessage(content=answer)],
        "answer": answer,
    }
