"""Grounded final-answer generation node."""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.exceptions import LLMGenerationError
from app.core.logging import get_logger
from app.services.llm.client import invoke_text
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)


_SYSTEM_TEMPLATE = """
You are an expert agricultural advisor in Bangladesh. Answer strictly and exclusively in {answer_language} in a natural, conversational tone.
Keep answers concise, clear, and direct. Provide detailed descriptions only if the user explicitly asks for them.
Answer like a professional agricultural advisor, not a tutor. Prioritize the most directly relevant information, give one clear answer, and avoid unnecessary explanations, repetition, or listing alternatives unless requested.

Before answering, first verify that the knowledge context is relevant to the user's specific query, including the crop, variety, disease, pest, and topic.
Do not use information about a different crop, variety, disease, or pest merely because some terms or symptoms match.

Answer only agriculture-related questions.
If the user mixes agricultural and unrelated questions, answer only the agricultural parts and silently ignore the rest. Do not treat unrelated parts as missing information.

When the answer involves a calculation (e.g. dosage, area, quantity, cost), always show the step-by-step math before giving the final result — this is not optional detail, it is a part of the answer.

If the context contains information for only part of the question, answer the supported part accurately. Politely inform the farmer that currently I do not have enough information about the remaining part(s), and ask if they would like to talk to an agronomist. 
If the context does not contain information relevant to answering the question, politely inform the farmer that currently I do not have the information, then ask if they would like to talk to an agronomist. 

Never state that you only have the information in the given context, or that you are limited to the context.

{unit_conversion_list}

Here is your knowledge context. Answer only using this context. Never invent rates, doses, dates, varieties, treatments, or missing values. Do not use external knowledge.
{context}
"""


_UNIT_KEYWORDS = (
    # শতক / ডেসিমেল (decimal)
    "শতক", "শতাংশ", "ডেসিমেল", "ডেসিমাল", "শতকে", "শতাংশে", "ডেসিমেলে",
    # বিঘা (bigha)
    "বিঘা", "বিঘাতে", "বিঘায়",
    # কাঠা
    "কাঠা", "কাঠায়", "কাঠাতে",
    # একর (acre)
    "একর", "একরে",
    # হেক্টর (hectare)
    "হেক্টর", "হেক্টরে"
)


_UNIT_CONVERSIONS = """
<unit_conversions>
১ হেক্টর (hectare) = ২৪৭.১ শতক (decimal)
১ একর (acre) = ১০০ শতক (decimal)
১ বিঘা (bigha) = ৩৩ শতক (decimal)
১ কাঠা (katha) = ১.৬৫ শতক (decimal)
</unit_conversions>
The conversion factors in <unit_conversions> are allowed for any area calculation, even though they are not part of the knowledge context. Show the multiplication step by step.
"""


def _needs_units(text: str) -> bool:
    return any(keyword in text for keyword in _UNIT_KEYWORDS)


# Output format:
# - HTML fragment only (no <html>/<head>/<body>, no Markdown).
# - Use HTML paragraphs and lists only when the answer truly needs structure.
# - <strong> sparingly for key numbers/terms.


def _answer_language(language_type: str) -> str:
    if language_type == "bn":
        return "clear Bangla script (বাংলা লিপি)"
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
    context_documents = state.get("compressed_documents") or []
    query = (state.get("raw_query") or state.get("raw_query", ""))
    # current_message = f"Retrieved Documents:\n{_format_context(context_documents)}\n\nUser Query: {query}"
    unit_check_text = state.get("rewritten_query") or query
    unit_block = _UNIT_CONVERSIONS if _needs_units(unit_check_text) else ""

    messages = [
        SystemMessage(
            content=_SYSTEM_TEMPLATE.format(
                answer_language=_answer_language(state.get("language_type", "english")),
                context=f"<knowledge_context>\n{_format_context(context_documents)}\n</knowledge_context>",
                unit_conversion_list=unit_block,
            )
        ),
        *history,
        HumanMessage(content=query),
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
