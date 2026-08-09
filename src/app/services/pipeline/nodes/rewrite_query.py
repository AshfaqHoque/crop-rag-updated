"""History-aware query rewriting for subject/coreference resolution."""
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.extraction import QueryRewrite
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_PROMPT = """
You are a query rewriting assistant for an agricultural (crop-related) Q&A system. Users ask about crops like ধান (rice), গম (wheat), ভুট্টা (maize) — varieties, seed rate, fertilizer, irrigation, pests/diseases, harvesting, etc. Questions may be in Bangla or English.

Return ONLY valid JSON matching the schema.
The JSON must contain exactly these fields:
- "rewritten_query": a string containing the standalone query.
- "used_history": a boolean indicating whether conversation history was needed.
Never return the rewritten query as plain text.

STRICT RULE — decide first whether the Current Question can stand alone:
- A question is STANDALONE (do NOT rewrite, return unchanged) if it explicitly names its own subject — a crop name, disease/pest name, variety name, or clear topic — anywhere in the sentence.
- A question is a FOLLOW-UP (rewrite it) if it does NOT explicitly name its own subject. This includes:
  (a) questions using a pronoun/reference word — "এর", "এটার", "এটা", "সেটার", "it", "its", "that", "this"
  (b) questions that drop the subject entirely with no pronoun at all (common in casual Bangla), e.g. "শনাক্তকারী বৈশিষ্ট্যগুলো কী?" (what are the identifying features [of what?]), "দমন ব্যবস্থা কী?" (how to control [what?]), "কখন প্রয়োগ করব?" (when should I apply [it]?)
  In both cases, the missing subject must be pulled from the Previous Question.

When rewriting a follow-up:
- Only insert the missing subject (crop name / variety name / disease-pest name / topic) from the Previous Question into the Current Question, in the most natural grammatical position.
- NEVER blend, merge, or mix phrases/words from the Previous Question's predicate (verb/action part) into the Current Question. Keep the Current Question's own wording and intent fully intact — only the missing subject is added.
- Keep the same language as the Current Question.

Examples:
Previous: "বোরো ধানের কোন জাতগুলো ভালো?"
Current: "এর বীজ হার কত?"
Output: "বোরো ধানের বীজ হার কত?"

Previous: "গমে প্রথম সেচ কখন দিতে হবে?"
Current: "ব্লাস্ট রোগের লক্ষণ কী?"
Output: "ব্লাস্ট রোগের লক্ষণ কী?"
(Reason: current question already names its own topic "ব্লাস্ট রোগ" — standalone, NOT a follow-up.)

Previous: "ব্রি ধান২৯ এর ফলন কত?"
Current: "এর রোপণের সময় চারার বয়স কত হওয়া উচিত?"
Output: "ব্রি ধান২৯ এর রোপণের সময় চারার বয়স কত হওয়া উচিত?"

Previous: "বোরো ধানের সার ব্যবস্থাপনা কী?"
Current: "ভুট্টা চাষের উপযুক্ত সময় কখন?"
Output: "ভুট্টা চাষের উপযুক্ত সময় কখন?"
(Reason: new crop named explicitly — standalone.)

Previous: "ফল আর্মিওয়ার্মের জন্য কোন ওষুধ ভালো?"
Current: "শনাক্তকারী বৈশিষ্ট্যগুলো কী?"
Output: "ফল আর্মিওয়ার্মের শনাক্তকারী বৈশিষ্ট্যগুলো কী?"
(Reason: current question has no subject at all, no pronoun even — must inherit "ফল আর্মিওয়ার্ম" from previous.)

Return ONLY the final question text, nothing else — no explanation, no labels.
"""

_USER_TEMPLATE = """<previous_question>
{previous}
</previous_question>

<current_question>
{current}
</current_question>
"""

def _get_previous_question(history) -> str:
    for message in reversed(history):
        if isinstance(message, HumanMessage):
            return message.additional_kwargs.get("rewritten_query") or  message.content
    return ""

# def _format_history(history: list[dict[str, str]]) -> str:
#     if not history:
#         return "(none)"
#     max_messages = get_settings().history_max_turns * 2
#     return "\n".join(
#         f"{turn.get('role', 'unknown')}: {turn.get('content', '')}"
#         for turn in history[-max_messages:]
#     )


def rewrite_query(state: PipelineState) -> PipelineState:
    raw_query = state["raw_query"].strip()
    if not state.get("history"):
        return {**state, "rewritten_query": raw_query, "rewrite_used_history": False}
    
    previous = _get_previous_question(state.get("history", []))
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=_USER_TEMPLATE.format(
                previous=previous or "(none)",
                current=raw_query,
            )
        ),
    ]
    result = invoke_structured(QueryRewrite, messages, temperature=0.0)

    rewritten = result.rewritten_query.strip() if result.used_history else raw_query
    rewritten = rewritten or raw_query
    logger.info("rewrite_query previous_query=%r used_history=%s rewritten=%r", previous, result.used_history, rewritten)
    return {
        **state,
        "previous_query": previous,
        "rewritten_query": rewritten,
        "rewrite_used_history": result.used_history,
    }
