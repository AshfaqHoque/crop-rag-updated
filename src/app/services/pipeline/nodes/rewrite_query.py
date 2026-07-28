"""History-aware query rewriting for subject/coreference resolution."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.extraction import QueryRewrite
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_PROMPT = """
You rewrite follow-up questions.

Rules:

- Extract the main agricultural subject from the previous question.
- If the current question already explicitly mentions a crop, disease, fertilizer, pesticide, or other agricultural subject, return it unchanged.
- Otherwise, if the current question uses a referring expression (it, its, this, that, these, those, they, them), replace that reference with the previous subject.
- Preserve the current question's wording and intent as much as possible.
- Never return the previous question itself.
- Never answer the question.
- Never add information.
- Output only the rewritten query.

Examples:

Previous: "বোরো ধানের বৈশিষ্ট্যগুলো কী কী?"
Current: "এতে কীভাবে সেচ দেব?"
Output: "বোরো ধানে কীভাবে সেচ দেব?"

Previous: "ফল আর্মিওয়ার্মের লক্ষণ কী?"
Current: "এটি কীভাবে দমন করব?"
Output: "ফল আর্মিওয়ার্ম কীভাবে দমন করব?"

Previous: "টমেটোর জন্য কোন সার ভালো?"
Current: "এটি কত দিন পর প্রয়োগ করব?"
Output: "টমেটোর জন্য সার কত দিন পর প্রয়োগ করব?"
"""

_USER_TEMPLATE = """<previous_question>
{previous}
</previous_question>

<current_question>
{current}
</current_question>
"""

def _get_previous_question(history: list[dict[str, str]]) -> str:
    """
    Returns the most recent user message before the current one.
    """
    for turn in reversed(history):
        if turn.get("role") == "user":
            return turn.get("content", "")

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
    logger.info("rewrite_query used_history=%s rewritten=%r", result.used_history, rewritten)
    return {
        **state,
        "rewritten_query": rewritten,
        "rewrite_used_history": result.used_history,
    }
