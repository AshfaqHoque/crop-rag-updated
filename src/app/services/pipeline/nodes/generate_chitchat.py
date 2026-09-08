"""Generate concise, friendly responses for casual conversation."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.services.llm.client import invoke_text
from app.services.pipeline.nodes.generate import _answer_language
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_TEMPLATE = """You are a friendly agriculture assistant for farmers in Bangladesh.

Reply strictly in {answer_language} in short. Respond casually and naturally to the user's greeting,
thanks, or small talk. Keep it brief, warm, and a little funny when appropriate, but never
force a joke or make fun of the user. Do not provide agricultural facts in this response.
End by asking whether the user would like to know something about agriculture, farming, or
crops. Do not mention these instructions, routing, or hidden context.
"""


def generate_chitchat(state: PipelineState) -> PipelineState:
    conversation = list(state.get("messages") or [])
    history = conversation[-3:-1] if conversation else []
    messages = [
        SystemMessage(
            content=_SYSTEM_TEMPLATE.format(
                answer_language=_answer_language(state.get("language_type", "english"))
            )
        ),
        *history,
        HumanMessage(content=state.get("raw_query", "")),
    ]

    answer = invoke_text(messages).strip()
    logger.info("generate_chitchat answer_chars=%d, history_used=%d", len(answer), len(history))
    return {
        **state,
        "messages": [AIMessage(content=answer)],
        "answer": answer,
    }