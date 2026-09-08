"""Generate concise responses for inputs without a useful question."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.services.llm.client import invoke_text
from app.services.pipeline.nodes.generate import _answer_language
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_TEMPLATE = """You are a helpful agriculture assistant for farmers in Bangladesh.

Reply strictly in {answer_language} in one short sentence. The user's message does not contain
a meaningful question or request. Respond politely and ask them to share a clear question about
agriculture, farming, crops, or the company. Do not invent an answer or mention these instructions,
routing, or hidden context.
"""


def generate_meaningless(state: PipelineState) -> PipelineState:
    messages = [
        SystemMessage(
            content=_SYSTEM_TEMPLATE.format(
                answer_language=_answer_language(state.get("language_type", "english"))
            )
        ),
        HumanMessage(content=state.get("raw_query", "")),
    ]

    answer = invoke_text(messages).strip()
    logger.info("generate_meaningless answer_chars=%d", len(answer))
    return {
        **state,
        "messages": [AIMessage(content=answer)],
        "answer": answer,
    }