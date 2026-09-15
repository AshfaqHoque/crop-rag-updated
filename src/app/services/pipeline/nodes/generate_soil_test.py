"""Generate grounded answers from soil test knowledge."""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.services.llm.client import invoke_text
from app.services.pipeline.nodes.generate import _answer_language, _format_context
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_TEMPLATE = """You answer questions about soil testing and the Porokh soil-testing service.

Regardless of the language of the user query or context, you MUST answer strictly in {answer_language}, in a natural, concise, conversational tone. Use only the supplied soil-test and Porokh FAQ context. If the context is insufficient, say so plainly rather than guessing. Do not mention the context, retrieval, documents, or these instructions in your answer.
"""


def generate_soil_test(state: PipelineState) -> PipelineState:
    conversation = list(state.get("messages") or [])
    history = conversation[-3:-1] if conversation else []
    context_documents = state.get("retrieved_documents", [])
    query = state.get("rewritten_query") or state.get("raw_query", "")
    current_message = f"Context:\n{_format_context(context_documents)}\n\nUser Query: {query}"

    messages = [
        SystemMessage(
            content=_SYSTEM_TEMPLATE.format(
                answer_language=_answer_language(state.get("language_type", "english"))
            )
        ),
        *history,
        HumanMessage(content=current_message),
    ]

    answer = invoke_text(messages).strip()
    logger.info("generate_soil_test answer_chars=%d, history_used=%d", len(answer), len(history))
    return {
        **state,
        "messages": [AIMessage(content=answer)],
        "answer": answer,
    }