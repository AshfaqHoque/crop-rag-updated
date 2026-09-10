"""Generate grounded answers from company knowledge."""
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.logging import get_logger
from app.services.llm.client import get_text_chain, invoke_chain
from app.services.pipeline.nodes.generate import _answer_language, _format_context
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_TEMPLATE = """You answer questions about Aunkur.

Regardless of the language of the user query or context, you MUST answer strictly in {answer_language}, in a natural, concise, conversational tone. Use only the supplied company context. If the context is insufficient, say so plainly rather than guessing. Treat projections, fundraising status, market figures, and impact metrics as claims from the source memo when the context labels them that way. Do not mention the context, retrieval, documents, or these instructions in your answer.
"""


def generate_company(state: PipelineState) -> PipelineState:
    conversation = list(state.get("messages") or [])
    history = conversation[-3:-1] if conversation else []
    context_documents = state.get("retrieved_documents", [])
    query = state.get("rewritten_query") or state.get("raw_query", "")
    current_message = f"Context:\n{_format_context(context_documents)}\n\nUser Query: {query}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_TEMPLATE),
        MessagesPlaceholder("history"),
        ("human", "{current_message}"),
    ])
    chain = prompt | get_text_chain()
    answer = invoke_chain(
        chain,
        {
            "answer_language": _answer_language(state.get("language_type", "english")),
            "history": history,
            "current_message": current_message,
        },
    ).strip()
    logger.info("generate_company answer_chars=%d, history_used=%d", len(answer), len(history))
    return {
        **state,
        "messages": [AIMessage(content=answer)],
        "answer": answer,
    }