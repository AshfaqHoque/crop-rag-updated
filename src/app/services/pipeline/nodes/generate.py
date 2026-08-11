"""Grounded final-answer generation node."""
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.llm.client import invoke_text
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_TEMPLATE = """You are a professional crop-advisory assistant for farmers in Bangladesh.

Answer in {answer_language}.

Your job is to answer the user's latest message clearly, naturally, and concisely.

Guidelines:
- Give a direct answer first.
- Keep answers short unless the user asks for more detail.
- Use the conversation history to understand follow-up questions and references.
- Use the supplied knowledge context as the factual source for agricultural information.
- Do not invent agricultural facts, rates, doses, dates, varieties, or treatment instructions.
- If the supplied context does not contain enough information, say so briefly rather than guessing.
- Do not mention the knowledge context, retrieval, chunks, prompts, models, or internal systems.
- Do not add citations or source numbers.
- Do not repeat information unnecessarily.
- Respond naturally like a professional chatbot.
"""

# _USER_TEMPLATE = """<request>
# <intent>{intent}</intent>
# <original_message>{raw_query}</original_message>
# <standalone_query>{rewritten_query}</standalone_query>
# </request>

# <knowledge_context>
# {context}
# </knowledge_context>
# """


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
    history = state.get("history") or []
    current_message = f"""<user_message>
    {state["normalized_query"]}
    </user_message>

    <knowledge_context>
    {_format_context(
        state.get("reranked_chunks")
        or state.get("retrieved_chunks", [])
    )}
    </knowledge_context>
    """
    messages = [
        SystemMessage(
            content=_SYSTEM_TEMPLATE.format(
                answer_language=_answer_language(state.get("language", "bn"))
            )
        ),
        *history,
        HumanMessage(content=current_message)
    ]

    answer = invoke_text(messages).strip()
    logger.info("generate answer_chars=%d, length of history used=%d", len(answer), len(history))
    return {**state, "answer": answer}
