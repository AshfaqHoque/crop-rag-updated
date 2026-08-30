"""History-aware query rewriting for subject/coreference resolution."""
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.extraction import QueryRewrite
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_PROMPT = """
Given the following conversation, rewrite the last user input to reflect what the user is actually asking. Write in Bangla Language.
"""

def rewrite_query(state: PipelineState) -> PipelineState:
    query = state.get("normalized_query", "").strip()
    conversation = list(state.get("messages") or [])
    history = conversation[:-1] if conversation else [] 

    if not history:
        logger.info("rewrite_query used_history=False")
        return {
            **state,
            "rewritten_query": query,
            "rewrite_used_history": False,
        }

    current_message = f"""<user_message>
    {query}
    </user_message>
    """
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        *history,
        HumanMessage(content=current_message,),
    ]

    result = invoke_structured(QueryRewrite, messages, temperature=0.0)

    rewritten = result.rewritten_query.strip() if result.used_history else query
    logger.info("rewrite_query used_history=%s rewritten=%r", result.used_history, rewritten)  # noqa: E501
    return {
        **state,
        "rewritten_query": rewritten,
        "rewrite_used_history": result.used_history,
    }
