"""History-aware query rewriting for subject/coreference resolution."""
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.schemas.extraction import QueryRewrite
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_PROMPT = """
You are an expert query transformer for an agricultural document retrieval system.
Your job is to resolve missing subjects, pronouns, or context in the New Query using the Conversation History so it can be looked up in a database.

Rules:
1. If the New Query relies on history (e.g., "how to cure it?", "how much dose?"), rewrite it into a single, fully independent agricultural search query in Bengali. Set used_history to true.
2. The rewritten query must include the explicit crop name, disease name, or topic from the history.
3. If the New Query is already self-contained and mentions the crop/subject explicitly, do NOT rewrite it. Set used_history to false.
4. Do NOT answer the question. Output ONLY the standalone search query in clear Bengali.
"""  # noqa: E501

def rewrite_query(state: PipelineState) -> PipelineState:
    query = state.get("normalized_query", "").strip()
    conversation = list(state.get("messages") or [])
    history = conversation[-3:-1] if conversation else [] 

    if not history:
        logger.info("rewrite_query used_history=False")
        return {
            **state,
            "rewritten_query": query,
            "rewrite_used_history": False,
        }

    current_message = f"New Query to Evaluate:\n{query}"
    
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        *history,
        HumanMessage(content=current_message,),
    ]

    result = invoke_structured(QueryRewrite, messages, temperature=0.0)

    rewritten = result.rewritten_query.strip() if result.used_history else query
    logger.info("rewrite_query used_history=%s rewritten=%r", result.used_history, rewritten)
    return {
        **state,
        "rewritten_query": rewritten,
        "rewrite_used_history": result.used_history,
    }
