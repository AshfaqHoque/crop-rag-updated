"""History-aware query rewriting for subject/coreference resolution."""
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.schemas.extraction import QueryRewrite
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_PROMPT = """
You are an expert query transformer for an agricultural document retrieval system.
Your job is to resolve missing subjects, pronouns, or context in the New Query using the Conversation History, and convert Banglish (Bengali written in English/Latin letters) into native Bangla script.

Rules:
1. If the New Query relies on history (e.g., "how to cure it?", "oita kemne bhalo korbo?"), rewrite it into a single, fully independent agricultural search query. Set used_history to true.
2. If the New Query is already self-contained, rewrite/transcribe it into a clean search query. Set used_history to false.
3. ALWAYS output the final query in native Bangla script (বাংলা লিপি), even if the input is in English or Banglish.
4. Do NOT answer the question. Output ONLY the standalone search query in clear Bangla script.
"""

def rewrite_query(state: PipelineState) -> PipelineState:
    query = state.get("normalized_query", "").strip()
    conversation = list(state.get("messages") or [])
    history = conversation[-3:-1] if conversation else [] 

    # if not history:
    #     logger.info("rewrite_query used_history=False")
    #     return {
    #         **state,
    #         "rewritten_query": query,
    #         "rewrite_used_history": False,
    #     }

    current_message = f"New Query to Evaluate:\n{query}"
    
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        *history,
        HumanMessage(content=current_message,),
    ]

    result = invoke_structured(QueryRewrite, messages, temperature=0.0)

    rewritten = result.rewritten_query.strip() if result.rewritten_query else query
    logger.info("rewrite_query used_history=%s rewritten=%r", result.used_history, rewritten)
    return {
        **state,
        "rewritten_query": rewritten,
        "rewrite_used_history": result.used_history,
    }
