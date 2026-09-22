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
1. If the New Query relies on history (e.g., "how to cure it?", "oita kemne bhalo korbo?"), rewrite it into a single, fully independent agricultural search query.
2. If the New Query is already self-contained, rewrite/transcribe it into a clean search query.
3. If the New Query is phrased negatively or as an exclusion (e.g., "X chara"), rewrite it into the positive underlying question — what the user is actually trying to find out — since documents state facts affirmatively and negated queries retrieve poorly.
4. ALWAYS output the final query in native Bangla script (বাংলা লিপি), even if the input is in English or Banglish.
4. Do NOT answer the question.

The JSON object MUST contain exactly this field:
   "rewritten_query": string
"""

def rewrite_query(state: PipelineState) -> PipelineState:
    query = (state.get("raw_query") or "").strip()
    conversation = list(state.get("messages") or [])
    history = conversation[-9:-1] if conversation else []

    current_message = f"New Query to Evaluate:\n{query}"
    
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        *history,
        HumanMessage(content=current_message,),
    ]

    result = invoke_structured(QueryRewrite, messages, temperature=0.0)

    rewritten = result.rewritten_query.strip() if result.rewritten_query else ""
    rewritten = rewritten or query
    logger.info("rewrite_query rewritten=%r", rewritten)
    return {
        **state,
        "rewritten_query": rewritten,
    }
