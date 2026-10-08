"""History-aware query rewriting for subject/coreference resolution."""
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Overwrite

from app.core.logging import get_logger
from app.schemas.extraction import QueryRewrite
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_PROMPT = """
You are an expert query rewriter for an agricultural document retrieval system. The documents are in Bangla.

Task: Rewrite the user message into one clean search query in Bangla Language.

Rules:
- Output ONLY native Bangla script (বাংলা লিপি).
- Resolve pronouns and coreferences using the conversation history.
- Preserve the user's exact intent.
- Convert entity names (crops, varieties, diseases, chemicals, etc.) carefully. Do not change their meaning.
- Do NOT answer the question.

The JSON object MUST contain exactly this field:
   "rewritten_query": string
"""

def rewrite_query(state: PipelineState) -> PipelineState:
    query = (state.get("raw_query") or "").strip()
    conversation = list(state.get("messages") or [])
    history = conversation[-9:-1] if conversation else []

    # current_message = f"Rewrite this query:\n{query}"
    
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        *history,
        HumanMessage(content=query),
    ]

    result = invoke_structured(QueryRewrite, messages, temperature=0.0, provider="vllm")

    rewritten = result.rewritten_query.strip() if result.rewritten_query else ""
    rewritten = rewritten or query
    logger.info("rewrite_query rewritten=%r", rewritten)
    return {
        "rewritten_query": rewritten,
        "retrieved_documents": Overwrite([]),
        "reranked_documents": Overwrite([]),
        "compressed_documents": Overwrite([]),
    }
