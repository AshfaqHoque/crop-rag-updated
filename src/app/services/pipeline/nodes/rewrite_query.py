"""History-aware query rewriting for subject/coreference resolution."""
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Overwrite

from app.core.logging import get_logger
from app.schemas.extraction import QueryRewrite
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM_PROMPT = """
You are a query rewriter for an agricultural RAG system.

Given the conversation history, rewrite the latest user query
into a self-contained Bangla search query that reflects exactly
what the user is asking.

Rules:
- Never reinterpret the question.
- Preserve the exact meaning and all specific entities, agricultural terms, and requested information. Never omit, generalize, or substitute them.
- If the latest query is self-contained, ignore conversation history.
- Do not answer the question.

Return only JSON:
{"rewritten_query": "বাংলা প্রশ্ন"}
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

    result = invoke_structured(QueryRewrite, messages, temperature=0.0)

    rewritten = result.rewritten_query.strip() if result.rewritten_query else ""
    rewritten = rewritten or query
    logger.info("rewrite_query rewritten=%r", rewritten)
    return {
        "rewritten_query": rewritten,
        "retrieved_documents": Overwrite([]),
        "reranked_documents": Overwrite([]),
        "compressed_documents": Overwrite([]),
    }
