"""Decide whether a crop_query should be answered via SQL or semantic RAG."""
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.schemas.sql import RetrievalModeDecision
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState
from app.services.sql.schema import DATABASE_SCHEMA   # we will put the schema string here

logger = get_logger(__name__)

_SYSTEM = """You are the retrieval-mode router of an agricultural assistant.

Database Schema:
{schema}

You must choose exactly one mode:

- "sql": any listing, sorting, ranking, counting, filtering, or lookup, as long as everything the question asks for or filters by is a column in the schema above.
- "semantic": everything else, including any question that asks for or filters by something not in the schema, plus advice, procedures, and reasoning.

If any part of the question is not in the schema, choose "semantic". Never answer the user.
Return two fields: "reason" (name the table/columns that answer it, or the part with no column) and "mode" ("sql" or "semantic").
"""

def decide_retrieval_mode(state: PipelineState) -> PipelineState:
    messages = [
        SystemMessage(content=_SYSTEM.format(schema=DATABASE_SCHEMA)),
        HumanMessage(content=(f"User Query:\n{state['rewritten_query']}")),
    ]
    result = invoke_structured(RetrievalModeDecision, messages, temperature=0)

    logger.info("retrieval_mode=%s , reason=%s", result.mode, result.reason)
    return {
        **state,
        "retrieval_mode": result.mode,
        "retrieval_mode_reason": result.reason,
    }