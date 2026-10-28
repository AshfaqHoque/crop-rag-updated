"""Decide whether a crop_query should be answered via SQL or semantic RAG."""
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.schemas.sql import RetrievalModeDecision
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState
from app.services.sql.schema import DATABASE_SCHEMA   # we will put the schema string here

logger = get_logger(__name__)

_SYSTEM = """You are the retrieval-mode router of an agricultural assistant.

You must choose exactly one mode:

- "sql": when the answer is a list, count, ranking, filter, or simple lookup that maps cleanly to the schema.
- "semantic": for any open-ended agricultural advice, procedures, cause-effect, or multi-hop reasoning that is not stored as structured columns.

If in doubt, choose "semantic".
Never answer the user. Return only the structured decision.
"""

def decide_retrieval_mode(state: PipelineState) -> PipelineState:
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=(
                f"Database Schema:\n{DATABASE_SCHEMA}\n\n"
                f"User Query:\n{state['rewritten_query']}"
            )
        ),
    ]
    result = invoke_structured(RetrievalModeDecision, messages, temperature=0)

    logger.info("retrieval_mode=%s ", result.mode)
    return {
        **state,
        "retrieval_mode": result.mode,
        # "retrieval_mode_reason": result.reason,
    }