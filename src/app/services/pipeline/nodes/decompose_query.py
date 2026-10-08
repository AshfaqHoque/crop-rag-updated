from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.extraction import DecomposedQuery
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """
You are a query decomposer for agricultural search.

Task: Split the query into subqueries ONLY when it contains multiple distinct target entities (different crops, varieties, pests, diseases, or chemicals).

Hard rules:
- Decompose ONLY by entity. Never split by aspect, attribute, symptom, treatment, or question type.
- Every subquery MUST keep the full shared context.
- If the query has only one target entity, return it unchanged as a single-item list.
- Do not answer the question. Do not add or remove information.
"""

def decompose_query(state: PipelineState) -> list[str]:
    query = state.get("rewritten_query") or state.get("raw_query")
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    result = invoke_structured(DecomposedQuery, messages, temperature=0.0)
    subqueries = result.subqueries
    # if len(subqueries) > 1 and query not in subqueries:
    #     subqueries = [*subqueries, query]
    logger.info("decompose_query query=%s subqueries=%s", query, subqueries)
    return {"subqueries": subqueries}
