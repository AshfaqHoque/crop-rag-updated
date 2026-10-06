from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.extraction import DecomposedQuery
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """
If the given query contains multiple entities (e.g. crops/varieties/pests/herbs/chemicals names), decompose the query into a list of subqueries, each focusing on one target entity.
If the query contains only one entity, return the query as it is, without any decomposition.

Decompose ONLY by entity, never by aspect, attribute, or topic. Always retain all core context entities (crop, disease, condition, etc.) across every subquery.
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
