from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.extraction import DecomposedQuery
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """
If the given query contains multiple entities (e.g. crops/varieties/pests/herbs/chemicals names), decompose the query into a list of subqueries, each containing a single entity.
If the query contains only one entity, return the query as it is, without any decomposition.
Even if the query contains multiple topics about a single entity, return the query as it is, without any decomposition.

Decompose ONLY by entity, never by aspect, attribute, or topic. Count entities first: if there is exactly one, return a list with the original query unchanged.
"""

def decompose_query(state: PipelineState) -> list[str]:
    query = state.get("rewritten_query") or state.get("raw_query")
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    result = invoke_structured(DecomposedQuery, messages, temperature=0.0)
    logger.info("decompose_query query=%s subqueries=%s", query, result.subqueries)
    return {"subqueries": result.subqueries}
