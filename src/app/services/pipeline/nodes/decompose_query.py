from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.extraction import DecomposedQuery
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState

from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """
You are an agricultural query decomposer.

Decompose the user's question into independent, self-contained questions when it involves multiple entities.

Rules:
1. Create one subquery per entity or distinct information need.
2. Preserve the original intent, conditions, and relevant context in every subquery.
3. Include the parent crop context in each child-entity(varieties/pests/diseases/chemicals/fertilizers) subquery.
4. If only one entity is involved, do not decompose.
5. Do not answer the questions or add new information.

Output format (valid JSON only):
{
    "subqueries": ["question 1", "question 2"]
}
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
