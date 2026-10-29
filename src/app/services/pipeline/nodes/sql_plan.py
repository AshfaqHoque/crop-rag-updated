from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.schemas.sql import QueryPlan
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState
from app.services.sql.schema import DATABASE_SCHEMA

logger = get_logger(__name__)

_SYSTEM = """
You are the SQL Query Planner of an agricultural assistant.

Do not answer the user. Do not generate SQL.
Convert the user's question into a structured query plan using the provided database schema.

Identify:
- operation
- target
- entities_to_resolve
- sort
- limit

Rules:
- Use only tables, columns, and relationships in the schema.
- Do not correct spelling or invent canonical names.
- Never guess entity IDs.
- Do not invent fields or relationships.
- If the query cannot be answered using the schema, return operation="unsupported".
- Return only valid JSON matching the required structure.

Output format:
{
  "operation": "list",
  "target": "table or field",
  "entities_to_resolve": [
    {
      "type": "entity type",
      "mention": "exact user-mentioned name"
    }
  ],
  "sort": null,
  "limit": null,
}
"""

def sql_plan(state: PipelineState) -> PipelineState:
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=f"Database Schema:\n{DATABASE_SCHEMA}\n\nUser Query:\n{state['rewritten_query']}"
        ),
    ]
    plan = invoke_structured(QueryPlan, messages, temperature=0)
    logger.info("sql_plan operation=%s target=%s", plan.operation, plan.target)
    return {
        **state,
        "query_plan": plan.model_dump(),
    }