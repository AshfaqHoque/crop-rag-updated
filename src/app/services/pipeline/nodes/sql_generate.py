from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.schemas.sql import SQLResult
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState
from app.services.sql.schema import DATABASE_SCHEMA

logger = get_logger(__name__)

_SYSTEM = """
You are an SQL Generator.

Convert the provided structured query plan into a valid SQL SELECT query.

Rules:
- Use only tables, columns, and relationships in the provided schema.
- Follow the query plan exactly.
- Generate only read-only SELECT queries.
- Use JOINs when relationships are required.
- Use DISTINCT when duplicate results are possible.
- Do not invent tables, columns, or relationships.
- Return only the SQL query in the "sql" field.
- Do not include markdown or explanations.
"""

def sql_generate(state: PipelineState) -> PipelineState:
    plan = state.get("query_plan") or {}
    if plan.get("operation") == "unsupported":
        return {**state, "sql_query": None, "sql_error": "unsupported operation"}

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=f"""
Database Schema:
{DATABASE_SCHEMA}

Query Plan:
{plan}
"""
        ),
    ]
    result = invoke_structured(SQLResult, messages, temperature=0)
    logger.info("generated_sql=%s", result.sql[:200])
    return {
        **state,
        "sql_query": result.sql,
    } 