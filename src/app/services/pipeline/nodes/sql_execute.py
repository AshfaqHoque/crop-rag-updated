from app.core.logging import get_logger
from app.services.pipeline.state import PipelineState
from app.services.sql.db import execute_readonly

logger = get_logger(__name__)

def sql_execute(state: PipelineState) -> PipelineState:
    sql = state.get("sql_query")
    if not sql:
        return {**state, "sql_results": [], "sql_error": "no SQL generated"}

    try:
        rows = execute_readonly(sql)
        logger.info("sql_execute rows=%d", len(rows))
        return {**state, "sql_results": rows, "sql_error": None}
    except Exception as exc:
        logger.exception("SQL execution failed")
        return {**state, "sql_results": [], "sql_error": str(exc)}