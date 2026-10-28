"""Generate a natural-language answer from SQL results."""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.services.llm.client import invoke_text
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

_SYSTEM = """
You are an expert agricultural advisor in Bangladesh.
Answer strictly and exclusively in {answer_language} in a natural, conversational tone.
Keep answers concise, clear and direct.

You are given:
1. The original user question
2. The rewritten user question that was used to generate the SQL query
3. The SQL query that was executed
4. The tabular result returned by the database

Rules:
- Base your answer ONLY on the SQL result.
- If the result is empty, say that you could not find matching data and offer to connect the user with an agronomist.
- Present lists cleanly (use bullet points or short numbered lists when helpful).
- Never invent rows or numbers that are not in the result.
- Do not mention SQL, tables, or the fact that you queried a database unless the user asks.
"""

def generate_sql_answer(state: PipelineState) -> PipelineState:
    language = "Bangla" if state.get("language_type") == "bn" else "English"
    results = state.get("sql_results") or []
    error = state.get("sql_error")

    context = {
        "query": state["raw_query"],
        "rewritten_query": state["rewritten_query"],
        "sql": state.get("sql_query"),
        "results": results,
        "error": error,
    }

    messages = [
        SystemMessage(content=_SYSTEM.format(answer_language=language)),
        HumanMessage(content=str(context)),
    ]

    answer = invoke_text(messages)
    logger.info("sql_answer_generated length=%d", len(answer))

    return {
        **state,
        "answer": answer.strip(),
        "messages": [AIMessage(content=answer.strip())],
    }