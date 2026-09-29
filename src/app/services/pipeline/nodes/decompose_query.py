from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.extraction import DecomposedQuery
from app.services.llm.client import invoke_structured

_SYSTEM_PROMPT = """
If the given query contains multiple entities (e.g. crops/varieties/pests/herbs/chemicals names), decompose the query into a list of subqueries, each containing a single entity.
If the query contains only one entity, return the query as it is, without any decomposition.
"""

query = "how many varieties of Aman Rice are there in Bangladesh?"

messages = [
    SystemMessage(content=_SYSTEM_PROMPT),
    HumanMessage(content=query),
]

result = invoke_structured(DecomposedQuery, messages, temperature=0.0)

print(result.subqueries)