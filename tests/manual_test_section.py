from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel

SectionName = Literal[
    "overview",
    "seed",
    "land_preparation",
    "intercultural",
    "irrigation",
    "harvest",
    "fertilizer",
    "climate",
    "variety",
    "pesticide",
    "herbicide",
]

PROMPT = """
Classify the user's agricultural query into one or more relevant sections.

Available sections:

- overview: general crop information or a broad cultivation overview
- seed: seed selection, seed rate, treatment, nursery, sowing, planting or transplanting
- land_preparation: tillage, ploughing, leveling, beds, pits or field preparation
- intercultural: crop-care operations after planting, excluding irrigation, fertilizer and chemical protection
- irrigation: watering, irrigation scheduling, drainage or water management
- harvest: maturity, harvesting, yield, post-harvest handling or storage
- fertilizer: fertilizers, manure, nutrients, deficiencies, doses or application schedules
- climate: soil suitability, season, temperature, rainfall, humidity, sunlight or other growing conditions
- variety: variety identification, selection, characteristics, suitability or comparison
- pesticide: pests, diseases and their prevention, diagnosis or control
- herbicide: chemical control of weeds

Rules:

1. Classify by meaning, not by exact words or language.
2. Return every section directly requested by the query.
3. Do not include sections that are only indirectly related.
4. Distinguish sections according to the purpose of the requested information.
5. A query may belong to multiple sections.
6. For a broad end-to-end cultivation question, return all major sections needed to answer it.
7. For general crop information without a specific topic, return only overview.
8. Return no duplicate sections.
9. If no section can be confidently identified, return an empty list.
10. If the query explicitly mentions a specific crop variety (e.g. BRRI Dhan-28, BINA Dhan-17, Hybrid Maize-9),
include "variety" together with the primary requested section, even if the user is asking about another topic such as seed, fertilizer or irrigation.

Output only the structured result.
"""  # noqa: E501

class SectionResult(BaseModel):
    sections: list[SectionName]

llm = ChatOllama(model="gemma3:4b", temperature=0)

structured_llm = llm.with_structured_output(SectionResult)

prompt = ChatPromptTemplate.from_messages([
    ("system", PROMPT),
    ("human", "{query}")
])

chain = prompt | structured_llm

while True:
    query = input("Enter your query: ")
    result = chain.invoke({"query": query})
    print(result.sections)