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
Classify the user's agricultural query into one or more sections.

Available sections:

- overview: general information about a crop
- seed: seed rate, seed quality, seed treatment, sowing
- land_preparation: soil preparation, ploughing, bed preparation
- intercultural: weeding, thinning, pruning, crop care
- irrigation: watering and irrigation
- harvest: harvesting time and harvesting method
- fertilizer: fertilizer, manure and nutrient application
- climate: temperature, rainfall, season, soil and weather requirements
- variety: crop varieties, variety comparison and variety selection
- pesticide: pests, diseases, insecticides and pesticides
- herbicide: weeds and herbicide application

Return every relevant section.
Only return values from the available section list.
"""

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