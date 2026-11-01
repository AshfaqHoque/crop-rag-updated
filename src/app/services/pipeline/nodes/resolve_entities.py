"""Resolve entity mentions via semantic search + LLM."""
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.services.llm.client import invoke_structured
from app.services.pipeline.state import PipelineState
from app.services.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)

_ALIAS = {
    # Crop
    "crop": "crop",
    "crop_name": "crop",
    "crop.crop_name": "crop",

    # Crop category
    "crop_category": "category",
    "category": "category",
    "crop_category.name": "category",

    # Variety
    "variety": "variety",
    "variety_name": "variety",
    "variety.variety_name": "variety",

    # Pest or disease
    "pesticide": "disease",
    "disease": "disease",
    "disease_name": "disease",
    "pesticide.disease_name": "disease",

    # Weed
    "herbicide": "weed",
    "weed": "weed",
    "weed_name": "weed",
    "herbicide.weed_name": "weed",

    # Chemical
    "chemical": "chemical",
    "trade_name": "chemical",
    "generic_name": "chemical",
    "chemical.trade_name": "chemical",
    "chemical.generic_name": "chemical",

    # Company
    "company": "company",
    "company.name": "company",
}

class EntityChoice(BaseModel):
    selected_name: Optional[str] = Field(description="Best match from candidates, or null")
    confidence: float = Field(ge=0.0, le=1.0)


def _search(etype: str, mention: str, k: int = 5) -> list[str]:
    etype = _ALIAS.get(etype, etype)
    hits = get_vector_store("sql_entities").similarity_search_with_score(
        mention, k=k, filter={"entity_type": etype}
    )
    seen, names = set(), []
    for doc, _ in sorted(hits, key=lambda x: x[1]):
        name = doc.metadata.get("canonical_name") or doc.page_content
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


_SYSTEM_PROMPT = """
Pick the best matching name from the candidates. Return null if none fit. Never invent names.
Output a JSON object with two fields:
- `selected_name`: The best matching name or null
- `confidence`: A confidence score between 0 and 1
"""

def _pick(etype: str, mention: str, candidates: list[str]) -> str | None:
    lines = "\n".join(f"{i}. {n}" for i, n in enumerate(candidates, 1))
    choice = invoke_structured(EntityChoice, [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Type: {etype}\nMention: {mention}\nCandidates:\n{lines}"),
    ], temperature=0)
    if choice.selected_name in candidates and choice.confidence >= 0.35:
        return choice.selected_name
    return None


def resolve_entities(state: PipelineState) -> PipelineState:
    plan = state.get("query_plan")
    if not plan or plan.get("operation") == "unsupported":
        return state
    entities = plan.get("entities_to_resolve") or []
    if not entities:
        return state

    resolved = []
    for ent in entities:
        etype = (ent.get("type") or "").strip().lower()
        mention = (ent.get("mention") or "").strip()
        if not etype or not mention:
            resolved.append(ent)
            continue
        cands = _search(etype, mention)
        name = _pick(etype, mention, cands) if cands else None
        if name:
            logger.info("resolved %s %r → %r", etype, mention, name)
        resolved.append({"type": etype, "mention": name or mention})

    return {**state, "query_plan": {**plan, "entities_to_resolve": resolved}}