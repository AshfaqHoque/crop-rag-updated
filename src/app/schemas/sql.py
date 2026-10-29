from typing import Literal, Optional
from pydantic import BaseModel, Field


class EntityToResolve(BaseModel):
    type: str
    mention: str


class QueryPlan(BaseModel):
    operation: Literal["list", "lookup", "count", "aggregate", "unsupported"]
    target: str | None = None
    entities_to_resolve: list[EntityToResolve] = Field(default_factory=list)
    sort: str | None = None
    limit: int | None = None
    relationship: str | None = None


class SQLResult(BaseModel):
    sql: str


class RetrievalModeDecision(BaseModel):
    mode: Literal["sql", "semantic"]
    # reason: Optional[str] = Field(description="One short sentence explaining the choice")