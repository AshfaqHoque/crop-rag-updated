from typing import Literal

from pydantic import BaseModel, Field


class QueryUnderstanding(BaseModel):
    language: Literal["bn", "en", "unsupported"] = Field(
        description="Language of the user's current message."
    )
    intent: Literal["small_talk", "unclear", "crop_query"] = Field(
        description="The conversational intent of the current message."
    )
    sections: list[str] = Field(
        default_factory=list,
        # max_length=10,
        description="Canonical section values from the supplied registry only.",
    )


class QueryRewrite(BaseModel):
    rewritten_query: str
    used_history: bool


class RelevantChunks(BaseModel):
    relevant_indexes: list[int] = Field(
        default_factory=list,
        description="Zero-based indexes of chunks that directly help answer the question.",
    )

class ChunkRelevance(BaseModel):
    relevant: bool = Field(
        description="True only if this single chunk contains information needed to answer the current question."
    )