from typing import Literal

from pydantic import BaseModel, Field


class QueryUnderstanding(BaseModel):
    language: Literal["bn", "en", "unsupported"] = Field(
        description="Language of the user's current message."
    )
    intent: Literal["small_talk", "unclear", "crop_query"] = Field(
        description="The conversational intent of the current message."
    )
    crops: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Canonical crop names from the supplied registry only.",
    )
    sections: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Canonical section values from the supplied registry only.",
    )


class QueryRewrite(BaseModel):
    rewritten_query: str = Field(
        min_length=1,
        max_length=4000,
        description="A standalone version of the user's question, without answering it.",
    )
    used_history: bool = Field(
        description="True only when conversation history was required to resolve the subject."
    )
