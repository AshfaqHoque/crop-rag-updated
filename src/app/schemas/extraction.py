from typing import Literal

from pydantic import BaseModel, Field


class QueryUnderstanding(BaseModel):
    intent: Literal["chitchat", "company_query", "crop_query", "meaningless", "soil_test_query", "request_agronomist"] = Field(
        description="The conversational intent of the current message."
    )


class QueryRewrite(BaseModel):
    rewritten_query: str


class DescriptiveQuery(BaseModel):
    descriptive: bool = Field(
        description="Whether the query requests broad coverage across multiple entities or sections."
    )


class CropExtraction(BaseModel):
    crops: list[str] = Field(
        default_factory=list,
        description="Crop names selected from the provided crop registry.",
    )


class RelevantChunks(BaseModel):
    relevant_indexes: list[int] = Field(
        default_factory=list,
        description="Zero-based indexes of chunks that directly help answer the question.",
    )

class ChunkRelevance(BaseModel):
    relevant: bool = Field(
        description="True only if this single chunk contains information needed to answer the current question."
    )

class LLMSplitResult(BaseModel):
    chunks: list[str]