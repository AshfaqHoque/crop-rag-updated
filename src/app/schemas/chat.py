from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=2000)

    @field_validator("session_id", "message")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class SourceChunk(BaseModel):
    chunk_id: str
    crop_name: str | None = None
    section: str | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    language: str
    rewritten_query: str | None = None
    retrieval_mode: str | None = None
    sources: list[SourceChunk] = Field(default_factory=list)
