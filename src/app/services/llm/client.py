"""Central Ollama chat-model wrapper with retries and normalized errors."""
from collections.abc import Sequence
from functools import lru_cache
from typing import TypeVar

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import LLMGenerationError
from app.core.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)
PromptInput = str | Sequence[BaseMessage]


@lru_cache
def get_chat_llm(temperature: float | None = None) -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.chat_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
    )


def get_structured_llm(schema: type[T], *, temperature: float | None = None):
    return get_chat_llm(temperature).with_structured_output(schema)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4), reraise=True)
def invoke_structured(
    schema: type[T],
    prompt: PromptInput,
    *,
    temperature: float | None = None,
) -> T:
    try:
        result = get_structured_llm(schema, temperature=temperature).invoke(prompt)
        if not isinstance(result, schema):
            raise LLMGenerationError(f"Model did not return expected schema {schema.__name__}")
        return result
    except LLMGenerationError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Structured LLM call failed: %s", exc)
        raise LLMGenerationError(str(exc)) from exc


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, max=4), reraise=True)
def invoke_text(prompt: PromptInput, *, temperature: float | None = None) -> str:
    try:
        result = get_chat_llm(temperature).invoke(prompt)
        content = result.content
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            ).strip()
        else:
            text = str(content or "").strip()
        if not text:
            raise LLMGenerationError("Model returned an empty response")
        return text
    except LLMGenerationError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Text LLM call failed: %s", exc)
        raise LLMGenerationError(str(exc)) from exc
