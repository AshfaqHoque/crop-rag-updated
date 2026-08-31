"""Central chat-model wrapper with provider selection, retries, and normalized errors."""
from collections.abc import Sequence
from functools import lru_cache
from typing import TypeVar

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import LLMGenerationError
from app.core.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)
PromptInput = str | Sequence[BaseMessage]


@lru_cache
def get_ollama_chat_llm(temperature: float | None = None) -> ChatOllama:
    settings = get_settings()
    logger.info("Initializing chat model provider=ollama model=%s", settings.ollama_chat_model)
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        reasoning=False,
        temperature=settings.llm_temperature if temperature is None else temperature,
    )


@lru_cache
def get_groq_chat_llm(temperature: float | None = None) -> ChatGroq:
    settings = get_settings()
    logger.info("Initializing chat model provider=groq model=%s", settings.groq_chat_model)
    api_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
    return ChatGroq(
        api_key=api_key,
        model=settings.groq_chat_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_retries=0,  # Retries are handled by invoke_text/invoke_structured below.
    )


@lru_cache
def get_vllm_chat_llm(temperature: float | None = None) -> ChatOpenAI:
    settings = get_settings()
    logger.info("Initializing chat model provider=vllm model=%s", settings.vllm_chat_model)
    return ChatOpenAI(
        base_url=settings.vllm_base_url,
        model=settings.vllm_chat_model,
        api_key=settings.vllm_api_key,
        temperature=1.0,
        top_p=0.95,
        extra_body={
            "top_k": 64,
        },
    )

@lru_cache
def get_chat_llm(temperature: float | None = None) -> ChatOllama | ChatGroq | ChatOpenAI:
    settings = get_settings()
    if settings.chat_provider == "groq":
        return get_groq_chat_llm(temperature)
    if settings.chat_provider == "vllm":
        return get_vllm_chat_llm(temperature)
    return get_ollama_chat_llm(temperature)


def get_structured_llm(schema: type[T], *, temperature: float | None = None):
    llm = get_chat_llm(temperature)
    if get_settings().chat_provider == "groq":
        # ChatGroq defaults to function/tool calling, which can fail when the
        # model emits plain text instead of the required tool call. GPT-OSS
        # supports Groq's native JSON Schema response format directly.
        return llm.with_structured_output(schema, method="json_mode")
    if get_settings().chat_provider == "vllm":
        return llm.with_structured_output(schema)
    return llm.with_structured_output(schema)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4), reraise=True)
def invoke_structured(
    schema: type[T],
    prompt: PromptInput,
    *,
    temperature: float | None = None,
) -> T:
    try:
        settings = get_settings()
        result = get_structured_llm(schema, temperature=temperature).invoke(prompt)
        if not isinstance(result, schema):
            raise LLMGenerationError(f"Model did not return expected schema {schema.__name__}")
        return result
    except LLMGenerationError:
        raise
    except Exception as exc:
        logger.warning("Structured LLM call failed: %s", exc)
        raise LLMGenerationError(str(exc)) from exc


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, max=4), reraise=True)
def invoke_text(prompt: PromptInput, *, temperature: float | None = None) -> str:
    try:
        settings = get_settings()
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
    except Exception as exc:
        logger.warning("Text LLM call failed: %s", exc)
        raise LLMGenerationError(str(exc)) from exc
