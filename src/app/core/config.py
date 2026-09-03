"""
Centralized configuration. Every tunable value in the app is read from
here so nodes remain deterministic and easy to test.
"""
import os
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Chat models. vLLM is the OpenAI-compatible local runtime used by this project;
    # Ollama and Groq remain available for other deployments.
    chat_provider: Literal["ollama", "groq", "vllm"] = "vllm"
    ollama_chat_model: str = "gemma4:12b"
    banglish_converter_model: str = "gemma4:12b"
    groq_chat_model: str = "openai/gpt-oss-20b"
    groq_api_key: SecretStr | None = None
    vllm_chat_model: str = "gemma4:12b"
    vllm_base_url: str = "http://localhost:8091/v1"
    vllm_api_key: str = "not-needed"

    # Ollama chat/embedding server
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3:latest"

    # Chroma. When chroma_host is empty, embedded/persistent Chroma is used.
    chroma_host: str | None = None
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "crop_knowledge_base"

    # Knowledge registry
    crop_registry_path: str = "./data/crops.json"

    # App
    app_env: str = "dev"
    log_level: str = "INFO"
    supported_languages: str = "bn,en"

    # Pipeline tuning
    retrieval_top_k: int = 20
    # dense_candidate_k: int = 20
    # bm25_candidate_k: int = 20
    # bm25_cache_ttl_seconds: int = 300
    # rrf_k: int = 60
    rerank_top_k: int = 10
    llm_temperature: float = 0.1
    history_max_turns: int = 1
    context_max_chars_per_chunk: int = 3000

    # External reranker service
    reranker_url: str = "http://localhost:8090/rerank"

    # LangSmith tracing
    langsmith_tracing: bool = True
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "crop-rag-chatbot"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Checkpointer
    checkpoint_backend: str = "memory"  # memory | redis | postgres | sqlite
    redis_url: str = "redis://localhost:6379/0"
    checkpoint_postgres_uri: str | None = None
    checkpoint_sqlite_path: str = "./data/checkpoints.db"

    @property
    def supported_languages_list(self) -> list[str]:
        return [lang.strip() for lang in self.supported_languages.split(",") if lang.strip()]

    @property
    def chat_model(self) -> str:
        if self.chat_provider == "groq":
            return self.groq_chat_model
        if self.chat_provider == "vllm":
            return self.vllm_chat_model
        return self.ollama_chat_model

def _apply_langsmith_env(settings: "Settings") -> None:
    """LangChain/LangGraph read these standard env vars directly (not our
    Settings object), so mirror the parsed config into os.environ once, before
    any ChatOllama/ChatGroq/graph instance gets constructed."""
    if not settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()

@lru_cache
def get_settings() -> Settings:
    """Settings are cached so the .env file is parsed once per process."""
    settings = Settings()
    _apply_langsmith_env(settings)
    return settings
