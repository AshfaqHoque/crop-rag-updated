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
    ollama_chat_model: str = "gemma4:31b-cloud"
    groq_chat_model: str = "openai/gpt-oss-20b"
    groq_api_key: SecretStr | None = None
    vllm_chat_model: str = "gemma4:12b"
    vllm_base_url: str = "http://localhost:8091/v1"
    vllm_api_key: str = "not-needed"
    vllm_top_p: float = 0.9
    vllm_top_k: int = 30

    # Ollama chat/embedding server
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3:latest"

    # Chroma. When chroma_host is empty, embedded/persistent Chroma is used.
    chroma_host: str | None = None
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "crop_knowledge_base"
    chroma_company_collection: str = "company_knowledge_base"
    chroma_soil_test_collection: str = "soil_test_knowledge_base"

    # Knowledge registry
    crop_registry_path: str = "./data/crops.json"
    graphql_endpoint: str = "https://aunkur-backend-311104304042.us-central1.run.app/graphql"
    graphql_timeout_seconds: float = 30.0
    # App
    app_env: str = "dev"
    log_level: str = "INFO"

    # Pipeline tuning
    retrieval_top_k: int = 20
    company_retrieval_top_k: int = 3
    soil_test_retrieval_top_k: int = 3
    rerank_top_k: int = 7
    llm_temperature: float = 0
    context_max_chars_per_chunk: int = 4000
    compression_temperature: float = 0.1
    compression_max_tokens: int = 1000
    rerank_gap_multiplier: float = 60.0
    rerank_min_score_delta: float = 0.65
    stream_chunk_delay_seconds: float = 0.1

    # External reranker service
    reranker_url: str = "http://localhost:8090/rerank"
    reranker_timeout_seconds: float = 30.0

    # LangSmith tracing
    langsmith_tracing: bool = True
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "aunkur-chat-api"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Checkpointer
    checkpoint_backend: str = "redis"  # memory | redis
    redis_url: str = "redis://localhost:6379/0"

    # Streamlit frontend
    ui_api_url: str = "http://localhost:8000/api/v1/chat/stream"
    ui_request_timeout_seconds: float = 120.0

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
