"""
Centralized configuration. Every tunable value in the app is read from
here so nodes remain deterministic and easy to test.
"""
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Chat models. Ollama remains the default; Groq can be selected per deployment.
    chat_provider: Literal["ollama", "groq"] = "groq"
    ollama_chat_model: str = "gemma3:4b"
    groq_chat_model: str = "openai/gpt-oss-20b"
    groq_api_key: SecretStr | None = None

    # Ollama chat/embedding server
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"

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
    retrieval_top_k: int = 5
    dense_candidate_k: int = 20
    bm25_candidate_k: int = 20
    bm25_cache_ttl_seconds: int = 300
    rrf_k: int = 60
    rerank_top_k: int = 4
    llm_temperature: float = 0.1
    history_max_turns: int = 6
    context_max_chars_per_chunk: int = 3000

    # Reranker. FlagEmbedding is loaded lazily only when this is enabled.
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_use_fp16: bool = False
    reranker_query_max_length: int = 256
    reranker_passage_max_length: int = 512
    reranker_fail_open: bool = True

    @property
    def supported_languages_list(self) -> list[str]:
        return [lang.strip() for lang in self.supported_languages.split(",") if lang.strip()]

    @property
    def chat_model(self) -> str:
        if self.chat_provider == "groq":
            return self.groq_chat_model
        return self.ollama_chat_model


@lru_cache
def get_settings() -> Settings:
    """Settings are cached so the .env file is parsed once per process."""
    return Settings()
