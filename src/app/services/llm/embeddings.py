"""bge-m3 dense embeddings served through Ollama.

Ollama exposes the dense vector path. The lexical side of hybrid retrieval
is implemented separately with BM25 in ``services/retrieval/hybrid.py``.
"""
from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from app.core.config import get_settings


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    settings = get_settings()
    return OllamaEmbeddings(base_url=settings.ollama_base_url, model=settings.embed_model)
