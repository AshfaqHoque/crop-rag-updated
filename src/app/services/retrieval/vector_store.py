"""Chroma access with consistent crop/section pre-filtering."""
from functools import lru_cache
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.services.llm.embeddings import get_embeddings

logger = get_logger(__name__)


@lru_cache
def get_vector_store() -> Chroma:
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "collection_name": settings.chroma_collection,
        "embedding_function": get_embeddings(),
    }
    if settings.chroma_host:
        kwargs["client"] = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            ssl=settings.chroma_ssl,
        )
    else:
        kwargs["persist_directory"] = settings.chroma_persist_dir
    return Chroma(**kwargs)


def build_metadata_filter(
    crops: list[str] | None = None,
    sections: list[str] | None = None,
) -> dict | None:
    clauses: list[dict] = []
    if crops:
        clauses.append({"crop_name": {"$in": crops}})
    if sections:
        clauses.append({"section": {"$in": sections}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def similarity_search(
    query: str,
    *,
    k: int = 8,
    crops: list[str] | None = None,
    sections: list[str] | None = None,
) -> list[tuple[Document, float]]:
    """Return (document, distance) pairs after metadata pre-filtering."""
    try:
        where = build_metadata_filter(crops, sections)
        return get_vector_store().similarity_search_with_score(query, k=k, filter=where)
    except Exception as exc:  # noqa: BLE001
        logger.error("Vector search failed: %s", exc)
        raise RetrievalError(str(exc)) from exc


def list_documents(
    *,
    crops: list[str] | None = None,
    sections: list[str] | None = None,
) -> list[Document]:
    """Load documents for lexical retrieval using the same metadata filter."""
    try:
        where = build_metadata_filter(crops, sections)
        result = get_vector_store().get(where=where, include=["documents", "metadatas"])
        ids = result.get("ids") or []
        texts = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        docs: list[Document] = []
        for index, text in enumerate(texts):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            if "chunk_id" not in metadata and index < len(ids):
                metadata["chunk_id"] = ids[index]
            docs.append(Document(page_content=text or "", metadata=metadata))
        return docs
    except Exception as exc:  # noqa: BLE001
        logger.error("Listing Chroma documents failed: %s", exc)
        raise RetrievalError(str(exc)) from exc
