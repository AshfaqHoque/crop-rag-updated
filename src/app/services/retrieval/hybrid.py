"""Dense retrieval with metadata filtering."""
from __future__ import annotations

from collections.abc import Callable

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.services.retrieval.vector_store import similarity_search

logger = get_logger(__name__)


class SemanticRetriever:
    def __init__(
        self,
        dense_search: Callable[..., list[tuple[Document, float]]] = similarity_search,
        collection_name: str | None = None,
    ) -> None:
        self._dense_search = dense_search
        self._collection_name = collection_name

    def retrieve(
        self,
        query: str,
        *,
        crops: list[str] | None = None,
        sections: list[str] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[Document], str]:
        settings = get_settings()
        limit = top_k or settings.retrieval_top_k

        try:
            search_kwargs = {
                "k": limit,
                "crops": crops,
                "sections": sections,
            }
            if self._collection_name:
                search_kwargs["collection_name"] = self._collection_name
            dense_results = self._dense_search(query, **search_kwargs)
        except RetrievalError:
            logger.exception("Dense retrieval failed")
            raise

        documents = []
        seen: set[str] = set()
        for document, distance in dense_results:
            chunk_id = str(document.metadata.get("chunk_id", document.page_content))
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            metadata = dict(document.metadata)
            metadata["distance"] = float(distance)
            documents.append(
                Document(page_content=document.page_content, metadata=metadata)
            )
        return documents[:limit], "dense_filtered"
