"""Hybrid dense and lexical retrieval with metadata filtering."""
from __future__ import annotations

from collections.abc import Callable
import re
import unicodedata

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.services.retrieval.vector_store import list_documents, similarity_search

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[\u0980-\u09FF]+|[a-z0-9]+")
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200c\u200d\u200b\ufeff"))


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.translate(_ZERO_WIDTH).translate(_BN_DIGITS).lower()
    return _TOKEN_RE.findall(normalized)


class SemanticRetriever:
    def __init__(
        self,
        dense_search: Callable[..., list[tuple[Document, float]]] = similarity_search,
        document_loader: Callable[..., list[Document]] = list_documents,
        collection_name: str | None = None,
    ) -> None:
        self._dense_search = dense_search
        self._document_loader = document_loader
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
            lexical_kwargs = {
                "crops": crops,
                "sections": sections,
            }
            if self._collection_name:
                lexical_kwargs["collection_name"] = self._collection_name
            corpus = self._document_loader(**lexical_kwargs)
        except RetrievalError:
            logger.exception("Hybrid retrieval failed")
            raise

        documents: list[Document] = []
        seen: set[str] = set()
        for document, distance in dense_results:
            metadata = dict(document.metadata)
            metadata["distance"] = float(distance)
            self._append_unique(documents, seen, Document(
                page_content=document.page_content,
                metadata=metadata,
            ))

        query_tokens = tokenize(query)
        if corpus and query_tokens:
            tokenized_documents = [
                (document, tokenize(document.page_content))
                for document in corpus
            ]
            tokenized_documents = [
                (document, tokens)
                for document, tokens in tokenized_documents
                if tokens
            ]
            if tokenized_documents:
                lexical_documents, tokenized_corpus = zip(*tokenized_documents)
                bm25 = BM25Okapi(list(tokenized_corpus))
                scores = bm25.get_scores(query_tokens)
                ranked_indices = sorted(
                    range(len(scores)),
                    key=lambda index: scores[index],
                    reverse=True,
                )[:limit]
                for index in ranked_indices:
                    self._append_unique(documents, seen, lexical_documents[index])

        return documents, "hybrid_filtered"

    @staticmethod
    def _append_unique(
        documents: list[Document],
        seen: set[str],
        document: Document,
    ) -> bool:
        chunk_id = document.metadata.get("chunk_id")
        unique_key = str(chunk_id or document.page_content)
        if unique_key in seen:
            return False
        seen.add(unique_key)
        documents.append(document)
        return True
