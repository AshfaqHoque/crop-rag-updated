"""Dense retrieval plus BM25 fallback/fusion when no crop is detected."""
from __future__ import annotations

import hashlib
import time
import unicodedata
from collections import defaultdict
from threading import RLock
from typing import Any, Callable

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.services.retrieval.vector_store import list_documents, similarity_search

logger = get_logger(__name__)


def tokenize(text: str) -> list[str]:
    """Tokenize Unicode letters, marks, and numbers without splitting Bangla words."""
    tokens: list[str] = []
    current: list[str] = []
    for char in text.casefold():
        category = unicodedata.category(char)
        if category[0] in {"L", "M", "N"}:
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _chunk_id(document: Document) -> str:
    value = document.metadata.get("chunk_id")
    if value is not None:
        return str(value)
    digest = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()[:20]
    return f"anonymous:{digest}"


def _document_payload(document: Document) -> dict[str, Any]:
    return {
        "chunk_id": _chunk_id(document),
        "content": document.page_content,
        "metadata": dict(document.metadata),
    }


class SemanticRetriever:
    def __init__(
        self,
        dense_search: Callable[..., list[tuple[Document, float]]] = similarity_search,
        document_loader: Callable[..., list[Document]] = list_documents,
    ) -> None:
        self._dense_search = dense_search
        self._document_loader = document_loader
        self._bm25_cache: dict[
            tuple[tuple[str, ...], tuple[str, ...]],
            tuple[float, list[Document], BM25Okapi],
        ] = {}
        self._cache_lock = RLock()

    def retrieve(
        self,
        query: str,
        *,
        crops: list[str] | None = None,
        sections: list[str] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        settings = get_settings()
        limit = top_k or settings.retrieval_top_k
        dense_results: list[tuple[Document, float]] = []

        try:
            dense_results = self._dense_search(
                query,
                k=limit,
                crops=crops,
                sections=sections,
            )
        except RetrievalError:
            if crops:
                raise
            logger.warning("Dense retrieval failed; attempting filtered BM25 fallback")

        # if crops:
        chunks = []
        seen: set[str] = set()
        for document, distance in dense_results:
            chunk_id = _chunk_id(document)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            item = _document_payload(document)
            item.update(
                {
                    "retrieval_score": float(distance),
                    # "retrieval_score": 1.0 / (settings.rrf_k + rank),
                }
            )
            chunks.append(item)
        return chunks[:limit], "dense_fi  ltered"

        # bm25_results = self._bm25_search(
        #     query,
        #     crops=crops,
        #     sections=sections,
        #     k=max(limit, settings.bm25_candidate_k),
        # )
        # fused = self._rrf_fuse(dense_results, bm25_results)
        # mode = "dense_bm25_rrf" if dense_results else "bm25_fallback"
        # return fused[:limit], mode

    def clear_bm25_cache(self) -> None:
        with self._cache_lock:
            self._bm25_cache.clear()

    def _get_bm25_index(
        self,
        *,
        crops: list[str] | None,
        sections: list[str] | None,
    ) -> tuple[list[Document], BM25Okapi] | None:
        settings = get_settings()
        key = (tuple(sorted(crops or [])), tuple(sorted(sections or [])))
        now = time.monotonic()

        if settings.bm25_cache_ttl_seconds > 0:
            with self._cache_lock:
                cached = self._bm25_cache.get(key)
                if cached and cached[0] > now:
                    return cached[1], cached[2]

        documents = self._document_loader(crops=crops, sections=sections)
        tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
        if not documents or not any(tokenized_corpus):
            return None
        index = BM25Okapi(tokenized_corpus)

        if settings.bm25_cache_ttl_seconds > 0:
            with self._cache_lock:
                expires_at = now + settings.bm25_cache_ttl_seconds
                self._bm25_cache[key] = (expires_at, documents, index)
        return documents, index

    def _bm25_search(
        self,
        query: str,
        *,
        crops: list[str] | None,
        sections: list[str] | None,
        k: int,
    ) -> list[tuple[Document, float]]:
        corpus = self._get_bm25_index(crops=crops, sections=sections)
        if corpus is None:
            return []
        documents, index = corpus
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = index.get_scores(query_tokens)
        ranked = sorted(
            zip(documents, scores, strict=False),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        return [
            (doc, float(score))
            for doc, score in ranked[:k]
            if abs(float(score)) > 1e-12
        ]

    def _rrf_fuse(
        self,
        dense_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
    ) -> list[dict[str, Any]]:
        rrf_k = get_settings().rrf_k
        scores: dict[str, float] = defaultdict(float)
        payloads: dict[str, dict[str, Any]] = {}

        seen_dense: set[str] = set()
        for rank, (document, distance) in enumerate(dense_results, start=1):
            chunk_id = _chunk_id(document)
            if chunk_id in seen_dense:
                continue
            seen_dense.add(chunk_id)
            payloads.setdefault(chunk_id, _document_payload(document))
            payloads[chunk_id]["dense_distance"] = float(distance)
            scores[chunk_id] += 1.0 / (rrf_k + rank)

        seen_bm25: set[str] = set()
        for rank, (document, score) in enumerate(bm25_results, start=1):
            chunk_id = _chunk_id(document)
            if chunk_id in seen_bm25:
                continue
            seen_bm25.add(chunk_id)
            payloads.setdefault(chunk_id, _document_payload(document))
            payloads[chunk_id]["bm25_score"] = float(score)
            scores[chunk_id] += 1.0 / (rrf_k + rank)

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        result = []
        for chunk_id in ranked_ids:
            item = payloads[chunk_id]
            item["retrieval_score"] = scores[chunk_id]
            result.append(item)
        return result
