"""Lazy wrapper for BAAI/bge-reranker-v2-m3 via FlagEmbedding."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from app.core.config import get_settings
from app.core.exceptions import RerankingError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _default_model_loader(
    model_name: str,
    use_fp16: bool,
    query_max_length: int,
    passage_max_length: int,
):
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RerankingError(
            "FlagEmbedding is not installed. Install the 'reranker' project extra."
        ) from exc
    return FlagReranker(
        model_name,
        use_fp16=use_fp16,
        query_max_length=query_max_length,
        passage_max_length=passage_max_length,
    )


class BGEReranker:
    def __init__(self, model_loader: Callable[..., Any] = _default_model_loader) -> None:
        self._model_loader = model_loader
        self._model: Any | None = None

    def _get_model(self):
        if self._model is None:
            settings = get_settings()
            self._model = self._model_loader(
                settings.reranker_model,
                settings.reranker_use_fp16,
                settings.reranker_query_max_length,
                settings.reranker_passage_max_length,
            )
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        settings = get_settings()
        limit = top_k or settings.rerank_top_k
        if not chunks:
            return []
        if not settings.reranker_enabled:
            return [dict(chunk) for chunk in chunks[:limit]]

        pairs = [[query, str(chunk.get("content", ""))] for chunk in chunks]
        try:
            raw_scores = self._get_model().compute_score(pairs, normalize=True)
            if isinstance(raw_scores, (int, float)):
                raw_scores = [float(raw_scores)]
            if len(raw_scores) != len(chunks):
                raise RerankingError("Reranker returned a score count that does not match chunks")

            scored = []
            for chunk, score in zip(chunks, raw_scores, strict=True):
                item = dict(chunk)
                item["rerank_score"] = float(score)
                scored.append(item)
            scored.sort(key=lambda item: item["rerank_score"], reverse=True)
            return scored[:limit]
        except RerankingError:
            if settings.reranker_fail_open:
                logger.exception("Reranker failed; returning retrieval order")
                return [dict(chunk) for chunk in chunks[:limit]]
            raise
        except Exception as exc:  # noqa: BLE001
            if settings.reranker_fail_open:
                logger.exception("Reranker failed; returning retrieval order: %s", exc)
                return [dict(chunk) for chunk in chunks[:limit]]
            raise RerankingError(str(exc)) from exc

    def filter_relevant(
        self,
        queries: list[str],
        chunks: list[dict[str, Any]],
        *,
        threshold: float,
    ) -> list[dict[str, Any]]:
        settings = get_settings()
        if not chunks or not settings.reranker_enabled:
            return [dict(chunk) for chunk in chunks]

        active_queries = [query.strip() for query in queries if query and query.strip()]
        if not active_queries:
            return [dict(chunk) for chunk in chunks]

        pairs = [
            [query, str(chunk.get("content", ""))]
            for chunk in chunks
            for query in active_queries
        ]
        try:
            raw_scores = self._get_model().compute_score(pairs, normalize=True)
            if isinstance(raw_scores, (int, float)):
                raw_scores = [float(raw_scores)]
            expected_scores = len(chunks) * len(active_queries)
            if len(raw_scores) != expected_scores:
                raise RerankingError("Relevance filter returned an invalid score count")

            filtered = []
            for index, chunk in enumerate(chunks):
                chunk_scores = raw_scores[
                    index * len(active_queries):(index + 1) * len(active_queries)
                ]
                item = dict(chunk)
                item["relevance_score"] = max(float(score) for score in chunk_scores)
                if item["relevance_score"] >= threshold:
                    filtered.append(item)
            return filtered
        except RerankingError:
            if settings.reranker_fail_open:
                logger.exception("Relevance filter failed; returning retrieval order")
                return [dict(chunk) for chunk in chunks]
            raise
        except Exception as exc:  # noqa: BLE001
            if settings.reranker_fail_open:
                logger.exception("Relevance filter failed; returning retrieval order: %s", exc)
                return [dict(chunk) for chunk in chunks]
            raise RerankingError(str(exc)) from exc


@lru_cache
def get_reranker() -> BGEReranker:
    return BGEReranker()
