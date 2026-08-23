from app.core.config import get_settings
from app.services.reranking.bge import BGEReranker


class FakeModel:
    def compute_score(self, pairs, normalize=True):
        assert normalize is True
        assert len(pairs) == 3
        return [0.1, 0.9, 0.4]


class FakeFilterModel:
    def compute_score(self, pairs, normalize=True):
        assert normalize is True
        assert len(pairs) == 4
        return [0.2, 0.3, 0.8, 0.4]


def test_reranker_orders_by_cross_encoder_score(monkeypatch):
    monkeypatch.setattr(get_settings(), "reranker_enabled", True)
    reranker = BGEReranker(model_loader=lambda *args: FakeModel())
    chunks = [
        {"chunk_id": "a", "content": "a"},
        {"chunk_id": "b", "content": "b"},
        {"chunk_id": "c", "content": "c"},
    ]
    result = reranker.rerank("query", chunks, top_k=2)
    assert [item["chunk_id"] for item in result] == ["b", "c"]
    assert result[0]["rerank_score"] == 0.9


def test_disabled_reranker_preserves_retrieval_order(monkeypatch):
    monkeypatch.setattr(get_settings(), "reranker_enabled", False)
    chunks = [{"chunk_id": "a", "content": "a"}, {"chunk_id": "b", "content": "b"}]
    assert [x["chunk_id"] for x in BGEReranker().rerank("q", chunks, top_k=1)] == ["a"]


def test_relevance_filter_uses_best_score_from_original_and_rewritten_queries(monkeypatch):
    monkeypatch.setattr(get_settings(), "reranker_enabled", True)
    reranker = BGEReranker(model_loader=lambda *args: FakeFilterModel())
    chunks = [
        {"chunk_id": "a", "content": "a"},
        {"chunk_id": "b", "content": "b"},
    ]

    result = reranker.filter_relevant(["original", "rewritten"], chunks, threshold=0.5)

    assert [item["chunk_id"] for item in result] == ["b"]
    assert result[0]["relevance_score"] == 0.8
