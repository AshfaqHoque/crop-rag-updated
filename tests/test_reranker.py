from app.core.config import get_settings
from langchain_core.documents import Document
from app.services.pipeline.nodes import rerank as rerank_module
from app.services.pipeline.nodes.rerank import rerank


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "results": [
                {"index": 0, "relevance_score": 0.1},
                {"index": 2, "relevance_score": 0.4},
                {"index": 1, "relevance_score": 0.9},
            ]
        }


def test_reranker_calls_service_and_orders_chunks(monkeypatch):
    monkeypatch.setattr(get_settings(), "rerank_top_k", 2)

    def post(url, *, json, timeout):
        assert url == get_settings().reranker_url
        assert json == {"query": "rewritten", "documents": ["a", "b", "c"]}
        assert timeout == get_settings().reranker_timeout_seconds
        return FakeResponse()

    monkeypatch.setattr(rerank_module.httpx, "post", post)
    result = rerank(
        {
            "raw_query": "original",
            "rewritten_query": "rewritten",
            "retrieved_documents": [
                Document(page_content="a", metadata={"chunk_id": "a"}),
                Document(page_content="b", metadata={"chunk_id": "b"}),
                Document(page_content="c", metadata={"chunk_id": "c"}),
            ],
        }
    )

    assert [document.metadata["chunk_id"] for document in result["reranked_documents"]] == ["b", "c"]
    assert result["reranked_documents"][0].metadata["relevance_score"] == 0.9


def test_reranker_skips_service_when_no_chunks(monkeypatch):
    monkeypatch.setattr(
        rerank_module.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call service")),
    )

    assert rerank({"raw_query": "query", "retrieved_documents": []})["reranked_documents"] == []
