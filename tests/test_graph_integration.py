from app.services.pipeline import graph as graph_module


def test_full_graph_crop_query_path(monkeypatch):
    visited = []

    def understand(state):
        visited.append("understand")
        return {
            **state,
            "language": "en",
            "intent": "crop_query",
            "crops": [],
            "sections": ["seed"],
        }

    def rewrite(state):
        visited.append("rewrite")
        return {**state, "rewritten_query": "seed rate for rice", "rewrite_used_history": True}

    def retrieve(state):
        visited.append("retrieve")
        return {
            **state,
            "retrieval_mode": "dense_bm25_rrf",
            "retrieved_chunks": [{"chunk_id": "x", "content": "rate", "metadata": {}}],
        }

    def rerank(state):
        visited.append("rerank")
        return {**state, "reranked_chunks": state["retrieved_chunks"]}

    def generate(state):
        visited.append("generate")
        return {**state, "answer": "answer [1]"}

    monkeypatch.setattr(graph_module, "understand_query", understand)
    monkeypatch.setattr(graph_module, "rewrite_query", rewrite)
    monkeypatch.setattr(graph_module, "retrieve", retrieve)
    monkeypatch.setattr(graph_module, "rerank", rerank)
    monkeypatch.setattr(graph_module, "generate", generate)

    result = graph_module.build_chat_graph().invoke(
        {"session_id": "s", "raw_query": "what about it?", "history": []}
    )
    assert visited == ["rewrite", "understand", "retrieve", "rerank", "generate"]
    assert result["answer"] == "answer [1]"
    assert result["retrieval_mode"] == "dense_bm25_rrf"


def test_full_graph_small_talk_skips_retrieval(monkeypatch):
    visited = []

    monkeypatch.setattr(
        graph_module,
        "understand_query",
        lambda state: {**state, "language": "en", "intent": "small_talk"},
    )
    monkeypatch.setattr(
        graph_module,
        "rewrite_query",
        lambda state: {**state, "rewritten_query": state["raw_query"]},
    )
    monkeypatch.setattr(
        graph_module,
        "retrieve",
        lambda state: (_ for _ in ()).throw(AssertionError("retrieve should be skipped")),
    )
    monkeypatch.setattr(
        graph_module,
        "rerank",
        lambda state: (_ for _ in ()).throw(AssertionError("rerank should be skipped")),
    )

    def generate(state):
        visited.append("generate")
        return {**state, "answer": "hello"}

    monkeypatch.setattr(graph_module, "generate", generate)
    result = graph_module.build_chat_graph().invoke(
        {"session_id": "s", "raw_query": "hello", "history": []}
    )
    assert visited == ["generate"]
    assert result["answer"] == "hello"
