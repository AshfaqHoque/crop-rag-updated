from langchain_core.documents import Document

from app.services.pipeline import graph as graph_module


def test_full_graph_crop_query_path(monkeypatch):
    visited = []

    def route(state):
        visited.append("route")
        return {
            **state,
            "intent": "crop_query",
            "crops": [],
        }

    def extract(state):
        visited.append("extract")
        return {**state, "crops": ["Boro Paddy"]}

    def rewrite(state):
        visited.append("rewrite")
        return {**state, "rewritten_query": "seed rate for rice", "rewrite_used_history": True}

    def retrieve(state):
        visited.append("retrieve")
        return {
            **state,
            "retrieval_mode": "dense_filtered",
            "retrieved_documents": [Document(page_content="rate", metadata={"chunk_id": "x"})],
        }

    def generate(state):
        visited.append("generate")
        return {**state, "answer": "answer [1]"}

    monkeypatch.setattr(graph_module, "route", route)
    monkeypatch.setattr(graph_module, "extract_crop", extract)
    monkeypatch.setattr(graph_module, "rewrite_query", rewrite)
    monkeypatch.setattr(graph_module, "retrieve", retrieve)
    monkeypatch.setattr(
        graph_module,
        "rerank",
        lambda state: {**state, "reranked_documents": state["retrieved_documents"]},
    )
    monkeypatch.setattr(
        graph_module,
        "compress_chunk",
        lambda state: {**state, "compressed_documents": state["reranked_documents"]},
    )
    monkeypatch.setattr(graph_module, "generate", generate)

    result = graph_module.build_chat_graph().invoke(
        {"session_id": "s", "raw_query": "what about it?", "messages": []},
        {"configurable": {"thread_id": "s"}},
    )
    assert visited == ["rewrite", "route", "extract", "retrieve", "generate"]
    assert result["answer"] == "answer [1]"
    assert result["retrieval_mode"] == "dense_filtered"


def test_full_graph_chitchat_skips_retrieval(monkeypatch):
    visited = []

    monkeypatch.setattr(
        graph_module,
        "route",
        lambda state: {**state, "language": "en", "intent": "chitchat"},
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
        "extract_crop",
        lambda state: (_ for _ in ()).throw(AssertionError("crop extraction should be skipped")),
    )
    def generate_chitchat(state):
        visited.append("generate_chitchat")
        return {**state, "answer": "hello"}

    monkeypatch.setattr(graph_module, "generate_chitchat", generate_chitchat)
    result = graph_module.build_chat_graph().invoke(
        {"session_id": "s", "raw_query": "hello", "messages": []},
        {"configurable": {"thread_id": "s"}},
    )
    assert visited == ["generate_chitchat"]
    assert result["answer"] == "hello"


def test_full_graph_company_query_skips_crop_pipeline(monkeypatch):
    visited = []

    monkeypatch.setattr(
        graph_module,
        "rewrite_query",
        lambda state: {**state, "rewritten_query": "what does Aunkur do?"},
    )
    monkeypatch.setattr(
        graph_module,
        "route",
        lambda state: {**state, "intent": "company_query"},
    )
    monkeypatch.setattr(
        graph_module,
        "retrieve_company",
        lambda state: (
            visited.append("retrieve_company")
            or {
                **state,
                "retrieval_mode": "company_dense",
                "retrieved_documents": [Document(page_content="company fact") for _ in range(3)],
            }
        ),
    )
    monkeypatch.setattr(
        graph_module,
        "rerank",
        lambda state: (_ for _ in ()).throw(AssertionError("rerank should be skipped")),
    )
    monkeypatch.setattr(
        graph_module,
        "generate_company",
        lambda state: visited.append("generate_company") or {**state, "answer": "company answer"},
    )

    result = graph_module.build_chat_graph().invoke(
        {"session_id": "s", "raw_query": "what does Aunkur do?", "messages": []},
        {"configurable": {"thread_id": "s"}},
    )

    assert visited == ["retrieve_company", "generate_company"]
    assert result["answer"] == "company answer"
    assert result["retrieval_mode"] == "company_dense"
