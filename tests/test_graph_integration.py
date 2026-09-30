from langchain_core.documents import Document

from app.services.pipeline import graph as graph_module
from app.services.pipeline.nodes import parallel_retrieve as parallel_retrieve_module


def test_full_graph_crop_query_path(monkeypatch):
    visited = []

    def route(state):
        visited.append("route")
        return {
            **state,
            "intent": "crop_query",
            "crops": [],
        }

    def rewrite(state):
        visited.append("rewrite")
        return {**state, "rewritten_query": "seed rate for rice", "rewrite_used_history": True}

    def decompose(state):
        visited.append("decompose")
        return {**state, "decomposed_queries": ["seed rate for rice"]}

    def retrieve_queries(state):
        visited.append("retrieve_queries")
        return {
            **state,
            "retrieval_mode": "dense_filtered",
            "retrieved_documents": [Document(page_content="rate", metadata={"chunk_id": "x"})],
            "reranked_documents": [Document(page_content="rate", metadata={"chunk_id": "x"})],
            "compressed_documents": [Document(page_content="rate", metadata={"chunk_id": "x"})],
        }

    def generate(state):
        visited.append("generate")
        return {**state, "answer": "answer [1]"}

    monkeypatch.setattr(graph_module, "route", route)
    monkeypatch.setattr(graph_module, "rewrite_query", rewrite)
    monkeypatch.setattr(graph_module, "decompose_query", decompose)
    monkeypatch.setattr(graph_module, "retrieve_decomposed_queries", retrieve_queries)
    monkeypatch.setattr(graph_module, "generate", generate)

    result = graph_module.build_chat_graph().invoke(
        {"session_id": "s", "raw_query": "what about it?", "messages": []},
        {"configurable": {"thread_id": "s"}},
    )
    assert visited == ["route", "rewrite", "decompose", "retrieve_queries", "generate"]
    assert result["answer"] == "answer [1]"
    assert result["retrieval_mode"] == "dense_filtered"


def test_decomposed_queries_each_run_retrieval_chain(monkeypatch):
    import asyncio

    visited = {"extract": [], "retrieve": [], "rerank": [], "compress": []}

    def extract(state):
        query = state["rewritten_query"]
        visited["extract"].append(query)
        return {**state, "crops": [query]}

    def retrieve(state):
        query = state["rewritten_query"]
        visited["retrieve"].append(query)
        document = Document(page_content=query, metadata={"chunk_id": query})
        return {**state, "retrieval_mode": "dense", "retrieved_documents": [document]}

    def rerank(state):
        query = state["rewritten_query"]
        visited["rerank"].append(query)
        return {**state, "reranked_documents": state["retrieved_documents"]}

    async def compress(state):
        query = state["rewritten_query"]
        visited["compress"].append(query)
        return {**state, "compressed_documents": state["reranked_documents"]}

    monkeypatch.setattr(parallel_retrieve_module, "extract_crop", extract)
    monkeypatch.setattr(parallel_retrieve_module, "retrieve", retrieve)
    monkeypatch.setattr(parallel_retrieve_module, "rerank", rerank)
    monkeypatch.setattr(parallel_retrieve_module, "compress_chunk", compress)

    result = asyncio.run(
        parallel_retrieve_module.retrieve_decomposed_queries(
            {
                "raw_query": "compare varieties",
                "rewritten_query": "compare varieties",
                "decomposed_queries": ["variety A", "variety B"],
            }
        )
    )

    expected_queries = ["variety A", "variety B"]
    assert all(sorted(queries) == expected_queries for queries in visited.values())
    assert [document.page_content for document in result["compressed_documents"]] == expected_queries
    assert result["retrieval_mode"] == "dense"


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


def test_full_graph_meaningless_skips_retrieval(monkeypatch):
    visited = []

    monkeypatch.setattr(
        graph_module,
        "route",
        lambda state: {**state, "language": "en", "intent": "meaningless"},
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

    def generate_meaningless(state):
        visited.append("generate_meaningless")
        return {**state, "answer": "Please ask a clear question."}

    monkeypatch.setattr(graph_module, "generate_meaningless", generate_meaningless)
    result = graph_module.build_chat_graph().invoke(
        {"session_id": "s", "raw_query": "asdf", "messages": []},
        {"configurable": {"thread_id": "s"}},
    )
    assert visited == ["generate_meaningless"]
    assert result["answer"] == "Please ask a clear question."


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
