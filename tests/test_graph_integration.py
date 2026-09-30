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


def test_full_graph_reranks_each_subquery_branch(monkeypatch):
    reranked_queries = []
    compression_inputs = []

    monkeypatch.setattr(
        graph_module,
        "rewrite_query",
        lambda state: {**state, "rewritten_query": "original"},
    )
    monkeypatch.setattr(
        graph_module,
        "route",
        lambda state: {**state, "intent": "crop_query"},
    )
    monkeypatch.setattr(
        graph_module,
        "extract_crop",
        lambda state: {**state, "crops": ["rice"]},
    )
    monkeypatch.setattr(
        graph_module,
        "decompose_query",
        lambda state: {**state, "subqueries": ["query one", "query two"]},
    )

    def retrieve(state):
        query = state["current_subquery"]
        return {
            "retrieved_documents": [
                Document(page_content=query, metadata={"chunk_id": query})
            ]
        }

    def rerank(state):
        query = state["current_subquery"]
        documents = state["retrieved_documents"]
        reranked_queries.append((query, [document.page_content for document in documents]))
        return {
            "reranked_documents": [
                Document(
                    page_content=documents[0].page_content,
                    metadata={"chunk_id": query, "relevance_score": 1.0},
                )
            ]
        }

    monkeypatch.setattr(graph_module, "retrieve", retrieve)
    monkeypatch.setattr(graph_module, "rerank", rerank)
    def compress(state):
        compression_inputs.extend(state["reranked_documents"])
        return {**state, "compressed_documents": state["reranked_documents"]}

    monkeypatch.setattr(graph_module, "compress_chunk", compress)
    monkeypatch.setattr(
        graph_module,
        "generate",
        lambda state: {**state, "answer": "answer"},
    )

    result = graph_module.build_chat_graph().invoke(
        {"session_id": "parallel-rerank", "raw_query": "original", "messages": []},
        {"configurable": {"thread_id": "parallel-rerank"}},
    )

    assert sorted(reranked_queries) == [
        ("query one", ["query one"]),
        ("query two", ["query two"]),
    ]
    assert {document.page_content for document in result["reranked_documents"]} == {
        "query one",
        "query two",
    }
    assert {document.page_content for document in compression_inputs} == {
        "query one",
        "query two",
    }


def test_full_graph_clears_document_reducers_between_turns(monkeypatch):
    compression_inputs_by_turn = []

    monkeypatch.setattr(
        graph_module,
        "rewrite_query",
        lambda state: {**state, "rewritten_query": state["raw_query"]},
    )
    monkeypatch.setattr(
        graph_module,
        "route",
        lambda state: {**state, "intent": "crop_query"},
    )
    monkeypatch.setattr(
        graph_module,
        "extract_crop",
        lambda state: {**state, "crops": ["rice"]},
    )
    monkeypatch.setattr(
        graph_module,
        "decompose_query",
        lambda state: {**state, "subqueries": [state["raw_query"]]},
    )
    monkeypatch.setattr(
        graph_module,
        "retrieve",
        lambda state: {
            "retrieved_documents": [
                Document(
                    page_content=state["current_subquery"],
                    metadata={"chunk_id": state["current_subquery"]},
                )
            ]
        },
    )
    monkeypatch.setattr(
        graph_module,
        "rerank",
        lambda state: {
            "reranked_documents": [
                Document(
                    page_content=state["retrieved_documents"][0].page_content,
                    metadata={"relevance_score": 0.5},
                )
            ]
        },
    )

    def compress(state):
        compression_inputs_by_turn.append(
            [document.page_content for document in state["reranked_documents"]]
        )
        return {**state, "compressed_documents": state["reranked_documents"]}

    monkeypatch.setattr(graph_module, "compress_chunk", compress)
    monkeypatch.setattr(
        graph_module,
        "generate",
        lambda state: {**state, "answer": "answer"},
    )

    graph = graph_module.build_chat_graph()
    config = {"configurable": {"thread_id": "reducer-reset"}}
    for query in ("first turn", "second turn"):
        graph.invoke(
            {"session_id": "reducer-reset", "raw_query": query, "messages": []},
            config,
        )

    assert compression_inputs_by_turn == [["first turn"], ["second turn"]]


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
