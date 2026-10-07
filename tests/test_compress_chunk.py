from unittest.mock import patch

from langchain_core.documents import Document

from app.services.pipeline.nodes.compress_chunk import compress_chunk, get_context_compressor

MODULE = "app.services.pipeline.nodes.compress_chunk"


def test_context_compressor_binds_max_tokens():
    get_context_compressor.cache_clear()

    with (
        patch(f"{MODULE}.get_chat_llm") as mock_get_chat_llm,
        patch(f"{MODULE}.LLMChainExtractor.from_llm") as mock_from_llm,
    ):
        llm = mock_get_chat_llm.return_value
        bound_llm = llm.bind.return_value

        get_context_compressor()

        llm.bind.assert_called_once_with(max_tokens=1000)
        mock_from_llm.assert_called_once_with(
            bound_llm,
            prompt=get_context_compressor.__globals__["CUSTOM_DEFAULT_PROMPT"],
        )

    get_context_compressor.cache_clear()


async def test_compress_chunk_bypasses_high_scores_and_compresses_the_rest(monkeypatch):
    documents = [
        Document(page_content="below", metadata={"relevance_score": 0.89}),
        Document(page_content="at threshold", metadata={"relevance_score": 0.9}),
        Document(page_content="above", metadata={"relevance_score": 0.95}),
    ]

    class FakeCompressor:
        async def acompress_documents(self, *, documents, query):
            assert query == "query"
            assert [document.page_content for document in documents] == ["below", "at threshold"]
            return [
                Document(
                    page_content=f"compressed {document.page_content}",
                    metadata=document.metadata,
                )
                for document in documents
            ]

    monkeypatch.setattr(
        "app.services.pipeline.nodes.compress_chunk.get_context_compressor",
        lambda: FakeCompressor(),
    )

    result = await compress_chunk(
        {
            "raw_query": "query",
            "reranked_documents": documents,
        }
    )

    assert [document.page_content for document in result["compressed_documents"]] == [
        "compressed below",
        "compressed at threshold",
        "above",
    ]


async def test_compress_chunk_drops_no_output(monkeypatch):
    class FakeCompressor:
        async def acompress_documents(self, *, documents, query):
            return [
                Document(page_content="NO_OUTPUT", metadata=documents[0].metadata),
                Document(page_content="Relevant answer text", metadata=documents[1].metadata),
            ]

    monkeypatch.setattr(
        "app.services.pipeline.nodes.compress_chunk.get_context_compressor",
        lambda: FakeCompressor(),
    )

    result = await compress_chunk(
        {
            "raw_query": "query",
            "reranked_documents": [
                Document(page_content="Irrelevant source", metadata={"chunk_id": "irrelevant"}),
                Document(page_content="Relevant source", metadata={"chunk_id": "relevant"}),
            ],
        }
    )

    assert [document.page_content for document in result["compressed_documents"]] == [
        "Relevant answer text"
    ]