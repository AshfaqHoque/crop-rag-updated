from unittest.mock import patch

from langchain_core.documents import Document

from app.schemas.extraction import DescriptiveQuery
from app.services.pipeline.nodes.classify_descriptive_query import classify_descriptive_query
from app.services.pipeline.nodes.summarize_chunks import SUMMARY_PROMPT, summarize_chunks


def test_classify_descriptive_query_uses_structured_boolean():
    result = DescriptiveQuery(descriptive=True)

    with patch(
        "app.services.pipeline.nodes.classify_descriptive_query.invoke_structured",
        return_value=result,
    ) as invoke_structured:
        state = classify_descriptive_query(
            {
                "raw_query": "Compare rice varieties and explain the full growing process",
                "rewritten_query": "সমস্ত ধানের জাত ও চাষ পদ্ধতি",
            }
        )

    assert state["descriptive"] is True
    schema, messages = invoke_structured.call_args.args
    assert schema is DescriptiveQuery
    assert "Compare rice varieties" in messages[-1].content
    assert "সমস্ত ধানের জাত" in messages[-1].content


async def test_summarize_chunks_preserves_metadata_and_uses_resolved_query(monkeypatch):
    source = Document(
        page_content="Original chunk",
        metadata={"chunk_id": "c-1", "section": "Harvesting", "crop": "Rice"},
    )

    class FakeSummarizer:
        async def acompress_documents(self, *, documents, query):
            assert documents == [source]
            assert query == "resolved query"
            return [
                Document(
                    page_content="Harvest at 120 days for Rice.",
                    metadata=source.metadata,
                )
            ]

    monkeypatch.setattr(
        "app.services.pipeline.nodes.summarize_chunks.get_chunk_summarizer",
        lambda: FakeSummarizer(),
    )

    result = await summarize_chunks(
        {
            "raw_query": "original query",
            "rewritten_query": "resolved query",
            "reranked_documents": [source],
        }
    )

    summary = result["compressed_documents"][0]
    assert summary.page_content == "[section: Harvesting | crop: Rice] Harvest at 120 days for Rice."
    assert summary.metadata == source.metadata


def test_summary_prompt_requires_preserving_values_and_names():
    prompt = SUMMARY_PROMPT.format(
        question="harvest timing",
        context="Harvest at 120 days for Rice.",
    )

    assert "Do not alter, normalize, or omit" in prompt
    assert "numerical values, units, dates, names" in prompt
    assert "preserve the original tone" in prompt