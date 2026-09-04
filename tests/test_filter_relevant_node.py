from app.schemas.extraction import ChunkRelevance
from app.services.pipeline.nodes import compress_chunk as filter_module
from app.services.pipeline.nodes.compress_chunk import filter_relevant


def test_filter_relevant_uses_reranked_chunks(monkeypatch):
    decisions = iter([True, False])

    def invoke(schema, messages):
        assert schema is ChunkRelevance
        assert "User Query: question" in messages[-1].content
        return ChunkRelevance(relevant=next(decisions))

    monkeypatch.setattr(filter_module, "invoke_structured", invoke)
    state = {
        "normalized_query": "question",
        "reranked_chunks": [
            {"chunk_id": "relevant", "content": "relevant"},
            {"chunk_id": "irrelevant", "content": "irrelevant"},
        ],
    }

    result = filter_relevant(state)

    assert [chunk["chunk_id"] for chunk in result["compressed_chunks"]] == ["relevant"]
