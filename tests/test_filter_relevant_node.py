from app.schemas.extraction import RelevantChunks
from app.services.pipeline.nodes import filter_relevant as filter_module
from app.services.pipeline.nodes.filter_relevant import filter_relevant


def test_filter_relevant_uses_llm_for_original_and_rewritten_queries(monkeypatch):
    def invoke(schema, messages, *, temperature):
        assert schema is RelevantChunks
        assert temperature == 0
        prompt = messages[1].content
        assert "original question" in prompt
        assert "rewritten question" in prompt
        assert "[0] relevant" in prompt
        assert "[1] irrelevant" in prompt
        return RelevantChunks(relevant_indexes=[0])

    monkeypatch.setattr(filter_module, "invoke_structured", invoke)
    state = {
        "raw_query": "original question",
        "rewritten_query": "rewritten question",
        "retrieved_chunks": [
            {"chunk_id": "relevant", "content": "relevant"},
            {"chunk_id": "irrelevant", "content": "irrelevant"},
        ],
    }

    result = filter_relevant(state)

    assert [chunk["chunk_id"] for chunk in result["filtered_chunks"]] == ["relevant"]