from unittest.mock import patch

from app.services.pipeline.nodes.generate import _format_context, generate

MODULE = "app.services.pipeline.nodes.generate"


def test_format_context_numbers_sources():
    chunks = [
        {
            "chunk_id": "5_seed",
            "content": "বীজের হার ১২০ কেজি",
            "metadata": {"crop_name": "Boro Paddy", "section": "seed"},
        }
    ]
    formatted = _format_context(chunks)
    assert "[1] chunk_id=5_seed" in formatted
    assert "বীজের হার" in formatted


@patch(f"{MODULE}.invoke_text", return_value="প্রতি হেক্টরে ১২০ কেজি [1]")
def test_generate_writes_answer(mock_invoke):
    state = {
        "raw_query": "বীজের হার কত?",
        "rewritten_query": "বোরো ধানের বীজের হার কত?",
        "language": "bn",
        "intent": "crop_query",
        "reranked_chunks": [
            {
                "chunk_id": "5_seed",
                "content": "বীজের হার ১২০ কেজি",
                "metadata": {"crop_name": "Boro Paddy", "section": "seed"},
            }
        ],
    }
    result = generate(state)
    assert result["answer"] == "প্রতি হেক্টরে ১২০ কেজি [1]"
    messages = mock_invoke.call_args.args[0]
    prompt = "\n".join(message.content for message in messages)
    assert "<standalone_query>বোরো ধানের" in prompt
    assert "[1] chunk_id=5_seed" in prompt
