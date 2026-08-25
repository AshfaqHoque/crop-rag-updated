from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

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


@patch(f"{MODULE}.invoke_text", return_value="Use the recommended rate.")
def test_generate_uses_and_appends_langgraph_messages(mock_invoke):
    state = {
        "messages": [
            HumanMessage(content="Tell me about Boro rice."),
            AIMessage(content="Boro rice is a rice crop."),
            HumanMessage(content="What about its seed rate?"),
        ],
        "normalized_query": "What about its seed rate?",
        "language": "en",
        "reranked_chunks": [],
    }

    result = generate(state)

    sent_messages = mock_invoke.call_args.args[0]
    assert sent_messages[1].content == "Tell me about Boro rice."
    assert sent_messages[2].content == "Boro rice is a rice crop."
    assert sent_messages[-1].content.startswith("<user_message>")
    assert isinstance(result["messages"][0], HumanMessage)
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "Use the recommended rate."
