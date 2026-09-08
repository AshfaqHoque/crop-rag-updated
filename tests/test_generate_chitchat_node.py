from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from app.services.pipeline.nodes.generate_chitchat import generate_chitchat


MODULE = "app.services.pipeline.nodes.generate_chitchat"


@patch(f"{MODULE}.invoke_text", return_value="Hello! I am doing well. Want to explore agriculture together?")
def test_generate_chitchat_writes_casual_answer(mock_invoke):
    state = {
        "raw_query": "How are you?",
        "language_type": "english",
        "messages": [HumanMessage(content="Hi"), AIMessage(content="Hello!")],
    }

    result = generate_chitchat(state)

    assert result["answer"].endswith("agriculture together?")
    assert isinstance(result["messages"][0], AIMessage)
    sent_messages = mock_invoke.call_args.args[0]
    assert sent_messages[1].content == "Hi"
    assert sent_messages[-1].content == "How are you?"
    assert "a little funny" in sent_messages[0].content