from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from app.services.pipeline.nodes.generate_chitchat import generate_chitchat


MODULE = "app.services.pipeline.nodes.generate_chitchat"


@patch(f"{MODULE}.invoke_chain", return_value="Hello! I am doing well. Want to explore agriculture together?")
def test_generate_chitchat_writes_casual_answer(mock_invoke):
    state = {
        "raw_query": "How are you?",
        "language_type": "english",
        "messages": [HumanMessage(content="Hi"), AIMessage(content="Hello!")],
    }

    result = generate_chitchat(state)

    assert result["answer"].endswith("agriculture together?")
    assert isinstance(result["messages"][0], AIMessage)
    chain_input = mock_invoke.call_args.args[1]
    assert chain_input["history"][0].content == "Hi"
    assert chain_input["query"] == "How are you?"