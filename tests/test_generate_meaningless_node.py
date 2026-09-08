from unittest.mock import patch

from langchain_core.messages import AIMessage

from app.services.pipeline.nodes.generate_meaningless import generate_meaningless


MODULE = "app.services.pipeline.nodes.generate_meaningless"


@patch(f"{MODULE}.invoke_text", return_value="Please ask a clear agriculture question.")
def test_generate_meaningless_requests_a_clear_question(mock_invoke):
    state = {
        "raw_query": "asdf qwerty",
        "language_type": "english",
        "messages": [],
    }

    result = generate_meaningless(state)

    assert result["answer"] == "Please ask a clear agriculture question."
    assert isinstance(result["messages"][0], AIMessage)
    sent_messages = mock_invoke.call_args.args[0]
    assert sent_messages[-1].content == "asdf qwerty"
    assert "meaningful question" in sent_messages[0].content