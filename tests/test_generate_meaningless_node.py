from unittest.mock import patch

from langchain_core.messages import AIMessage

from app.services.pipeline.nodes.generate_meaningless import generate_meaningless


MODULE = "app.services.pipeline.nodes.generate_meaningless"


@patch(f"{MODULE}.invoke_chain", return_value="Please ask a clear agriculture question.")
def test_generate_meaningless_requests_a_clear_question(mock_invoke):
    state = {
        "raw_query": "asdf qwerty",
        "language_type": "english",
        "messages": [],
    }

    result = generate_meaningless(state)

    assert result["answer"] == "Please ask a clear agriculture question."
    assert isinstance(result["messages"][0], AIMessage)
    chain_input = mock_invoke.call_args.args[1]
    assert chain_input["query"] == "asdf qwerty"