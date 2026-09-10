from unittest.mock import patch

from langchain_core.messages import HumanMessage

from app.schemas.extraction import QueryUnderstanding
from app.services.pipeline.nodes.route import route

MODULE = "app.services.pipeline.nodes.route"


@patch(f"{MODULE}.invoke_chain")
def test_route_returns_intent_only(mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(intent="crop_query")
    state = {"raw_query": "বোরো ধানের বীজ হার কত?", "history": []}

    result = route(state)

    assert result["intent"] == "crop_query"
    assert "sections" not in result
    assert result["raw_query"] == "বোরো ধানের বীজ হার কত?"


@patch(f"{MODULE}.invoke_chain")
def test_route_does_not_detect_sections(mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(intent="crop_query")
    state = {"raw_query": "some query", "history": []}

    result = route(state)

    assert result["intent"] == "crop_query"
    assert "sections" not in result


@patch(f"{MODULE}.invoke_chain")
def test_route_chitchat(mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(intent="chitchat")
    state = {"raw_query": "hi there", "history": []}

    result = route(state)

    assert result["intent"] == "chitchat"


@patch(f"{MODULE}.invoke_chain")
def test_route_uses_raw_query_and_history(mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(intent="crop_query")
    state = {
        "raw_query": "এতে কতবার সেচ দিতে হয়?",
        "messages": [
            HumanMessage(content="বোরো ধানে কীভাবে চাষ করব?"),
            HumanMessage(content="এতে কতবার সেচ দিতে হয়?"),
        ],
        "rewritten_query": "must not be sent to routing",
    }

    route(state)

    chain_input = mock_invoke.call_args.args[1]
    assert chain_input["history"][0].content == "বোরো ধানে কীভাবে চাষ করব?"
    assert chain_input["query"] == state["raw_query"]
    assert "must not be sent to routing" not in chain_input["query"]
