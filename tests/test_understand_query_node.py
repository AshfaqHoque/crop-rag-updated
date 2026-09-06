from unittest.mock import patch

from app.schemas.extraction import QueryUnderstanding
from app.services.pipeline.nodes.route import route

MODULE = "app.services.pipeline.nodes.route"


@patch(f"{MODULE}.invoke_structured")
def test_route_returns_intent_only(mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(intent="crop_query")
    state = {"raw_query": "বোরো ধানের বীজ হার কত?", "history": []}

    result = route(state)

    assert result["intent"] == "crop_query"
    assert "sections" not in result
    assert result["raw_query"] == "বোরো ধানের বীজ হার কত?"


@patch(f"{MODULE}.invoke_structured")
def test_route_does_not_detect_sections(mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(intent="crop_query")
    state = {"raw_query": "some query", "history": []}

    result = route(state)

    assert result["intent"] == "crop_query"
    assert "sections" not in result


@patch(f"{MODULE}.invoke_structured")
def test_route_chitchat(mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(intent="chitchat")
    state = {"raw_query": "hi there", "history": []}

    result = route(state)

    assert result["intent"] == "chitchat"
