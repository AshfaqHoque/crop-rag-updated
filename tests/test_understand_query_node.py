from unittest.mock import patch

from app.schemas.extraction import QueryUnderstanding
from app.services.pipeline.nodes.understand_query import _format_history, understand_query

MODULE = "app.services.pipeline.nodes.understand_query"


def test_format_history_empty():
    assert _format_history([]) == "(none)"


def test_format_history_formats_recent_turns():
    history = [
        {"role": "user", "content": "what is seed rate for aman rice?"},
        {"role": "assistant", "content": "..."},
    ]
    formatted = _format_history(history)
    assert "user: what is seed rate for aman rice?" in formatted
    assert "assistant: ..." in formatted


def test_format_history_only_keeps_last_four_turns():
    history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    formatted = _format_history(history)
    assert "msg 9" in formatted
    assert "msg 0" not in formatted


@patch(f"{MODULE}.invoke_structured")
def test_understand_query_keeps_valid_extractions(mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(
        language="bn", intent="crop_query", sections=["seed"]
    )
    state = {"raw_query": "বোরো ধানের বীজ হার কত?", "history": []}

    result = understand_query(state)

    assert result["language"] == "bn"
    assert result["intent"] == "crop_query"
    assert result["sections"] == ["seed"]
    assert result["raw_query"] == "বোরো ধানের বীজ হার কত?"


@patch(f"{MODULE}.invoke_structured")
def test_understand_query_drops_hallucinated_section(mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(
        language="en", intent="crop_query", sections=["marketing"]
    )
    state = {"raw_query": "some query", "history": []}

    result = understand_query(state)

    assert result["sections"] == []


@patch(f"{MODULE}.invoke_structured")
def test_understand_query_small_talk(mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(
        language="en", intent="small_talk", sections=[]
    )
    state = {"raw_query": "hi there", "history": []}

    result = understand_query(state)

    assert result["intent"] == "small_talk"
    assert result["sections"] == []


@patch(f"{MODULE}.invoke_structured")
def test_understand_query_unsupported_language(mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(
        language="unsupported", intent="crop_query", sections=[]
    )
    state = {"raw_query": "ami dhan chas korte chai", "history": []}

    result = understand_query(state)

    assert result["language"] == "unsupported"
