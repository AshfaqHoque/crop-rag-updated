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
@patch(f"{MODULE}.crop_names", return_value=["Boro Paddy", "Mango"])
@patch(f"{MODULE}.format_crop_list_for_prompt", return_value="Boro Paddy (বোরো ধান)")
def test_understand_query_keeps_valid_extractions(mock_fmt, mock_crops, mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(
        language="bn", intent="crop_query", crops=["Boro Paddy"], sections=["seed"]
    )
    state = {"raw_query": "বোরো ধানের বীজ হার কত?", "history": []}

    result = understand_query(state)

    assert result["language"] == "bn"
    assert result["intent"] == "crop_query"
    assert result["crops"] == ["Boro Paddy"]
    assert result["sections"] == ["seed"]
    # original state fields are preserved
    assert result["raw_query"] == "বোরো ধানের বীজ হার কত?"


@patch(f"{MODULE}.invoke_structured")
@patch(f"{MODULE}.crop_names", return_value=["Boro Paddy"])
def test_understand_query_drops_hallucinated_crop(mock_crops, mock_invoke):
    # Model invents a crop name that isn't in the registry — must be filtered out,
    # not passed through to the retriever's metadata filter.
    mock_invoke.return_value = QueryUnderstanding(
        language="en", intent="crop_query", crops=["Watermelon"], sections=["pesticide"]
    )
    state = {"raw_query": "How to grow watermelon?", "history": []}

    result = understand_query(state)

    assert result["crops"] == []
    assert result["sections"] == ["pesticide"]


@patch(f"{MODULE}.invoke_structured")
@patch(f"{MODULE}.crop_names", return_value=["Boro Paddy"])
def test_understand_query_drops_hallucinated_section(mock_crops, mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(
        language="en", intent="crop_query", crops=["Boro Paddy"], sections=["marketing"]
    )
    state = {"raw_query": "some query", "history": []}

    result = understand_query(state)

    assert result["crops"] == ["Boro Paddy"]
    assert result["sections"] == []


@patch(f"{MODULE}.invoke_structured")
def test_understand_query_small_talk_has_no_crops(mock_invoke):
    mock_invoke.return_value = QueryUnderstanding(
        language="en", intent="small_talk", crops=[], sections=[]
    )
    state = {"raw_query": "hi there", "history": []}

    result = understand_query(state)

    assert result["intent"] == "small_talk"
    assert result["crops"] == []
    assert result["sections"] == []


@patch(f"{MODULE}.invoke_structured")
def test_understand_query_unsupported_language(mock_invoke):
    # e.g. Banglish: "ami dhan chas korte chai"
    mock_invoke.return_value = QueryUnderstanding(
        language="unsupported", intent="crop_query", crops=[], sections=[]
    )
    state = {"raw_query": "ami dhan chas korte chai", "history": []}

    result = understand_query(state)

    assert result["language"] == "unsupported"
