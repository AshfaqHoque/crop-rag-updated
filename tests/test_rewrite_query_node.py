from unittest.mock import patch

from app.schemas.extraction import QueryRewrite
from app.services.pipeline.nodes.rewrite_query import rewrite_query

MODULE = "app.services.pipeline.nodes.rewrite_query"


def test_rewrite_skips_llm_without_history():
    state = {"raw_query": "বোরো ধানের বীজ হার কত?", "history": [], "intent": "crop_query"}
    with patch(f"{MODULE}.invoke_structured") as invoke:
        result = rewrite_query(state)
    invoke.assert_not_called()
    assert result["rewritten_query"] == state["raw_query"]
    assert result["rewrite_used_history"] is False


@patch(f"{MODULE}.invoke_structured")
def test_rewrite_resolves_subject_from_history(mock_invoke):
    mock_invoke.return_value = QueryRewrite(
        rewritten_query="বোরো ধানে কতবার সেচ দিতে হয়?",
        used_history=True,
    )
    state = {
        "raw_query": "এতে কতবার সেচ দিতে হয়?",
        "history": [{"role": "user", "content": "বোরো ধান কীভাবে চাষ করব?"}],
        "intent": "crop_query",
    }
    result = rewrite_query(state)
    assert result["rewritten_query"] == "বোরো ধানে কতবার সেচ দিতে হয়?"
    assert result["rewrite_used_history"] is True


@patch(f"{MODULE}.invoke_structured")
def test_rewrite_preserves_original_when_model_returns_blank(mock_invoke):
    # Pydantic disallows a blank at construction, so simulate a model-like object.
    mock_invoke.return_value = type(
        "Result", (), {"rewritten_query": "   ", "used_history": False}
    )()
    state = {
        "raw_query": "what about irrigation?",
        "history": [{"role": "user", "content": "Tell me about potato"}],
        "intent": "crop_query",
    }
    assert rewrite_query(state)["rewritten_query"] == "what about irrigation?"


@patch(f"{MODULE}.invoke_structured")
def test_rewrite_ignores_paraphrase_when_history_not_used(mock_invoke):
    mock_invoke.return_value = QueryRewrite(
        rewritten_query="How much irrigation should Boro Paddy receive?",
        used_history=False,
    )
    state = {
        "raw_query": "How many irrigations for boro paddy?",
        "history": [{"role": "user", "content": "Earlier unrelated question"}],
    }
    assert rewrite_query(state)["rewritten_query"] == state["raw_query"]
