from unittest.mock import patch

from app.schemas.extraction import DecomposedQuery
from app.services.pipeline.nodes.decompose_query import decompose_query

MODULE = "app.services.pipeline.nodes.decompose_query"


@patch(f"{MODULE}.invoke_structured")
def test_adds_rewritten_query_when_decomposed(mock_invoke):
    mock_invoke.return_value = DecomposedQuery(subqueries=["query one", "query two"])
    state = {
        "raw_query": "follow-up question",
        "rewritten_query": "rewritten question",
    }

    result = decompose_query(state)

    assert result["subqueries"] == ["query one", "query two", "rewritten question"]


@patch(f"{MODULE}.invoke_structured")
def test_does_not_add_query_for_single_subquery(mock_invoke):
    mock_invoke.return_value = DecomposedQuery(subqueries=["rewritten question"])

    result = decompose_query({"rewritten_query": "rewritten question"})

    assert result["subqueries"] == ["rewritten question"]