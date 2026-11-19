from unittest.mock import patch

from app.schemas.extraction import SectionIdentifier
from app.schemas.sql import EntityToResolve
from app.services.pipeline.nodes.extract_sections import extract_sections
from app.services.pipeline.nodes.resolve_entities import resolve_entities_semantic
from app.services.pipeline.graph import route_after_decide

MODULE = "app.services.pipeline.nodes.extract_sections"


@patch(f"{MODULE}.invoke_structured")
def test_extract_sections_extracts_crop_like_other_entities(mock_invoke):
    mock_invoke.return_value = SectionIdentifier(
        sections=["pesticide"],
        entities=[
            EntityToResolve(type="crop", mention="বোরো ধান"),
            EntityToResolve(type="pesticide", mention="মাজরা পোকা"),
        ],
    )

    result = extract_sections({"raw_query": "বোরো ধানে মাজরা পোকা দমন করব কিভাবে?"})

    assert result["crops"] == []
    assert result["sections"] == ["pesticide"]
    assert result["entities_to_resolve"] == [
        {"type": "crop", "mention": "বোরো ধান"},
        {"type": "pesticide", "mention": "মাজরা পোকা"},
    ]
    prompt = mock_invoke.call_args.args[1][0].content
    assert "Entity types:" in prompt
    assert "- crop: Crop names." in prompt


@patch("app.services.pipeline.nodes.resolve_entities._pick")
@patch("app.services.pipeline.nodes.resolve_entities._search")
def test_entity_resolution_sets_canonical_crop_filter(mock_search, mock_pick):
    mock_search.side_effect = [["Boro Paddy"], ["Stem Borer"]]
    mock_pick.side_effect = ["Boro Paddy", "Stem Borer"]

    result = resolve_entities_semantic(
        {
            "entities_to_resolve": [
                {"type": "crop", "mention": "বোরো ধান"},
                {"type": "pesticide", "mention": "মাজরা পোকা"},
            ],
        }
    )

    assert result["crops"] == ["Boro Paddy"]
    assert result["resolved_entities"] == [
        {"type": "crop", "mention": "Boro Paddy"},
        {"type": "pesticide", "mention": "Stem Borer"},
    ]
    assert result["entity_filters"] == {"disease_name": ["Stem Borer"]}


@patch("app.services.pipeline.nodes.resolve_entities._pick")
@patch("app.services.pipeline.nodes.resolve_entities._search")
def test_entity_resolution_builds_variety_and_pest_filters(mock_search, mock_pick):
    mock_search.side_effect = [["BRRI Dhan 28"], ["Stem Borer"]]
    mock_pick.side_effect = ["BRRI Dhan 28", "Stem Borer"]

    result = resolve_entities_semantic(
        {
            "entities_to_resolve": [
                {"type": "variety", "mention": "Dhan 28"},
                {"type": "pesticide", "mention": "stem borer"},
            ],
        }
    )

    assert result["entity_filters"] == {
        "variety_name": ["BRRI Dhan 28"],
        "disease_name": ["Stem Borer"],
    }


@patch("app.services.pipeline.nodes.resolve_entities._search", return_value=[])
def test_unresolved_entities_do_not_create_hard_filters(mock_search):
    result = resolve_entities_semantic(
        {
            "entities_to_resolve": [
                {"type": "variety", "mention": "unknown variety"},
                {"type": "pesticide", "mention": "unknown pest"},
            ],
        }
    )

    assert result["entity_filters"] == {}


def test_semantic_route_skips_separate_crop_node():
    assert route_after_decide({"retrieval_mode": "semantic"}) == "extract_sections"
