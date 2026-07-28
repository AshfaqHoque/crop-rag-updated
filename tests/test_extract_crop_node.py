from unittest.mock import patch

from app.services.pipeline.nodes.extract_crop import extract_crop
from app.services.pipeline.registry import CropInfo

MODULE = "app.services.pipeline.nodes.extract_crop"

CROPS = [
    CropInfo(crop_id="5", crop_name="Boro Paddy", crop_bangla_name="বোরো ধান"),
    CropInfo(crop_id="16", crop_name="Mango", crop_bangla_name="আম"),
    CropInfo(crop_id="220", crop_name="Brinjal/Eggplant", crop_bangla_name="বেগুন"),
]


@patch(f"{MODULE}.get_known_crops", return_value=CROPS)
def test_extracts_english_canonical_name(mock_crops):
    result = extract_crop({"raw_query": "What is the seed rate for boro paddy?"})

    assert result["crops"] == ["Boro Paddy"]


@patch(f"{MODULE}.get_known_crops", return_value=CROPS)
def test_extracts_bangla_name_with_grammatical_suffix(mock_crops):
    result = extract_crop({"raw_query": "বোরো ধানের বীজ হার কত?"})

    assert result["crops"] == ["Boro Paddy"]


@patch(f"{MODULE}.get_known_crops", return_value=CROPS)
def test_uses_rewritten_query_for_follow_up(mock_crops):
    result = extract_crop(
        {
            "raw_query": "What about its seed rate?",
            "rewritten_query": "What is the seed rate for Mango?",
        }
    )

    assert result["crops"] == ["Mango"]


@patch(f"{MODULE}.get_known_crops", return_value=CROPS)
def test_extracts_slash_separated_alias(mock_crops):
    result = extract_crop({"raw_query": "How do I control pests in eggplant?"})

    assert result["crops"] == ["Brinjal/Eggplant"]


@patch(f"{MODULE}.get_known_crops", return_value=CROPS)
def test_does_not_match_english_substrings_or_bangla_homographs(mock_crops):
    result = extract_crop({"raw_query": "My management question is আমার প্রশ্ন"})

    assert result["crops"] == []


@patch(f"{MODULE}.get_known_crops", return_value=CROPS)
def test_extracts_multiple_crops_in_registry_order(mock_crops):
    result = extract_crop({"raw_query": "Compare mango and বেগুন"})

    assert result["crops"] == ["Mango", "Brinjal/Eggplant"]
