from app.services.retrieval.vector_store import build_metadata_filter


def test_no_filters_returns_none():
    assert build_metadata_filter(None, None) is None
    assert build_metadata_filter([], []) is None


def test_crop_only_filter():
    assert build_metadata_filter(crops=["Boro Paddy"]) == {"crop_name": {"$in": ["Boro Paddy"]}}


def test_section_only_filter():
    assert build_metadata_filter(sections=["seed"]) == {"section": {"$in": ["seed"]}}


def test_crop_and_section_combined_with_and():
    result = build_metadata_filter(crops=["Boro Paddy"], sections=["seed", "irrigation"])
    assert result == {
        "$and": [
            {"crop_name": {"$in": ["Boro Paddy"]}},
            {"section": {"$in": ["seed", "irrigation"]}},
        ]
    }


def test_multiple_crops_uses_in_clause():
    # e.g. a query mentioning both "boro" and "aman" paddy
    result = build_metadata_filter(crops=["Boro Paddy", "Aman Paddy"])
    assert result == {"crop_name": {"$in": ["Boro Paddy", "Aman Paddy"]}}
