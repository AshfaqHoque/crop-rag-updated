import json

from app.core.config import get_settings
from app.services.pipeline import registry


def _write_crops(tmp_path, crops):
    path = tmp_path / "crops.json"
    path.write_text(json.dumps(crops), encoding="utf-8")
    return path


def test_get_known_crops_loads_from_file(tmp_path, monkeypatch):
    crops = [
        {"crop_id": "5", "crop_name": "Boro Paddy", "crop_bangla_name": "বোরো ধান"},
        {"crop_id": "12", "crop_name": "Mango", "crop_bangla_name": "আম"},
    ]
    path = _write_crops(tmp_path, crops)
    monkeypatch.setattr(get_settings(), "crop_registry_path", str(path))
    registry.get_known_crops.cache_clear()
    try:
        result = registry.get_known_crops()
        assert len(result) == 2
        assert result[0].crop_name == "Boro Paddy"
        assert result[1].crop_bangla_name == "আম"
    finally:
        registry.get_known_crops.cache_clear()


def test_get_known_crops_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "crop_registry_path", str(tmp_path / "missing.json"))
    registry.get_known_crops.cache_clear()
    try:
        assert registry.get_known_crops() == []
    finally:
        registry.get_known_crops.cache_clear()


def test_crop_names_and_prompt_formatting(tmp_path, monkeypatch):
    crops = [{"crop_id": "5", "crop_name": "Boro Paddy", "crop_bangla_name": "বোরো ধান"}]
    path = _write_crops(tmp_path, crops)
    monkeypatch.setattr(get_settings(), "crop_registry_path", str(path))
    registry.get_known_crops.cache_clear()
    try:
        assert registry.crop_names() == ["Boro Paddy"]
        assert "Boro Paddy (বোরো ধান)" in registry.format_crop_list_for_prompt()
    finally:
        registry.get_known_crops.cache_clear()


def test_sections_list_has_eleven_entries():
    # Matches the user's original spec: 11 sections (general info, seed, irrigation,
    # pesticide, herbicide, etc.)
    assert len(registry.SECTIONS) == 11
    assert "seed" in registry.SECTIONS
    assert "pesticide" in registry.SECTIONS
