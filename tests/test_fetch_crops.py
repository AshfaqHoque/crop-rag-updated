import json

import pytest

from app.core.config import get_settings
from app.ingestion.fetch_crops import fetch_crops


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_fetch_crops_writes_successful_payload(monkeypatch, tmp_path):
    registry = tmp_path / "crops.json"
    payload = {"data": {"getAllCropsFullDetails": {"rows": [{"id": 5}]}}}
    calls = {}

    def post(endpoint, **kwargs):
        calls.update(endpoint=endpoint, **kwargs)
        return FakeResponse(payload)

    monkeypatch.setattr("app.ingestion.fetch_crops.requests.post", post)
    monkeypatch.setenv("GRAPHQL_ENDPOINT", "https://example.test/graphql")
    monkeypatch.setenv("CROP_REGISTRY_PATH", str(registry))
    get_settings.cache_clear()

    assert fetch_crops(7) == payload
    assert json.loads(registry.read_text(encoding="utf-8")) == payload
    assert calls["endpoint"] == "https://example.test/graphql"
    assert calls["json"]["variables"] == {"updated_within_days": 7}


def test_fetch_crops_does_not_write_graphql_errors(monkeypatch, tmp_path):
    registry = tmp_path / "crops.json"
    registry.write_text('{"old": true}', encoding="utf-8")
    payload = {"errors": [{"message": "failed"}]}

    monkeypatch.setattr("app.ingestion.fetch_crops.requests.post", lambda *args, **kwargs: FakeResponse(payload))
    monkeypatch.setenv("GRAPHQL_ENDPOINT", "https://example.test/graphql")
    monkeypatch.setenv("CROP_REGISTRY_PATH", str(registry))
    get_settings.cache_clear()

    with pytest.raises(RuntimeError):
        fetch_crops()
    assert registry.read_text(encoding="utf-8") == '{"old": true}'