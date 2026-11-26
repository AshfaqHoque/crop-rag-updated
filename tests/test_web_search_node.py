from types import SimpleNamespace

import pytest

from app.services.pipeline.nodes import web_search as web_search_module


@pytest.mark.asyncio
async def test_web_search_fetches_and_returns_grounding_documents(monkeypatch):
    requests = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {
                    "title": "Rice cultivation",
                    "url": "https://ais.gov.bd/rice",
                    "body": "Plant rice seedlings at the recommended spacing.",
                },
                {"title": None, "url": "https://ais.gov.bd/empty", "body": "  "},
            ]

    class FakeClient:
        def __init__(self, *, timeout):
            assert timeout == 20

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            pass

        async def get(self, url, *, params):
            requests.append((url, params))
            return FakeResponse()

    monkeypatch.setattr(web_search_module.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(
        web_search_module,
        "get_settings",
        lambda: SimpleNamespace(
            web_search_url="http://localhost:8000/search",
            web_search_timeout_seconds=20,
        ),
    )

    result = await web_search_module.web_search(
        {"raw_query": "rice spacing", "rewritten_query": "rice seedling spacing"}
    )

    assert requests == [
        ("http://localhost:8000/search", {"q": "rice seedling spacing"})
    ]
    assert len(result["compressed_documents"]) == 1
    document = result["compressed_documents"][0]
    assert "Title: Rice cultivation" in document.page_content
    assert "Source: https://ais.gov.bd/rice" in document.page_content
    assert "Plant rice seedlings" in document.page_content
    assert document.metadata == {
        "source": "web_search",
        "url": "https://ais.gov.bd/rice",
        "title": "Rice cultivation",
    }
