"""AIS site-restricted web search via SearXNG + trafilatura extraction."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import trafilatura

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _extract_body(url: str) -> str | None:
    """Sync helper: fetch + extract main text (run in a thread)."""
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded)
        return text.strip() if text else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("trafilatura failed for %s: %s", url, exc)
        return None


async def search_ais(query: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    """
    Search site:ais.gov.bd via SearXNG and return [{title, url, body}, ...].
    Mirrors the previous external API response shape.
    """
    settings = get_settings()
    limit = limit or settings.web_search_limit
    site = settings.web_search_site

    params = {
        "q": f"site:{site} {query}",
        "format": "json",
    }

    async with httpx.AsyncClient(timeout=settings.web_search_timeout_seconds) as client:
        resp = await client.get(settings.searxng_url, params=params)
        resp.raise_for_status()
        data = resp.json()

    raw_results = [
        r for r in data.get("results", [])
        if site in r.get("url", "")
    ][:limit]

    # Extract bodies concurrently (trafilatura is sync)
    async def _one(r: dict) -> dict[str, Any] | None:
        url = r.get("url")
        if not url:
            return None
        body = await asyncio.to_thread(_extract_body, url)
        if not body:
            return None
        return {
            "title": r.get("title") or "",
            "url": url,
            "body": body,
        }

    results = await asyncio.gather(*(_one(r) for r in raw_results))
    return [r for r in results if r is not None]