"""Fetch AIS web-search results when local retrieval has no relevant context."""

import httpx
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)


async def web_search(state: PipelineState) -> PipelineState:
    query = state.get("rewritten_query") or state.get("raw_query", "")
    settings = get_settings()

    async with httpx.AsyncClient(timeout=settings.web_search_timeout_seconds) as client:
        response = await client.get(settings.web_search_url, params={"q": query})
    response.raise_for_status()
    results = response.json()

    if not isinstance(results, list):
        raise ValueError("Web search response must be a list of results")

    documents: list[Document] = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Web search results must be JSON objects")

        body = result.get("body")
        url = result.get("url")
        title = result.get("title")
        if not isinstance(body, str) or not isinstance(url, str):
            raise ValueError("Web search results must include string 'body' and 'url' fields")

        body = body.strip()
        url = url.strip()
        if not body or not url:
            continue

        title = title.strip() if isinstance(title, str) else ""
        content = f"Title: {title}\nSource: {url}\n\n{body}" if title else f"Source: {url}\n\n{body}"
        metadata = {"source": "web_search", "url": url}
        if title:
            metadata["title"] = title
        documents.append(Document(page_content=content, metadata=metadata))

    logger.info("web_search results=%d query=%r", len(documents), query)
    return {"compressed_documents": documents}
