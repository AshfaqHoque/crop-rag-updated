"""Fetch AIS web-search results when local retrieval has no relevant context."""

from langchain_core.documents import Document

from app.core.logging import get_logger
from app.services.pipeline.state import PipelineState
from app.services.web_search import search_ais

logger = get_logger(__name__)


async def web_search(state: PipelineState) -> PipelineState:
    query = state.get("rewritten_query") or state.get("raw_query", "")

    results = await search_ais(query)

    documents: list[Document] = []
    for result in results:
        body = (result.get("body") or "").strip()
        url = (result.get("url") or "").strip()
        title = (result.get("title") or "").strip()
        if not body or not url:
            continue

        content = (
            f"Title: {title}\nSource: {url}\n\n{body}"
            if title
            else f"Source: {url}\n\n{body}"
        )
        metadata = {"source": "web_search", "url": url}
        if title:
            metadata["title"] = title
        documents.append(Document(page_content=content, metadata=metadata))

    logger.info("web_search results=%d query=%r", len(documents), query)
    return {"compressed_documents": documents}
