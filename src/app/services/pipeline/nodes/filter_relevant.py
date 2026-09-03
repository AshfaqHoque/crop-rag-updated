"""Compress reranked chunks using LangChain LLMChainExtractor."""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from app.core.logging import get_logger
from app.services.llm.client import get_chat_llm
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)


@lru_cache
def get_context_compressor() -> LLMChainExtractor:
    """Create the LangChain LLM context compressor."""

    llm = get_chat_llm(temperature=0.0)

    # Uses LangChain's default extraction prompt.
    return LLMChainExtractor.from_llm(llm)


def filter_relevant(state: PipelineState) -> PipelineState:
    """Extract only query-relevant content from reranked chunks."""

    chunks = state.get("reranked_chunks", [])

    if not chunks:
        return {
            **state,
            "compressed_chunks": [],
        }

    query = (
        state.get("rewritten_query")
        or state.get("normalized_query")
        or state.get("raw_query", "")
    )

    documents: list[Document] = []

    for index, chunk in enumerate(chunks):
        content = str(chunk.get("content", "")).strip()

        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "_chunk_index": index,
                },
            )
        )

    if not documents:
        return {
            **state,
            "compressed_chunks": [],
        }

    try:
        compressor = get_context_compressor()

        compressed_documents = compressor.compress_documents(
            documents=documents,
            query=query,
        )

        compressed_chunks = []

        for document in compressed_documents:
            content = document.page_content.strip()

            if not content:
                continue

            index = document.metadata["_chunk_index"]

            # Restore your original chunk object
            chunk = dict(chunks[index])

            # Replace full content with compressed content
            chunk["content"] = content

            compressed_chunks.append(chunk)

    except Exception:
        logger.exception(
            "Context compression failed; using reranked chunks"
        )

        # Fail open:
        # if compression fails, generation still gets the reranked data.
        compressed_chunks = [
            dict(chunk)
            for chunk in chunks
        ]

    original_chars = sum(
        len(str(chunk.get("content", "")))
        for chunk in chunks
    )

    compressed_chars = sum(
        len(str(chunk.get("content", "")))
        for chunk in compressed_chunks
    )

    logger.info(
        "context_compression chunks_before=%d chunks_after=%d "
        "chars_before=%d chars_after=%d",
        len(chunks),
        len(compressed_chunks),
        original_chars,
        compressed_chars,
    )

    return {
        **state,
        "filtered_chunks": compressed_chunks,
    }