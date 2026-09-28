"""Summarize retrieved chunks while preserving their factual details."""

from functools import lru_cache

from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.llm.client import get_chat_llm
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

SUMMARY_TEMPLATE = """Summarize and shorten the following context for the user's query.
Keep all useful facts and preserve the original tone. Do not alter, normalize, or omit
numerical values, units, dates, names, or other identifying details. 
Do not add facts that are not present in the chunk. Do not answer the question.
If the context contains no useful information for the query, return NO_OUTPUT.

Query: {question}
Context:
{context}

Summary:"""

SUMMARY_PROMPT = PromptTemplate(
    template=SUMMARY_TEMPLATE,
    input_variables=["question", "context"],
)


@lru_cache
def get_chunk_summarizer() -> LLMChainExtractor:
    """Create the LLM chunk summarizer."""
    settings = get_settings()
    llm = get_chat_llm(temperature=settings.compression_temperature).bind(
        max_tokens=settings.compression_max_tokens
    )
    return LLMChainExtractor.from_llm(llm, prompt=SUMMARY_PROMPT)


def _metadata_prefix(metadata: dict) -> str:
    skip = {
        "_chunk_index",
        "crop_id",
        "variety_id",
        "chunk_id",
        "relevance_score",
        "distance",
        "crop_bangla_name",
    }
    pairs = [
        f"{key}: {value}"
        for key, value in metadata.items()
        if key not in skip and value not in (None, "")
    ]
    return f"[{' | '.join(pairs)}] " if pairs else ""


async def summarize_chunks(state: PipelineState) -> PipelineState:
    """Summarize reranked chunks and retain their identifying metadata."""
    documents = state.get("reranked_documents", []) or state.get("retrieved_documents", [])
    if not documents:
        return {**state, "compressed_documents": []}

    query = state.get("rewritten_query") or state.get("raw_query", "")
    try:
        summaries = await get_chunk_summarizer().acompress_documents(
            documents=documents,
            query=query,
        )
        output_documents = []
        for summary in summaries:
            content = summary.page_content.strip()
            if not content:
                continue
            metadata = dict(summary.metadata)
            prefix = _metadata_prefix(metadata)
            if prefix:
                content = f"{prefix}{content}"
            output_documents.append(Document(page_content=content, metadata=metadata))
    except Exception:
        logger.exception("Chunk summarization failed; using reranked chunks")
        output_documents = documents

    logger.info(
        "chunk_summarization chunks_before=%d chunks_after=%d",
        len(documents),
        len(output_documents),
    )
    return {**state, "compressed_documents": output_documents}