"""Compress reranked chunks using LangChain LLMChainExtractor."""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import PromptTemplate

from app.core.logging import get_logger
from app.services.llm.client import get_chat_llm
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

# Standard LangChain LLMChainExtractor prompt with your custom rule added
DEFAULT_EXTRACTION_TEMPLATE = """Given the following question and context, extract any part of the context AS IS that is relevant to answer the question. Preserve context liberally. Return NO_OUTPUT ONLY if the context is completely irrelevant.
Remember, DO NOT edit the extracted parts of the context.

Question: {question}
Context:
{context}

Extracted Content:"""

CUSTOM_DEFAULT_PROMPT = PromptTemplate(
    template=DEFAULT_EXTRACTION_TEMPLATE,
    input_variables=["question", "context"],
)

@lru_cache
def get_context_compressor() -> LLMChainExtractor:
    """Create the LangChain LLM context compressor."""

    llm = get_chat_llm(temperature=0.0)
    # Uses LangChain's default extraction prompt.
    return LLMChainExtractor.from_llm(llm, prompt=CUSTOM_DEFAULT_PROMPT)

def _metadata_prefix(metadata: dict) -> str:
    """Deterministically render identifying metadata as context, so it
    can never be dropped by the compressor -- no per-section mapping
    needed, generalizes to any section/field automatically.
    """

    # Drop fields that are pure plumbing, not identity.
    skip = {"_chunk_index", "crop_id", "variety_id", "chunk_id"}

    pairs = [
        f"{key}: {value}"
        for key, value in metadata.items()
        if key not in skip and value not in (None, "")
    ]

    if not pairs:
        return ""

    return "[" + " | ".join(pairs) + "] "

def compress_chunk(state: PipelineState) -> PipelineState:
    """Extract only query-relevant content from reranked chunks."""

    chunks = state.get("reranked_chunks", []) or state.get("retrieved_chunks", [])

    if not chunks:
        return {**state,"compressed_chunks": [],}

    query = ( state.get("rewritten_query") or state.get("normalized_query")or state.get("raw_query", ""))
    documents: list[Document] = []

    for index, chunk in enumerate(chunks):
        content = str(chunk.get("content", "")).strip()

        if not content:
            continue

        documents.append(Document(page_content=content,metadata={"_chunk_index": index,},))

    try:
        compressor = get_context_compressor()
        compressed_documents = compressor.compress_documents(documents=documents, query=query,)
        compressed_chunks = []

        for document in compressed_documents:
            content = document.page_content.strip()

            if not content:
                continue

            index = document.metadata["_chunk_index"]
            chunk = dict(chunks[index])

            prefix = _metadata_prefix(chunk.get("metadata", {}))
            if prefix:
                content = f"{prefix}{content}"

            chunk["content"] = content
            compressed_chunks.append(chunk)

    except Exception:
        logger.exception("Context compression failed; using reranked chunks")
        compressed_chunks = [dict(chunk) for chunk in chunks]

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

    return { **state,"compressed_chunks": compressed_chunks, }