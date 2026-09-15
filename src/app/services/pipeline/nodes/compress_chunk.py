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
DEFAULT_EXTRACTION_TEMPLATE = """Given the following question and context, extract any part of the context AS IS that is directly useful for answering the question. Preserve context liberally. Return NO_OUTPUT only if the context is completely irrelevant.
Remember, DO NOT edit the extracted parts of the context. If the question asks about multiple items (e.g. a comparison) and the context only contains information about one of them, still extract that item's full content as-is — do not return NO_OUTPUT just because the other item is missing.

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
    #for vllm it is max_tokens=1000, for ollama it is num_predict=1000
    llm = get_chat_llm(temperature=0.1)
    # Uses LangChain's default extraction prompt.
    return LLMChainExtractor.from_llm(llm, prompt=CUSTOM_DEFAULT_PROMPT)

def _metadata_prefix(metadata: dict) -> str:
    """Deterministically render identifying metadata as context, so it
    can never be dropped by the compressor -- no per-section mapping
    needed, generalizes to any section/field automatically.
    """

    # Drop fields that are pure plumbing, not identity.
    skip = {"_chunk_index", "crop_id", "variety_id", "chunk_id", "relevance_score", "distance", "crop_bangla_name"}

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

    documents = state.get("reranked_documents", []) or state.get("retrieved_documents", [])

    if not documents:
        return {**state, "compressed_documents": []}

    query = ( state.get("rewritten_query") or state.get("normalized_query")or state.get("raw_query", ""))
    compression_inputs: list[Document] = []

    for index, document in enumerate(documents):
        content = document.page_content.strip()

        if not content:
            continue

        metadata = dict(document.metadata)
        metadata["_source_index"] = index
        compression_inputs.append(Document(page_content=content, metadata=metadata))

    try:
        compressor = get_context_compressor()
        compressed_documents = compressor.compress_documents(
            documents=compression_inputs,
            query=query,
        )
        output_documents = []

        for compressed_document in compressed_documents:
            content = compressed_document.page_content.strip()

            if not content:
                continue

            index = compressed_document.metadata["_source_index"]
            source_document = documents[index]
            metadata = dict(source_document.metadata)
            prefix = _metadata_prefix(metadata)

            if prefix:
                content = f"{prefix}{content}"

            output_documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )
    except Exception:
        logger.exception("Context compression failed; using reranked chunks")
        output_documents = list(documents)

    original_chars = sum(
        len(document.page_content)
        for document in documents
    )

    compressed_chars = sum(
        len(document.page_content)
        for document in output_documents
    )

    logger.info(
        "context_compression chunks_before=%d chunks_after=%d "
        "chars_before=%d chars_after=%d",
        len(documents),
        len(output_documents),
        original_chars,
        compressed_chars,
    )

    return {**state, "compressed_documents": output_documents}