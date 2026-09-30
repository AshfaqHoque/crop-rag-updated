"""Run retrieval, reranking, and compression concurrently per subquery."""

import asyncio

from app.services.pipeline.nodes.compress_chunk import compress_chunk
from app.services.pipeline.nodes.extract_crop import extract_crop
from app.services.pipeline.nodes.rerank import rerank
from app.services.pipeline.nodes.retrieve import retrieve
from app.services.pipeline.state import PipelineState


async def _retrieve_one(state: PipelineState, query: str) -> PipelineState:
    query_state: PipelineState = {
        **state,
        "rewritten_query": query,
        "retrieved_documents": [],
        "reranked_documents": [],
        "compressed_documents": [],
    }
    extracted_state = await asyncio.to_thread(extract_crop, query_state)
    retrieved_state = await asyncio.to_thread(retrieve, extracted_state)
    reranked_state = await asyncio.to_thread(rerank, retrieved_state)
    return await compress_chunk(reranked_state)


async def retrieve_decomposed_queries(state: PipelineState) -> PipelineState:
    queries = state.get("decomposed_queries") or [
        state.get("rewritten_query") or state.get("raw_query", "")
    ]
    results = await asyncio.gather(*(_retrieve_one(state, query) for query in queries))

    retrieved_documents = [
        document for result in results for document in result.get("retrieved_documents", [])
    ]
    reranked_documents = [
        document for result in results for document in result.get("reranked_documents", [])
    ]
    compressed_documents = [
        document for result in results for document in result.get("compressed_documents", [])
    ]
    modes = {result.get("retrieval_mode") for result in results if result.get("retrieval_mode")}

    return {
        **state,
        "retrieved_documents": retrieved_documents,
        "reranked_documents": reranked_documents,
        "compressed_documents": compressed_documents,
        "retrieval_mode": next(iter(modes)) if len(modes) == 1 else "multi_query",
    }