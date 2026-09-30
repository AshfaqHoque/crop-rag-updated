"""Retrieve and rerank documents independently for each query branch."""

from app.services.pipeline.nodes.rerank import rerank
from app.services.pipeline.nodes.retrieve import retrieve
from app.services.pipeline.state import PipelineState


def retrieve_and_rerank(state: PipelineState) -> PipelineState:
    retrieved = retrieve(state)
    reranked = rerank({**state, **retrieved})
    return {
        "retrieved_documents": retrieved.get("retrieved_documents", []),
        "reranked_documents": reranked.get("reranked_documents", []),
    }