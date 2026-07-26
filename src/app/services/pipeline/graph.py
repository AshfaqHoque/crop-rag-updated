"""LangGraph assembly for the complete chat pipeline."""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.services.pipeline.nodes.generate import generate
from app.services.pipeline.nodes.rerank import rerank
from app.services.pipeline.nodes.retrieve import retrieve
from app.services.pipeline.nodes.rewrite_query import rewrite_query
from app.services.pipeline.nodes.understand_query import understand_query
from app.services.pipeline.state import PipelineState


def route_after_understanding(state: PipelineState) -> str:
    is_crop_query = state.get("intent") == "crop_query"
    is_supported = state.get("language") != "unsupported"
    return "retrieve" if is_crop_query and is_supported else "generate"


def build_chat_graph():
    builder = StateGraph(PipelineState)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("understand_query", understand_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("generate", generate)

    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "understand_query")
    builder.add_conditional_edges(
        "understand_query",
        route_after_understanding,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


@lru_cache
def get_chat_graph():
    return build_chat_graph()
