"""LangGraph assembly for the complete chat pipeline."""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.services.pipeline.checkpointer import make_checkpointer
from app.services.pipeline.nodes.compress_chunk import compress_chunk
from app.services.pipeline.nodes.extract_crop import extract_crop
from app.services.pipeline.nodes.generate import generate
from app.services.pipeline.nodes.generate_company import generate_company
from app.services.pipeline.nodes.generate_chitchat import generate_chitchat
from app.services.pipeline.nodes.generate_meaningless import generate_meaningless
from app.services.pipeline.nodes.generate_soil_test import generate_soil_test
from app.services.pipeline.nodes.rerank import rerank
from app.services.pipeline.nodes.retrieve import retrieve
from app.services.pipeline.nodes.retrieve_company import retrieve_company
from app.services.pipeline.nodes.retrieve_soil_test import retrieve_soil_test
from app.services.pipeline.nodes.rewrite_query import rewrite_query
from app.services.pipeline.nodes.route import route
from app.services.pipeline.state import PipelineState


def route_after_route(state: PipelineState) -> str:
    if state.get("intent") == "chitchat":
        return "generate_chitchat"
    if state.get("intent") == "meaningless":
        return "generate_meaningless"
    if state.get("intent") == "crop_query":
        return "extract_crop"
    if state.get("intent") == "company_query":
        return "retrieve_company"
    if state.get("intent") == "soil_test_query":
        return "retrieve_soil_test"

def build_chat_graph():
    builder = StateGraph(PipelineState)
    
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("route", route)
    builder.add_node("extract_crop", extract_crop)
    builder.add_node("retrieve", retrieve)
    builder.add_node("retrieve_company", retrieve_company)
    builder.add_node("retrieve_soil_test", retrieve_soil_test)
    builder.add_node("rerank", rerank)
    builder.add_node("compress_chunk", compress_chunk)
    builder.add_node("generate", generate)
    builder.add_node("generate_company", generate_company)
    builder.add_node("generate_soil_test", generate_soil_test)
    builder.add_node("generate_chitchat", generate_chitchat)
    builder.add_node("generate_meaningless", generate_meaningless)

    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "route")
    builder.add_conditional_edges(
        "route",
        route_after_route,
        {
            "extract_crop": "extract_crop",
            "retrieve_company": "retrieve_company",
            "retrieve_soil_test": "retrieve_soil_test",
            "generate_chitchat": "generate_chitchat",
            "generate_meaningless": "generate_meaningless",
        },
    )
    builder.add_edge("extract_crop", "retrieve")
    builder.add_edge("retrieve_company", "generate_company")
    builder.add_edge("retrieve_soil_test", "generate_soil_test")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "compress_chunk")
    builder.add_edge("compress_chunk", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("generate_company", END)
    builder.add_edge("generate_chitchat", END)
    builder.add_edge("generate_meaningless", END)

    return builder.compile(checkpointer=make_checkpointer())


@lru_cache
def get_chat_graph():
    return build_chat_graph()
