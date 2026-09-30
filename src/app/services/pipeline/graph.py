"""LangGraph assembly for the complete chat pipeline."""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.services.pipeline.checkpointer import make_checkpointer
from app.services.pipeline.nodes.generate import generate
from app.services.pipeline.nodes.generate_company import generate_company
from app.services.pipeline.nodes.generate_chitchat import generate_chitchat
from app.services.pipeline.nodes.generate_meaningless import generate_meaningless
from app.services.pipeline.nodes.generate_soil_test import generate_soil_test
from app.services.pipeline.nodes.handle_agronomist_request import handle_agronomist_request
from app.services.pipeline.nodes.decompose_query import decompose_query
from app.services.pipeline.nodes.parallel_retrieve import retrieve_decomposed_queries
from app.services.pipeline.nodes.retrieve_company import retrieve_company
from app.services.pipeline.nodes.retrieve_soil_test import retrieve_soil_test
from app.services.pipeline.nodes.rewrite_query import rewrite_query
from app.services.pipeline.nodes.route import route
from app.services.pipeline.state import PipelineState


def route_after_route(state: PipelineState) -> str:
    if state.get("intent") == "request_agronomist":
        return "handle_agronomist_request"
    if state.get("intent") == "chitchat":
        return "generate_chitchat"
    if state.get("intent") == "meaningless":
        return "generate_meaningless"
    if state.get("intent") in {"crop_query", "company_query", "soil_test_query"}:
        return "rewrite_query"


def route_after_rewrite(state: PipelineState) -> str:
    if state.get("intent") == "crop_query":
        return "decompose_query"
    if state.get("intent") == "company_query":
        return "retrieve_company"
    if state.get("intent") == "soil_test_query":
        return "retrieve_soil_test"

def build_chat_graph():
    builder = StateGraph(PipelineState)
    
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("route", route)
    builder.add_node("decompose_query", decompose_query)
    builder.add_node("retrieve_decomposed_queries", retrieve_decomposed_queries)
    builder.add_node("retrieve_company", retrieve_company)
    builder.add_node("retrieve_soil_test", retrieve_soil_test)
    builder.add_node("generate", generate)
    builder.add_node("generate_company", generate_company)
    builder.add_node("generate_soil_test", generate_soil_test)
    builder.add_node("generate_chitchat", generate_chitchat)
    builder.add_node("generate_meaningless", generate_meaningless)
    builder.add_node("handle_agronomist_request", handle_agronomist_request)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        route_after_route,
        {
            "rewrite_query": "rewrite_query",
            "generate_chitchat": "generate_chitchat",
            "generate_meaningless": "generate_meaningless",
            "handle_agronomist_request": "handle_agronomist_request",
        },
    )
    builder.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "decompose_query": "decompose_query",
            "retrieve_company": "retrieve_company",
            "retrieve_soil_test": "retrieve_soil_test",
        },
    )
    builder.add_edge("decompose_query", "retrieve_decomposed_queries")
    builder.add_edge("retrieve_decomposed_queries", "generate")
    builder.add_edge("retrieve_company", "generate_company")
    builder.add_edge("retrieve_soil_test", "generate_soil_test")
    builder.add_edge("generate", END)
    builder.add_edge("generate_company", END)
    builder.add_edge("generate_chitchat", END)
    builder.add_edge("generate_meaningless", END)
    builder.add_edge("handle_agronomist_request", END)
    return builder.compile(checkpointer=make_checkpointer())


@lru_cache
def get_chat_graph():
    return build_chat_graph()
