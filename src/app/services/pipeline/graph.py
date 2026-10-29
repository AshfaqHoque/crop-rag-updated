"""LangGraph assembly for the complete chat pipeline."""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.services.pipeline.checkpointer import get_checkpointer
from app.services.pipeline.nodes.decompose_query import decompose_query
from app.services.pipeline.nodes.extract_crop import extract_crop
from app.services.pipeline.nodes.generate import generate
from app.services.pipeline.nodes.generate_company import generate_company
from app.services.pipeline.nodes.generate_chitchat import generate_chitchat
from app.services.pipeline.nodes.generate_capability import generate_capability
from app.services.pipeline.nodes.generate_meaningless import generate_meaningless
from app.services.pipeline.nodes.generate_out_of_scope import generate_out_of_scope
from app.services.pipeline.nodes.generate_soil_test import generate_soil_test
from app.services.pipeline.nodes.handle_agronomist_request import handle_agronomist_request
from app.services.pipeline.nodes.retrieve_and_rerank import retrieve_rerank_and_compress
from app.services.pipeline.nodes.retrieve_company import retrieve_company
from app.services.pipeline.nodes.retrieve_soil_test import retrieve_soil_test
from app.services.pipeline.nodes.rewrite_query import rewrite_query
from app.services.pipeline.nodes.route import route
from app.services.pipeline.nodes.decide_retrieval_mode import decide_retrieval_mode
from app.services.pipeline.nodes.sql_plan import sql_plan
from app.services.pipeline.nodes.sql_generate import sql_generate
from app.services.pipeline.nodes.sql_execute import sql_execute
from app.services.pipeline.nodes.generate_sql_answer import generate_sql_answer
from app.services.pipeline.state import PipelineState


def route_after_route(state: PipelineState) -> str:
    if state.get("intent") == "request_agronomist":
        return "handle_agronomist_request"
    if state.get("intent") == "chitchat":
        return "generate_chitchat"
    if state.get("intent") == "meaningless":
        return "generate_meaningless"
    if state.get("intent") == "out_of_scope":
        return "generate_out_of_scope"
    if state.get("intent") == "capability_query":
        return "generate_capability"
    if state.get("intent") == "crop_query":
        return "decide_retrieval_mode"
    if state.get("intent") == "company_query":
        return "retrieve_company"
    if state.get("intent") == "soil_test_query":
        return "retrieve_soil_test"


def route_after_decide(state: PipelineState) -> str:
    if state.get("retrieval_mode") == "sql":
        return "sql_plan"
    return "extract_crop"


def fan_out_subqueries(state: PipelineState) -> list[Send]:
    return [
        Send("retrieve_rerank_and_compress", {
            **state,
            "rewritten_query": q,
            "retrieved_documents": [],
            "reranked_documents": [],
            "compressed_documents": [],
        })
        for q in state["subqueries"]
    ]


def build_chat_graph():
    builder = StateGraph(PipelineState)
    
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("route", route)
    builder.add_node("extract_crop", extract_crop)
    builder.add_node("retrieve_rerank_and_compress", retrieve_rerank_and_compress)
    builder.add_node("decompose_query", decompose_query)
    builder.add_node("retrieve_company", retrieve_company)
    builder.add_node("retrieve_soil_test", retrieve_soil_test)
    builder.add_node("generate", generate)
    builder.add_node("generate_company", generate_company)
    builder.add_node("generate_soil_test", generate_soil_test)
    builder.add_node("generate_chitchat", generate_chitchat)
    builder.add_node("generate_capability", generate_capability)
    builder.add_node("generate_meaningless", generate_meaningless)
    builder.add_node("generate_out_of_scope", generate_out_of_scope)
    builder.add_node("handle_agronomist_request", handle_agronomist_request)
    builder.add_node("decide_retrieval_mode", decide_retrieval_mode)
    builder.add_node("sql_plan", sql_plan)
    builder.add_node("sql_generate", sql_generate)
    builder.add_node("sql_execute", sql_execute)
    builder.add_node("generate_sql_answer", generate_sql_answer)

    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "route")
    builder.add_conditional_edges(
        "route",
        route_after_route,
        {
            "decide_retrieval_mode": "decide_retrieval_mode",
            "retrieve_company": "retrieve_company",
            "retrieve_soil_test": "retrieve_soil_test",
            "generate_chitchat": "generate_chitchat",
            "generate_capability": "generate_capability",
            "generate_meaningless": "generate_meaningless",
            "generate_out_of_scope": "generate_out_of_scope",
            "handle_agronomist_request": "handle_agronomist_request",
        },
    )

    builder.add_conditional_edges(
        "decide_retrieval_mode",
        route_after_decide,
        {
            "sql_plan": "sql_plan",
            "extract_crop": "extract_crop",
        },
    )

    builder.add_edge("sql_plan", "sql_generate")
    builder.add_edge("sql_generate", "sql_execute")
    builder.add_edge("sql_execute", "generate_sql_answer")
    builder.add_edge("generate_sql_answer", END)
    builder.add_edge("extract_crop", "decompose_query")
    builder.add_conditional_edges("decompose_query", fan_out_subqueries, ["retrieve_rerank_and_compress"])
    builder.add_edge("retrieve_company", "generate_company")
    builder.add_edge("retrieve_soil_test", "generate_soil_test")
    builder.add_edge("retrieve_rerank_and_compress", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("generate_company", END)
    builder.add_edge("generate_chitchat", END)
    builder.add_edge("generate_capability", END)
    builder.add_edge("generate_meaningless", END)
    builder.add_edge("generate_out_of_scope", END)
    builder.add_edge("handle_agronomist_request", END)
    return builder.compile(checkpointer=get_checkpointer())


@lru_cache
def get_chat_graph():
    return build_chat_graph()

def reset_chat_graph():
    """Call after checkpointer is initialized so the graph picks it up."""
    get_chat_graph.cache_clear()