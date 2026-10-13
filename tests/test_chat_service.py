import json

import pytest
from langchain_core.documents import Document

from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.thinking_messages import BENGALI_THINKING_MESSAGES


class FakeGraph:
    def __init__(self):
        self.states = []

    def invoke(self, state, config):
        self.states.append(state)
        return {
            **state,
            "language": "en",
            "rewritten_query": "standalone query",
            "answer": "grounded answer [1]",
            "reranked_documents": [
                Document(
                    page_content="seed rate",
                    metadata={
                        "chunk_id": "5_seed",
                        "crop_name": "Boro Paddy",
                        "section": "seed",
                        "relevance_score": 0.91,
                    },
                )
            ],
        }

    async def astream(self, state, config, stream_mode):
        if False:
            yield None


@pytest.mark.asyncio
async def test_chat_service_uses_langgraph_thread_state():
    graph = FakeGraph()
    service = ChatService(graph=graph)

    response = await service.chat(
        ChatRequest(session_id="session", message="follow up", language_type="english")
    )

    assert graph.states[0]["messages"][0].content == "follow up"
    assert graph.states[0]["language_type"] == "english"
    assert response.answer == "grounded answer [1]"
    assert response.sources[0].chunk_id == "5_seed"
    assert response.sources[0].distance == 0.91


@pytest.mark.asyncio
async def test_stream_chat_uses_bangla_thinking_message():
    service = ChatService(graph=FakeGraph())

    chunks = service.stream_chat(
        ChatRequest(session_id="session", message="ধানের পরিচর্যা", language_type="bangla")
    )
    status_event = json.loads((await anext(chunks)).removeprefix("data: ").strip())

    assert status_event["type"] == "status"
    assert status_event["content"] in BENGALI_THINKING_MESSAGES

    await chunks.aclose()


@pytest.mark.asyncio
async def test_stream_chat_emits_out_of_scope_fallback_from_node_update():
    class OutOfScopeGraph:
        async def astream(self, state, config, stream_mode):
            yield "updates", {"route": {"intent": "out_of_scope"}}
            yield "updates", {
                "generate_out_of_scope": {
                    "answer": "Sorry, I can't help with this topic. Would you like to know something about agriculture?"
                }
            }

    service = ChatService(graph=OutOfScopeGraph())
    chunks = service.stream_chat(
        ChatRequest(session_id="session", message="tell me a joke", language_type="en")
    )
    events = [json.loads(event.removeprefix("data: ").strip()) async for event in chunks]

    assert any(event.get("node") == "generate_out_of_scope" for event in events)
    assert any(event["type"] == "token" and event["content"].startswith("Sorry,") for event in events)
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"].startswith("Sorry,")
