import pytest
from langchain_core.documents import Document

from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService


class FakeGraph:
    def __init__(self):
        self.states = []

    def invoke(self, state, config):
        self.states.append(state)
        return {
            **state,
            "language": "en",
            "rewritten_query": "standalone query",
            "retrieval_mode": "dense_filtered",
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


@pytest.mark.asyncio
async def test_chat_service_uses_langgraph_thread_state():
    graph = FakeGraph()
    service = ChatService(graph=graph)

    response = await service.chat(ChatRequest(session_id="session", message="follow up"))

    assert graph.states[0]["messages"][0].content == "follow up"
    assert response.answer == "grounded answer [1]"
    assert response.sources[0].chunk_id == "5_seed"
    assert response.sources[0].distance == 0.91
