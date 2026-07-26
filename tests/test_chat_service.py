import pytest

from app.memory.store import InMemoryConversationStore
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService


class FakeGraph:
    def __init__(self):
        self.states = []

    def invoke(self, state):
        self.states.append(state)
        return {
            **state,
            "language": "en",
            "rewritten_query": "standalone query",
            "retrieval_mode": "dense_filtered",
            "answer": "grounded answer [1]",
            "reranked_chunks": [
                {
                    "chunk_id": "5_seed",
                    "metadata": {"crop_name": "Boro Paddy", "section": "seed"},
                    "rerank_score": 0.91,
                }
            ],
        }


@pytest.mark.asyncio
async def test_chat_service_loads_and_persists_history():
    store = InMemoryConversationStore(max_turns=3)
    store.append_turn("session", "old question", "old answer")
    graph = FakeGraph()
    service = ChatService(history_store=store, graph=graph)

    response = await service.chat(ChatRequest(session_id="session", message="follow up"))

    assert graph.states[0]["history"][0]["content"] == "old question"
    assert response.answer == "grounded answer [1]"
    assert response.sources[0].chunk_id == "5_seed"
    assert store.get_history("session")[-1]["content"] == "grounded answer [1]"
