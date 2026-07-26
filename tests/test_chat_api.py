from fastapi.testclient import TestClient

from app.main import app
from app.schemas.chat import ChatResponse
from app.services.chat_service import get_chat_service


class FakeChatService:
    async def chat(self, request):
        return ChatResponse(
            session_id=request.session_id,
            answer="ok",
            language="en",
            rewritten_query=request.message,
            retrieval_mode="dense_filtered",
            sources=[],
        )


def test_chat_endpoint_uses_service():
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    try:
        response = TestClient(app).post(
            "/api/v1/chat",
            json={"session_id": "s1", "message": "seed rate?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "ok"
    assert response.json()["rewritten_query"] == "seed rate?"
