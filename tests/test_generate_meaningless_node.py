from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from app.services.pipeline.nodes.generate_meaningless import generate_meaningless


MODULE = "app.services.pipeline.nodes.generate_meaningless"


@patch(
    f"{MODULE}.astream_text",
    new_callable=AsyncMock,
    return_value="Please ask a clear agriculture question.",
)
async def test_generate_meaningless_requests_a_clear_question(mock_astream):
    state = {
        "raw_query": "asdf qwerty",
        "language_type": "english",
        "messages": [],
    }

    result = await generate_meaningless(state)

    assert result["answer"] == "Please ask a clear agriculture question."
    assert isinstance(result["messages"][0], AIMessage)
    sent_messages = mock_astream.await_args.args[0]
    assert sent_messages[-1].content == "asdf qwerty"
    assert "meaningful question" in sent_messages[0].content