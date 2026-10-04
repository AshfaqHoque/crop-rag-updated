"""Service layer that orchestrates LangGraph chat requests and API response formatting."""

import asyncio
from collections import defaultdict
from functools import lru_cache
import json

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.schemas.chat import ChatRequest, ChatResponse, SourceChunk
from app.services.pipeline.graph import get_chat_graph
from app.services.thinking_messages import get_thinking_message

_TERMINAL_GENERATE_NODES = {"generate", "generate_company", "generate_chitchat", "generate_meaningless", "generate_soil_test", "handle_agronomist_request", "generate_capability"}

class ChatService:
    def __init__(self, graph=None) -> None:
        self._graph = graph or get_chat_graph()
        self._session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        # Serializing each session prevents two simultaneous follow-ups from reading
        # the same stale history and being persisted out of order.
        async with self._session_locks[request.session_id]:
            config = {
                "configurable": {"thread_id": request.session_id},
                "run_name": "crop_rag_chat",
                "tags": [f"session:{request.session_id}"],
                "metadata": {"session_id": request.session_id},
            }
            initial_state = {
                "messages": [HumanMessage(content=request.message.strip())],
                "session_id": request.session_id,
                "raw_query": request.message.strip(),
                "language_type": request.language_type,
            }
            result = await run_in_threadpool(self._graph.invoke, initial_state, config)
            answer = result.get("answer", "").strip()

            return self._to_response(request.session_id, result, answer)

    async def stream_chat(self, request: ChatRequest):
        """Yields SSE-formatted chunks of the final answer as it's generated."""
        async with self._session_locks[request.session_id]:
            config = {
                "configurable": {"thread_id": request.session_id},
                "run_name": "crop_rag_chat",
                "tags": [f"session:{request.session_id}"],
                "metadata": {"session_id": request.session_id},
            }
            initial_state = {
                "messages": [HumanMessage(content=request.message.strip())],
                "session_id": request.session_id,
                "raw_query": request.message.strip(),
                "language_type": request.language_type,
            }
            
            thinking_message = get_thinking_message(request.language_type)
            yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_message}, ensure_ascii=False)}\n\n"

            active_message_id = None
            answer = ""
            async for msg_chunk, metadata in self._graph.astream(initial_state, config, stream_mode="messages"):
                # Only forward tokens from the actual answer-generating nodes —
                # not rewrite_query/route/extract_crop, which also call the LLM.
                if metadata.get("langgraph_node") not in _TERMINAL_GENERATE_NODES:
                    continue
                if active_message_id is None:
                    active_message_id = msg_chunk.id
                elif msg_chunk.id != active_message_id:
                    continue  # second LLM run (e.g. an invoke_text retry) — skip it
                if msg_chunk.content:
                    answer += msg_chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'content': msg_chunk.content}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(get_settings().stream_chunk_delay_seconds)
            yield f"data: {json.dumps({'type': 'done', 'answer': answer}, ensure_ascii=False)}\n\n"


    @staticmethod
    def _to_response(session_id: str, result: dict, answer: str) -> ChatResponse:
        documents: list[Document] = (
            result.get("compressed_documents")
            or result.get("reranked_documents")
            or result.get("retrieved_documents")
            or []
        )
        sources = []
        for document in documents:
            metadata = document.metadata
            distance = metadata.get("relevance_score", metadata.get("distance"))
            sources.append(
                SourceChunk(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    crop_name=metadata.get("crop_name"),
                    section=metadata.get("section"),
                    distance=float(distance) if distance is not None else None,
                )
            )

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            language=result.get("language_type", "unknown"),
            status=result.get("status"),
            rewritten_query=result.get("rewritten_query"),
            sources=sources,
            messages=result.get("messages", []),
        )


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService()
