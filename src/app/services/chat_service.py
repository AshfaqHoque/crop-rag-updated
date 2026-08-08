"""Application service that joins session history, LangGraph, and API schemas."""
import asyncio
from collections import defaultdict
from functools import lru_cache

from starlette.concurrency import run_in_threadpool

from app.memory.store import InMemoryConversationStore, get_conversation_store
from app.schemas.chat import ChatRequest, ChatResponse, SourceChunk
from app.services.pipeline.graph import get_chat_graph


class ChatService:
    def __init__(self, history_store: InMemoryConversationStore | None = None, graph=None) -> None:
        self._history_store = history_store or get_conversation_store()
        self._graph = graph or get_chat_graph()
        self._session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        # Serializing each session prevents two simultaneous follow-ups from reading
        # the same stale history and being persisted out of order.
        async with self._session_locks[request.session_id]:
            history = self._history_store.get_history(request.session_id)
            initial_state = {
                "session_id": request.session_id,
                "raw_query": request.message.strip(),
                "history": history,
            }
            result = await run_in_threadpool(self._graph.invoke, initial_state)
            answer = result.get("answer", "").strip()            
            self._history_store.append_turn(request.session_id, request.message.strip(), answer,)  # noqa: E501

            return self._to_response(request.session_id, result, answer)

    @staticmethod
    def _to_response(session_id: str, result: dict, answer: str) -> ChatResponse:
        ranked = result.get("reranked_chunks") or result.get("retrieved_chunks") or []
        sources = []
        for chunk in ranked:
            metadata = chunk.get("metadata") or {}
            score = chunk.get("rerank_score", chunk.get("retrieval_score"))
            sources.append(
                SourceChunk(
                    chunk_id=str(chunk.get("chunk_id", metadata.get("chunk_id", ""))),
                    crop_name=metadata.get("crop_name"),
                    section=metadata.get("section"),
                    score=float(score) if score is not None else None,
                )
            )

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            language=result.get("language", "unknown"),
            rewritten_query=result.get("rewritten_query"),
            retrieval_mode=result.get("retrieval_mode"),
            sources=sources,
        )


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService()
