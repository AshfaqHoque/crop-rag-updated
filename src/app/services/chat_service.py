"""Application service that joins session history, LangGraph, and API schemas."""
import asyncio
from collections import defaultdict
from functools import lru_cache

from langchain_core.messages import HumanMessage
from starlette.concurrency import run_in_threadpool

from app.schemas.chat import ChatRequest, ChatResponse, SourceChunk
from app.services.pipeline.graph import get_chat_graph


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
            }
            result = await run_in_threadpool(self._graph.invoke, initial_state, config)
            answer = result.get("answer", "").strip()

            return self._to_response(request.session_id, result, answer)

    @staticmethod
    def _to_response(session_id: str, result: dict, answer: str) -> ChatResponse:
        chunks = (
            result["filtered_chunks"]
            if "filtered_chunks" in result
            else result.get("reranked_chunks") or result.get("retrieved_chunks") or []
        )
        sources = []
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            distance = chunk.get(
                "relevance_score",
                chunk.get("rerank_score", chunk.get("retrieval_score")),
            )
            sources.append(
                SourceChunk(
                    chunk_id=str(chunk.get("chunk_id", metadata.get("chunk_id", ""))),
                    crop_name=metadata.get("crop_name"),
                    section=metadata.get("section"),
                    distance=float(distance) if distance is not None else None,
                )
            )

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            language=result.get("language", "unknown"),
            rewritten_query=result.get("raw_query"),
            retrieval_mode=result.get("retrieval_mode"),
            sources=sources,
            messages=result.get("messages", []),
        )


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService()
