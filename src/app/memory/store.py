"""Thread-safe in-process conversation history store.

Use Redis or a database-backed implementation for multi-replica deployments.
The interface is intentionally small so that replacement is straightforward.
"""
from collections import defaultdict, deque
from functools import lru_cache
from threading import RLock

from app.core.config import get_settings


class InMemoryConversationStore:
    def __init__(self, max_turns: int = 6):
        self._max_messages = max_turns * 2
        self._sessions: dict[str, deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=self._max_messages)
        )
        self._lock = RLock()

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return [dict(message) for message in self._sessions.get(session_id, ())]

    def append_turn(self, session_id: str, user_message: str, assistant_message: str, *, standalone_query: str | None = None,) -> None:
        canonical_query = (standalone_query or user_message).strip()
        with self._lock:
            history = self._sessions[session_id]
            history.append({"role": "user", "content": user_message, "canonical_query": canonical_query})
            history.append({"role": "assistant", "content": assistant_message})

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


@lru_cache
def get_conversation_store() -> InMemoryConversationStore:
    return InMemoryConversationStore(max_turns=get_settings().history_max_turns)
