"""Thread-safe in-process conversation history store.

Use Redis or a database-backed implementation for multi-replica deployments.
The interface is intentionally small so that replacement is straightforward.
"""
from collections import defaultdict, deque
from functools import lru_cache
from threading import RLock

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from app.core.config import get_settings


class InMemoryConversationStore:
    def __init__(self, max_turns: int = 1):
        # One turn = one HumanMessage + one AIMessage
        self._max_messages = max_turns * 2
        self._sessions: dict[str, deque[BaseMessage]] = defaultdict(
            lambda: deque(maxlen=self._max_messages)
        )
        self._lock = RLock()

    def get_history(self, session_id: str) -> list[BaseMessage]:
        """Return conversation history as LangChain messages."""
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def append_turn(self, session_id: str, user_message: str, assistant_message: str, rewritten_query: str) -> None:
        with self._lock:
            history = self._sessions[session_id]
            history.append( HumanMessage(content=user_message, additional_kwargs={"rewritten_query": rewritten_query,}))
            history.append(AIMessage(content=assistant_message))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


@lru_cache
def get_conversation_store() -> InMemoryConversationStore:
    return InMemoryConversationStore(max_turns=get_settings().history_max_turns)
