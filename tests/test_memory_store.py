from app.memory.store import InMemoryConversationStore


def test_memory_store_appends_complete_turn():
    store = InMemoryConversationStore(max_turns=2)
    store.append_turn("s1", "hello", "hi")
    assert store.get_history("s1") == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_memory_store_keeps_only_configured_turns():
    store = InMemoryConversationStore(max_turns=2)
    for i in range(3):
        store.append_turn("s1", f"q{i}", f"a{i}")
    history = store.get_history("s1")
    assert [item["content"] for item in history] == ["q1", "a1", "q2", "a2"]


def test_memory_store_sessions_are_isolated():
    store = InMemoryConversationStore(max_turns=2)
    store.append_turn("a", "qa", "aa")
    store.append_turn("b", "qb", "ab")
    assert store.get_history("a")[0]["content"] == "qa"
    assert store.get_history("b")[0]["content"] == "qb"
