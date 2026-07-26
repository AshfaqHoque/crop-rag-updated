from langchain_core.documents import Document

from app.services.retrieval.hybrid import HybridRetriever, tokenize


def _doc(chunk_id: str, text: str, section: str = "seed") -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "crop_name": "Boro Paddy", "section": section},
    )


def test_tokenize_supports_bangla_and_english():
    tokens = tokenize("বোরো ধান Seed-rate 120 kg")
    assert "বোরো" in tokens
    assert "ধান" in tokens
    assert "seed" in tokens
    assert "120" in tokens


def test_crop_detected_uses_filtered_dense_only():
    calls = {}

    def dense(query, **kwargs):
        calls["dense"] = kwargs
        return [(_doc("d1", "seed rate"), 0.12)]

    def loader(**kwargs):
        raise AssertionError("BM25 corpus should not be loaded when crop is detected")

    chunks, mode = HybridRetriever(dense, loader).retrieve(
        "boro seed rate", crops=["Boro Paddy"], sections=["seed"], top_k=3
    )
    assert mode == "dense_filtered"
    assert chunks[0]["chunk_id"] == "d1"
    assert calls["dense"]["crops"] == ["Boro Paddy"]
    assert calls["dense"]["sections"] == ["seed"]


def test_no_crop_fuses_dense_and_bm25_with_same_filter():
    calls = {}
    dense_doc = _doc("dense", "general cultivation advice")
    lexical_doc = _doc("lexical", "seed rate is 120 kg")

    def dense(query, **kwargs):
        calls["dense"] = kwargs
        return [(dense_doc, 0.2), (lexical_doc, 0.3)]

    def loader(**kwargs):
        calls["loader"] = kwargs
        return [dense_doc, lexical_doc]

    chunks, mode = HybridRetriever(dense, loader).retrieve(
        "seed rate 120", crops=[], sections=["seed"], top_k=5
    )
    assert mode == "dense_bm25_rrf"
    assert {item["chunk_id"] for item in chunks} == {"dense", "lexical"}
    assert calls["dense"]["sections"] == ["seed"]
    assert calls["loader"]["sections"] == ["seed"]
    lexical = next(item for item in chunks if item["chunk_id"] == "lexical")
    assert "bm25_score" in lexical
    assert "dense_distance" in lexical


def test_bm25_corpus_is_cached(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "bm25_cache_ttl_seconds", 300)
    load_count = 0
    document = _doc("cached", "seed rate 120 kg")

    def dense(query, **kwargs):
        return []

    def loader(**kwargs):
        nonlocal load_count
        load_count += 1
        return [document]

    retriever = HybridRetriever(dense, loader)
    retriever.retrieve("seed rate", crops=[], sections=["seed"])
    retriever.retrieve("120 kg", crops=[], sections=["seed"])
    assert load_count == 1
