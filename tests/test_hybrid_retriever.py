from langchain_core.documents import Document

from app.services.retrieval.hybrid import SemanticRetriever, tokenize


def _doc(chunk_id: str, text: str, section: str = "seed") -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "crop_name": "Boro Paddy", "section": section},
    )


def test_retrieval_combines_filtered_dense_and_bm25_candidates():
    calls = {}
    dense_doc = _doc("dense", "seed rate")
    lexical_doc = _doc("lexical", "seed rate is 120 kg")
    duplicate_doc = _doc("dense", "seed rate duplicate")

    def dense(query, **kwargs):
        calls["dense"] = kwargs
        return [(dense_doc, 0.12)]

    def loader(**kwargs):
        calls["loader"] = kwargs
        return [
            lexical_doc,
            duplicate_doc,
            _doc("unrelated-1", "pest management"),
            _doc("unrelated-2", "irrigation schedule"),
            _doc("unrelated-3", "soil preparation"),
        ]

    documents, mode = SemanticRetriever(dense, loader).retrieve(
        "seed rate", crops=["Boro Paddy"], sections=["seed"], top_k=4
    )
    assert mode == "hybrid_filtered"
    chunk_ids = [document.metadata["chunk_id"] for document in documents]
    assert chunk_ids[:2] == ["dense", "lexical"]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert len(documents) <= 8
    assert calls["dense"]["crops"] == ["Boro Paddy"]
    assert calls["dense"]["sections"] == ["seed"]
    assert calls["loader"]["crops"] == ["Boro Paddy"]
    assert calls["loader"]["sections"] == ["seed"]
    assert calls["dense"]["k"] == 4
    assert documents[0].metadata["distance"] == 0.12


def test_dense_retrieval_preserves_document_metadata_and_score():
    calls = {}
    dense_doc = _doc("dense", "general cultivation advice")
    lexical_doc = _doc("lexical", "seed rate is 120 kg")

    def dense(query, **kwargs):
        calls["dense"] = kwargs
        return [(dense_doc, 0.2), (lexical_doc, 0.3)]

    documents, mode = SemanticRetriever(dense, lambda **kwargs: []).retrieve(
        "seed rate 120", crops=[], sections=["seed"], top_k=5
    )
    assert mode == "hybrid_filtered"
    assert {document.metadata["chunk_id"] for document in documents} == {"dense", "lexical"}
    assert calls["dense"]["sections"] == ["seed"]
    assert len(documents) == 2
    lexical = next(document for document in documents if document.metadata["chunk_id"] == "lexical")
    assert lexical.metadata["distance"] == 0.3


def test_retriever_can_target_company_collection():
    calls = {}

    def dense(query, **kwargs):
        calls["dense"] = kwargs
        return [(_doc("company", "company information"), 0.1)]

    documents, mode = SemanticRetriever(
        dense,
        lambda **kwargs: [],
        collection_name="company_knowledge_base",
    ).retrieve("what does Aunkur do?", top_k=3)

    assert mode == "hybrid_filtered"
    assert len(documents) == 1
    assert calls["dense"]["collection_name"] == "company_knowledge_base"
    assert calls["dense"]["k"] == 3


def test_bangla_tokenizer_normalizes_digits_and_zero_width_characters():
    assert tokenize("হীরা ২\u200c জাত") == ["হীরা", "2", "জাত"]
