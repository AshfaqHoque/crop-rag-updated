from langchain_core.documents import Document

from app.services.retrieval.hybrid import SemanticRetriever


def _doc(chunk_id: str, text: str, section: str = "seed") -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "crop_name": "Boro Paddy", "section": section},
    )


def test_crop_detected_uses_filtered_dense_only():
    calls = {}

    def dense(query, **kwargs):
        calls["dense"] = kwargs
        return [(_doc("d1", "seed rate"), 0.12)]

    def loader(**kwargs):
        raise AssertionError("BM25 corpus should not be loaded when crop is detected")

    documents, mode = SemanticRetriever(dense).retrieve(
        "boro seed rate", crops=["Boro Paddy"], sections=["seed"], top_k=3
    )
    assert mode == "dense_filtered"
    assert documents[0].metadata["chunk_id"] == "d1"
    assert calls["dense"]["crops"] == ["Boro Paddy"]
    assert calls["dense"]["sections"] == ["seed"]


def test_dense_retrieval_preserves_document_metadata_and_score():
    calls = {}
    dense_doc = _doc("dense", "general cultivation advice")
    lexical_doc = _doc("lexical", "seed rate is 120 kg")

    def dense(query, **kwargs):
        calls["dense"] = kwargs
        return [(dense_doc, 0.2), (lexical_doc, 0.3)]

    documents, mode = SemanticRetriever(dense).retrieve(
        "seed rate 120", crops=[], sections=["seed"], top_k=5
    )
    assert mode == "dense_filtered"
    assert {document.metadata["chunk_id"] for document in documents} == {"dense", "lexical"}
    assert calls["dense"]["sections"] == ["seed"]
    lexical = next(document for document in documents if document.metadata["chunk_id"] == "lexical")
    assert lexical.metadata["distance"] == 0.3


def test_retriever_can_target_company_collection():
    calls = {}

    def dense(query, **kwargs):
        calls["dense"] = kwargs
        return [(_doc("company", "company information"), 0.1)]

    documents, mode = SemanticRetriever(
        dense,
        collection_name="company_knowledge_base",
    ).retrieve("what does Aunkur do?", top_k=3)

    assert mode == "dense_filtered"
    assert len(documents) == 1
    assert calls["dense"]["collection_name"] == "company_knowledge_base"
    assert calls["dense"]["k"] == 3
