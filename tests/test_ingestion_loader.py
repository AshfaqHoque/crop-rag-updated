from pathlib import Path

from app.ingestion.loader import _sanitize_metadata, load_chunks_file

FIXTURES = Path(__file__).parent / "fixtures"


def test_sanitize_metadata_drops_none_values():
    raw = {"crop_name": "Boro Paddy", "variety_id": None, "section": "seed"}
    clean = _sanitize_metadata(raw)
    assert clean == {"crop_name": "Boro Paddy", "section": "seed"}


def test_sanitize_metadata_stringifies_unsupported_types():
    raw = {"tags": ["a", "b"]}  # Chroma can't store lists
    clean = _sanitize_metadata(raw)
    assert clean == {"tags": "['a', 'b']"}


def test_sanitize_metadata_keeps_primitives():
    raw = {"crop_id": "5", "priority": 3, "score": 0.9, "verified": True}
    assert _sanitize_metadata(raw) == raw


def test_load_chunks_file_parses_sample_fixture():
    docs = load_chunks_file(FIXTURES / "sample_chunks.jsonl")
    assert len(docs) == 4

    seed_doc = next(d for d in docs if d.metadata["chunk_id"] == "5_seed")
    assert seed_doc.metadata["crop_name"] == "Boro Paddy"
    assert seed_doc.metadata["section"] == "seed"
    assert "বীজ" in seed_doc.page_content


def test_load_chunks_file_skips_malformed_lines(tmp_path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(
        '{"chunk_id": "ok_1", "text": "valid", "metadata": {"section": "seed"}}\n'
        "not valid json\n"
        '{"text": "missing chunk_id"}\n'
        '{"chunk_id": "ok_2", "text": "also valid", "metadata": {}}\n',
        encoding="utf-8",
    )
    docs = load_chunks_file(bad_file)
    assert [d.metadata["chunk_id"] for d in docs] == ["ok_1", "ok_2"]


def test_load_chunks_file_skips_blank_lines(tmp_path):
    f = tmp_path / "chunks.jsonl"
    f.write_text(
        '{"chunk_id": "a", "text": "x", "metadata": {}}\n\n\n'
        '{"chunk_id": "b", "text": "y", "metadata": {}}\n',
        encoding="utf-8",
    )
    docs = load_chunks_file(f)
    assert len(docs) == 2
