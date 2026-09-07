"""
Full ingestion entry point for company data: markdown -> chunks -> embedded
Chroma collection. Mirrors build_index.py's role for the crop pipeline --
this script IS the orchestrator (there's no separate loader.py step here,
since prepare_company_data.parse_all already returns in-memory Chunk
objects and there's no need to round-trip them through a file just to
read them back in).

It still writes the JSONL dump build_index.py writes, for the same reason:
inspecting chunk boundaries before trusting them into Chroma.

Run:
    python -m app.ingestion.load_company_data --input data/aunkur_company_info.md
    python -m app.ingestion.load_company_data --input data/aunkur_company_info.md --reset
"""
import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.ingestion.prepare_company_data import parse_all
from app.schemas.chunk_schema import Chunk
from app.services.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)

# Chroma metadata values must be str/int/float/bool -- no None, no nested dicts/lists.
_ALLOWED_METADATA_TYPES = (str, int, float, bool)


def _sanitize_metadata(metadata: dict) -> dict:
    clean = {}
    for key, value in metadata.items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, _ALLOWED_METADATA_TYPES) else str(value)
    return clean


def _chunks_to_documents(chunks: list[Chunk]) -> list[Document]:
    docs = []
    for c in chunks:
        metadata = _sanitize_metadata(c.metadata)
        metadata["chunk_id"] = c.chunk_id
        docs.append(Document(page_content=c.text, metadata=metadata))
    return docs


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    """Dumps chunks for human inspection / debugging chunk boundaries --
    same purpose as build_index.py's write_chunks_jsonl for the crop
    pipeline. Not read back in by anything; it's a debug artifact only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def reset_collection() -> None:
    """Wipes the existing company Chroma collection so the next ingest
    starts clean."""
    settings = get_settings()
    store = get_vector_store(settings.chroma_company_collection)
    try:
        store.delete_collection()
        logger.info("Deleted existing collection '%s'", settings.chroma_company_collection)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not delete existing collection '%s': %s", settings.chroma_company_collection, exc)
    # get_vector_store() is lru_cache'd; clear it so the next call recreates
    # the (now-deleted) collection instead of reusing the stale handle.
    get_vector_store.cache_clear()


def ingest_chunks(chunks: list[Chunk], batch_size: int = 64) -> int:
    """Embeds (via bge-m3) and upserts already-parsed company chunks into
    the dedicated company Chroma collection, in batches."""
    docs = _chunks_to_documents(chunks)
    if not docs:
        logger.warning("No valid company chunks to ingest")
        return 0

    settings = get_settings()
    store = get_vector_store(settings.chroma_company_collection)
    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        ids = [d.metadata["chunk_id"] for d in batch]
        store.add_documents(batch, ids=ids)  # upsert: same id overwrites the existing chunk
        total += len(batch)
        logger.info("Ingested %d/%d company chunks", total, len(docs))
    return total


def ingest(paths: list[Path], batch_size: int = 64) -> int:
    """Parses markdown files straight through to embedded Chroma upserts.
    Convenience wrapper for programmatic use; main() below does the same
    two steps explicitly so it can write the debug dump in between."""
    return ingest_chunks(parse_all(paths), batch_size=batch_size)


def main() -> None:
    configure_logging()
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Load company-info chunks (Markdown) into Chroma")
    parser.add_argument("--input", type=Path, nargs="+", default=[Path("data/aunkur_company_info.md")], help="Path(s) to company-info markdown file(s)")
    parser.add_argument("--chunks-output", type=Path, default=Path("data/company_chunks.jsonl"), help="Where to write the inspectable JSONL chunk dump (default: data/company_chunks.jsonl)")
    parser.add_argument("--reset", action="store_true", help="Wipe the existing company Chroma collection first")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logger.info("Parsing %s", args.input)
    chunks = parse_all(args.input)
    logger.info("Produced %d chunk(s)", len(chunks))

    write_chunks_jsonl(chunks, args.chunks_output)
    logger.info("Wrote inspectable dump to %s", args.chunks_output)

    if args.reset:
        logger.info("Resetting collection '%s'...", settings.chroma_company_collection)
        reset_collection()

    logger.info("Embedding and loading into Chroma (collection '%s')...", settings.chroma_company_collection)
    count = ingest_chunks(chunks, batch_size=args.batch_size)
    logger.info("Done. Ingested %d chunk(s) into collection '%s'.", count, settings.chroma_company_collection)


if __name__ == "__main__":
    main()