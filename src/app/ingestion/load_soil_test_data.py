"""
Full ingestion entry point for soil-test (Porokh) FAQ data: CSV -> chunks
-> embedded Chroma collection. Simplified version of load_company_data.py's
shape -- same orchestrator role, but the metadata-sanitizing / doc-building
/ JSONL-dumping steps are folded into main() instead of being split into
their own functions, since none of them are reused anywhere else.

Run (single file):
    python -m app.ingestion.load_soil_test_data --input data/soil_test_faq.csv

Run (default files):
    python -m app.ingestion.load_soil_test_data --reset

Run (multiple FAQ CSVs into the same collection -- e.g. Porokh device FAQ
+ general Soil Test FAQ; chunk ids are namespaced per source filename in
prepare_soil_test_data.py, so rows with the same row-number from different
files won't collide on upsert):
    python -m app.ingestion.load_soil_test_data --input data/porokh_faq.csv data/soil_test_faq.csv --reset
"""
import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.ingestion.prepare_soil_test_data import parse_all
from app.schemas.chunk_schema import Chunk
from app.services.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)

# Chroma metadata values must be str/int/float/bool -- no None, no nested dicts/lists.
_ALLOWED_METADATA_TYPES = (str, int, float, bool)


def reset_collection() -> None:
    """Wipes the existing soil-test Chroma collection so the next ingest
    starts clean."""
    settings = get_settings()
    store = get_vector_store(settings.chroma_soil_test_collection)
    try:
        store.delete_collection()
        logger.info("Deleted existing collection '%s'", settings.chroma_soil_test_collection)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not delete existing collection '%s': %s", settings.chroma_soil_test_collection, exc)
    # get_vector_store() is lru_cache'd; clear it so the next call recreates
    # the (now-deleted) collection instead of reusing the stale handle.
    get_vector_store.cache_clear()


def ingest_chunks(chunks: list[Chunk], batch_size: int = 64) -> int:
    """Sanitizes metadata, embeds (via bge-m3), and upserts already-parsed
    soil-test chunks into the dedicated soil-test Chroma collection, in
    batches. Metadata sanitizing and Document-building live here inline
    since nothing else needs them."""
    docs = []
    for c in chunks:
        metadata = {
            k: v if isinstance(v, _ALLOWED_METADATA_TYPES) else str(v)
            for k, v in c.metadata.items()
            if v is not None
        }
        metadata["chunk_id"] = c.chunk_id
        docs.append(Document(page_content=c.text, metadata=metadata))

    if not docs:
        logger.warning("No valid soil-test chunks to ingest")
        return 0

    settings = get_settings()
    store = get_vector_store(settings.chroma_soil_test_collection)
    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        ids = [d.metadata["chunk_id"] for d in batch]
        store.add_documents(batch, ids=ids)  # upsert: same id overwrites the existing chunk
        total += len(batch)
        logger.info("Ingested %d/%d soil-test chunks", total, len(docs))
    return total


def main() -> None:
    configure_logging()
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Load soil-test FAQ chunks (CSV) into Chroma")
    parser.add_argument("--input", type=Path, nargs="+", default=[Path("data/porokh_faq.csv"), Path("data/soil_test_faq.csv")], help="Path(s) to soil-test FAQ CSV file(s)",)
    parser.add_argument("--chunks-output", type=Path, default=Path("data/soil_test_chunks.jsonl"), help="Where to write the inspectable JSONL chunk dump (default: data/soil_test_chunks.jsonl)")
    parser.add_argument("--reset", action="store_true", help="Wipe the existing soil-test Chroma collection first")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logger.info("Parsing %s", args.input)
    chunks = parse_all(args.input)
    logger.info("Produced %d chunk(s)", len(chunks))

    # Dump for human inspection / debugging chunk boundaries before trusting
    # them into Chroma -- not read back in by anything, debug artifact only.
    args.chunks_output.parent.mkdir(parents=True, exist_ok=True)
    with args.chunks_output.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Wrote inspectable dump to %s", args.chunks_output)

    if args.reset:
        logger.info("Resetting collection '%s'...", settings.chroma_soil_test_collection)
        reset_collection()

    logger.info("Embedding and loading into Chroma (collection '%s')...", settings.chroma_soil_test_collection)
    count = ingest_chunks(chunks, batch_size=args.batch_size)
    logger.info("Done. Ingested %d chunk(s) into collection '%s'.", count, settings.chroma_soil_test_collection)


if __name__ == "__main__":
    main()