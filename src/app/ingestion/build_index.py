"""
Full ingestion entry point: raw crop export -> chunks -> embedded Chroma
collection.

This is intentionally a thin orchestrator. It does not duplicate any
chunking, embedding, or storage logic -- it just calls the two stages that
already exist elsewhere in this package, in order:

    1. app.ingestion.prepare_data.chunk_all  -- crop JSON rows -> Chunk objects
    2. app.ingestion.loader.ingest           -- JSONL chunks -> embedded upserts into Chroma

(html_cleaner.py is used internally by prepare_data.py; you don't call it
directly here.)

Run:
    python -m app.ingestion.build_index                        # settings.crop_registry_path -> data/chunks.jsonl -> Chroma
    python -m app.ingestion.build_index --input data/other.json
    python -m app.ingestion.build_index --reset                # wipe the collection first
"""
import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.ingestion.loader import ingest
from app.ingestion.prepare_data import chunk_all
from app.schemas.chunk_schema import Chunk
from app.services.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)


def load_crop_rows(path: Path) -> list[dict]:
    """Parses the GraphQL-shaped crop export into a flat list of crop dicts.

    Expects the same shape as data/crops.json:
        {"data": {"getAllCropsFullDetails": {"rows": [ {...crop...}, ... ]}}}

    This mirrors app.services.pipeline.registry.get_known_crops, which reads
    the same file at query time to build the crop-name registry -- keep the
    two in sync if the export format ever changes.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level, got {type(raw).__name__}")

    rows = raw.get("data", {}).get("getAllCropsFullDetails", {}).get("rows")
    if rows is None:
        raise ValueError(f"{path}: missing data.getAllCropsFullDetails.rows")
    return rows


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    """Dumps chunks for human inspection / debugging chunk boundaries.

    This is also exactly the file format app.ingestion.loader.load_chunks_file
    expects to read back in, so the same file doubles as the ingest input.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def reset_collection() -> None:
    """Wipes the existing Chroma collection so the next ingest starts clean."""
    settings = get_settings()
    store = get_vector_store()
    try:
        store.delete_collection()
        logger.info("Deleted existing collection '%s'", settings.chroma_collection)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not delete existing collection '%s': %s", settings.chroma_collection, exc)
    # get_vector_store() is lru_cache'd; clear it so the next call recreates
    # the (now-deleted) collection instead of reusing the stale handle.
    get_vector_store.cache_clear()


def main() -> None:
    configure_logging()
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Chunk raw crop data and load it into the Chroma vector store.")
    parser.add_argument("--input", type=Path, default=Path(settings.crop_registry_path), help="Path to the crop JSON export (default: settings.crop_registry_path)")
    parser.add_argument("--chunks-output", type=Path, default=Path("data/chunks.jsonl"), help="Where to write the inspectable JSONL chunk dump (default: data/chunks.jsonl)")
    parser.add_argument("--reset", action="store_true", help="Wipe the existing Chroma collection first")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logger.info("Loading crop rows from %s", args.input)
    crops = load_crop_rows(args.input)
    logger.info("Loaded %d crop(s)", len(crops))

    logger.info("Chunking...")
    chunks = chunk_all(crops)
    logger.info("Produced %d chunk(s)", len(chunks))

    write_chunks_jsonl(chunks, args.chunks_output)
    logger.info("Wrote inspectable dump to %s", args.chunks_output)

    if args.reset:
        logger.info("Resetting collection '%s'...", settings.chroma_collection)
        reset_collection()

    logger.info("Embedding and loading into Chroma (collection '%s')...", settings.chroma_collection)
    count = ingest(args.chunks_output, batch_size=args.batch_size)
    logger.info("Done. Ingested %d chunk(s) into collection '%s'.", count, settings.chroma_collection)


if __name__ == "__main__":
    main()
