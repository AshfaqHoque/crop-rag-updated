"""
Loads pre-built crop chunks into Chroma.

Expects a JSONL file — one JSON object per line — matching the format
already used to build your existing collection:

    {"chunk_id": "5_seed", "text": "...",
     "metadata": {"crop_id": "5", "crop_name": "Boro Paddy",
                  "section": "seed", ...}}

Run:
    python -m app.ingestion.loader --input data/chunks.jsonl
"""
import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)

# Chroma metadata values must be str/int/float/bool — no None, no nested dicts/lists.
_ALLOWED_METADATA_TYPES = (str, int, float, bool)


def _sanitize_metadata(metadata: dict) -> dict:
    clean = {}
    for key, value in metadata.items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, _ALLOWED_METADATA_TYPES) else str(value)
    return clean


def load_chunks_file(path: Path) -> list[Document]:
    """Parses the JSONL chunk file into LangChain Documents. Pure parsing, no I/O to Chroma."""
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d in %s: %s", line_num, path, exc)
                continue

            if "chunk_id" not in obj or "text" not in obj:
                logger.warning("Skipping line %d: missing chunk_id or text", line_num)
                continue

            metadata = _sanitize_metadata(obj.get("metadata", {}))
            metadata["chunk_id"] = obj["chunk_id"]
            docs.append(Document(page_content=obj["text"], metadata=metadata))
    return docs


def ingest(path: Path, batch_size: int = 64) -> int:
    """Loads, embeds (via bge-m3), and upserts chunks into Chroma in batches."""
    docs = load_chunks_file(path)
    if not docs:
        logger.warning("No valid documents found in %s", path)
        return 0

    store = get_vector_store()
    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        ids = [d.metadata["chunk_id"] for d in batch]
        store.add_documents(batch, ids=ids)  # upsert: same id overwrites the existing chunk
        total += len(batch)
        logger.info("Ingested %d/%d chunks", total, len(docs))
    return total


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Load crop chunks (JSONL) into Chroma")
    parser.add_argument("--input", required=True, type=Path, help="Path to JSONL chunk file")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    count = ingest(args.input, batch_size=args.batch_size)
    settings = get_settings()
    logger.info("Done. Ingested %d chunks into collection '%s'", count, settings.chroma_collection)


if __name__ == "__main__":
    main()
