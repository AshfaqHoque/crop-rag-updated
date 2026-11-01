"""Build entity embeddings from crops.json into one Chroma collection.

    python -m app.ingestion.build_entity_index
    python -m app.ingestion.build_entity_index --reset
"""
import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)
COLLECTION = "sql_entities"


def load_crops(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]["getAllCropsFullDetails"]["rows"]


def extract_entities(crops: list[dict]) -> list[Document]:
    docs, seen = [], set()

    def add(etype: str, eid, *names):
        key = (etype, str(eid))
        if key in seen:
            return
        names = list(dict.fromkeys(n.strip() for n in names if n and n.strip()))
        if not names:
            return
        seen.add(key)
        primary = names[0]
        for i, name in enumerate(names):
            docs.append(Document(
                page_content=name,
                metadata={
                    "entity_type": etype,
                    "canonical_name": primary,
                    "entity_id": str(eid),
                    "chunk_id": f"{etype}_{eid}_{i}",
                },
            ))

    for crop in crops:
        add("crop", crop["id"], crop.get("crop_name"))

        cat = crop.get("crop_category") or {}
        if cat.get("id"):
            add("category", cat["id"], cat.get("category_name"))

        for v in crop.get("variety") or []:
            add("variety", v["id"], v.get("variety_name"))

        for p in crop.get("pesticide") or []:
            add("disease", p["id"], p.get("disease_name"))
            for c in p.get("chemical") or []:
                add("chemical", c["id"], c.get("trade_name"), c.get("generic_name"))
                if c.get("company_id"):
                    add("company", c["company_id"], c.get("company_name"))

        for h in crop.get("herbicide") or []:
            add("weed", h["id"], h.get("weed_name"))
            if h.get("company_id"):
                add("company", h["company_id"], h.get("company_name"))

    return docs


def main():
    configure_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(settings.crop_registry_path))
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    crops = load_crops(args.input)
    docs = extract_entities(crops)
    logger.info("extracted %d entity docs from %d crops", len(docs), len(crops))

    store = get_vector_store(COLLECTION)
    if args.reset:
        try:
            store.delete_collection()
            get_vector_store.cache_clear()
            store = get_vector_store(COLLECTION)
        except Exception:
            pass

    for i in range(0, len(docs), 64):
        batch = docs[i:i + 64]
        store.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
        logger.info("ingested %d/%d", min(i + 64, len(docs)), len(docs))


if __name__ == "__main__":
    main()