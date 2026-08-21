"""
Usage:
    python -m scripts.build_index                     # reads every *.json in data/raw/
    python -m scripts.build_index --input data/raw/rice.json
    python -m scripts.build_index --reset              # wipe and rebuild Chroma collection

Run this once after dropping your full crop export(s) into data/raw/,
and again any time the source data or EMBED_MODEL changes.
"""
import argparse
import json

import config
from chunker import chunk_all
from parser import load_crops, load_crops_from_dir
from ollama_embedder import OllamaEmbedder
from bm25_retriever import BM25Retriever
from crop_matcher import build_crop_index
from chroma_store import ChromaStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Single JSON file. Defaults to all files in data/raw/.")
    parser.add_argument("--reset", action="store_true", help="Wipe the existing Chroma collection first.")
    args = parser.parse_args()

    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw crop data...")
    if args.input:
        crops = load_crops(args.input)
    else:
        crops = load_crops_from_dir(config.DATA_RAW_DIR)
    print(f"  loaded {len(crops)} crop(s)")

    print("Chunking...")
    chunks = chunk_all(crops)
    print(f"  produced {len(chunks)} chunk(s)")

    # Dump for human inspection / debugging chunk boundaries.
    with open(config.CHUNKS_JSONL_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    print(f"  wrote inspectable dump to {config.CHUNKS_JSONL_PATH}")

    print(f"Embedding {len(chunks)} chunks via Ollama ({config.EMBED_MODEL})...")
    embedder = OllamaEmbedder()
    embeddings = embedder.embed_batch([c.text for c in chunks])

    print("Writing to Chroma...")
    store = ChromaStore()
    if args.reset:
        store.reset()
    store.add(chunks, embeddings)
    print(f"  collection now has {store.count()} vectors")

    print("Building BM25 index...")
    bm25 = BM25Retriever.build(chunks)
    bm25.save()

    print("Building crop-name index...")
    crop_index = build_crop_index(crops)
    with open(config.CROP_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(crop_index, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()