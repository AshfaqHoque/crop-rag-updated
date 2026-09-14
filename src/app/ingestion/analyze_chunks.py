"""Analyze text sizes in a JSONL chunk file.

Run from the repository root:

    python -m app.ingestion.analyze_chunks --input data/chunks.jsonl
"""

import argparse
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def analyze_chunks(
    path: Path, top_n: int = 10
) -> tuple[int, int, int, float, list[tuple[int, str]], list[tuple[int, str]]]:
    """Return count, size statistics, and largest/smallest chunks."""
    sizes: list[tuple[int, str]] = []

    with path.open("r", encoding="utf-8") as chunks_file:
        for line_number, raw_line in enumerate(chunks_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", line_number, exc)
                continue

            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            if not isinstance(chunk_id, str) or not isinstance(text, str):
                logger.warning("Skipping line %d: missing string chunk_id or text", line_number)
                continue

            sizes.append((len(text), chunk_id))

    if not sizes:
        return 0, 0, 0, 0.0, [], []

    largest = sorted(sizes, key=lambda item: (-item[0], item[1]))[:top_n]
    smallest = sorted(sizes, key=lambda item: (item[0], item[1]))[:top_n]
    lengths = [size for size, _ in sizes]
    return len(sizes), min(lengths), max(lengths), sum(lengths) / len(lengths), largest, smallest


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze chunk sizes in a JSONL file")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/chunks.jsonl"),
        help="Path to the JSONL chunk file (default: data/chunks.jsonl)",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of largest chunks to show")
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be at least 1")

    total, minimum, maximum, average, largest, smallest = analyze_chunks(args.input, top_n=args.top)

    print(f"File: {args.input}")
    print("Size unit: Unicode characters in the text field")
    print(f"Total chunks: {total}")
    print(f"Minimum chunk size: {minimum}")
    print(f"Maximum chunk size: {maximum}")
    print(f"Average chunk size: {average:.2f}")
    print(f"Top {args.top} chunk sizes:")
    for rank, (size, chunk_id) in enumerate(largest, start=1):
        print(f"  {rank:>2}. {size:>6} characters  {chunk_id}")
    print(f"Smallest {args.top} chunk sizes:")
    for rank, (size, chunk_id) in enumerate(smallest, start=1):
        print(f"  {rank:>2}. {size:>6} characters  {chunk_id}")


if __name__ == "__main__":
    main()