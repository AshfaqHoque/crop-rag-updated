"""
DESIGN DECISION (mirrors prepare_company_data.py -- read that file's
docstring first, and prepare_data.py before that)
==============================================================================
The soil-test FAQ sources are CSV exports from spreadsheets, not Markdown
and not the crop JSON. Here the "correct chunk boundary" is simply one
row = one Q&A pair = one chunk -- there's no heading structure to walk,
so this is the simplest of the three prepare_*.py modules.

There is now more than one soil-test FAQ CSV feeding the same collection
(Porokh device FAQ, general Soil Test FAQ, ...), and they don't agree on
header text -- e.g. the answer column is named " উত্তর" in one file and
"এক লাইনের উত্তর" in another. They DO agree on column order, so columns
are read positionally (0=id, 1=category, 2=question, 3=answer) instead of
by header name, which makes this robust to header-wording differences
across source files.

Expected file shape (see data/porokh_faq.csv, data/soil_test_faq.csv):

    Porokh FAQ,,,                              <- title row, discarded
    ,,,                                        <- blank row, discarded
    ক্রমিক,ক্যাটাগরি,প্রশ্ন, উত্তর              <- header row (line 3), discarded
    1,Porokh ,এই ডিভাইসের নাম কী?,"..."         <- data rows, read by position

So unlike a plain CSV, the real data starts on line 4 -- the first three
lines (title, blank, header) are skipped unconditionally rather than
parsed, since the header text isn't trustworthy across files anyway.
Cell values are still whitespace-stripped on the way in (e.g. "Porokh "
category values have a trailing space).

Each chunk's text is self-identifying (category + question prepended
before the answer), same reasoning as the crop and company chunks: dense
embeddings should be able to place it correctly even without a metadata
filter, but the metadata filter (category) is there for hard-filtering
too. Rows with an empty answer carry no answerable content and are
dropped rather than embedded as an empty chunk.

chunk_id is namespaced with the source file's stem (e.g.
"soiltest_porokh_faq_12") rather than just the row id, since two
different source files can both have a row 12 and both feed this same
collection -- without the file stem, the second file's rows would
silently overwrite the first's on upsert.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from app.schemas.chunk_schema import Chunk

# Header lives on line 3 (1-indexed); real data starts on line 4. The two
# lines before the header (title row, blank row) are spreadsheet-export
# noise, not data -- and the header itself is skipped too since its exact
# wording isn't consistent across source files (see module docstring).
_DATA_START_LINE = 4

_COL_ID, _COL_CATEGORY, _COL_QUESTION, _COL_ANSWER = 0, 1, 2, 3


def _slugify(title: str) -> str:
    slug = title.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


@dataclass
class _Row:
    row_id: str
    category: str
    question: str
    answer: str


def _read_rows(path: Path) -> list[_Row]:
    """Skips the title/blank/header lines, then reads the remaining rows
    positionally (id, category, question, answer) regardless of what the
    header text says."""
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for _ in range(_DATA_START_LINE - 1):
            next(f)
        rows = []
        for cells in csv.reader(f):
            if not any(c.strip() for c in cells):
                continue  # skip stray blank rows
            cells = [c.strip() for c in cells] + [""] * 4  # pad short rows
            rows.append(
                _Row(
                    row_id=cells[_COL_ID],
                    category=cells[_COL_CATEGORY],
                    question=cells[_COL_QUESTION],
                    answer=cells[_COL_ANSWER],
                )
            )
        return rows


def parse_soil_test_csv(path: Path) -> list[Chunk]:
    """Entry point: turn one soil-test FAQ CSV into its full list of
    chunks -- one chunk per row, skipping rows with no answer."""
    rows = _read_rows(path)
    source = path.stem
    chunks: list[Chunk] = []
    skipped_empty = 0
    fallback_counter = 0

    for row in rows:
        if not row.answer:
            skipped_empty += 1
            continue

        fallback_counter += 1
        row_id = row.row_id or str(fallback_counter)
        category = row.category or "General"

        header = f"Category: {category}\nQuestion: {row.question}\n\n"
        chunk_text = header + f"Answer: {row.answer}"

        chunks.append(
            Chunk(
                chunk_id=f"soiltest_{_slugify(source)}_{row_id}",
                text=chunk_text,
                metadata={
                    "doc_type": "soil_test_faq",
                    "source_file": source,
                    "category": category,
                    "question": row.question,
                    "source_row_id": row_id,
                },
            )
        )

    if skipped_empty:
        print(f"[prepare_soil_test_data] {source}: skipped {skipped_empty} row(s) with empty answers")

    return chunks


def parse_all(paths: list[Path]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for p in paths:
        all_chunks.extend(parse_soil_test_csv(p))
    return all_chunks