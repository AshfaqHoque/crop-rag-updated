"""
DESIGN DECISION (mirrors prepare_data.py -- read that file's docstring first)
==============================================================================
Company-info source docs are hand-authored Markdown, not JSON, so the
"correct chunk boundary" signal comes from the heading structure instead
of a schema. This module treats every "###" as one self-contained,
independently-answerable fact-unit -- exactly the same granularity
philosophy as one-chunk-per-variety / one-chunk-per-pest in prepare_data.py.

Structure assumed (see data/aunkur_company_info.md):

    # <Doc title>                      <- used only to derive company name
    ## <Parent section>                <- e.g. "Company Overview", "Revenue"
    ### <Subsection>                   <- ONE CHUNK
    <paragraph text>
    *Source pages: 1, 3*               <- optional, parsed into metadata
    *Retrieval note: ...*              <- optional, kept in chunk text as-is

Each chunk's text is self-identifying (company name + parent + subsection
header prepended), same reasoning as crop chunks: dense embeddings should
be able to place it correctly even without a metadata filter, but the
metadata filter (parent_section/section) is there for hard-filtering too.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from app.schemas.chunk_schema import Chunk

_H1_RE = re.compile(r"^#\s+(.*)$", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.*)$", re.MULTILINE)
_SOURCE_PAGES_RE = re.compile(r"\*Source pages:\s*([^*]+)\*")


def _slugify(title: str) -> str:
    slug = title.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def _company_name(full_text: str) -> str:
    """Pulls the company name from the H1, e.g.
    '# Aunkur - Section-wise Paragraph Resource' -> 'Aunkur'."""
    match = _H1_RE.search(full_text)
    if not match:
        return "Unknown"
    title = match.group(1).strip()
    return title.split(" - ")[0].strip()


def _extract_source_pages(body: str) -> str | None:
    match = _SOURCE_PAGES_RE.search(body)
    if not match:
        return None
    return match.group(1).strip()


@dataclass
class _Heading:
    level: int  # 2 or 3
    title: str
    start: int  # char offset right after the heading line


def _iter_headings(text: str) -> list[_Heading]:
    headings = []
    for m in _HEADING_RE.finditer(text):
        level = len(m.group(1))
        title = m.group(2).strip()
        headings.append(_Heading(level=level, title=title, start=m.end()))
    return headings


def parse_company_markdown(path: Path) -> list[Chunk]:
    """Entry point: turn one company-info markdown file into its full list
    of chunks -- one chunk per '###' subsection, tagged with its parent
    '##' section."""
    text = path.read_text(encoding="utf-8")
    company = _company_name(text)
    company_id = _slugify(company)

    headings = _iter_headings(text)
    chunks: list[Chunk] = []

    # Body of a heading runs from just after its own heading line to the
    # start of the *line* containing the next heading (level 2 or 3).
    def _line_start_at_or_before(offset: int) -> int:
        idx = text.rfind("\n", 0, offset) + 1
        return idx

    current_parent: str | None = None
    for i, h in enumerate(headings):
        body_start = h.start
        if i + 1 < len(headings):
            body_end = _line_start_at_or_before(headings[i + 1].start)
        else:
            body_end = len(text)
        body = text[body_start:body_end].strip()

        if h.level == 2:
            current_parent = h.title
            continue

        # h.level == 3 -> one chunk
        parent_title = current_parent or "General"
        source_pages = _extract_source_pages(body)

        header = f"Company: {company}\nSection: {parent_title} / {h.title}\n\n"
        chunk_text = header + body

        chunks.append(
            Chunk(
                chunk_id=f"{company_id}_{_slugify(parent_title)}_{_slugify(h.title)}",
                text=chunk_text,
                metadata={
                    "doc_type": "company",
                    "company_id": company_id,
                    "company_name": company,
                    "parent_section": parent_title,
                    "section": h.title,
                    **({"source_pages": source_pages} if source_pages else {}),
                },
            )
        )

    return chunks


def parse_all(paths: list[Path]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for p in paths:
        all_chunks.extend(parse_company_markdown(p))
    return all_chunks