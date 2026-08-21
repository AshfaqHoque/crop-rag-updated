"""
The source data stores almost every descriptive field (general_info,
special_character, damage_control, control_measure, ...) as raw HTML
(<p>, <strong>, <ul>, &nbsp;, stray <img> tags, etc).

Embedding models were not trained on HTML soup -- tags waste tokens and
can distort similarity. We strip everything down to clean, readable text
before it ever becomes a chunk.
"""
import re
from bs4 import BeautifulSoup


def clean_html(raw: str | None) -> str:
    if not raw:
        return ""

    soup = BeautifulSoup(raw, "html.parser")

    # Drop non-content elements outright (images just leave a URL/alt text
    # that isn't useful for a text retriever).
    for tag in soup(["img", "script", "style"]):
        tag.decompose()

    # Turn list items into "- item" lines and block elements into
    # newlines so structure survives as plain text.
    for li in soup.find_all("li"):
        li.insert_before("\n- ")
    for block in soup.find_all(["p", "div", "br", "h1", "h2", "h3", "h4"]):
        block.insert_after("\n")

    text = soup.get_text()

    # Collapse the &nbsp;-driven whitespace mess and repeated blank lines.
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = text.strip()
    return text


def is_meaningful(text: str, min_len: int = 3) -> bool:
    """Guards against emitting chunks that are empty or just punctuation."""
    return bool(text) and len(text.strip()) >= min_len