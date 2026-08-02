"""Deterministic crop extraction from the canonical crop registry.

Scope: matches against crop.crop_name (English) and crop.crop_bangla_name
(Bangla script) only. Romanized/Banglish queries (e.g. "gomer" typed in Latin
script for what is written "গমের" in Bangla) are NOT handled -- that would
require a transliteration layer, which is intentionally out of scope for now.

Matching strategy
------------------
1. Normalize text (NFKC, casefold, punctuation -> space, preserve Bangla
   vowel signs / combining marks).
2. Tokenize on whitespace.
3. Compare each *whole token* (or, for multi-word aliases, a contiguous run
   of whole tokens) against known crop aliases -- never a raw substring
   check. This is what stops "আম" (mango) from matching inside "আমি" or
   "আমন" (Aman rice).
4. Bangla attaches case/plural suffixes directly onto the stem with no space
   (e.g. "গমের" = "গম" + "ের"), so the *last* token in a candidate window is
   compared via its stem, not just its raw form.

No hardcoded suffix list
-------------------------
Stemming is delegated entirely to bnltk (BanglaStemmer), a real Bangla NLP
library trained on actual corpora -- this module does not encode any suffix
or morphology rules of its own. If bnltk isn't installed, stemming is
simply skipped: matching degrades to exact whole-token comparison only
(so "গমের" won't match "গম" until the library is installed), rather than
guessing at suffixes. This is a deliberate choice -- a hand-rolled, partial
suffix list is worse than no stemming at all, because it silently gets
morphology wrong for cases nobody thought to test. A missing dependency
should fail loudly (via the startup log warning below), not quietly
reintroduce incorrect matching.

Why not just alias in normalized_query?
------------------------------------------
Plain substring containment has no concept of word boundaries, so it treats
"আম" as a match anywhere those two characters appear -- including inside
totally unrelated words. Whole-token / stem-aware matching fixes that while
still catching real inflected forms.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache

from app.core.logging import get_logger
from app.services.pipeline.registry import CropInfo, get_known_crops
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# bnltk gives real Bangla stemming trained on actual corpora. It is the
# ONLY source of morphology knowledge in this module -- there is no
# hand-maintained suffix list. If it's missing, we log once at import time
# and matching degrades to exact whole-token comparison (see module
# docstring for why we don't paper over this with our own suffix guesses).
# Install with: pip install bnltk
# (Note: the similarly-named bnlp_toolkit package does NOT provide a
# stemmer as of this writing -- bnltk is the correct package for this.)
# --------------------------------------------------------------------------
try:
    from bnltk.stemmer import BanglaStemmer  # type: ignore

    _stemmer = BanglaStemmer()
    _HAS_STEMMER = True
except ImportError:
    _stemmer = None
    _HAS_STEMMER = False
    logger.warning(
        "bnltk not installed -- crop matching will not handle Bangla "
        "inflected forms (e.g. 'গমের' will not match 'গম'). "
        "Install with: pip install bnltk"
    )


def normalize_text(text: str) -> str:
    """
    Normalize text while preserving Bengali vowel signs and combining marks.

    Examples:
        "Boro   Paddy" -> "boro paddy"
        "বোরো-ধান" -> "বোরো ধান"
        "গমের" -> "গমের"
        "তুলনা" -> "তুলনা"
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    characters: list[str] = []

    for character in text:
        category = unicodedata.category(character)

        if character.isspace() or character == "_":
            characters.append(" ")
        # Preserve letters, numbers and combining marks.
        elif category[0] in {"L", "N", "M"}:
            characters.append(character)
        else:
            characters.append(" ")

    return " ".join("".join(characters).split())


def stem_token(token: str) -> str:
    """
    Reduce a Bangla-script token to its stem so inflected forms (e.g.
    "গমের") match their dictionary form (e.g. "গম").

    Delegates entirely to bnltk's BanglaStemmer. If it's unavailable or
    errors on a given token, returns the token unchanged (no local suffix
    guessing) -- that token will then only match via exact whole-token
    comparison.
    """
    if not _HAS_STEMMER:
        return token

    try:
        stemmed = _stemmer.stem(token)
        return stemmed if stemmed else token
    except Exception:
        logger.warning("stemming failed for token=%r; using token as-is", token)
        return token


@lru_cache
def get_crop_aliases() -> list[tuple[str, str]]:
    """Return (normalized_alias, canonical_crop_name) pairs, longest alias first."""
    aliases: list[tuple[str, str]] = []

    for crop in get_known_crops():
        english_name = normalize_text(crop.crop_name)
        bangla_name = normalize_text(crop.crop_bangla_name)

        aliases.append((english_name, crop.crop_name))
        aliases.append((bangla_name, crop.crop_name))

    aliases.sort(key=lambda item: len(item[0]), reverse=True)

    return aliases


def extract_crop_names(query: str) -> list[str]:
    normalized_query = normalize_text(query)
    query_tokens = normalized_query.split(" ")
    if not query_tokens or query_tokens == [""]:
        return []

    # Stem once per token up front, reused across every alias comparison.
    stemmed_tokens = [stem_token(t) for t in query_tokens]

    found_crops: list[str] = []

    for alias, crop_name in get_crop_aliases():
        if crop_name in found_crops:
            continue

        alias_tokens = alias.split(" ")
        n = len(alias_tokens)
        if n == 0:
            continue

        for i in range(len(query_tokens) - n + 1):
            window_raw = query_tokens[i : i + n]
            window_stemmed = stemmed_tokens[i : i + n]

            # Interior tokens of a multi-word alias must match exactly --
            # Bangla inflection lands on the final word of a phrase, not
            # the middle of it (e.g. "বোরো ধানের" -> stem last token only).
            interior_ok = all(
                w == a for w, a in zip(window_raw[:-1], alias_tokens[:-1])
            )
            last_ok = window_stemmed[-1] == alias_tokens[-1] or window_raw[-1] == alias_tokens[-1]

            if interior_ok and last_ok:
                found_crops.append(crop_name)
                break

    return found_crops


def extract_crop(state: PipelineState) -> PipelineState:
    query = state.get("rewritten_query") or state["raw_query"]
    crops = extract_crop_names(query)
    logger.info("extract_crops query=%r crops=%s", query, crops)
    return {
        "crops": crops
    }