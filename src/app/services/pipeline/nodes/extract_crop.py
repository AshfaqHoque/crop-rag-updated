"""Deterministic crop extraction from the canonical crop registry."""
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from app.core.logging import get_logger
from app.services.pipeline.registry import get_known_crops
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)

@dataclass(frozen=True)
class CropAlias:
    crop_name: str
    alias: str
    tokens: tuple[str, ...]
    registry_order: int

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

def is_non_ascii_token(token: str) -> bool:
    """
    Bengali and other non-ASCII words may contain grammatical additions
    directly after the registered word.
    """
    return any(ord(character) > 127 for character in token)

def token_matches(alias_token: str, query_token: str) -> bool:
    """
    Match a registry token against a query token.

    Exact examples:
        ধান == ধান
        wheat == wheat

    Attached-form examples:
        গম matches গমের
        ধান matches ধানের

    No suffix names are hardcoded.
    """
    if alias_token == query_token:
        return True

    # Prefix matching is only used for non-ASCII words.
    # This prevents English words such as "rice" matching arbitrary
    # longer English words.
    return (
        is_non_ascii_token(alias_token)
        and len(query_token) > len(alias_token)
        and query_token.startswith(alias_token)
    )
    
@lru_cache
def get_crop_aliases() -> tuple[CropAlias, ...]:
    aliases: list[CropAlias] = []

    for registry_order, crop in enumerate(get_known_crops()):
        crop_aliases = {
            normalize_text(crop.crop_name),
            normalize_text(crop.crop_bangla_name),
        }

        for alias in crop_aliases:
            if not alias:
                continue

            aliases.append(
                CropAlias(
                    crop_name=crop.crop_name,
                    alias=alias,
                    tokens=tuple(alias.split()),
                    registry_order=registry_order,
                )
            )

    return tuple(aliases)

def match_complete_alias(
    query_tokens: list[str],
    start_index: int,
) -> CropAlias | None:
    """
    Match a complete crop alias starting at the current query token.

    Examples:
        বোরো ধান       -> Boro Rice
        আমন ধানের      -> Aman Rice
        গমের           -> Wheat
    """
    candidates: list[CropAlias] = []

    for crop_alias in get_crop_aliases():
        alias_tokens = crop_alias.tokens
        end_index = start_index + len(alias_tokens)

        if end_index > len(query_tokens):
            continue

        query_part = query_tokens[start_index:end_index]

        if all(
            token_matches(alias_token, query_token)
            for alias_token, query_token in zip(alias_tokens, query_part)
        ):
            candidates.append(crop_alias)

    if not candidates:
        return None

    # Prefer the most specific alias.
    # Registry order resolves equal matches deterministically.
    candidates.sort(
        key=lambda item: (
            -len(item.tokens),
            -len(item.alias),
            item.registry_order,
        )
    )

    return candidates[0]

def match_generic_crop_token(query_token: str) -> CropAlias | None:
    """
    Resolve a generic query token against words inside crop aliases.

    Example registry:

        বোরো ধান
        আমন ধান
        গম

    Query token:

        ধান

    Both rice entries contain ধান, so the first one in the registry wins.
    """
    candidates: list[CropAlias] = []

    for crop_alias in get_crop_aliases():
        if any(
            token_matches(alias_token, query_token)
            for alias_token in crop_alias.tokens
        ):
            candidates.append(crop_alias)

    if not candidates:
        return None

    candidates.sort(key=lambda item: item.registry_order)

    return candidates[0]

def extract_crop_names(query: str) -> list[str]:
    normalized_query = normalize_text(query)
    query_tokens = normalized_query.split()

    found_crops: list[str] = []
    token_index = 0

    while token_index < len(query_tokens):
        # First try a complete and specific crop name.
        matched_alias = match_complete_alias(
            query_tokens=query_tokens,
            start_index=token_index,
        )

        if matched_alias is not None:
            if matched_alias.crop_name not in found_crops:
                found_crops.append(matched_alias.crop_name)

            token_index += len(matched_alias.tokens)
            continue

        # Otherwise resolve a generic token such as ধান.
        matched_alias = match_generic_crop_token(
            query_token=query_tokens[token_index]
        )

        if (
            matched_alias is not None
            and matched_alias.crop_name not in found_crops
        ):
            found_crops.append(matched_alias.crop_name)

        token_index += 1

    return found_crops

def extract_crop(state: PipelineState) -> PipelineState:
    query = state.get("rewritten_query") or state["raw_query"]
    crops = extract_crop_names(query)
    logger.info("extract_crops query=%r crops=%s", query, crops)
    return {
        "crops": crops
    }