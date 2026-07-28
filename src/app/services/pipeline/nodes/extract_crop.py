"""Deterministic crop extraction from the canonical crop registry."""
import re
import unicodedata
from functools import lru_cache

from app.core.logging import get_logger
from app.services.pipeline.registry import CropInfo, get_known_crops
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """
    Make text easier to compare.
    Example:
        "Boro   Paddy" -> "boro paddy"
        "বোরো-ধান" -> "বোরো ধান"
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.replace("_", " ")

    return " ".join(text.split())

@lru_cache
def get_crop_aliases() -> list[tuple[str, str]]:
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
    found_crops: list[str] = []
    
    for alias, crop_name in get_crop_aliases():
        if alias in normalized_query and crop_name not in found_crops:
            found_crops.append(crop_name)
        
    return found_crops

def extract_crop(state: PipelineState) -> PipelineState:
    query = state.get("rewritten_query") or state["raw_query"]
    crops = extract_crop_names(query)
    logger.info("extract_crops query=%r crops=%s", query, crops)
    return {
        "crops": crops
    }