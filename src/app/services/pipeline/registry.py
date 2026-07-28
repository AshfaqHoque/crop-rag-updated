"""
Canonical crop and section registry. Crops are matched deterministically,
while sections extracted by the LLM are validated against this fixed
vocabulary before either value reaches the retriever's metadata filters.
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# The 11 sections that exist as top-level fields in the crop schema /
# chunk metadata. Keep this in sync with however chunks are built —
# these must exactly match the "section" value in chunk metadata.
SECTIONS: list[str] = [
    "overview",
    "seed",
    "land_preparation",
    "intercultural",
    "irrigation",
    "harvest",
    "fertilizer",
    "climate",
    "variety",
    "pesticide",
    "herbicide",
]


@dataclass(frozen=True)
class CropInfo:
    crop_id: str
    crop_name: str  # canonical English name — must match Chroma metadata "crop_name" exactly
    crop_bangla_name: str


@lru_cache
def get_known_crops() -> list[CropInfo]:
    """
    Loads the canonical crop list from a JSON file (crop_id, crop_name,
    crop_bangla_name). This should be exported from your crop database —
    see data/crops.json for the expected format and README for the
    export step. Cached for the process lifetime; restart the app (or
    call .cache_clear()) after updating the file.
    """
    settings = get_settings()
    path = Path(settings.crop_registry_path)
    if not path.exists():
        logger.warning("Crop registry file not found at %s — extraction will match no crops.", path)
        return []    
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
     
    if isinstance(raw, dict):
        rows = raw.get("data", {}).get("getAllCropsFullDetails", {}).get("rows", [])
    else:
        logger.warning("Crop registry at %s has an unsupported JSON shape.", path)
        return []
    
    crops: list[CropInfo] = []
    
    for item in rows:
        crop_id = item.get("crop_id") or item.get("id")
        crop_name = item.get("crop_name")
        crop_bangla_name = item.get("crop_bangla_name")
        
        if crop_id is None or not crop_name or not crop_bangla_name:
            continue
        
        crops.append(
            CropInfo(
                crop_id=str(crop_id),
                crop_name=str(crop_name).strip(),
                crop_bangla_name=str(crop_bangla_name).strip(),
            )
        )
    
    return crops


def crop_names() -> list[str]:
    return [c.crop_name for c in get_known_crops()]


def format_crop_list_for_prompt() -> str:
    """'Boro Paddy (বোরো ধান), Mango (আম), ...' — fed into the extraction prompt."""
    return ", ".join(f"{c.crop_name} ({c.crop_bangla_name})" for c in get_known_crops())
