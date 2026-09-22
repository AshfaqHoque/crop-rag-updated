"""Minimal runtime crop registry used by extraction.

This module only keeps the canonical crop names needed for entity extraction and
retrieval. Section and variety registries are intentionally omitted because the
application does not currently use them at runtime.
"""
from dataclasses import dataclass
from functools import lru_cache

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CropInfo:
    crop_name: str  # canonical English name — used for Chroma filtering and reranking
    crop_id: str | None = None
    crop_bangla_name: str = ""


def _build_crop_infos(rows: list[dict]) -> list[CropInfo]:
    crops: list[CropInfo] = []
    for item in rows:
        crop_name = item.get("crop_name")
        if not crop_name:
            continue

        crops.append(
            CropInfo(
                crop_id=str(item.get("crop_id") or item.get("id") or "").strip() or None,
                crop_name=str(crop_name).strip(),
                crop_bangla_name=str(item.get("crop_bangla_name") or "").strip(),
            )
        )
    return crops


@lru_cache
def get_known_crops() -> list[CropInfo]:
    """Load the canonical crop list from GraphQL and cache it in memory."""
    from app.ingestion.fetch_crops import fetch_crops

    payload = fetch_crops(persist=False)
    rows = payload.get("data", {}).get("getAllCropsFullDetails", {}).get("rows", [])
    if not isinstance(rows, list):
        logger.warning("GraphQL crop response did not include rows; returning an empty registry.")
        return []

    crops: list[CropInfo] = []
    for item in rows:
        crop_name = item.get("crop_name")
        if not crop_name:
            continue

        crops.append(
            CropInfo(
                crop_id=str(item.get("crop_id") or item.get("id") or "").strip() or None,
                crop_name=str(crop_name).strip(),
                crop_bangla_name=str(item.get("crop_bangla_name") or "").strip(),
            )
        )
    return crops

