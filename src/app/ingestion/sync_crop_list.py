"""Fetch crops and save a lightweight local crop registry."""

import json
import os
import tempfile
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.ingestion.fetch_crops import fetch_crops
from app.services.pipeline.registry import _build_crop_infos, get_known_crops

logger = get_logger(__name__)


def sync_crop_list() -> int:
    # Existing function also saves the full response to data/crops.json.
    payload = fetch_crops()

    rows = payload["data"]["getAllCropsFullDetails"]["rows"]
    crops = _build_crop_infos(rows)

    if not crops:
        raise ValueError("No valid crops received; keeping existing local list")

    path = Path(get_settings().crop_list_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as file:
            temporary = file.name

            json.dump(
                [
                    {
                        "crop_id": crop.crop_id,
                        "crop_name": crop.crop_name,
                        "crop_bangla_name": crop.crop_bangla_name,
                    }
                    for crop in crops
                ],
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

        os.replace(temporary, path)

    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)

    get_known_crops.cache_clear()

    logger.info("Saved %d crops to %s", len(crops), path)

    return len(crops)


if __name__ == "__main__":
    configure_logging()
    sync_crop_list()