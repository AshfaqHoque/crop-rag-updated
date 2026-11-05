
import json
import sqlite3
from pathlib import Path
from urllib.parse import urlsplit

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.pipeline.state import PipelineState

logger = get_logger(__name__)


# entity_type -> (table_name, name_column, image_column)
IMAGE_SOURCES = {
    "pesticide": (
        "pesticide",
        "disease_name",
        "image_urls",
    ),
}


def parse_image_urls(value: str | None) -> list[str]:
    """
    Convert the SQLite image_urls value into a list.

    Example input:
        '["https://s3.example.com/a.jpg"]'

    Output:
        ["https://s3.example.com/a.jpg"]
    """
    if not value:
        return []

    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        decoded = value
    if isinstance(decoded, str):
        decoded = [decoded]
    if not isinstance(decoded, list):
        return []
    urls = []
    for item in decoded:
        url = item.strip()
        try:
            parts = urlsplit(url)
            valid = (parts.scheme.lower() in ("http", "https")and bool(parts.hostname))
        except ValueError:
            valid = False
        if valid and url not in urls:
            urls.append(url)
    return urls


def lookup_images(entity_type: str,name: str,) -> list[str]:
    """
    Look up images using the resolved entity name.

    Example:
        entity_type = "pesticide"
        name = "Stem borer"

    SQLite query:
        SELECT image_urls
        FROM pesticide
        WHERE TRIM(disease_name) = 'Stem borer'
    """

    source = IMAGE_SOURCES.get(entity_type)
    if not source or not name:
        return []

    table, name_column, image_column = source
    db_path = Path(get_settings().crop_db_path)
    if not db_path.is_file():
        logger.warning("Crop database not found: %s", db_path)
        return []
    query = (
        f"SELECT {image_column} "
        f"FROM {table} "
        f"WHERE TRIM({name_column}) = ? COLLATE NOCASE"
    )
    try:
        db_uri = db_path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(db_uri, uri=True) as conn:
            rows = conn.execute(query,(name.strip(),)).fetchall()
    except sqlite3.Error as exc:
        logger.warning("Image lookup failed: %s", exc)
        return []
    image_urls = []
    for (value,) in rows:
        for url in parse_image_urls(value):
            if url not in image_urls:
                image_urls.append(url)
    return image_urls


def get_image(state: PipelineState) -> dict:
    entities_to_resolve = state.get("entities_to_resolve") or []
    if not entities_to_resolve:
        return {"image_urls": []}

    resolved_entities = state.get("resolved_entities") or []
    image_urls = []

    for entity in resolved_entities:
        if not isinstance(entity, dict):
            continue

        entity_type = (entity.get("type") or "").strip().lower()
        entity_name = (entity.get("mention") or "").strip()

        # Only retrieve pesticide images for now.
        if entity_type != "pesticide":
            continue

        urls = lookup_images(entity_type=entity_type,name=entity_name,)
        for url in urls:
            if url not in image_urls:
                image_urls.append(url)

    logger.info("Get Image node: found %d image URLs", len(image_urls))
    return {"image_urls": image_urls,}
