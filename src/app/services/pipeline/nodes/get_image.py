
import json
import sqlite3
from pathlib import Path
from urllib.parse import urlsplit
import requests

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

def is_valid_image_url(url: str) -> bool:
    """
    Check whether the URL actually contains a valid image.
    Skip broken URLs, 403/404 errors, and non-image responses.
    """
    try:
        with requests.get(
            url,
            headers={"Range": "bytes=0-2047"},
            stream=True,
            allow_redirects=True,
            timeout=(2, 5),
        ) as response:

            # Check HTTP status
            if response.status_code not in (200, 206):
                logger.info(
                    "Skipping image: HTTP %s from %s",
                    response.status_code,
                    url,
                )
                return False

            # Check content type
            content_type = (
                response.headers.get("Content-Type", "")
                .split(";", 1)[0]
                .strip()
                .lower()
            )

            if content_type in {
                "text/html",
                "text/xml",
                "application/xml",
                "application/json",
            }:
                return False

            # Read first 2048 bytes
            data = next(
                response.iter_content(chunk_size=2048),
                b"",
            )

            if not data:
                return False

            # JPEG, PNG, GIF, BMP
            if data.startswith((
                b"\xff\xd8\xff",
                b"\x89PNG\r\n\x1a\n",
                b"GIF87a",
                b"GIF89a",
                b"BM",
            )):
                return True

            # WEBP
            if (
                len(data) >= 12
                and data[:4] == b"RIFF"
                and data[8:12] == b"WEBP"
            ):
                return True

            # TIFF, ICO
            if data.startswith((
                b"II*\x00",
                b"MM\x00*",
                b"\x00\x00\x01\x00",
            )):
                return True

            # AVIF, HEIC
            if (
                len(data) >= 12
                and data[4:8] == b"ftyp"
                and data[8:12] in {
                    b"avif",
                    b"avis",
                    b"heic",
                    b"heix",
                    b"mif1",
                }
            ):
                return True

            # SVG
            if (
                content_type == "image/svg+xml"
                and b"<svg" in data[:2048].lower()
            ):
                return True

            logger.info("Invalid image content: %s", url)
            return False

    except requests.RequestException as exc:
        logger.warning(
            "Skipping unreachable image %s: %s",
            url,
            exc,
        )
        return False

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
        if not isinstance(item, str):
            continue
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
            if url not in image_urls and is_valid_image_url(url):
                image_urls.append(url)
    return image_urls


def get_image(state: PipelineState) -> dict:
    entities_to_resolve = state.get("entities_to_resolve") or []
    if not entities_to_resolve:
        return {"image_urls": []}

    resolved_entities = state.get("resolved_entities") or []

    # Previously shown images from this session
    shown_image_urls = list(state.get("shown_image_urls") or [])
    seen = set(shown_image_urls)

    image_urls = []

    for entity in resolved_entities:
        if not isinstance(entity, dict):
            continue

        entity_type = (entity.get("type") or "").strip().lower()
        entity_name = (entity.get("mention") or "").strip()

        if entity_type != "pesticide":
            continue

        urls = lookup_images(entity_type=entity_type,name=entity_name,)
        for url in urls:
            if url not in seen:
                image_urls.append(url)
                seen.add(url)

    logger.info("Get Image node: %d unseen image URLs", len(image_urls))

    return {"image_urls": image_urls,"shown_image_urls": shown_image_urls + image_urls,}