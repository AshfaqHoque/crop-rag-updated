"""Simple read-only SQLite helper."""
import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def execute_readonly(sql: str, max_rows: int | None = None) -> list[dict[str, Any]]:
    settings = get_settings()
    max_rows = max_rows or settings.sql_max_rows
    db_path = Path(settings.crop_db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Crop database not found at {db_path}")

    cleaned = sql.strip().rstrip(";").strip()
    upper = cleaned.upper()

    # Basic safety
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError("Only SELECT statements are allowed")

    forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "PRAGMA")
    for token in forbidden:
        if token in upper:
            raise ValueError(f"Forbidden keyword: {token}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = ON")
        cur = conn.execute(cleaned)
        rows = cur.fetchmany(max_rows)
        columns = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()