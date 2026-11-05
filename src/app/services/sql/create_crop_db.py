# scripts/create_crops_db.py
import json
import sqlite3
from pathlib import Path

JSON_PATH = Path("data/crops.json")
DB_PATH = Path("data/crops.db")


def create_crops_db(json_path: Path = JSON_PATH, db_path: Path = DB_PATH) -> None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = data["data"]["getAllCropsFullDetails"]["rows"]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS crop_category (
            id   INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS company (
            id   INTEGER PRIMARY KEY,
            name TEXT
        );

        CREATE TABLE IF NOT EXISTS crop (
            id               INTEGER PRIMARY KEY,
            crop_name        TEXT NOT NULL,
            crop_category_id INTEGER,
            FOREIGN KEY (crop_category_id) REFERENCES crop_category(id)
        );

        CREATE TABLE IF NOT EXISTS variety (
            id              INTEGER PRIMARY KEY,
            variety_name    TEXT NOT NULL,
            crop_id         INTEGER,
            rating          REAL,
            price           REAL,
            FOREIGN KEY (crop_id) REFERENCES crop(id)
        );

        CREATE TABLE IF NOT EXISTS pesticide (
            id           INTEGER PRIMARY KEY,
            disease_name TEXT,
            disease_type TEXT,
            crop_id      INTEGER,
            image_urls    TEXT,
            FOREIGN KEY (crop_id) REFERENCES crop(id)
        );

        CREATE TABLE IF NOT EXISTS herbicide (
            id          INTEGER PRIMARY KEY,
            weed_name   TEXT,
            crop_id     INTEGER,
            company_id  INTEGER,
            FOREIGN KEY (crop_id) REFERENCES crop(id),
            FOREIGN KEY (company_id) REFERENCES company(id)
        );

        CREATE TABLE IF NOT EXISTS chemical (
            id           INTEGER PRIMARY KEY,
            trade_name   TEXT,
            generic_name TEXT,
            company_id   INTEGER,
            pesticide_id INTEGER,
            FOREIGN KEY (company_id)   REFERENCES company(id),
            FOREIGN KEY (pesticide_id) REFERENCES pesticide(id)
        );
    """)

    categories: dict[int, str] = {}
    companies: dict[int, str | None] = {}

    for crop in rows:
        cat = crop.get("crop_category") or {}
        if cat.get("id"):
            categories[cat["id"]] = cat.get("category_name")

        # for h in crop.get("herbicide") or []:
        #     if h.get("company_id"):
        #         companies[h["company_id"]] = h.get("company_name")

        for p in crop.get("pesticide") or []:
            for c in p.get("chemical") or []:
                if c.get("company_id"):
                    companies[c["company_id"]] = c.get("company_name")

        for v in crop.get("variety") or []:
            if v.get("company_id"):
                companies[v["company_id"]] = v.get("company_name")

    cur.executemany(
        "INSERT OR IGNORE INTO crop_category (id, name) VALUES (?, ?)",
        categories.items(),
    )
    cur.executemany(
        "INSERT OR IGNORE INTO company (id, name) VALUES (?, ?)",
        companies.items(),
    )

    for crop in rows:
        crop_name = (crop.get("crop_name") or "").strip()
        bangla_name = (crop.get("crop_bangla_name") or "").strip()
        combined_name = f"{bangla_name} ({crop_name})"
        
        cur.execute(
            "INSERT OR IGNORE INTO crop (id, crop_name, crop_category_id) VALUES (?, ?, ?)",
            (crop["id"], combined_name, (crop.get("crop_category") or {}).get("id")),
        )

        for v in crop.get("variety") or []:
            cur.execute(
                "INSERT OR IGNORE INTO variety (id, variety_name, crop_id, rating, price) VALUES (?, ?, ?, ?, ?)",
                (v["id"], v["variety_name"], v.get("crop_id"), v.get("rating"), v.get("price")),
            )

        for p in crop.get("pesticide") or []:
            images = p.get("image") or []
            if isinstance(images, dict):
                images = [images]
            image_urls = [ img["image"] for img in images if isinstance(img, dict) and img.get("image")]
            cur.execute(
                "INSERT OR IGNORE INTO pesticide (id, disease_name, disease_type, crop_id, image_urls) VALUES (?, ?, ?, ?, ?)",
                (p["id"], p.get("disease_name"), p.get("disease_type"), p.get("crop_id"), json.dumps(image_urls, ensure_ascii=False),),
            )
            for c in p.get("chemical") or []:
                cur.execute(
                    "INSERT OR IGNORE INTO chemical (id, trade_name, generic_name, company_id, pesticide_id) VALUES (?, ?, ?, ?, ?)",
                    (c["id"], c.get("trade_name"), c.get("generic_name"), c.get("company_id"), c.get("pesticide_id")),
                )

        for h in crop.get("herbicide") or []:
            cur.execute(
                "INSERT OR IGNORE INTO herbicide (id, weed_name, crop_id, company_id) VALUES (?, ?, ?, ?)",
                (h["id"], h.get("weed_name"), h.get("crop_id"), h.get("company_id")),
            )

    conn.commit()
    conn.close()
    print(f"Created {db_path}")


if __name__ == "__main__":
    create_crops_db()

# python -m app.services.sql.create_crop_db