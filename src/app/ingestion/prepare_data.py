"""
DESIGN DECISION (read this before changing anything)
=====================================================
We do NOT do generic fixed-size chunking (e.g. "split into 500-token
windows"). That approach is fine for prose documents but wrong for this
data, because the JSON already tells us the correct chunk boundaries:
one crop has many *independent* sub-topics (seed, climate, irrigation,
harvest, cost, ...) and, critically, many *independent varieties* and
many *independent pest/disease entries*.

If we concatenated everything about "Boro Paddy" into one giant chunk,
a query like "what is the seed rate for BRRI dhan88" would retrieve a
huge blob containing 20 other varieties too, and a small 4B local model
has no chance of picking the right number out of that noise.

So the rule is: **one chunk = one self-contained, independently
answerable fact-unit.**
    - crop overview            -> 1 chunk
    - crop-level seed info     -> 1 chunk  (NOT the same as variety seed_rate!)
    - crop-level climate       -> 1 chunk
    - crop-level land prep     -> 1 chunk
    - crop-level intercultural -> 1 chunk
    - crop-level irrigation    -> 1 chunk
    - crop-level harvest       -> 1 chunk
    - crop-level fertilizer    -> 1 chunk
    - crop-level cost info     -> 1 chunk
    - EACH variety             -> 1 chunk each
    - EACH pesticide/disease   -> 1 chunk each (chemicals nested inside)
    - EACH herbicide entry     -> 1 chunk each

Every chunk's text is *self-identifying*: it starts with the crop name
(Bangla + English) and a section label, so it makes sense in isolation
and dense embeddings pick up the crop identity even without a metadata
filter. Every chunk's metadata also carries crop_id/crop_name/section/
variety_name/disease_name so the retriever can hard-filter, not just
hope semantic similarity sorts it out.
"""
from typing import Any

from app.schemas.chunk_schema import Chunk
from app.ingestion.html_cleaner import clean_html, is_meaningful


def _header(crop: dict, section_label: str) -> str:
    name = crop.get("crop_name") or ""
    bn_name = crop.get("crop_bangla_name") or ""
    return f"ফসল (Crop): {bn_name} ({name})\nবিভাগ (Section): {section_label}\n"


def _crop_id(crop: dict) -> str:
    return str(crop.get("id"))


def chunk_overview(crop: dict) -> Chunk | None:
    info = clean_html(crop.get("general_info"))
    extra_lines = []
    if crop.get("scientific_name"):
        extra_lines.append(f"বৈজ্ঞানিক নাম (Scientific name): {crop['scientific_name']}")
    if crop.get("crop_family"):
        extra_lines.append(f"পরিবার (Family): {crop['crop_family']}")
    # if crop.get("average_production") is not None:
    #     extra_lines.append(f"গড় উৎপাদন (Average production): {crop['average_production']}")

    body = "\n".join(extra_lines + ([info] if info else []))
    if not is_meaningful(body):
        return None
    text = _header(crop, "সাধারণ তথ্য / Overview") + body
    return Chunk(
        chunk_id=f"{_crop_id(crop)}_overview",
        text=text,
        metadata={
            "crop_id": _crop_id(crop),
            "crop_name": crop.get("crop_name"),
            "crop_bangla_name": crop.get("crop_bangla_name"),
            "section": "overview",
        },
    )


def _simple_section(crop: dict, key: str, section_label: str, section_tag: str,
                     text_field: str = "description") -> Chunk | None:
    """Handles the many sections that are shaped like
    {id, crop_id, description: "<html>"} -- harvest, intercultural,
    irrigation, landPreparation, climate.general_info."""
    obj = crop.get(key)
    if not obj:
        return None
    body = clean_html(obj.get(text_field))
    if not is_meaningful(body):
        return None
    text = _header(crop, section_label) + body
    return Chunk(
        chunk_id=f"{_crop_id(crop)}_{section_tag}",
        text=text,
        metadata={
            "crop_id": _crop_id(crop),
            "crop_name": crop.get("crop_name"),
            "crop_bangla_name": crop.get("crop_bangla_name"),
            "section": section_tag,
        },
    )


def chunk_fertilizer(crop: dict) -> Chunk | None:
    obj = crop.get("fertilizer")
    if not obj:
        return None
    body = clean_html(obj.get("fertilizer"))
    if not is_meaningful(body):
        return None
    text = _header(crop, "সার ব্যবস্থাপনা / Fertilizer") + body
    return Chunk(
        chunk_id=f"{_crop_id(crop)}_fertilizer",
        text=text,
        metadata={
            "crop_id": _crop_id(crop),
            "crop_name": crop.get("crop_name"),
            "crop_bangla_name": crop.get("crop_bangla_name"),
            "section": "fertilizer",
        },
    )


def chunk_seed(crop: dict) -> Chunk | None:
    """Crop-level seed info: treatment, sowing method, seedbed, and the
    crop-level seed_rate (hectare basis). This is DELIBERATELY separate
    from each variety's own seed_rate field -- they answer different
    questions and mixing them is exactly the confusion you flagged."""
    obj = crop.get("seed")
    if not obj:
        return None

    parts = []
    seed_rate = obj.get("seed_rate")
    if is_meaningful(seed_rate):
        parts.append(f"বীজের হার: {seed_rate}")
    treatment = clean_html(obj.get("treatment"))
    if is_meaningful(treatment):
        parts.append("বীজ শোধন ও নির্বাচন (Seed treatment/selection):\n" + treatment)
    seedbed = clean_html(obj.get("seedbed"))
    if is_meaningful(seedbed):
        parts.append("বীজতলা তৈরি (Seedbed preparation):\n" + seedbed)
    showing = clean_html(obj.get("showing_method"))
    if is_meaningful(showing):
        parts.append("রোপণ পদ্ধতি (Sowing/transplanting method):\n" + showing)

    if not parts:
        return None
    text = _header(crop, "বীজ (সাধারণ) / Seed (crop-level, not variety-specific)") + "\n\n".join(parts)
    return Chunk(
        chunk_id=f"{_crop_id(crop)}_seed",
        text=text,
        metadata={
            "crop_id": _crop_id(crop),
            "crop_name": crop.get("crop_name"),
            "crop_bangla_name": crop.get("crop_bangla_name"),
            "section": "seed",
        },
    )


# def chunk_cost(crop: dict) -> Chunk | None:
#     items = crop.get("cropAdditionalCostInfo") or []
#     if not items:
#         return None
#     lines = []
#     for item in items:
#         unit = (item.get("unitInfo") or {}).get("unit_name", "")
#         lines.append(f"- {item.get('cost_type')}: {item.get('amount')} টাকা/{unit}")
#     text = _header(crop, "উৎপাদন খরচ / Additional cost breakdown") + "\n".join(lines)
#     return Chunk(
#         chunk_id=f"{_crop_id(crop)}_cost",
#         text=text,
#         metadata={
#             "crop_id": _crop_id(crop),
#             "crop_name": crop.get("crop_name"),
#             "crop_bangla_name": crop.get("crop_bangla_name"),
#             "section": "cost",
#         },
#     )


def chunk_varieties(crop: dict) -> list[Chunk]:
    """ONE chunk per variety. This is the key isolation mechanism: a
    query about a specific variety's seed rate/yield/duration retrieves
    exactly that variety's chunk and nothing else."""
    chunks = []
    for v in crop.get("variety") or []:
        name = v.get("variety_name") or "Unknown variety"
        lines = [f"জাতের নাম (Variety): {name}"]
        if v.get("company_name"):
            lines.append(f"উদ্ভাবক/কোম্পানি (Company): {v['company_name'].strip()}")
        # if v.get("duration_start") or v.get("duration_end"):
        #     lines.append(
        #         f"জীবনকাল (Duration): {v.get('duration_start')}-{v.get('duration_end')} দিন"
        #     )
        # if v.get("avg_expected_yield"):
        #     lines.append(
        #         f"গড় প্রত্যাশিত ফলন (Avg expected yield): {v['avg_expected_yield']} "
        #         f"মণ/একর অথবা প্রাসঙ্গিক একক (unit as per source)"
        #     )
        if v.get("seed_rate"):
            unit = (v.get("seedRateUnit") or {}).get("unit_name", "")
            lines.append(f"এই জাতের বীজ হার (This variety's seed rate): {v['seed_rate']} {unit}")
        if v.get("price"):
            lines.append(f"মূল্য (Price): {v['price']}")
        if v.get("rating") is not None:
            lines.append(f"রেটিং (Rating): {v['rating']}/5")

        special = clean_html(v.get("special_character"))
        if is_meaningful(special):
            lines.append("বিস্তারিত বৈশিষ্ট্য (Detailed characteristics):\n" + special)

        body = "\n".join(lines)
        if not is_meaningful(body):
            continue

        text = _header(crop, f"জাত / Variety -- {name}") + body
        chunks.append(
            Chunk(
                chunk_id=f"{_crop_id(crop)}_variety_{v.get('id')}",
                text=text,
                metadata={
                    "crop_id": _crop_id(crop),
                    "crop_name": crop.get("crop_name"),
                    "crop_bangla_name": crop.get("crop_bangla_name"),
                    "section": "variety",
                    "variety_id": str(v.get("id")),
                    "variety_name": name,
                },
            )
        )
    return chunks


def chunk_pesticides(crop: dict) -> list[Chunk]:
    """ONE chunk per disease/pest entry, chemicals folded in as text
    (chemicals are few enough per entry that splitting them further
    would hurt more than help -- they're only meaningful alongside the
    disease they treat)."""
    chunks = []
    for p in crop.get("pesticide") or []:
        disease = p.get("disease_name") or "Unknown pest/disease"
        dtype = p.get("disease_type") or ""
        lines = [f"রোগ/পোকা (Pest/Disease): {disease} ({dtype})"]

        symptoms = clean_html(p.get("damage_control"))
        if is_meaningful(symptoms):
            lines.append("লক্ষণ ও ক্ষতি (Symptoms/damage):\n" + symptoms)
        control = clean_html(p.get("control_measure"))
        if is_meaningful(control):
            lines.append("দমন ব্যবস্থাপনা (Control measures):\n" + control)

        chem_lines = []
        for c in p.get("chemical") or []:
            dose_unit = (c.get("applicationDoseUnitInfo") or {}).get("unit_name", "")
            chem_lines.append(
                f"- {c.get('trade_name', '').strip()} (generic: {c.get('generic_name', '').strip()}): "
                f"মাত্রা {c.get('application_dose')} {dose_unit} প্রতি {c.get('pesticide_amount')} "
                f"{(c.get('pesticideAmountUnitInfo') or {}).get('unit_name', '')} পানিতে, "
                f"মূল্য প্রায় {c.get('price')} টাকা"
            )
        if chem_lines:
            lines.append("সুপারিশকৃত কীটনাশক/ছত্রাকনাশক (Recommended chemicals):\n" + "\n".join(chem_lines))

        body = "\n".join(lines)
        text = _header(crop, f"রোগবালাই / Pest-Disease -- {disease}") + body
        chunks.append(
            Chunk(
                chunk_id=f"{_crop_id(crop)}_pest_{p.get('id')}",
                text=text,
                metadata={
                    "crop_id": _crop_id(crop),
                    "crop_name": crop.get("crop_name"),
                    "crop_bangla_name": crop.get("crop_bangla_name"),
                    "section": "pesticide",
                    "disease_name": disease,
                    "disease_type": dtype,
                },
            )
        )
    return chunks


def chunk_herbicides(crop: dict) -> list[Chunk]:
    chunks = []
    for h in crop.get("herbicide") or []:
        target = h.get("pesticide_name") or "Unknown weed target"
        lines = [
            f"আগাছা/লক্ষ্য (Weed target): {target}",
            f"ট্রেড নাম (Trade name): {h.get('trade_name', '').strip()}",
            f"জেনেরিক নাম (Generic name): {h.get('generic_name', '').strip()}",
        ]
        # if h.get("application_dose"):
        #     unit = (h.get("applicationDoseUnitInfo") or {}).get("unit_name", "")
        #     lines.append(f"প্রয়োগ মাত্রা (Application dose): {h['application_dose']} {unit}")
        guide = clean_html(h.get("application_guide"))
        if is_meaningful(guide):
            lines.append("প্রয়োগ নির্দেশিকা (Application guide):\n" + guide)

        body = "\n".join(lines)
        text = _header(crop, f"আগাছানাশক / Herbicide -- {target}") + body
        chunks.append(
            Chunk(
                chunk_id=f"{_crop_id(crop)}_herbicide_{h.get('id')}",
                text=text,
                metadata={
                    "crop_id": _crop_id(crop),
                    "crop_name": crop.get("crop_name"),
                    "crop_bangla_name": crop.get("crop_bangla_name"),
                    "section": "herbicide",
                    "weed_target": target,
                },
            )
        )
    return chunks


def chunk_crop(crop: dict) -> list[Chunk]:
    """Entry point: turn one crop dict into its full list of chunks."""
    chunks: list[Chunk] = []

    overview = chunk_overview(crop)
    if overview:
        chunks.append(overview)

    seed = chunk_seed(crop)
    if seed:
        chunks.append(seed)

    climate = _simple_section(crop, "climate", "জলবায়ু ও মাটি / Climate & soil", "climate", text_field="general_info")
    if climate:
        chunks.append(climate)

    land_prep = _simple_section(crop, "landPreparation", "জমি তৈরি / Land preparation", "land_preparation")
    if land_prep:
        chunks.append(land_prep)

    intercultural = _simple_section(crop, "intercultural", "আন্তঃপরিচর্যা / Intercultural operations", "intercultural")
    if intercultural:
        chunks.append(intercultural)

    irrigation = _simple_section(crop, "irrigation", "সেচ / Irrigation", "irrigation")
    if irrigation:
        chunks.append(irrigation)

    harvest = _simple_section(crop, "harvest", "ফসল কাটা / Harvest", "harvest")
    if harvest:
        chunks.append(harvest)

    fertilizer = chunk_fertilizer(crop)
    if fertilizer:
        chunks.append(fertilizer)

    # cost = chunk_cost(crop)
    # if cost:
    #     chunks.append(cost)

    chunks.extend(chunk_varieties(crop))
    chunks.extend(chunk_pesticides(crop))
    chunks.extend(chunk_herbicides(crop))

    return chunks


def chunk_all(crops: list[dict]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for crop in crops:
        all_chunks.extend(chunk_crop(crop))
    return all_chunks