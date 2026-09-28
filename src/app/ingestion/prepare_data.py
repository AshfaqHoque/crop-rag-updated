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
from app.schemas.extraction import LLMSplitResult
from app.services.llm.client import invoke_structured
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_CHUNK_SIZE = 5000

def safe_str(val: Any) -> str:
    """Safely convert None, int, float, or string values to a stripped string."""
    if val is None:
        return ""
    return str(val).strip()

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


def chunk_climate(crop: dict) -> Chunk | None:
    """Crop-level climate and environmental requirements.
    Includes temperature, rainfall, pH, humidity, EC, salinity,
    land type, soil texture, and general climate-related guidance.
    """
    obj = crop.get("climate")
    if not obj:
        return None

    parts = []

    temperature_start = obj.get("climate_temperature_start")
    temperature_end = obj.get("climate_temperature_end")
    if is_meaningful(temperature_start) and is_meaningful(temperature_end):
        parts.append(f"উপযুক্ত তাপমাত্রা: {temperature_start}°C থেকে {temperature_end}°C")

    rainfall_start = obj.get("climate_rainfall_start")
    rainfall_end = obj.get("climate_rainfall_end")
    if is_meaningful(rainfall_start) and is_meaningful(rainfall_end):
        parts.append(f"উপযুক্ত বৃষ্টিপাত: {rainfall_start} থেকে {rainfall_end} মিমি")

    ph_start = obj.get("climate_ph_start")
    ph_end = obj.get("climate_ph_end")
    if is_meaningful(ph_start) and is_meaningful(ph_end):
        parts.append(f"উপযুক্ত pH: {ph_start} থেকে {ph_end}")

    humidity_start = obj.get("climate_humidity")
    humidity_end = obj.get("climate_humidity_end")
    if is_meaningful(humidity_start) and is_meaningful(humidity_end):
        parts.append(f"উপযুক্ত আর্দ্রতা: {humidity_start}% থেকে {humidity_end}%")

    ec_start = obj.get("climate_ec_start")
    ec_end = obj.get("climate_ec_end")
    if is_meaningful(ec_start) and is_meaningful(ec_end):
        parts.append(f"উপযুক্ত EC: {ec_start} থেকে {ec_end} dS/m")

    salinity_start = obj.get("salinity_start")
    salinity_end = obj.get("salinity_end")
    if is_meaningful(salinity_start) and is_meaningful(salinity_end):
        parts.append(f"সহনীয় লবণাক্ততা: {salinity_start} থেকে {salinity_end}")

    general_info = clean_html(obj.get("general_info"))
    if is_meaningful(general_info):
        parts.append("জলবায়ু ও পরিবেশ সংক্রান্ত সাধারণ তথ্য:\n" + general_info)

    if not parts:
        return None

    text = _header(crop, "জলবায়ু ও পরিবেশ / Climate & Environment") + "\n\n".join(parts)

    return Chunk(
        chunk_id=f"{_crop_id(crop)}_climate",
        text=text,
        metadata={
            "crop_id": _crop_id(crop),
            "crop_name": crop.get("crop_name"),
            "crop_bangla_name": crop.get("crop_bangla_name"),
            "section": "climate",
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
    showing_method = clean_html(obj.get("showing_method"))
    if is_meaningful(showing_method):
        parts.append("রোপণ পদ্ধতি (Sowing/transplanting method):\n" + showing_method)
    if not parts:
        return None
    text = _header(crop, "বীজ / Seed") + "\n\n".join(parts)

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


def chunk_cost(crop: dict) -> Chunk | None:
    """Crop-level additional production costs, grouped into one chunk."""
    costs = crop.get("cropAdditionalCostInfo")
    if not costs:
        return None
    parts = []
    for cost in costs:
        cost_type = (cost.get("cost_type") or "").strip()
        amount = cost.get("amount")
        unit_name = "টাকা প্রতি শতক" #(cost.get("unitInfo") or {}).get("unit_name", "")
        if not is_meaningful(cost_type) or not is_meaningful(amount):
            continue
        label = cost_type.replace("_", " ").strip().title()
        if is_meaningful(unit_name):
            parts.append(f"{label}: {amount} {unit_name}")
        else:
            parts.append(f"{label}: {amount}")
    if not parts:
        return None
    text = _header(crop, "খরচ / Cost") + "\n\n".join(parts)

    return Chunk(
        chunk_id=f"{_crop_id(crop)}_cost",
        text=text,
        metadata={
            "crop_id": _crop_id(crop),
            "crop_name": crop.get("crop_name"),
            "crop_bangla_name": crop.get("crop_bangla_name"),
            "section": "cost",
        },
    )

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
        if v.get("duration_start") or v.get("duration_end"):
            lines.append(f"জীবনকাল (Duration): {v.get('duration_start')}-{v.get('duration_end')} দিন")
        if v.get("avg_expected_yield"):
            lines.append(f"গড় প্রত্যাশিত ফলন (Average expected yield): {v['avg_expected_yield']} ")
        if v.get("seed_rate"):
            lines.append(f"এই জাতের বীজ হার (This variety's seed rate): {v['seed_rate']}")
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
    """ONE chunk per pest/disease entry, with all recommended chemicals
    included in the same chunk so their relationship is preserved.
    """
    chunks = []
    for p in crop.get("pesticide") or []:
        disease_name = p.get("disease_name") or "Unknown pest/disease"
        lines = [
            f"বালাই/রোগের নাম (Pest/Disease name): {safe_str(p.get('disease_name'))}",
            f"বালাই/রোগের ধরন (Pest/Disease type): {safe_str(p.get('disease_type'))}",
            f"রোগজীবাণুর নাম (Disease-causing organism): {safe_str(p.get('disease_germs'))}",
        ]
        favorable_environment = clean_html(p.get("favorable_environment"))
        if is_meaningful(favorable_environment):
            lines.append("অনুকূল পরিবেশ (Favorable environment): " + favorable_environment)
        damage_control = clean_html(p.get("damage_control"))
        if is_meaningful(damage_control):
            lines.append("ক্ষতির লক্ষণ (Damage symptoms): "+ damage_control)
        control_measure = clean_html(p.get("control_measure"))
        if is_meaningful(control_measure):
            lines.append("দমন ব্যবস্থাপনা (Control measures): "+ control_measure)
        # Recommended chemicals/pesticides
        chemicals = p.get("chemical") or []
        for c in chemicals:
            chemical_lines = [
                f"ট্রেড নাম (Trade name): {safe_str(c.get('trade_name'))}",
                f"জেনেরিক নাম (Generic name): {safe_str(c.get('generic_name'))}",
                f"কোম্পানির নাম (Company name): {safe_str(c.get('company_name'))}",
                f"কীটনাশকের প্রয়োগ মাত্রা (Applicable Amount of Pesticide): {safe_str(c.get('application_dose'))}",
                f"পানির প্রয়োগ মাত্রা (Applicable Amount of Water): {safe_str(c.get('pesticide_amount'))}",
                f"মূল্য (Price): {safe_str(c.get('price'))}",
                f"রেটিং (Rating): {c.get('rating', '')}",
            ]
            guide = clean_html(c.get("application_guide"))
            if is_meaningful(guide):
                chemical_lines.append("প্রয়োগ নির্দেশিকা (Application guide):\n" + guide)
            lines.append("প্রস্তাবিত কীটনাশক (Recommended pesticide):\n" + "\n".join(chemical_lines))
        body = "\n".join(lines)

        text = (_header(crop, f"কীটনাশক / Pesticide -- {disease_name}")+ body)

        chunks.append(
            Chunk(
                chunk_id=f"{_crop_id(crop)}_pesticide_{p.get('id')}",
                text=text,
                metadata={
                    "crop_id": _crop_id(crop),
                    "crop_name": crop.get("crop_name"),
                    "crop_bangla_name": crop.get("crop_bangla_name"),
                    "section": "pesticide",
                    "disease_name": disease_name,
                    "disease_type": p.get("disease_type"),
                },
            )
        )

    return chunks

def chunk_herbicides(crop: dict) -> list[Chunk]:
    chunks = []
    for h in crop.get("herbicide") or []:
        weed_name = h.get("pesticide_name") or "Unknown weed"
        lines = [
            f"আগাছার নাম (Weed name): {weed_name}",
            f"ট্রেড নাম (Trade name): {safe_str(h.get('trade_name'))}",
            f"জেনেরিক নাম (Generic name): {safe_str(h.get('generic_name'))}",
            f"কোম্পানির নাম (Company name): {safe_str(h.get('company_name'))}",
            f"আগাছানাশকের প্রয়োগ মাত্রা (Applicable Amount of Herbicide): {safe_str(h.get('application_dose'))}",
            f"পানির প্রয়োগ মাত্রা (Applicable Amount of Water): {safe_str(h.get('pesticide_amount'))}",
            f"মূল্য (Price): {safe_str(h.get('price'))}",
            f"রেটিং (Rating): {h.get('rating', '')}",
        ]
        guide = clean_html(h.get("application_guide"))
        if is_meaningful(guide):
            lines.append("প্রয়োগ নির্দেশিকা (Application guide):\n" + guide)
        body = "\n".join(lines)
        text = _header(crop, f"আগাছানাশক / Herbicide -- {weed_name}") + body
        chunks.append(
            Chunk(
                chunk_id=f"{_crop_id(crop)}_herbicide_{h.get('id')}",
                text=text,
                metadata={
                    "crop_id": _crop_id(crop),
                    "crop_name": crop.get("crop_name"),
                    "crop_bangla_name": crop.get("crop_bangla_name"),
                    "section": "herbicide",
                    "weed_name": weed_name,
                },
            )
        )
    return chunks


def _chunk_name_list(crop: dict, names: list[str], section_tag: str, label_bn: str) -> Chunk | None:
    """Builds a single 'index' chunk out of a flat list of names (e.g.
    crop['children']['varieties']) -- NOT the same as chunk_varieties()/
    chunk_pesticides()/chunk_herbicides(), which build one detailed chunk
    per item. This is a lightweight, self-contained answer to "what
    varieties/pests/herbicides exist for this crop?" without pulling in
    every variety's/pest's/herbicide's full detail chunk.

    Deliberately does NOT use _header(): a plain "<crop> এর <label> হলো:"
    opening line is enough for the embedding to identify the crop, and it
    reads like a direct answer rather than a labeled record.
    """
    cleaned = [n.strip() for n in (names or []) if n and n.strip()]
    if not cleaned:
        return None

    bn_name = crop.get("crop_bangla_name") or crop.get("crop_name") or ""
    text = f"{bn_name} এর {label_bn} হলো:\n" + "\n".join(cleaned)
    return Chunk(
        chunk_id=f"{_crop_id(crop)}_total_{section_tag}",
        text=text,
        metadata={
            "crop_id": _crop_id(crop),
            "crop_name": crop.get("crop_name"),
            "crop_bangla_name": crop.get("crop_bangla_name"),
            "section": section_tag,
        },
    )


def chunk_children(crop: dict) -> list[Chunk]:
    """crop['children'] holds three flat name lists (varieties, pesticides,
    herbicides) -- summary indexes, distinct from the per-item detail
    chunks built by chunk_varieties()/chunk_pesticides()/chunk_herbicides().
    One chunk per non-empty list; empty lists (e.g. no herbicides on record
    for this crop) are skipped rather than emitting a useless chunk."""
    children = crop.get("children") or {}
    chunks = []
    varieties = _chunk_name_list(crop, children.get("varieties"), "varieties", "জাতগুলো")
    if varieties:
        chunks.append(varieties)
    pesticides = _chunk_name_list(crop, children.get("pesticides"), "pesticides", "রোগবালাই/পোকামাকড়সমূহ")
    if pesticides:
        chunks.append(pesticides)
    herbicides = _chunk_name_list(crop, children.get("herbicides"), "herbicides", "আগাছানাশক সমূহ")
    if herbicides:
        chunks.append(herbicides)

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

    climate = chunk_climate(crop)
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

    cost = chunk_cost(crop)
    if cost:
        chunks.append(cost)

    chunks.extend(chunk_varieties(crop))
    chunks.extend(chunk_pesticides(crop))
    chunks.extend(chunk_herbicides(crop))
    chunks.extend(chunk_children(crop))

    return chunks

_LLM_SPLIT_PROMPT = """
Split the following agricultural knowledge into exactly two semantically coherent chunks.

Rules:
- Preserve all information exactly.
- Do not summarize.
- Do not rewrite.
- Do not add or remove information.
- Split only at a natural semantic boundary.
- Keep related information together.

You MUST return ONLY a valid JSON object with a single key "chunks" containing a list of 2 strings:
     {{"chunks": ["chunk 1 text", "chunk 2 text"]}}

Text:
{text}
"""
def llm_splitter(chunks: list[Chunk]) -> list[Chunk]:
    """Keep splitting any chunk > MAX_CHUNK_SIZE until all pieces are ≤ limit."""
    final: list[Chunk] = []
    queue: list[Chunk] = list(chunks)          # work list

    while queue:
        chunk = queue.pop(0)

        if len(chunk.text) <= MAX_CHUNK_SIZE:
            final.append(chunk)
            continue

        prompt = _LLM_SPLIT_PROMPT.format(text=chunk.text)
        logger.info("Splitting chunk %s (size %d)", chunk.chunk_id, len(chunk.text))
        result = invoke_structured(LLMSplitResult, prompt)

        for i, text in enumerate(result.chunks, start=1):
            new_id = f"{chunk.chunk_id}_{i}"
            logger.info("Created chunk %s (size %d)", new_id, len(text))
            queue.append(                       # put back on the work list
                Chunk(
                    chunk_id=new_id,
                    text=text,
                    metadata=chunk.metadata,
                )
            )

    return final

def chunk_all(crops: list[dict]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for crop in crops:
        all_chunks.extend(chunk_crop(crop))
    return all_chunks