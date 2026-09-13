"""Fetch the crop registry from GraphQL and persist it locally.

Configure the endpoint in the repository's ``.env`` file before use::

  GRAPHQL_ENDPOINT=https://your-crop-service.example/graphql

Use it from the command line::

  python -m app.ingestion.fetch_crops
  python -m app.ingestion.fetch_crops --updated-within-days 7

Or call it from Python::

  from app.ingestion.fetch_crops import fetch_crops

  payload = fetch_crops(updated_within_days=7)

After a successful request, the complete response is written to the path in
``CROP_REGISTRY_PATH`` (``data/crops.json`` by default). The existing file is
left unchanged when the request fails or GraphQL returns errors.
"""
import argparse
import json
import os
import tempfile
from typing import Optional

import requests

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.pipeline.registry import get_known_crops

logger = get_logger(__name__)

QUERY = """
query getAllCropsFullDetails($updated_within_days: Int) {
  getAllCropsFullDetails(updated_within_days: $updated_within_days) {
    result_code
    status
    count
    rows {
      id creator_id crop_name crop_bangla_name scientific_name crop_family general_info
      average_production is_verified
      harvest { id crop_id description }
      intercultural { id crop_id description }
      irrigation { id crop_id description }
      landPreparation { id crop_id description }
      fertilizer { fertilizer }
      seed { id crop_id treatment showing_method time_showing seedbed seed_rate }
      climate {
        id crop_id general_info climate_temperature_start climate_temperature_end
        climate_rainfall_start climate_rainfall_end climate_ph_start climate_ph_end
        climate_humidity climate_humidity_end climate_ec_start climate_ec_end
        land_type soil_texture salinity_start salinity_end
      }
      cropInfestationGuidelines {
        id crop_id infestation_id application_guide infestation_slug herbicide_message
      }
      cropAdditionalCostInfo { id crop_id cost_type amount unit unitInfo { id unit_name unit_code } }
      herbicide {
        id crop_id pesticide_name trade_name generic_name company_name company_id
        company { id name }
        application_dose application_dose_unit application_dose_unit_name
        applicationDoseUnitInfo { id unit_name unit_code }
        pesticide_amount pesticide_amount_unit pesticideAmountUnitInfo { id unit_name unit_code }
        rating price price_unit priority application_guide is_deleted is_verified
      }
      pesticide {
        id crop_id priority infestation_id disease_type disease_name damage_control
        control_measure is_verified
        chemical {
          id pesticide_id pesticide_name trade_name generic_name company_name company_id
          company { id name }
          application_dose application_dose_unit application_dose_unit_name
          applicationDoseUnitInfo { id unit_name unit_code }
          pesticide_amount pesticide_amount_unit pesticideAmountUnitInfo { id unit_name unit_code }
          rating price priority packetSizeAndPrice { size price }
          application_guide is_deleted is_verified
        }
      }
      variety {
        id crop_id variety_name variety_duration variety_yield avg_expected_yield yield_up yield_low
        company_name company_id company { id name }
        seed_rate rating price duration_start duration_end seed_rate_unit seed_rate_unit_name
        seedRateUnit { id unit_name unit_code }
        production production_unit productionUnit { id unit_name unit_code }
        special_character time_showing
        seasons { variety_id season_id season }
        variety_seed {
          id variety_id crops_name company_name per_shotok per_shotok_unit
          potential_yield_text potential_yield_value potential_yield_unit rating
        }
        is_deleted is_verified
      }
      crop_category {
        id category_name category_code
        feature { id feature_name feature_desc feature_price }
      }
    }
  }
}
"""


def _write_registry(payload: dict, path: str) -> None:
    """Write beside the destination, then replace it to avoid partial JSON."""
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=os.path.dirname(destination), delete=False) as file:
        temporary_path = file.name
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary_path, destination)


def fetch_crops(updated_within_days: Optional[int] = None) -> dict:
    """Fetch crops and update the configured crop registry.

    Args:
        updated_within_days: If provided, asks GraphQL for crops changed in
            this many previous days. ``None`` requests the complete registry.

    Returns:
        The complete decoded GraphQL response that was written to the registry.

    Raises:
        ValueError: If the endpoint is not configured or the response shape is
            not the expected GraphQL crop response.
        RuntimeError: If GraphQL returns an ``errors`` array.
        requests.RequestException: If the HTTP request fails.
    """
    settings = get_settings()
    if not settings.graphql_endpoint:
        raise ValueError("GRAPHQL_ENDPOINT must be configured before fetching crops")

    response = requests.post(
        settings.graphql_endpoint,
        json={"query": QUERY, "variables": {"updated_within_days": updated_within_days}},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("GraphQL response must be a JSON object")
    if payload.get("errors"):
        raise RuntimeError(f"GraphQL returned errors: {payload['errors']}")
    rows = payload.get("data", {}).get("getAllCropsFullDetails", {}).get("rows")
    if not isinstance(rows, list):
        raise ValueError("GraphQL response is missing data.getAllCropsFullDetails.rows")

    _write_registry(payload, settings.crop_registry_path)
    logger.info("Updated crop registry %s with %d crop(s)", settings.crop_registry_path, len(rows))
    get_known_crops.cache_clear()
    return payload


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Fetch the crop registry from GraphQL")
    parser.add_argument("--updated-within-days", type=int)
    args = parser.parse_args()
    fetch_crops(args.updated_within_days)


if __name__ == "__main__":
    main()