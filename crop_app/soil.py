from __future__ import annotations

from typing import Any

import requests

SOILGRIDS_QUERY_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOILGRIDS_DOCS_URL = "https://rest.isric.org/soilgrids/v2.0/docs"
SOILGRIDS_PORTAL_URL = "https://soilgrids.org/"


class SoilEstimateError(RuntimeError):
    """An actionable SoilGrids request or response error."""


def get_soil_estimates(
    latitude: float, longitude: float, timeout: float = 12.0
) -> dict[str, Any]:
    """Return SoilGrids topsoil point estimates where that API has data."""
    try:
        response = requests.get(
            SOILGRIDS_QUERY_URL,
            params=[
                ("lon", longitude),
                ("lat", latitude),
                ("property", "nitrogen"),
                ("property", "phh2o"),
                ("depth", "0-5cm"),
                ("value", "mean"),
            ],
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise SoilEstimateError(
            "SoilGrids lookup timed out. You can still enter soil-test values."
        ) from exc
    except requests.RequestException as exc:
        raise SoilEstimateError(
            "SoilGrids is unavailable right now. You can still enter soil-test values."
        ) from exc
    except ValueError as exc:
        raise SoilEstimateError(
            "SoilGrids returned an unreadable response. You can enter soil-test values."
        ) from exc

    try:
        layers = {
            layer["name"]: layer for layer in payload["properties"]["layers"]
        }
        nitrogen = _property_estimate(layers.get("nitrogen"), "nitrogen")
        ph = _property_estimate(layers.get("phh2o"), "ph")
    except (KeyError, TypeError, ValueError) as exc:
        raise SoilEstimateError(
            "SoilGrids returned incomplete data. You can enter soil-test values."
        ) from exc

    return {
        "nitrogen": nitrogen,
        "phosphorus": None,
        "potassium": None,
        "ph": ph,
        "depth": "0–5 cm topsoil",
        "period": (
            "SoilGrids 2.0 modelled map; no observation date or temporal averaging "
            "period is supplied for a point estimate."
        ),
        "source": "ISRIC SoilGrids 2.0",
        "source_url": SOILGRIDS_PORTAL_URL,
        "api_docs_url": SOILGRIDS_DOCS_URL,
        "not_available_note": (
            "SoilGrids v2.0's point API does not provide plant-available phosphorus "
            "or potassium."
        ),
        "point_note": (
            "These are modelled estimates at the selected city/town coordinates, "
            "not a spatial average for its district and not a soil-test result."
        ),
    }


def _property_estimate(layer: Any, property_name: str) -> dict[str, Any]:
    if not isinstance(layer, dict):
        return {
            "value": None,
            "unit": "g/kg" if property_name == "nitrogen" else "pH",
        }
    depth = next(
        (
            item
            for item in layer.get("depths", [])
            if item.get("label") == "0-5cm"
        ),
        None,
    )
    raw_value = depth.get("values", {}).get("mean") if depth else None
    if raw_value is None:
        return {
            "value": None,
            "unit": "g/kg" if property_name == "nitrogen" else "pH",
        }

    divisor = float(layer["unit_measure"]["d_factor"])
    if divisor <= 0:
        raise ValueError(f"SoilGrids returned an invalid scale for {property_name}.")
    return {
        "value": float(raw_value) / divisor,
        "unit": "g/kg" if property_name == "nitrogen" else "pH",
    }
