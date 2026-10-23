from __future__ import annotations

from typing import Any

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ATTRIBUTION_URL = "https://open-meteo.com/"


class WeatherServiceError(RuntimeError):
    """An actionable error from location search or the weather provider."""


def search_locations(
    query: str, timeout: float = 8.0, country_code: str | None = None
) -> list[dict[str, Any]]:
    cleaned_query = query.strip()
    if len(cleaned_query) < 2:
        raise WeatherServiceError("Enter at least two characters for a location.")

    try:
        params: dict[str, Any] = {
            "name": cleaned_query,
            "count": 10,
            "language": "en",
            "format": "json",
        }
        if country_code:
            params["countryCode"] = country_code
        response = requests.get(GEOCODING_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise WeatherServiceError("Location search timed out. Please try again.") from exc
    except requests.RequestException as exc:
        raise WeatherServiceError(
            "The location service is unavailable. Check your connection and try again."
        ) from exc
    except ValueError as exc:
        raise WeatherServiceError("The location service returned an unreadable response.") from exc

    results = payload.get("results") if isinstance(payload, dict) else None
    if not results:
        raise WeatherServiceError(
            "No matching location was found. Try adding the state, region, or country."
        )
    return results


def get_current_weather(
    location: dict[str, Any], timeout: float = 8.0
) -> dict[str, Any]:
    try:
        latitude = float(location["latitude"])
        longitude = float(location["longitude"])
        response = requests.get(
            WEATHER_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation",
                "timezone": "auto",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        current = payload["current"]
        units = payload["current_units"]
        return {
            "time": current["time"],
            "temperature": float(current["temperature_2m"]),
            "temperature_unit": units["temperature_2m"],
            "humidity": float(current["relative_humidity_2m"]),
            "humidity_unit": units["relative_humidity_2m"],
            "precipitation": float(current["precipitation"]),
            "precipitation_unit": units["precipitation"],
            "timezone": payload.get("timezone", "local time"),
        }
    except requests.Timeout as exc:
        raise WeatherServiceError("Weather lookup timed out. Please try again.") from exc
    except requests.RequestException as exc:
        raise WeatherServiceError(
            "The weather service is unavailable. Check your connection and try again."
        ) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherServiceError(
            "The weather service returned incomplete or invalid data for this location."
        ) from exc
