from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from typing import Any

import requests

from crop_app.locations import INDIAN_STATES_AND_UTS

GEONAMES_DOWNLOAD_URL = "https://download.geonames.org/export/dump/"
GEONAMES_ATTRIBUTION_URL = "https://www.geonames.org/"
GEONAMES_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
PLACE_FEATURE_CODES = {
    "PPL",
    "PPLA",
    "PPLA2",
    "PPLA3",
    "PPLA4",
    "PPLA5",
    "PPLC",
    "PPLG",
    "PPLS",
}
ADMINISTRATIVE_SEAT_CODES = PLACE_FEATURE_CODES.difference({"PPL"})


class LocationCatalogError(RuntimeError):
    """A clear error while downloading or parsing location hierarchy data."""


def _normalized_name(name: str) -> str:
    return " ".join(name.split()).casefold()


def _fetch_text(session: requests.Session, filename: str, timeout: float) -> str:
    response = session.get(f"{GEONAMES_DOWNLOAD_URL}{filename}", timeout=timeout)
    response.raise_for_status()
    return response.content.decode("utf-8-sig")


def load_india_location_catalog(timeout: float = 45.0) -> dict[str, Any]:
    """Load GeoNames administrative divisions and populated places for India."""
    try:
        with requests.Session() as session:
            admin1_text = _fetch_text(session, "admin1CodesASCII.txt", timeout)
            admin2_text = _fetch_text(session, "admin2Codes.txt", timeout)
            response = session.get(f"{GEONAMES_DOWNLOAD_URL}IN.zip", timeout=timeout)
            response.raise_for_status()
            places_archive = response.content
            with zipfile.ZipFile(io.BytesIO(places_archive)) as archive:
                place_filename = next(
                    name for name in archive.namelist() if name.endswith("IN.txt")
                )
    except requests.RequestException as exc:
        raise LocationCatalogError(
            "The GeoNames location directory is unavailable. Check your connection "
            "and try again."
        ) from exc
    except (OSError, UnicodeError, zipfile.BadZipFile, StopIteration) as exc:
        raise LocationCatalogError(
            "The GeoNames location directory returned invalid or incomplete data."
        ) from exc

    canonical_by_name = {
        _normalized_name(state): state for state, _capital in INDIAN_STATES_AND_UTS
    }
    canonical_by_name["andaman and nicobar"] = "Andaman and Nicobar Islands"

    states_by_code: dict[str, str] = {}
    for row in csv.reader(io.StringIO(admin1_text), delimiter="\t"):
        if len(row) < 2 or not row[0].startswith("IN."):
            continue
        name = canonical_by_name.get(_normalized_name(row[1]))
        if name:
            states_by_code[row[0]] = name

    if set(states_by_code.values()) != set(canonical_by_name.values()):
        missing_states = sorted(
            set(canonical_by_name.values()) - set(states_by_code.values())
        )
        raise LocationCatalogError(
            "GeoNames did not provide all Indian states and union territories: "
            + ", ".join(missing_states)
        )

    districts_by_state: dict[str, dict[str, str]] = defaultdict(dict)
    for row in csv.reader(io.StringIO(admin2_text), delimiter="\t"):
        if len(row) < 2 or not row[0].startswith("IN."):
            continue
        parts = row[0].split(".")
        if len(parts) != 3:
            continue
        state_code, district_id = f"{parts[0]}.{parts[1]}", parts[2]
        if state_code in states_by_code:
            districts_by_state[state_code][district_id] = row[1]

    cities_by_district: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    try:
        with zipfile.ZipFile(io.BytesIO(places_archive)) as archive:
            with archive.open(place_filename) as place_file:
                place_rows = csv.reader(
                    io.TextIOWrapper(place_file, encoding="utf-8"),
                    delimiter="\t",
                )
                for row in place_rows:
                    if (
                        len(row) < 15
                        or row[6] != "P"
                        or row[7] not in PLACE_FEATURE_CODES
                    ):
                        continue
                    if not row[8] or not row[10] or not row[11]:
                        continue
                    state_code = f"{row[8]}.{row[10]}"
                    district_id = row[11]
                    if state_code not in states_by_code:
                        continue
                    try:
                        latitude = float(row[4])
                        longitude = float(row[5])
                        population = int(row[14] or 0)
                    except ValueError:
                        continue
                    if row[7] not in ADMINISTRATIVE_SEAT_CODES and population < 1000:
                        continue

                    key = f"{state_code}.{district_id}"
                    if district_id not in districts_by_state.get(state_code, {}):
                        continue
                    place = {
                        "name": row[1],
                        "latitude": latitude,
                        "longitude": longitude,
                        "population": population,
                        "feature_code": row[7],
                        "geoname_id": row[0],
                    }
                    cities_by_district[key][row[0]] = place
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise LocationCatalogError(
            "The GeoNames India place data could not be read."
        ) from exc

    districts: dict[str, list[dict[str, str]]] = {}
    for state_code, entries in districts_by_state.items():
        districts[state_code] = [
            {"name": name, "code": f"{state_code}.{district_id}"}
            for district_id, name in sorted(
                entries.items(), key=lambda entry: entry[1].casefold()
            )
        ]

    return {
        "states": [
            {"name": name, "code": code}
            for code, name in sorted(
                states_by_code.items(), key=lambda entry: entry[1].casefold()
            )
        ],
        "districts": districts,
        "cities": {
            district_code: sorted(
                entries.values(),
                key=lambda place: (
                    place["name"].casefold(),
                    -place["population"],
                    place["geoname_id"],
                ),
            )
            for district_code, entries in cities_by_district.items()
        },
        "source": "GeoNames India administrative and populated-place dump",
        "source_url": GEONAMES_ATTRIBUTION_URL,
        "license_url": GEONAMES_LICENSE_URL,
        "places_note": (
            "Populated places with a reported population of at least 1,000, plus "
            "GeoNames administrative seats. Smaller places or places without a "
            "reported population may not appear."
        ),
    }
