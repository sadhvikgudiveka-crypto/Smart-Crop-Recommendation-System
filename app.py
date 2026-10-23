from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from crop_app.crop_reference import (
    CROP_DISPLAY_NAMES,
    CropReferenceError,
    get_crop_reference,
)
from crop_app.geography import LocationCatalogError, load_india_location_catalog
from crop_app.inference import explain_top_crop, rank_crops
from crop_app.soil import SoilEstimateError, get_soil_estimates
from crop_app.validation import InputValidationError, validate_soil_inputs
from crop_app.weather import (
    WEATHER_ATTRIBUTION_URL,
    WeatherServiceError,
    get_current_weather,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "crop_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"

st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
)
def _timeout_seconds() -> float:
    value: Any = os.environ.get("WEATHER_TIMEOUT_SECONDS", "8")
    try:
        secret_value = st.secrets.get("weather_timeout_seconds")
    except StreamlitSecretNotFoundError:
        secret_value = None
    if secret_value is not None:
        value = secret_value
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "weather_timeout_seconds must be a positive number in Streamlit secrets "
            "or WEATHER_TIMEOUT_SECONDS."
        ) from exc
    if timeout <= 0:
        raise ValueError("Weather service timeout must be positive.")
    return timeout


@st.cache_resource
def _load_artifacts(model_path: str, metadata_path: str) -> tuple[Any, dict[str, Any]]:
    with Path(metadata_path).open(encoding="utf-8") as file:
        metadata = json.load(file)
    return joblib.load(model_path), metadata


@st.cache_data(ttl=86400, max_entries=1, show_spinner=False)
def _cached_location_catalog() -> dict[str, Any]:
    return load_india_location_catalog()


@st.cache_data(ttl=86400, max_entries=512, show_spinner=False)
def _cached_soil_estimates(latitude: float, longitude: float) -> dict[str, Any]:
    return get_soil_estimates(latitude, longitude)


@st.cache_data(ttl=600, max_entries=256, show_spinner=False)
def _cached_current_weather(
    latitude: float, longitude: float, timeout: float
) -> dict[str, Any]:
    return get_current_weather(
        {"latitude": latitude, "longitude": longitude}, timeout=timeout
    )


@st.cache_data(ttl=86400, max_entries=128, show_spinner=False)
def _cached_crop_reference(crop_label: str) -> dict[str, str]:
    return get_crop_reference(crop_label)


def _rank_ordinal(rank: int) -> str:
    if 10 <= rank % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
    return f"{rank}{suffix}"


def _clear_selected_location_data() -> None:
    for key in (
        "selected_location",
        "selected_location_name",
        "weather_data",
        "soil_estimates",
    ):
        st.session_state.pop(key, None)


with st.container(border=True):
    st.title("🌾 Smart Crop Recommendation System")
    st.caption("Explore how soil-feature patterns relate to crop labels.")
st.warning(
    "Educational prototype only — recommendations reflect patterns in one training "
    "dataset and are not definitive farming advice. They do not replace soil testing "
    "or guidance from local agricultural experts."
)

with st.expander("How to interpret this prototype"):
    st.markdown(
        """
        - Crop rankings use only the entered N, P, K, and pH values. Live weather is
          shown as context and **does not affect the ranking**.
        - N/P/K use the dataset's numeric scale; its publisher does not document their
          units. Values outside the model's training ranges are rejected rather than
          extrapolated.
        - Scores are calibrated model probabilities among the dataset's crop labels,
          not probabilities of a successful harvest or measures of agronomic suitability.
        - SoilGrids provides modelled point estimates for some soil properties; they
          are not measurements and do not cover every location. Enter measured values
          when available. Estimates are not substituted for model inputs when their
          units or meanings differ.
        """
    )

st.header("1. Location and live weather")
st.caption(
    "Choose an Indian country, state/union territory, district, and city/town. "
    "The selected place's coordinates are used for live weather and SoilGrids "
    "point estimates. Weather remains contextual and does not affect the soil-only model."
)

catalog: dict[str, Any] | None
try:
    with st.spinner("Loading India's location directory…"):
        catalog = _cached_location_catalog()
except LocationCatalogError as exc:
    catalog = None
    st.error(f"Could not load the location directory: {exc}")

selected_location: dict[str, Any] | None = None
selected_location_name: str | None = None
if catalog:
    country = st.selectbox("Country", ["India"], index=0, key="country_choice")
    if country == "India":
        state_names = [item["name"] for item in catalog["states"]]
        selected_state = st.selectbox(
            "State or union territory",
            state_names,
            index=None,
            placeholder="Select a state or union territory",
            key="state_choice",
        )
        if selected_state:
            state = next(
                item for item in catalog["states"] if item["name"] == selected_state
            )
            district_options = catalog["districts"].get(state["code"], [])
            if district_options:
                selected_district_code = st.selectbox(
                    "District",
                    [item["code"] for item in district_options],
                    index=None,
                    format_func=lambda code: next(
                        item["name"]
                        for item in district_options
                        if item["code"] == code
                    ),
                    placeholder="Select a district",
                    key=f"district_choice_{state['code']}",
                )
            else:
                selected_district_code = None
                st.info("No districts are listed for that selection in the directory.")

            if selected_district_code:
                city_options = catalog["cities"].get(selected_district_code, [])
                if city_options:
                    selected_city_index = st.selectbox(
                        "City or town",
                        range(len(city_options)),
                        index=None,
                        format_func=lambda index: (
                            f"{city_options[index]['name']} "
                            f"(population {city_options[index]['population']:,})"
                            if city_options[index]["population"] > 0
                            else city_options[index]["name"]
                        ),
                        placeholder="Select a city or town",
                        key=f"city_choice_{selected_district_code}",
                    )
                    if selected_city_index is not None:
                        selected_location = city_options[selected_city_index]
                        selected_location_name = (
                            f"{selected_location['name']}, "
                            f"{next(item['name'] for item in district_options if item['code'] == selected_district_code)}, "
                            f"{selected_state}, India"
                        )
                else:
                    st.info(
                        "No listed city/town matches this district. The directory "
                        "may omit smaller places; choose another district or enter "
                        "soil-test values without a location."
                    )

if selected_location:
    location_key = (
        float(selected_location["latitude"]),
        float(selected_location["longitude"]),
    )
    previous_key = st.session_state.get("selected_location_key")
    if previous_key != location_key:
        _clear_selected_location_data()
        st.session_state.selected_location_key = location_key
    st.session_state.selected_location = selected_location
    st.session_state.selected_location_name = selected_location_name
    st.info(
        f"Selected location: **{selected_location_name}** "
        f"({location_key[0]:.4f}, {location_key[1]:.4f})"
    )
    weather_error = None
    soil_error = None
    if "weather_data" not in st.session_state:
        try:
            with st.spinner("Loading current weather for the selected place…"):
                st.session_state.weather_data = _cached_current_weather(
                    location_key[0], location_key[1], _timeout_seconds()
                )
        except (WeatherServiceError, ValueError) as exc:
            weather_error = str(exc)
    if "soil_estimates" not in st.session_state:
        try:
            with st.spinner("Checking for SoilGrids estimates…"):
                st.session_state.soil_estimates = _cached_soil_estimates(*location_key)
        except SoilEstimateError as exc:
            soil_error = str(exc)
    if weather_error:
        st.warning(f"Weather is unavailable: {weather_error}")
    if soil_error:
        st.warning(soil_error)
else:
    if st.session_state.get("selected_location_key") is not None:
        _clear_selected_location_data()
        st.session_state.pop("selected_location_key", None)
    st.info("Choose each location level to load weather and check for soil estimates.")
st.caption(
    "Location directory: [GeoNames](https://www.geonames.org/), "
    "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Records may be "
    "incomplete or out of date."
)

selected_location_name = st.session_state.get("selected_location_name")
weather_data = st.session_state.get("weather_data")
soil_estimates = st.session_state.get("soil_estimates")
if weather_data:
    st.caption(
        f"Current model conditions at {weather_data['time']} "
        f"({weather_data['timezone']}) for the selected city/town."
    )
    weather_columns = st.columns(3)
    weather_columns[0].metric(
        "Temperature",
        f"{weather_data['temperature']:.1f} {weather_data['temperature_unit']}",
    )
    weather_columns[1].metric(
        "Relative humidity",
        f"{weather_data['humidity']:.0f} {weather_data['humidity_unit']}",
    )
    weather_columns[2].metric(
        "Precipitation",
        f"{weather_data['precipitation']:.1f} "
        f"{weather_data['precipitation_unit']}",
    )
    st.caption(
        f"Weather data by [Open-Meteo]({WEATHER_ATTRIBUTION_URL}). These current "
        "conditions are displayed for context; they are not model inputs."
    )

st.header("2. Soil values")
if soil_estimates:
    st.markdown(
        "**Location estimates (not measurements):** ISRIC SoilGrids 2.0 modelled "
        f"point data at {soil_estimates['depth']}. {soil_estimates['point_note']} "
        f"{soil_estimates['period']}"
    )
    estimate_columns = st.columns(4)
    displayed_estimates = [
        ("Total nitrogen", soil_estimates["nitrogen"]),
        ("Phosphorus", soil_estimates["phosphorus"]),
        ("Potassium", soil_estimates["potassium"]),
        ("Soil pH", soil_estimates["ph"]),
    ]
    for column, (label, estimate) in zip(estimate_columns, displayed_estimates):
        value_text = (
            f"{estimate['value']:.2f} {estimate['unit']}"
            if estimate and estimate["value"] is not None
            else "Unavailable"
        )
        if estimate is None:
            estimate_note = "No value or unit is provided by this source."
        elif estimate["value"] is None:
            estimate_note = (
                f"No estimate was returned for this point; expected unit: "
                f"{estimate['unit']}."
            )
        else:
            estimate_note = f"Unit: {estimate['unit']}."
        column.metric(label, value_text)
        column.caption(
            f"{estimate_note} [{soil_estimates['source']}]"
            f"({soil_estimates['source_url']}); {soil_estimates['depth']}; "
            f"{soil_estimates['period']}"
        )
    st.caption(
        f"{soil_estimates['not_available_note']} The total-nitrogen estimate is "
        "reported in g/kg, while the crop dataset's N feature has an undocumented "
        "scale; it is shown for context only and is not used by the model."
    )
    st.info(
        "Enter N, P, and K from a soil test using the model dataset's numeric scale. "
        "If a SoilGrids pH estimate is available, it is prefilled as an editable "
        "starting value; confirm or replace it with a soil-test result."
    )
else:
    st.caption(
        "Enter soil measurements. If location-based estimates are unavailable, "
        "the model still requires N, P, K, and pH values from a soil test."
    )
st.caption(
    "N/P/K inputs must follow the training dataset's numeric scale, whose units "
    "are not specified by its publisher. Use soil-test results; do not enter the "
    "SoilGrids total-N g/kg value as model N."
)
soil_context_key = (
    selected_location["geoname_id"] if selected_location else "manual"
)
with st.form("soil_inputs"):
    input_columns = st.columns(4)
    nitrogen = input_columns[0].number_input(
        "Nitrogen (N, dataset scale)",
        min_value=0.0,
        value=None,
        placeholder="Soil-test value",
        key=f"soil_nitrogen_{soil_context_key}",
    )
    phosphorus = input_columns[1].number_input(
        "Phosphorus (P, dataset scale)",
        min_value=0.0,
        value=None,
        placeholder="Soil-test value",
        key=f"soil_phosphorus_{soil_context_key}",
    )
    potassium = input_columns[2].number_input(
        "Potassium (K, dataset scale)",
        min_value=0.0,
        value=None,
        placeholder="Soil-test value",
        key=f"soil_potassium_{soil_context_key}",
    )
    estimated_ph = (
        soil_estimates["ph"]["value"]
        if soil_estimates and soil_estimates["ph"]["value"] is not None
        else None
    )
    ph = input_columns[3].number_input(
        "Soil pH (editable; estimate if available)",
        min_value=0.0,
        max_value=14.0,
        value=estimated_ph,
        placeholder="Soil-test value (0–14)",
        key=f"soil_ph_{soil_context_key}",
    )
    recommend_submitted = st.form_submit_button("Recommend crops")

if recommend_submitted:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        st.error(
            "The trained model is not installed yet. Follow the README steps to download "
            "the dataset and run `python scripts/train_model.py`."
        )
    else:
        try:
            model, metadata = _load_artifacts(str(MODEL_PATH), str(METADATA_PATH))
            values = validate_soil_inputs(
                {"N": nitrogen, "P": phosphorus, "K": potassium, "ph": ph}, metadata
            )
            rankings = rank_crops(model, values, top_k=3)
            explanations = explain_top_crop(
                model, values, metadata["training_medians"]
            )
            if not selected_location_name:
                st.caption(
                    "No location is selected. Rankings use soil values only; "
                    "location and weather are not required for this model."
                )
            else:
                st.caption(
                    f"Location: {selected_location_name}. Current weather is shown "
                    "above but is not a feature in this soil-only model."
                )
            st.header("3. Ranked dataset matches")
            st.caption(metadata["probability_note"])
            for rank, item in enumerate(rankings, start=1):
                crop_label = str(item["crop"])
                crop_name = CROP_DISPLAY_NAMES.get(
                    crop_label.casefold(), crop_label.replace("_", " ").title()
                )
                with st.container(border=True):
                    st.subheader(f"{_rank_ordinal(rank)} — {crop_name}")
                    image_column, details_column = st.columns([1, 2])
                    try:
                        with st.spinner(f"Loading {crop_name} image and reference…"):
                            crop_reference = _cached_crop_reference(crop_label)
                    except CropReferenceError as exc:
                        crop_reference = None
                        details_column.warning(str(exc))

                    if crop_reference and crop_reference["image_url"]:
                        image_column.image(
                            crop_reference["image_url"],
                            alt=f"{crop_name} crop",
                            width="stretch",
                        )
                        image_column.text(
                            f"Image credit: {crop_reference['image_artist']}. "
                            f"License: {crop_reference['image_license']}."
                        )
                        image_column.caption(
                            f"[Image source and license details]"
                            f"({crop_reference['image_page_url']})"
                        )
                    else:
                        image_column.info(
                            crop_reference.get("image_error")
                            if crop_reference
                            else "No crop image with verified creator and license "
                            "details is available right now."
                        )

                    details_column.metric(
                        "Calibrated model score",
                        f"{float(item['probability']):.1%}",
                    )
                    if crop_reference:
                        details_column.text(crop_reference["description"])
                        details_column.caption(
                            f"Crop overview: [{crop_reference['article_title']}]"
                            f"({crop_reference['article_url']}) · "
                            "Wikimedia Foundation, CC BY-SA 4.0."
                        )

                    profile = metadata.get("crop_profiles", {}).get(crop_label)
                    if profile:
                        profile_parts: list[str] = []
                        if "temperature" in profile:
                            profile_parts.append(
                                f"temperature {float(profile['temperature']):.1f} °C"
                            )
                        if "rainfall" in profile:
                            profile_parts.append(
                                f"rainfall {float(profile['rainfall']):.1f} mm "
                                "(period unspecified)"
                            )
                        if "ph" in profile:
                            profile_parts.append(f"soil pH {float(profile['ph']):.2f}")
                        if all(feature in profile for feature in ("N", "P", "K")):
                            profile_parts.append(
                                "dataset-scale N/P/K "
                                f"{float(profile['N']):g}/"
                                f"{float(profile['P']):g}/"
                                f"{float(profile['K']):g}"
                            )
                        details_column.markdown(
                            "**Training-dataset profile (median example values):** "
                            + "; ".join(profile_parts)
                            + "."
                        )
                        details_column.caption(
                            "These figures describe records associated with this crop "
                            f"label in the source dataset (median of "
                            f"{int(profile.get('sample_count', 0))} records). They are "
                            "not optimal growing ranges or fertiliser recommendations. "
                            "Dataset source: "
                            "[Crop Recommendation Dataset]"
                            "(https://www.kaggle.com/datasets/atharvaingle/"
                            "crop-recommendation-dataset)."
                        )
                    else:
                        details_column.info(
                            "No crop-specific reference profile is present in the "
                            "model metadata. Retrain using the README steps to include it."
                        )

            st.subheader(f"Factors for the top-ranked crop: {rankings[0]['crop']}")
            st.caption(metadata["explanation_note"])
            for factor in explanations:
                change = float(factor["change"])
                if abs(change) < 1e-12:
                    effect = "does not change"
                else:
                    direction = "raises" if change > 0 else "lowers"
                    effect = f"{direction} the model's probability"
                st.write(
                    f"- **{factor['feature']}** = {factor['submitted_value']:g}: "
                    f"compared with its training median "
                    f"({factor['training_median']:g}), this {effect} for "
                    f"{rankings[0]['crop']} by {abs(change):.1%}."
                )

            st.subheader("Values used")
            st.dataframe(
                [{"Feature": key, "Value": value} for key, value in values.items()],
                hide_index=True,
                width="stretch",
            )
            if st.session_state.get("weather_data"):
                st.caption(
                    "Weather shown above was fetched for the selected location but was "
                    "not used in these rankings."
                )
        except InputValidationError as exc:
            st.error(str(exc))
        except (KeyError, OSError, ValueError, TypeError) as exc:
            st.error(f"Could not use the trained model or its metadata: {exc}")

st.divider()
st.caption(
    "Dataset: [Crop Recommendation Dataset on Kaggle]"
    "(https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset), "
    "Apache 2.0 as declared by its publisher. See the README for provenance and limitations."
)
