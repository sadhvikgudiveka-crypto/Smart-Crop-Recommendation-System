from __future__ import annotations

import io
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from crop_app.locations import INDIAN_STATES_AND_UTS

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _geonames_responses() -> dict[str, bytes]:
    state_lines = []
    assam_code = ""
    for index, (name, _capital) in enumerate(INDIAN_STATES_AND_UTS, start=1):
        code = f"IN.{index:02}"
        state_lines.append(f"{code}\t{name}\t{name}\t{index}")
        if name == "Assam":
            assam_code = code

    admin2 = f"{assam_code}.01\tKamrup Metropolitan\tKamrup Metropolitan\t1\n"
    fields = [""] * 19
    fields[0:8] = [
        "12345",
        "Guwahati",
        "Guwahati",
        "",
        "26.1445",
        "91.7362",
        "P",
        "PPLA",
    ]
    fields[8] = "IN"
    fields[10] = assam_code.rsplit(".", 1)[1]
    fields[11] = "01"
    fields[14] = "957352"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("IN.txt", "\t".join(fields) + "\n")
    return {
        "admin1CodesASCII.txt": "\n".join(state_lines).encode(),
        "admin2Codes.txt": admin2.encode(),
        "IN.zip": archive.getvalue(),
    }


def _response(payload: dict) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


class CascadingLocationTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_data.clear()
        self.catalog_files = _geonames_responses()

    def _session_get(self, _session: requests.Session, url: str, **_kwargs: object) -> Mock:
        filename = url.rsplit("/", 1)[-1]
        response = Mock()
        response.raise_for_status.return_value = None
        response.content = self.catalog_files[filename]
        return response

    @staticmethod
    def _api_get(url: str, **_kwargs: object) -> Mock:
        if "forecast" in url:
            return _response(
                {
                    "current": {
                        "time": "2026-10-23T10:00",
                        "temperature_2m": 25.0,
                        "relative_humidity_2m": 68,
                        "precipitation": 0.0,
                    },
                    "current_units": {
                        "temperature_2m": "°C",
                        "relative_humidity_2m": "%",
                        "precipitation": "mm",
                    },
                    "timezone": "Asia/Kolkata",
                }
            )
        return _response(
            {
                "properties": {
                    "layers": [
                        {
                            "name": "nitrogen",
                            "unit_measure": {"d_factor": 100},
                            "depths": [{"label": "0-5cm", "values": {"mean": 127}}],
                        },
                        {
                            "name": "phh2o",
                            "unit_measure": {"d_factor": 10},
                            "depths": [{"label": "0-5cm", "values": {"mean": 76}}],
                        },
                    ]
                }
            }
        )

    def test_cascade_uses_selected_place_coordinates_and_keeps_recommendations(self) -> None:
        with (
            patch.object(requests.Session, "get", autospec=True, side_effect=self._session_get),
            patch("requests.get", side_effect=self._api_get) as api_get,
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
            self.assertEqual(app.selectbox[0].label, "Country")
            self.assertIn("India", app.selectbox[0].options)

            state_box = next(item for item in app.selectbox if item.label == "State or union territory")
            state_box.set_value("Assam")
            app.run()
            district_box = next(item for item in app.selectbox if item.label == "District")
            district_box.set_value(district_box.options[0])
            app.run()
            city_box = next(item for item in app.selectbox if item.label == "City or town")
            city_box.set_value(0)
            app.run()

            self.assertTrue(any("Guwahati, Kamrup Metropolitan, Assam, India" in item.value for item in app.info))
            self.assertEqual([item.label for item in app.metric[:3]], [
                "Temperature",
                "Relative humidity",
                "Precipitation",
            ])
            self.assertTrue(any(item.label == "Total nitrogen" and "1.27 g/kg" in item.value for item in app.metric))
            self.assertTrue(any(item.label == "Phosphorus" and "Unavailable" in item.value for item in app.metric))
            self.assertTrue(any(item.label == "Potassium" and "Unavailable" in item.value for item in app.metric))
            self.assertTrue(any(item.label == "Soil pH" and "7.60 pH" in item.value for item in app.metric))
            self.assertEqual([item.value for item in app.number_input[:3]], [None, None, None])
            self.assertEqual(app.number_input[3].value, 7.6)

            for field, value in zip(app.number_input, (50.0, 50.0, 50.0, 6.5)):
                field.set_value(value)
            next(item for item in app.button if item.label == "Recommend crops").click()
            app.run()
            if (
                (APP_PATH.parent / "artifacts" / "crop_model.joblib").exists()
                and (APP_PATH.parent / "artifacts" / "model_metadata.json").exists()
            ):
                self.assertIn(
                    "3. Ranked dataset matches",
                    [item.value for item in app.header],
                )
                score_metrics = [
                    item
                    for item in app.metric
                    if item.label == "Calibrated model score"
                ]
                self.assertEqual(len(score_metrics), 3)
            else:
                self.assertTrue(
                    any(
                        "trained model is not installed yet" in item.value
                        for item in app.error
                    )
                )

            weather_call = next(call for call in api_get.call_args_list if "forecast" in call.args[0])
            soil_call = next(call for call in api_get.call_args_list if "soilgrids" in call.args[0])
            self.assertEqual(weather_call.kwargs["params"]["latitude"], 26.1445)
            self.assertEqual(weather_call.kwargs["params"]["longitude"], 91.7362)
            self.assertEqual(soil_call.kwargs["params"][0], ("lon", 91.7362))
            self.assertEqual(soil_call.kwargs["params"][1], ("lat", 26.1445))

    def test_catalog_loading_failure_is_actionable(self) -> None:
        with patch.object(
            requests.Session,
            "get",
            autospec=True,
            side_effect=requests.ConnectionError("offline"),
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=45).run()
        self.assertTrue(any("location directory is unavailable" in item.value for item in app.error))


if __name__ == "__main__":
    unittest.main()
