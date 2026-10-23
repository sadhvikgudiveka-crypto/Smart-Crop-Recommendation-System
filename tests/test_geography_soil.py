from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from crop_app.soil import SoilEstimateError, get_soil_estimates


class SoilEstimateTests(unittest.TestCase):
    @staticmethod
    def _response(payload: dict) -> Mock:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        return response

    def test_converts_reported_soilgrids_scale_and_marks_unavailable_p_and_k(self) -> None:
        payload = {
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
        with patch("crop_app.soil.requests.get", return_value=self._response(payload)):
            estimates = get_soil_estimates(26.14, 91.73)
        self.assertEqual(estimates["nitrogen"], {"value": 1.27, "unit": "g/kg"})
        self.assertEqual(estimates["ph"], {"value": 7.6, "unit": "pH"})
        self.assertIsNone(estimates["phosphorus"])
        self.assertIsNone(estimates["potassium"])
        self.assertIn("no observation date", estimates["period"])

    def test_null_coverage_is_not_replaced_with_fabricated_values(self) -> None:
        payload = {
            "properties": {
                "layers": [
                    {
                        "name": "nitrogen",
                        "unit_measure": {"d_factor": 100},
                        "depths": [{"label": "0-5cm", "values": {"mean": None}}],
                    },
                    {
                        "name": "phh2o",
                        "unit_measure": {"d_factor": 10},
                        "depths": [{"label": "0-5cm", "values": {"mean": None}}],
                    },
                ]
            }
        }
        with patch("crop_app.soil.requests.get", return_value=self._response(payload)):
            estimates = get_soil_estimates(28.0, 77.0)
        self.assertIsNone(estimates["nitrogen"]["value"])
        self.assertIsNone(estimates["ph"]["value"])

    def test_network_errors_are_reported_as_soil_estimate_errors(self) -> None:
        with patch(
            "crop_app.soil.requests.get",
            side_effect=requests.Timeout("timeout"),
        ):
            with self.assertRaisesRegex(SoilEstimateError, "timed out"):
                get_soil_estimates(28.0, 77.0)


if __name__ == "__main__":
    unittest.main()
