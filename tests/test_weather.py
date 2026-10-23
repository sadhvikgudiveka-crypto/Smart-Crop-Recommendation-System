import unittest
from unittest.mock import Mock, patch

import requests

from crop_app.weather import (
    WeatherServiceError,
    get_current_weather,
    search_locations,
)


class WeatherTests(unittest.TestCase):
    @patch("crop_app.weather.requests.get")
    def test_search_locations_returns_results(self, get) -> None:
        response = Mock()
        response.json.return_value = {"results": [{"name": "Pune", "country": "India"}]}
        get.return_value = response

        locations = search_locations("Pune")

        self.assertEqual(locations[0]["name"], "Pune")
        self.assertEqual(get.call_args.kwargs["params"]["name"], "Pune")

    @patch("crop_app.weather.requests.get")
    def test_search_no_results_is_actionable(self, get) -> None:
        response = Mock()
        response.json.return_value = {}
        get.return_value = response

        with self.assertRaisesRegex(WeatherServiceError, "No matching location"):
            search_locations("Nowhere")

    @patch("crop_app.weather.requests.get")
    def test_weather_values_keep_provider_units_and_time(self, get) -> None:
        response = Mock()
        response.json.return_value = {
            "current": {
                "time": "2026-10-22T12:00",
                "temperature_2m": 22.5,
                "relative_humidity_2m": 61,
                "precipitation": 0.0,
            },
            "current_units": {
                "temperature_2m": "°C",
                "relative_humidity_2m": "%",
                "precipitation": "mm",
            },
            "timezone": "Asia/Kolkata",
        }
        get.return_value = response

        current = get_current_weather({"latitude": 18.5, "longitude": 73.8})

        self.assertEqual(current["temperature_unit"], "°C")
        self.assertEqual(current["humidity_unit"], "%")
        self.assertEqual(current["precipitation_unit"], "mm")
        self.assertEqual(current["time"], "2026-10-22T12:00")

    @patch("crop_app.weather.requests.get", side_effect=requests.Timeout)
    def test_weather_timeout_is_reported(self, _get) -> None:
        with self.assertRaisesRegex(WeatherServiceError, "timed out"):
            search_locations("Pune")


if __name__ == "__main__":
    unittest.main()
