from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from crop_app.crop_reference import CropReferenceError, get_crop_reference


def _response(payload: dict) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


class CropReferenceTests(unittest.TestCase):
    def test_crop_label_selects_matching_wikipedia_page_and_credited_image(self) -> None:
        summary = _response(
            {
                "extract": "Rice is a cereal crop.",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Rice"}},
                "thumbnail": {
                    "source": (
                        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
                        "a/ab/Rice_plant.jpg/320px-Rice_plant.jpg"
                    )
                },
            }
        )
        commons = _response(
            {
                "query": {
                    "pages": {
                        "1": {
                            "imageinfo": [
                                {
                                    "thumburl": "https://upload.wikimedia.org/rice-small.jpg",
                                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Rice_plant.jpg",
                                    "extmetadata": {
                                        "Artist": {"value": "<a>User Name</a>"},
                                        "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                        "LicenseUrl": {
                                            "value": "https://creativecommons.org/licenses/by-sa/4.0/"
                                        },
                                    },
                                }
                            ]
                        }
                    }
                }
            }
        )
        with patch("crop_app.crop_reference.requests.get", side_effect=[summary, commons]) as get:
            reference = get_crop_reference("rice")

        self.assertEqual(reference["description"], "Rice is a cereal crop.")
        self.assertEqual(reference["image_artist"], "User Name")
        self.assertEqual(reference["image_license"], "CC BY-SA 4.0")
        self.assertIn("Rice_plant.jpg", get.call_args_list[1].kwargs["params"]["titles"])

    def test_unmapped_crop_and_missing_image_license_are_not_silently_substituted(self) -> None:
        with self.assertRaises(CropReferenceError):
            get_crop_reference("unknown crop")

        summary = _response(
            {
                "extract": "Rice is a cereal crop.",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Rice"}},
                "thumbnail": {
                    "source": "https://upload.wikimedia.org/wikipedia/commons/thumb/Rice.jpg"
                },
            }
        )
        commons = _response(
            {
                "query": {
                    "pages": {
                        "1": {
                            "imageinfo": [
                                {
                                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Rice.jpg",
                                    "extmetadata": {},
                                }
                            ]
                        }
                    }
                }
            }
        )
        with patch(
            "crop_app.crop_reference.requests.get", side_effect=[summary, commons]
        ):
            reference = get_crop_reference("rice")
        self.assertEqual(reference["description"], "Rice is a cereal crop.")
        self.assertFalse(reference["image_url"])
        self.assertIn("license details", reference["image_error"])

    def test_network_failure_is_reported(self) -> None:
        with patch(
            "crop_app.crop_reference.requests.get",
            side_effect=requests.Timeout("timeout"),
        ):
            with self.assertRaisesRegex(CropReferenceError, "could not load"):
                get_crop_reference("rice")

    def test_curated_crop_image_fallback_is_used_with_its_own_credit(self) -> None:
        summary = _response(
            {
                "extract": "Cotton is a crop grown for its fibre.",
                "content_urls": {
                    "desktop": {"page": "https://en.wikipedia.org/wiki/Cotton"}
                },
                "thumbnail": {
                    "source": (
                        "https://upload.wikimedia.org/wikipedia/commons/"
                        "thumb/c/c1/Cotton_article_image.jpg/320px-Cotton_article_image.jpg"
                    )
                },
            }
        )
        no_image = _response({"query": {"pages": {"-1": {"missing": ""}}}})
        fallback_image = _response(
            {
                "query": {
                    "pages": {
                        "2": {
                            "imageinfo": [
                                {
                                    "thumburl": "https://upload.wikimedia.org/cotton.jpg",
                                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:COTTON_CROP_PLANT.jpg",
                                    "extmetadata": {
                                        "Artist": {"value": "Example Artist"},
                                        "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                        "LicenseUrl": {
                                            "value": "https://creativecommons.org/licenses/by-sa/4.0/"
                                        },
                                    },
                                }
                            ]
                        }
                    }
                }
            }
        )
        with patch(
            "crop_app.crop_reference.requests.get",
            side_effect=[summary, no_image, fallback_image],
        ) as get:
            reference = get_crop_reference("cotton")

        self.assertEqual(reference["image_artist"], "Example Artist")
        self.assertIn(
            "COTTON CROP PLANT.jpg",
            get.call_args_list[2].kwargs["params"]["titles"],
        )


if __name__ == "__main__":
    unittest.main()
