from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, unquote, urlparse

import requests

WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKIMEDIA_COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
WIKIMEDIA_USER_AGENT = (
    "SmartCropRecommendationSystem/1.0 "
    "(educational crop-reference display)"
)
FALLBACK_COMMONS_FILES = {
    "cotton": "COTTON CROP PLANT.jpg",
    "muskmelon": "Cucumis melo var. Queen Anne's Pocket.jpg",
}
CROP_DISPLAY_NAMES = {
    "blackgram": "Black gram",
    "kidneybeans": "Kidney bean",
    "mothbeans": "Moth bean",
    "mungbean": "Mung bean",
    "pigeonpeas": "Pigeon pea",
    "grapes": "Grapes",
}
WIKIPEDIA_PAGES = {
    "apple": "Apple",
    "banana": "Banana",
    "blackgram": "Vigna mungo",
    "chickpea": "Chickpea",
    "coconut": "Coconut",
    "coffee": "Coffee",
    "cotton": "Cotton",
    "grapes": "Grape",
    "jute": "Jute",
    "kidneybeans": "Kidney bean",
    "lentil": "Lentil",
    "maize": "Maize",
    "mango": "Mango",
    "mothbeans": "Moth bean",
    "mungbean": "Mung bean",
    "muskmelon": "Cucumis melo",
    "orange": "Orange (fruit)",
    "papaya": "Papaya",
    "pigeonpeas": "Pigeon pea",
    "pomegranate": "Pomegranate",
    "rice": "Rice",
    "watermelon": "Watermelon",
}


class CropReferenceError(RuntimeError):
    """A crop-reference source could not provide usable information."""


class _PlainText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    parser = _PlainText()
    parser.feed(value)
    return " ".join(" ".join(parser.parts).split())


def _commons_image(file_title: str, timeout: float) -> dict[str, str]:
    response = requests.get(
        WIKIMEDIA_COMMONS_API_URL,
        params={
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": 900,
            "titles": f"File:{file_title}",
        },
        headers={"User-Agent": WIKIMEDIA_USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    file_page = next(iter(payload["query"]["pages"].values()))
    image_info = file_page["imageinfo"][0]
    metadata = image_info["extmetadata"]
    artist = _plain_text(metadata.get("Artist", {}).get("value"))
    license_name = _plain_text(metadata.get("LicenseShortName", {}).get("value"))
    license_url = metadata.get("LicenseUrl", {}).get("value")
    image_url = image_info.get("thumburl") or image_info.get("url")
    image_page_url = image_info.get("descriptionurl")
    if license_name.casefold() == "public domain" and not license_url:
        license_url = image_page_url
    if not all((artist, license_name, license_url, image_url, image_page_url)):
        raise ValueError("image attribution or license metadata is missing")
    return {
        "image_url": image_url,
        "image_page_url": image_page_url,
        "image_artist": artist,
        "image_license": license_name,
        "image_license_url": license_url,
    }


def get_crop_reference(crop_label: str, timeout: float = 10.0) -> dict[str, str]:
    """Fetch crop-specific encyclopedic context and a credited Commons image."""
    page_title = WIKIPEDIA_PAGES.get(crop_label.casefold())
    if page_title is None:
        raise CropReferenceError(
            f"No crop reference has been configured for {crop_label!r}."
        )

    try:
        response = requests.get(
            f"{WIKIPEDIA_SUMMARY_URL}{quote(page_title, safe='')}",
            headers={"User-Agent": WIKIMEDIA_USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
        summary = response.json()
        description = summary["extract"].strip()
        article_url = summary["content_urls"]["desktop"]["page"]
        thumbnail_url = summary.get("thumbnail", {}).get("source", "")
        source_image_url = summary.get("originalimage", {}).get("source", thumbnail_url)
        if not description or not article_url:
            raise KeyError("missing crop description or article link")
        if len(description) > 500:
            sentence_end = description.rfind(". ", 0, 500)
            description = (
                description[: sentence_end + 1]
                if sentence_end > 200
                else description[:497].rstrip() + "..."
            )
    except requests.RequestException as exc:
        raise CropReferenceError(
            f"Wikimedia could not load the {crop_label} reference right now."
        ) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise CropReferenceError(
            f"Wikimedia returned incomplete crop details for {crop_label}."
        ) from exc

    reference = {
        "description": description,
        "article_title": str(summary.get("title", page_title)),
        "article_url": article_url,
        "image_url": "",
        "image_page_url": "",
        "image_artist": "",
        "image_license": "",
        "image_license_url": "",
        "image_error": "",
    }
    file_titles: list[str] = []
    if source_image_url:
        image_filename = unquote(urlparse(source_image_url).path.rsplit("/", 1)[-1])
        file_titles.append(re.sub(r"^\d+px-", "", image_filename))
    fallback_title = FALLBACK_COMMONS_FILES.get(crop_label.casefold())
    if fallback_title and fallback_title not in file_titles:
        file_titles.append(fallback_title)

    image_error = f"No crop image with verifiable credits is available for {crop_label}."
    for file_title in file_titles:
        try:
            reference.update(_commons_image(file_title, timeout))
            return reference
        except requests.RequestException:
            image_error = f"Wikimedia could not load the {crop_label} image right now."
        except (KeyError, StopIteration, TypeError, ValueError):
            image_error = (
                f"Wikimedia did not provide complete image license details for {crop_label}."
            )
    reference["image_error"] = image_error
    return reference
