"""Photo labels preserve class identity and media provenance."""

import hashlib

import pytest

from dev.prototype_space import serve


def test_reference_membership_and_media_provenance():
    image = {"type": "StillImage", "identifier": "https://example.org/moth.jpg", "creator": "Photographer"}
    records = [
        {"key": 1, "taxonKey": 999, "media": [image]},
        {"key": 2, "speciesKey": 42, "license": "Occurrence data license", "media": [image, image]},
        {"key": 3, "speciesKey": 42, "media": [{**image, "type": "Sound"}]},
        {"key": 4, "speciesKey": 42, "media": [{**image, "identifier": "file:///tmp/moth.jpg"}]},
    ]
    album = serve.reference_photos("17", {"acceptedKey": 42, "canonicalName": "Example moth"}, records)
    assert album["class_id"] == "17"
    assert album["accepted_id"] == "42"
    assert album["display_name"] == "Example moth"
    assert len(album["photos"]) == 1
    photo = album["photos"][0]
    digest = hashlib.md5(image["identifier"].encode(), usedforsecurity=False).hexdigest()
    assert photo["image_path"] == f"/gbif-image/2/{digest}"
    assert photo["license"] == "Image license not supplied"
    assert photo["creator"] == "Photographer"
    assert photo["source"] == "https://www.gbif.org/occurrence/2"


def test_photo_service_restricts_requests_and_registers_images(monkeypatch):
    calls = []

    def retrieve(url):
        calls.append(url)
        if "/species/" in url:
            return {"canonicalName": "Example"}
        return {"results": [{"key": 9, "taxonKey": 42, "media": [{"type": "StillImage", "identifier": "https://example.org/a.jpg"}]}]}

    monkeypatch.setattr(serve.gbif, "retrive_request", retrieve)
    service = serve.PhotoService({"42"}, {})
    with pytest.raises(ValueError):
        service.metadata("43")
    with pytest.raises(ValueError):
        service.image("/gbif-image/9/unresolved")
    assert calls == []
    album = service.metadata("42")
    path = album["photos"][0]["image_path"]
    service.cache["prototype-image:" + path] = (b"cached image", "image/jpeg")
    assert service.image(path) == (b"cached image", "image/jpeg")
    assert len(calls) == 2
