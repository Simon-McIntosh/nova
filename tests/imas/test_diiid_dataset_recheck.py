"""Tests for the live competition-dataset recheck receipt."""

from __future__ import annotations

from datetime import datetime, timezone
from email.message import Message
import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from benchmarks.diiid_dataset_recheck import (
    HUB_API_URL,
    LANDED_CURRENT_CHANNEL_COUNT,
    LANDED_GEOMETRY_DIGEST,
    REFERENCE_OBJECT,
    SCHEMA_URL,
    TARGET_CONDUCTORS,
    inspect_publication,
    retrieve_publication,
)


GEOMETRY_NAMES = [
    *(f"F{number}{side}" for side in ("A", "B") for number in range(1, 10)),
    "ECOILA",
]
CURRENT_CHANNELS = [
    *(f"magnetics_{name}" for name in GEOMETRY_NAMES),
    "magnetics_bcoil",
]


def _schema_payload() -> dict:
    support = [
        "magnetics_time",
        "magnetics_plasma_current",
        "magnetics_plasma_current_times",
        "magnetics_dsep",
        "magnetics_dsep_times",
    ]
    return {"features": [{"name": name} for name in [*CURRENT_CHANNELS, *support]]}


def _landed_baseline(object_bytes: bytes) -> dict:
    return {
        "geometry_digest": LANDED_GEOMETRY_DIGEST,
        "geometry_names": GEOMETRY_NAMES,
        "current_channels": CURRENT_CHANNELS,
        "source_object_sha256": hashlib.sha256(object_bytes).hexdigest(),
    }


class _Response:
    def __init__(self, body: bytes, url: str, *, content_type: str):
        self._body = body
        self._url = url
        self.status = 200
        self.headers = Message()
        self.headers["content-type"] = content_type

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._body

    def geturl(self) -> str:
        return self._url


def test_identical_publication_object_preserves_baseline() -> None:
    object_bytes = b"published parquet object"
    measurement = inspect_publication(
        object_bytes,
        _schema_payload(),
        landed_baseline=_landed_baseline(object_bytes),
    )
    assert measurement["geometry"]["conductor_count"] == 19
    assert measurement["geometry"]["digest_matches_landed"] is True
    assert measurement["currents"]["channel_count"] == LANDED_CURRENT_CHANNEL_COUNT
    assert measurement["currents"]["channel_count_difference"] == 0
    assert measurement["newly_published_target_count"] == 0
    assert measurement["all_five_still_absent_from_geometry_and_currents"] is True
    assert all(
        status["geometry_present"] is False
        and status["current_channel_present"] is False
        for status in measurement["target_conductors"].values()
    )


def test_retrieval_records_revision_source_and_timestamp(tmp_path: Path) -> None:
    revision = "abc123"
    metadata = json.dumps(
        {"sha": revision, "lastModified": "2026-08-21T00:00:00.000Z"}
    ).encode()
    parquet_bytes = b"published parquet object"
    source_object = tmp_path / "source.parquet"
    source_object.write_bytes(parquet_bytes)
    landed = tmp_path / "landed.json"
    landed.write_text(
        json.dumps(
            {
                "physical_geometry_digest": LANDED_GEOMETRY_DIGEST,
                "source_row": str(source_object),
                "quantities": {
                    "poloidal_conductors": {
                        "conductors": [{"name": name} for name in GEOMETRY_NAMES]
                    },
                    "bcoil": {"input_column": "magnetics_bcoil"},
                },
            }
        )
    )

    def opener(request, *, timeout):
        assert timeout == 30.0
        if request.full_url == HUB_API_URL:
            return _Response(
                metadata, request.full_url, content_type="application/json"
            )
        if request.full_url == SCHEMA_URL:
            return _Response(
                json.dumps(_schema_payload()).encode(),
                request.full_url,
                content_type="application/json",
            )
        assert request.full_url.endswith(f"/{revision}/{REFERENCE_OBJECT}")
        return _Response(
            parquet_bytes, request.full_url, content_type="application/octet-stream"
        )

    receipt = retrieve_publication(
        opener=opener,
        timestamp=datetime(2026, 8, 21, 9, 45, tzinfo=timezone.utc),
        landed_receipt=landed,
    )
    assert receipt["status"] == "retrieved"
    assert receipt["retrieval_timestamp_utc"] == "2026-08-21T09:45:00Z"
    assert receipt["source"]["revision"] == revision
    assert len(receipt["source"]["channels"]) == 3
    assert all(
        "?" not in channel["resolved_url"] for channel in receipt["source"]["channels"]
    )
    assert receipt["target_conductors"].keys() == set(TARGET_CONDUCTORS)


def test_http_refusal_is_preserved_verbatim() -> None:
    refusal_text = '{"error":"Repository access denied"}\n'

    def opener(request, *, timeout):
        raise HTTPError(
            request.full_url,
            403,
            "Forbidden",
            Message(),
            io.BytesIO(refusal_text.encode()),
        )

    receipt = retrieve_publication(
        opener=opener,
        timestamp=datetime(2026, 8, 21, tzinfo=timezone.utc),
    )
    assert receipt["status"] == "source_unreachable"
    assert receipt["source"]["attempted_channel"] == (
        "Hugging Face Hub dataset metadata REST API"
    )
    assert receipt["refusal"]["http_status"] == 403
    assert receipt["refusal"]["reason"] == "Forbidden"
    assert receipt["refusal"]["response_text_exact"] == refusal_text
    assert receipt["newly_published_target_count"] is None


@pytest.mark.parametrize("name", TARGET_CONDUCTORS)
def test_every_target_has_geometry_and_current_fields(name: str) -> None:
    object_bytes = b"published parquet object"
    measurement = inspect_publication(
        object_bytes,
        _schema_payload(),
        landed_baseline=_landed_baseline(object_bytes),
    )
    assert set(measurement["target_conductors"][name]) == {
        "geometry_present",
        "geometry_row_count",
        "current_channel",
        "current_channel_present",
    }
