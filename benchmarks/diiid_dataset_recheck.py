"""Recheck the public competition dataset for omitted DIII-D conductors.

The check resolves the current Hugging Face revision, reads its published
feature schema, and retrieves one revision-pinned parquet object. The parquet
object is compared byte-for-byte with the landed source object, which makes the
landed geometry digest applicable without adding a parquet engine dependency.
Network refusals are captured verbatim and do not make the benchmark fail.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


DATASET_ID = "Sophelio/fusion-equilibrium-challenge"
HUB_API_URL = f"https://huggingface.co/api/datasets/{DATASET_ID}"
SCHEMA_URL = (
    "https://datasets-server.huggingface.co/first-rows?"
    f"dataset={DATASET_ID}&config=diii_d_train&split=train"
)
REFERENCE_OBJECT = "data/diii_d_train/d3d_shot_00000a10ac.parquet"
LANDED_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/machine-description/"
    "machine_description_receipt.json"
)
OUTPUT_DIRECTORY = Path("docs/figures/diiid-forward-onboarding/dataset-recheck")
LANDED_GEOMETRY_DIGEST = (
    "782e9e08f02e610e252e9cf6d6cccfb3a9aefa62b56f14865553ba2f35d213dc"
)
LANDED_CURRENT_CHANNEL_COUNT = 20
TARGET_CONDUCTORS = ("ECOILB", "E567UP", "E567DN", "E89UP", "E89DN")
NON_ACTUATOR_MAGNETICS_COLUMNS = {
    "magnetics_dsep",
    "magnetics_dsep_times",
    "magnetics_plasma_current",
    "magnetics_plasma_current_times",
    "magnetics_time",
}
OpenUrl = Callable[..., Any]


@dataclass(frozen=True)
class SourceRefusal:
    """Exact failure returned by one attempted publication channel."""

    channel: str
    url: str
    error_type: str
    http_status: int | None
    reason: str
    response_text_exact: str


class PublicationUnavailable(RuntimeError):
    """Carry a structured source refusal to the receipt writer."""

    def __init__(self, refusal: SourceRefusal):
        super().__init__(refusal.reason)
        self.refusal = refusal


def _fetch_bytes(
    url: str, *, channel: str, opener: OpenUrl = urlopen, timeout: float = 30.0
) -> tuple[bytes, dict[str, Any]]:
    """Fetch one URL or raise with the exact refusal returned by that channel."""
    request = Request(url, headers={"User-Agent": "nova-dataset-recheck/1"})
    try:
        with opener(request, timeout=timeout) as response:
            body = response.read()
            resolved_url = urlsplit(response.geturl())
            return body, {
                "channel": channel,
                "requested_url": url,
                "resolved_url": urlunsplit(
                    (
                        resolved_url.scheme,
                        resolved_url.netloc,
                        resolved_url.path,
                        "",
                        "",
                    )
                ),
                "http_status": int(response.status),
                "content_type": response.headers.get("content-type"),
                "content_length": len(body),
                "etag": response.headers.get("etag"),
            }
    except HTTPError as error:
        response_text = error.read().decode("utf-8", errors="replace")
        error.close()
        raise PublicationUnavailable(
            SourceRefusal(
                channel=channel,
                url=url,
                error_type=type(error).__name__,
                http_status=error.code,
                reason=str(error.reason),
                response_text_exact=response_text,
            )
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise PublicationUnavailable(
            SourceRefusal(
                channel=channel,
                url=url,
                error_type=type(error).__name__,
                http_status=None,
                reason=str(reason),
                response_text_exact=str(reason),
            )
        ) from error


def _read_landed_baseline(path: Path = LANDED_RECEIPT) -> dict[str, Any]:
    """Read the source object, geometry, and channels defining the baseline."""
    receipt = json.loads(path.read_text())
    digest = receipt["physical_geometry_digest"]
    if digest != LANDED_GEOMETRY_DIGEST:
        raise ValueError(f"landed geometry digest changed unexpectedly: {digest}")
    geometry_names = [
        item["name"]
        for item in receipt["quantities"]["poloidal_conductors"]["conductors"]
    ]
    current_channels = [f"magnetics_{name}" for name in geometry_names]
    current_channels.append(receipt["quantities"]["bcoil"]["input_column"])
    source_path = Path(receipt["source_row"])
    source_bytes = source_path.read_bytes()
    return {
        "receipt_path": str(path),
        "source_path": str(source_path),
        "source_object_size_bytes": len(source_bytes),
        "source_object_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "geometry_digest": digest,
        "geometry_names": geometry_names,
        "current_channels": sorted(current_channels),
    }


def inspect_publication(
    parquet_bytes: bytes,
    schema_payload: dict[str, Any],
    *,
    landed_baseline: dict[str, Any],
) -> dict[str, Any]:
    """Measure live schema and infer geometry through full-object identity."""
    feature_names = [str(item["name"]) for item in schema_payload["features"]]
    current_channels = sorted(
        name
        for name in feature_names
        if name.startswith("magnetics_")
        and name not in NON_ACTUATOR_MAGNETICS_COLUMNS
        and not name.endswith("_times")
    )
    remote_sha256 = hashlib.sha256(parquet_bytes).hexdigest()
    byte_identical = remote_sha256 == landed_baseline["source_object_sha256"]
    geometry_names = landed_baseline["geometry_names"] if byte_identical else None
    published_digest = landed_baseline["geometry_digest"] if byte_identical else None
    target_status = {
        name: {
            "geometry_present": (
                name in geometry_names if geometry_names is not None else None
            ),
            "geometry_row_count": (
                geometry_names.count(name) if geometry_names is not None else None
            ),
            "current_channel": f"magnetics_{name}",
            "current_channel_present": f"magnetics_{name}" in current_channels,
        }
        for name in TARGET_CONDUCTORS
    }
    newly_published = [
        name
        for name, status in target_status.items()
        if status["geometry_present"] or status["current_channel_present"]
    ]
    unknown_geometry = any(
        status["geometry_present"] is None for status in target_status.values()
    )
    return {
        "geometry": {
            "conductor_count": len(geometry_names) if geometry_names else None,
            "conductor_names": geometry_names,
            "provenance_digest": published_digest,
            "landed_digest": landed_baseline["geometry_digest"],
            "digest_matches_landed": byte_identical,
            "digest_comparison_basis": (
                "current revision-pinned parquet object is byte-identical to the "
                "landed source object"
                if byte_identical
                else "remote object changed; geometry digest was not inferred"
            ),
            "remote_reference_object_sha256": remote_sha256,
            "landed_source_object_sha256": landed_baseline["source_object_sha256"],
            "reference_object_byte_identical": byte_identical,
        },
        "currents": {
            "channel_count": len(current_channels),
            "channel_names": current_channels,
            "landed_channel_count": LANDED_CURRENT_CHANNEL_COUNT,
            "channel_count_difference": (
                len(current_channels) - LANDED_CURRENT_CHANNEL_COUNT
            ),
        },
        "target_conductors": target_status,
        "newly_published_target_count": len(newly_published),
        "newly_published_targets": newly_published,
        "all_five_still_absent_from_geometry_and_currents": (
            not newly_published if not unknown_geometry else None
        ),
    }


def retrieve_publication(
    *,
    opener: OpenUrl = urlopen,
    timestamp: datetime | None = None,
    landed_receipt: Path = LANDED_RECEIPT,
) -> dict[str, Any]:
    """Retrieve the live revision and return a success or refusal receipt."""
    retrieved_at = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    timestamp_text = retrieved_at.isoformat().replace("+00:00", "Z")
    attempted_channels = []
    try:
        metadata_bytes, metadata_channel = _fetch_bytes(
            HUB_API_URL,
            channel="Hugging Face Hub dataset metadata REST API",
            opener=opener,
        )
        attempted_channels.append(metadata_channel)
        metadata = json.loads(metadata_bytes)
        revision = str(metadata["sha"])
        schema_bytes, schema_channel = _fetch_bytes(
            SCHEMA_URL,
            channel="Hugging Face dataset-server published feature schema",
            opener=opener,
        )
        attempted_channels.append(schema_channel)
        parquet_url = (
            f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
            f"{revision}/{REFERENCE_OBJECT}"
        )
        parquet_bytes, parquet_channel = _fetch_bytes(
            parquet_url,
            channel="revision-pinned Hugging Face parquet object",
            opener=opener,
        )
        attempted_channels.append(parquet_channel)
        measurement = inspect_publication(
            parquet_bytes,
            json.loads(schema_bytes),
            landed_baseline=_read_landed_baseline(landed_receipt),
        )
        return {
            "measurement": "DIII-D competition dataset conductor publication recheck",
            "retrieval_timestamp_utc": timestamp_text,
            "status": "retrieved",
            "source": {
                "dataset": DATASET_ID,
                "repository_url": (f"https://huggingface.co/datasets/{DATASET_ID}"),
                "revision": revision,
                "repository_last_modified": metadata.get("lastModified"),
                "reference_object": REFERENCE_OBJECT,
                "reference_object_sha256": hashlib.sha256(parquet_bytes).hexdigest(),
                "channels": attempted_channels,
            },
            **measurement,
        }
    except PublicationUnavailable as error:
        return {
            "measurement": "DIII-D competition dataset conductor publication recheck",
            "retrieval_timestamp_utc": timestamp_text,
            "status": "source_unreachable",
            "source": {
                "dataset": DATASET_ID,
                "repository_url": (f"https://huggingface.co/datasets/{DATASET_ID}"),
                "channels_completed_before_refusal": attempted_channels,
                "attempted_channel": error.refusal.channel,
            },
            "refusal": asdict(error.refusal),
            "geometry": {
                "provenance_digest": None,
                "landed_digest": LANDED_GEOMETRY_DIGEST,
                "digest_matches_landed": None,
            },
            "currents": {
                "channel_count": None,
                "landed_channel_count": LANDED_CURRENT_CHANNEL_COUNT,
                "channel_count_difference": None,
            },
            "target_conductors": {
                name: {
                    "geometry_present": None,
                    "current_channel": f"magnetics_{name}",
                    "current_channel_present": None,
                }
                for name in TARGET_CONDUCTORS
            },
            "newly_published_target_count": None,
            "newly_published_targets": None,
            "all_five_still_absent_from_geometry_and_currents": None,
        }


def main() -> None:
    """Run the live recheck and write its durable receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--landed-receipt", type=Path, default=LANDED_RECEIPT)
    args = parser.parse_args()
    receipt = retrieve_publication(landed_receipt=args.landed_receipt)
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "dataset_recheck_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "status": receipt["status"],
                "current_channel_count": receipt["currents"]["channel_count"],
                "newly_published_target_count": receipt["newly_published_target_count"],
                "geometry_digest_matches": receipt["geometry"]["digest_matches_landed"],
            }
        )
    )


if __name__ == "__main__":
    main()
