"""Enumerate MAST equilibrium slices from catalog and Zarr metadata only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_CATALOG_INDEX = "https://mastapp.site/parquet/level2/shots"
DEFAULT_MIRROR_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")
DEFAULT_OUTPUT = Path("docs/figures/mast-catalog-gpu-solve/slice-census.json")
DEFAULT_REPORT = Path("scripts/mast_slice_census/report.md")
EXPECTED_CATALOG_SHOTS = 11_573
TARGET_WALL_SECONDS = 3_600
TARGET_DEVICES = 8
TIME_METADATA_RELATIVE_PATH = Path("equilibrium/time/zarr.json")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _campaign_label(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    label = str(value).strip()
    return label or "Unknown"


def _shot_id_from_store(path: Path) -> int | None:
    if path.suffix != ".zarr":
        return None
    try:
        return int(path.stem)
    except ValueError:
        return None


def _read_slice_count(path: Path) -> int:
    metadata = json.loads(path.read_text())
    shape = metadata.get("shape")
    if (
        metadata.get("zarr_format") != 3
        or metadata.get("node_type") != "array"
        or not isinstance(shape, list)
        or len(shape) != 1
        or not isinstance(shape[0], int)
        or isinstance(shape[0], bool)
        or shape[0] < 0
    ):
        raise ValueError("expected a one-dimensional Zarr v3 array shape")
    dimension_names = metadata.get("dimension_names")
    if dimension_names not in (None, ["time"]):
        raise ValueError("equilibrium time metadata has an unexpected dimension")
    return shape[0]


def census(
    catalog_index: str,
    mirror_root: Path,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return a metadata-only census joined by catalog shot identifier."""
    index = pd.read_parquet(catalog_index, columns=["shot_id", "campaign"])
    rows = [
        {"shot_id": int(row.shot_id), "campaign": _campaign_label(row.campaign)}
        for row in index.itertuples(index=False)
    ]
    duplicate_ids = sorted(
        shot
        for shot, count in Counter(row["shot_id"] for row in rows).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            f"catalog index contains duplicate shot identifiers: {duplicate_ids}"
        )
    rows.sort(key=lambda row: row["shot_id"])

    mirror_shots = sorted(
        shot
        for path in mirror_root.iterdir()
        if path.is_dir() and (shot := _shot_id_from_store(path)) is not None
    )
    mirror_shot_set = set(mirror_shots)
    index_shot_set = {row["shot_id"] for row in rows}

    campaign_index_shots = Counter(row["campaign"] for row in rows)
    campaign_reachable_shots: Counter[str] = Counter()
    campaign_equilibrium_shots: Counter[str] = Counter()
    campaign_slices: Counter[str] = Counter()
    slice_counts: list[dict[str, int]] = []
    missing_stores: list[int] = []
    shots_without_equilibrium: list[int] = []
    missing_time_metadata: list[int] = []
    unreadable_time_metadata: list[dict[str, str | int]] = []
    empty_equilibrium_shots: list[int] = []

    for row in rows:
        shot = row["shot_id"]
        campaign = row["campaign"]
        if shot not in mirror_shot_set:
            missing_stores.append(shot)
            continue
        campaign_reachable_shots[campaign] += 1
        shot_store = mirror_root / f"{shot}.zarr"
        if not (shot_store / "equilibrium").is_dir():
            shots_without_equilibrium.append(shot)
            continue
        metadata_path = shot_store / TIME_METADATA_RELATIVE_PATH
        if not metadata_path.is_file():
            missing_time_metadata.append(shot)
            continue
        try:
            slices = _read_slice_count(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            unreadable_time_metadata.append({"shot_id": shot, "error": str(exc)})
            continue
        campaign_equilibrium_shots[campaign] += 1
        campaign_slices[campaign] += slices
        slice_counts.append({"shot_id": shot, "slices": slices})
        if slices == 0:
            empty_equilibrium_shots.append(shot)

    total_slices = sum(campaign_slices.values())
    readable_equilibrium_shots = sum(campaign_equilibrium_shots.values())
    reachable_shots = sum(campaign_reachable_shots.values())
    per_campaign = []
    for campaign in sorted(campaign_index_shots):
        slices = campaign_slices[campaign]
        equilibrium_shots = campaign_equilibrium_shots[campaign]
        per_campaign.append(
            {
                "campaign": campaign,
                "catalog_index_shots": campaign_index_shots[campaign],
                "reachable_shot_stores": campaign_reachable_shots[campaign],
                "shots_with_equilibrium_slices": equilibrium_shots,
                "shots_without_equilibrium_group": (
                    campaign_reachable_shots[campaign] - equilibrium_shots
                ),
                "equilibrium_slices": slices,
                "mean_slices_per_equilibrium_shot": (
                    round(slices / equilibrium_shots, 6) if equilibrium_shots else None
                ),
                "required_aggregate_slices_per_second": round(
                    slices / TARGET_WALL_SECONDS, 9
                ),
                "required_slices_per_second_per_device": round(
                    slices / (TARGET_WALL_SECONDS * TARGET_DEVICES), 9
                ),
            }
        )

    generated = generated_at or datetime.now(UTC).isoformat()
    return {
        "schema": "mast-equilibrium-slice-census",
        "generated_at": generated,
        "scope": {
            "catalog_tier": "FAIR-MAST level-2",
            "slice_definition": (
                "One entry in the Zarr v3 equilibrium/time dimension; no validity "
                "or plasma-current filter is applied."
            ),
            "target_wall_seconds": TARGET_WALL_SECONDS,
            "target_devices": TARGET_DEVICES,
        },
        "provenance": {
            "catalog_index": catalog_index,
            "catalog_fields_read": ["shot_id", "campaign"],
            "catalog_identity_sha256": _sha256_json(rows),
            "mirror_root": str(mirror_root),
            "mirror_shot_ids_sha256": _sha256_json(mirror_shots),
            "per_shot_metadata_path": str(TIME_METADATA_RELATIVE_PATH),
            "slice_counts_sha256": _sha256_json(slice_counts),
            "enumeration_method": (
                "Join the FAIR-MAST level-2 shot catalog to numeric *.zarr "
                "directories in the local mirror, then read only each "
                "equilibrium/time/zarr.json shape. The first and only shape "
                "dimension is the equilibrium-slice count."
            ),
            "data_access": {
                "catalog_metadata_downloaded": True,
                "zarr_metadata_files_read": len(slice_counts),
                "bulk_signal_arrays_downloaded": False,
                "equilibrium_solves_run": 0,
            },
        },
        "coverage": {
            "plan_asserted_catalog_shots": EXPECTED_CATALOG_SHOTS,
            "catalog_index_rows": len(index),
            "catalog_index_unique_shots": len(rows),
            "mirrored_shot_stores": len(mirror_shots),
            "reachable_catalog_shot_stores": reachable_shots,
            "shots_with_equilibrium_slices": readable_equilibrium_shots,
            "shots_without_equilibrium_group": len(shots_without_equilibrium),
            "shot_store_shortfall_against_plan_assertion": max(
                EXPECTED_CATALOG_SHOTS - reachable_shots, 0
            ),
            "equilibrium_bearing_shot_shortfall_against_plan_assertion": max(
                EXPECTED_CATALOG_SHOTS - readable_equilibrium_shots, 0
            ),
            "index_shots_missing_from_mirror": missing_stores,
            "mirror_shots_absent_from_index": sorted(mirror_shot_set - index_shot_set),
            "catalog_shots_without_equilibrium_group": shots_without_equilibrium,
            "shots_missing_equilibrium_time_metadata": missing_time_metadata,
            "shots_with_unreadable_equilibrium_time_metadata": unreadable_time_metadata,
            "shots_with_zero_equilibrium_slices": empty_equilibrium_shots,
        },
        "totals": {
            "equilibrium_slices": total_slices,
            "required_aggregate_slices_per_second": round(
                total_slices / TARGET_WALL_SECONDS, 9
            ),
            "required_slices_per_second_per_device": round(
                total_slices / (TARGET_WALL_SECONDS * TARGET_DEVICES), 9
            ),
            "aggregate_formula": f"{total_slices} / {TARGET_WALL_SECONDS}",
            "per_device_formula": (
                f"{total_slices} / ({TARGET_WALL_SECONDS} * {TARGET_DEVICES})"
            ),
        },
        "per_campaign": per_campaign,
    }


def report(census_data: dict[str, Any]) -> str:
    """Render the census method, coverage, and throughput requirement."""
    coverage = census_data["coverage"]
    totals = census_data["totals"]
    scope = census_data["scope"]
    access = census_data["provenance"]["data_access"]
    store_shortfall = coverage["shot_store_shortfall_against_plan_assertion"]
    equilibrium_shortfall = coverage[
        "equilibrium_bearing_shot_shortfall_against_plan_assertion"
    ]
    coverage_sentence = (
        (
            "There is no catalog or mirror coverage shortfall: every asserted "
            "shot was enumerated."
        )
        if store_shortfall == 0
        else (
            f"Catalog coverage is short by {store_shortfall} reachable shot stores "
            "against "
            f"the asserted {coverage['plan_asserted_catalog_shots']:,} shots."
        )
    )
    lines = [
        "# MAST equilibrium-slice census",
        "",
        (
            f"The reachable FAIR-MAST level-2 catalog contains "
            f"**{totals['equilibrium_slices']:,} equilibrium slices**. Finishing "
            f"them in {scope['target_wall_seconds']:,} s requires "
            f"**{totals['required_aggregate_slices_per_second']:,.3f} slices/s "
            f"aggregate**, or "
            f"**{totals['required_slices_per_second_per_device']:,.3f} "
            f"slices/s/device** across {scope['target_devices']} devices."
        ),
        "",
        "## Coverage",
        "",
        (
            f"The catalog index has {coverage['catalog_index_rows']:,} rows and "
            f"{coverage['catalog_index_unique_shots']:,} unique shot identifiers, "
            f"against the asserted {coverage['plan_asserted_catalog_shots']:,}. "
            f"The local mirror exposes {coverage['mirrored_shot_stores']:,} numeric "
            f"shot stores; {coverage['reachable_catalog_shot_stores']:,} join to "
            f"the index and "
            f"{coverage['shots_with_equilibrium_slices']:,} contain equilibrium "
            f"slices. The remaining {equilibrium_shortfall:,} reachable stores "
            "contain no `equilibrium` group and therefore contribute zero slices."
        ),
        "",
        coverage_sentence,
        "",
        (
            "| Campaign | Index shots | Reachable stores | Equilibrium shots | "
            "Slices | Aggregate slices/s | Slices/s/device |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in census_data["per_campaign"]:
        lines.append(
            f"| {row['campaign']} | {row['catalog_index_shots']:,} | "
            f"{row['reachable_shot_stores']:,} | "
            f"{row['shots_with_equilibrium_slices']:,} | "
            f"{row['equilibrium_slices']:,} | "
            f"{row['required_aggregate_slices_per_second']:,.3f} | "
            f"{row['required_slices_per_second_per_device']:,.3f} |"
        )
    lines.extend(
        [
            "",
            "## Method and provenance",
            "",
            (
                "The enumeration reads only `shot_id` and `campaign` from the "
                f"FAIR-MAST metadata index at "
                f"`{census_data['provenance']['catalog_index']}`. It joins those "
                "identifiers to the local level-2 mirror and parses only each "
                "`equilibrium/time/zarr.json` file. The declared one-dimensional "
                "Zarr shape is counted as that shot's equilibrium slices; no "
                "plasma-current, validity, or topology filter is applied."
            ),
            "",
            (
                "The JSON receipt records hashes of the normalized catalog identity, "
                "mirror shot identifiers, and per-shot slice counts so the exact "
                "enumeration can be compared with a later catalog snapshot."
            ),
            "",
            "## Execution boundary",
            "",
            (
                f"No equilibrium solve was run (count: "
                f"{access['equilibrium_solves_run']}). No bulk signal data was "
                "downloaded for counting. The only network read was the compact "
                "shot-catalog metadata; all per-shot reads were local Zarr metadata "
                f"({access['zarr_metadata_files_read']:,} JSON files), not array "
                "chunks."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(data: dict[str, Any], output: Path, report_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    report_path.write_text(report(data))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-index", default=DEFAULT_CATALOG_INDEX)
    parser.add_argument("--mirror-root", type=Path, default=DEFAULT_MIRROR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    data = census(args.catalog_index, args.mirror_root)
    _write_outputs(data, args.output, args.report)
    summary = {
        "catalog_index_shots": data["coverage"]["catalog_index_unique_shots"],
        "reachable_shot_stores": data["coverage"]["reachable_catalog_shot_stores"],
        "shots_with_equilibrium_slices": data["coverage"][
            "shots_with_equilibrium_slices"
        ],
        "shots_without_equilibrium_group": data["coverage"][
            "shots_without_equilibrium_group"
        ],
        "totals": data["totals"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
