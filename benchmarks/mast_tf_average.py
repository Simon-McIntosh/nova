"""Audit independent observables for the MAST axisymmetric toroidal field.

The agreement threshold is fixed before reading any shot. A reconstruction's
radius-field product may set an axisymmetric reference, but cannot validate
itself. Feed current needs an independently sourced effective linked turn count;
a probe needs an established orientation and absolute-field calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path

import numpy as np

from nova.imas.mast_calibration_cohort import (
    ExperimentClass,
    calibration_experiments,
)
from nova.imas.mast_error_field_screen import (
    DRIVEN_CURRENT,
    ERROR_FIELD_ALIASES,
    ERROR_FIELD_CHANNELS,
)
from nova.imas.mast_vacuum_cohort import (
    KILO,
    SHOT_STORE,
    ShotSurvey,
)

RELATIVE_TOLERANCE = 0.02
MU_OVER_TWO_PI = 2.0e-7
DEFAULT_CENSUS = Path.home() / ".cache/nova-mast/mast_vacuum_census.json"
LEVEL2_STORE = Path("/work/projects/imas_gpu/mast/level2/shots")
CANDIDATE_PATTERN = re.compile(r"toroidal|\btf\b|bphi|bvac|irod", re.IGNORECASE)


def current_rbphi(current: np.ndarray, *, linked_turns: float | None) -> np.ndarray:
    """Convert amperes to tesla metres with an explicitly supplied winding count."""
    if linked_turns is None or not np.isfinite(linked_turns) or linked_turns <= 0:
        raise ValueError(
            "an independently sourced positive linked turn count is required"
        )
    return MU_OVER_TWO_PI * linked_turns * np.asarray(current, dtype=float)


def compare_shot(
    nominal: np.ndarray,
    measured: np.ndarray,
    *,
    nominal_sources: set[str],
    measured_sources: set[str],
) -> dict:
    """Score aligned signed products using RMS error relative to measured RMS.

    Both source sets name acquisition ancestry, not merely output channel names.
    Missing and shared ancestry refuse the comparison. No gain, offset, sign or
    passive correction is fitted, and a nonfinite sample refuses the whole arm.
    """
    if (
        not nominal_sources
        or not measured_sources
        or nominal_sources & measured_sources
    ):
        raise ValueError("independent acquisition ancestry is required")
    nominal = np.asarray(nominal, dtype=float)
    measured = np.asarray(measured, dtype=float)
    if nominal.ndim != 1 or nominal.shape != measured.shape or not nominal.size:
        raise ValueError("nonempty aligned one-dimensional products are required")
    if not np.isfinite(nominal).all() or not np.isfinite(measured).all():
        raise ValueError("all compared samples must be finite")
    norm = float(np.linalg.norm(measured))
    if norm == 0:
        raise ValueError("a nonzero measured field is required")
    difference = float(np.linalg.norm(nominal - measured) / norm)
    return {
        "relative_difference": difference,
        "within_tolerance": difference <= RELATIVE_TOLERANCE,
    }


def aggregate_comparisons(rows: list[dict], *, expected_shots: int) -> dict:
    """Promote only complete independent coverage with both quantiles in tolerance."""
    values = [row["relative_difference"] for row in rows]
    if any(not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("relative differences must be finite and nonnegative")
    median, upper = (
        np.quantile(values, [0.5, 0.95]).tolist() if values else (None, None)
    )
    complete = expected_shots > 0 and len(rows) == expected_shots
    passed = complete and median <= RELATIVE_TOLERANCE and upper <= RELATIVE_TOLERANCE
    return {
        "verdict": "promoted" if passed else "unresolved",
        "scored_shots": len(rows),
        "expected_shots": expected_shots,
        "median_relative_difference": median,
        "p95_relative_difference": upper,
        "relative_agreement_tolerance": RELATIVE_TOLERANCE,
    }


def cohort_shots(census: dict) -> list[int]:
    """Reclassify the pinned census with the production calibration owner."""
    keys = {item.name for item in fields(ShotSurvey)}
    surveys = [
        ShotSurvey(**{k: v for k, v in row.items() if k in keys})
        for row in census["surveys"]
    ]
    return [
        row.shot
        for row in calibration_experiments(surveys)
        if row.experiment == ExperimentClass.TOROIDAL_FIELD_ONLY
    ]


def array_metadata(path: Path) -> dict[str, dict]:
    """Read the store's consolidated array inventory, including acquisition labels."""
    if not (path / ".zmetadata").exists():
        document = json.loads((path / "zarr.json").read_text())
        metadata = document["consolidated_metadata"]["metadata"]
        return {
            key: value.get("attributes", {}) | {"stored_shape": value["shape"]}
            for key, value in metadata.items()
            if value.get("node_type") == "array"
        }
    document = json.loads((path / ".zmetadata").read_text())["metadata"]
    return {
        key.removesuffix("/.zarray"): document.get(
            key.removesuffix(".zarray") + ".zattrs", {}
        )
        | {"stored_shape": value["shape"]}
        for key, value in document.items()
        if key.endswith("/.zarray")
    }


def candidate_reason(path: str) -> str:
    """State why each discovered source cannot yet validate an absolute TF average."""
    if path == "amc/tf_current":
        return (
            "feed current exists; effective linked turns and conductor mapping "
            "are unsourced"
        )
    if path.startswith(("efm/", "equilibrium/", "summary/")):
        return (
            "reconstruction product; independent acquisition ancestry "
            "is not established"
        )
    if (
        "b_field_tor_probe" in path
        or "cc/mt" in path.lower()
        or "omaha" in path.lower()
    ):
        return (
            "probe orientation unresolved; voltage/derivative calibration "
            "does not establish absolute R*Bphi"
        )
    return (
        "metadata does not establish an absolute axisymmetric TF measurement "
        "with radius and orientation"
    )


def finite_summary(group, path: str) -> dict:
    """Read a small scalar waveform and distinguish absent, empty and finite data."""
    if path not in group:
        return {"status": "absent"}
    values = np.asarray(group[path][...], dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "status": "finite" if finite.size else "no-finite-samples",
        "sample_count": int(values.size),
        "finite_count": int(finite.size),
        "minimum": float(finite.min()) if finite.size else None,
        "maximum": float(finite.max()) if finite.size else None,
    }


def inspect_shot(shot: int, store: Path, level2: Path) -> dict:
    """Inventory both stores and retain each refused candidate with its provenance."""
    import zarr

    row = {
        "shot": shot,
        "verdict": "unresolved",
        "relative_difference": None,
        "measured_rbphi": None,
        "nominal_rbphi": None,
        "candidates": [],
        "paths_tried": [],
        "errors": [],
    }
    for level, root in (("level1", store), ("level2", level2)):
        path = root / f"{shot}.zarr"
        row["paths_tried"].append(str(path))
        if not path.exists():
            row[level] = "absent"
            continue
        try:
            metadata = array_metadata(path)
        except (OSError, ValueError, KeyError) as error:
            row["errors"].append(f"{level}: {error}")
            continue
        row[level] = "readable"
        for name, attrs in sorted(metadata.items()):
            text = name + " " + str(attrs.get("description", ""))
            if CANDIDATE_PATTERN.search(text) or "b_field_tor_probe" in name:
                row["candidates"].append(
                    {
                        "level": level,
                        "path": name,
                        "metadata": attrs,
                        "accepted": False,
                        "reason": candidate_reason(name),
                    }
                )
        if level != "level1":
            continue
        group = zarr.open_group(str(path), mode="r")
        row["tf_current_kA"] = finite_summary(group, "amc/tf_current")
        row["positive_control"] = (
            "amc/tf_current" in metadata and row["tf_current_kA"]["status"] == "finite"
        )
        row["error_field_peaks_A"] = {}
        for channel in ERROR_FIELD_CHANNELS + ERROR_FIELD_ALIASES:
            summary = finite_summary(group, f"amc/{channel}")
            if summary["status"] == "finite":
                row["error_field_peaks_A"][channel] = KILO * max(
                    abs(summary["minimum"]), abs(summary["maximum"])
                )
        peaks = row["error_field_peaks_A"]
        row["error_field_screen"] = (
            "unmeasured"
            if not peaks
            else "driven"
            if max(peaks.values()) >= DRIVEN_CURRENT
            else "quiet"
        )
        row["reconstruction_product"] = {"status": "absent"}
        if "efm/bvac_r" in group and "efm/bvac_val" in group:
            radius = np.asarray(group["efm/bvac_r"][...], dtype=float)
            field = np.asarray(group["efm/bvac_val"][...], dtype=float)
            if radius.shape == field.shape:
                product = radius * field
                finite = product[np.isfinite(product)]
                row["reconstruction_product"] = {
                    "status": "finite" if finite.size else "no-finite-samples",
                    "finite_count": int(finite.size),
                    "minimum_T_m": float(finite.min()) if finite.size else None,
                    "maximum_T_m": float(finite.max()) if finite.size else None,
                    "independent": False,
                }
                if "efm/irod" in group:
                    rod = np.asarray(group["efm/irod"][...], dtype=float)
                    if rod.shape == product.shape:
                        keep = np.isfinite(product) & np.isfinite(rod) & (product != 0)
                        if keep.any():
                            row["reconstruction_product"][
                                "irod_identity_max_relative_error"
                            ] = float(
                                np.max(
                                    np.abs(MU_OVER_TWO_PI * rod[keep] - product[keep])
                                    / np.abs(product[keep])
                                )
                            )
            else:
                row["reconstruction_product"] = {"status": "shape-mismatch"}
    return row


def main() -> None:
    """Write a compact cohort receipt and the complete source inventory separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--level2-store", type=Path, default=LEVEL2_STORE)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/figures/mast-tf-average/receipt.json")
    )
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--inventory-control-shot", type=int, default=30420)
    args = parser.parse_args()
    census_bytes = args.census.read_bytes()
    shots = cohort_shots(json.loads(census_bytes))
    if not shots:
        raise ValueError("the calibration census contains no TF-only shots")
    print(
        f"Pre-registered tolerance: {RELATIVE_TOLERANCE}; cohort: {len(shots)}",
        flush=True,
    )
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(
            lambda shot: inspect_shot(shot, args.store, args.level2_store), shots
        ):
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f"Inventoried {len(rows)}/{len(shots)}", flush=True)
    control = inspect_shot(args.inventory_control_shot, args.store, args.level2_store)
    control_paths = [
        item["path"]
        for item in control["candidates"]
        if item["level"] == "level2" and "b_field_tor_probe" in item["path"]
    ]
    detail = {
        "rows": rows,
        "inventory_control_outside_scoring": control,
        "positive_control": (
            "finite amc/tf_current found by the same inventory and waveform reader"
        ),
    }
    detail_bytes = (json.dumps(detail, sort_keys=True, allow_nan=False) + "\n").encode()
    args.detail.parent.mkdir(parents=True, exist_ok=True)
    args.detail.write_bytes(detail_bytes)
    if not control_paths:
        raise ValueError("level-2 inventory failed to see the known toroidal channels")
    counts = Counter()
    catalog = {}
    for row in rows:
        for candidate in row["candidates"]:
            key = candidate["level"] + "/" + candidate["path"]
            counts[key] += 1
            catalog.setdefault(
                key,
                {
                    "reason": candidate["reason"],
                    "representative_metadata": candidate["metadata"],
                    "representative_shot": row["shot"],
                },
            )
    compact = []
    for row in rows:
        product = row.get("reconstruction_product", {})
        current = row.get("tf_current_kA", {})
        compact.append(
            {
                "shot": row["shot"],
                "verdict": row["verdict"],
                "relative_difference": None,
                "feed_current_peak_A": KILO
                * max(
                    abs(current.get("minimum") or 0),
                    abs(current.get("maximum") or 0),
                )
                if current.get("status") == "finite"
                else None,
                "reconstruction_rbphi_range_T_m": [
                    product.get("minimum_T_m"),
                    product.get("maximum_T_m"),
                ],
                "level2": row.get("level2", "unreadable"),
                "error_field_screen": row.get("error_field_screen", "unmeasured"),
            }
        )
    summary = aggregate_comparisons([], expected_shots=len(shots))
    summary.update(
        {
            "finite_feed_current_shots": sum(
                row.get("positive_control", False) for row in rows
            ),
            "finite_reconstruction_product_shots": sum(
                row.get("reconstruction_product", {}).get("status") == "finite"
                for row in rows
            ),
            "level2_present_shots": sum(
                row.get("level2") == "readable" for row in rows
            ),
            "inventory_errors": sum(len(row["errors"]) for row in rows),
            "error_field_screen": dict(
                Counter(row.get("error_field_screen", "unmeasured") for row in rows)
            ),
        }
    )
    receipt = {
        "schema": "mast-tf-average",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "census": {
            "path": str(args.census),
            "sha256": hashlib.sha256(census_bytes).hexdigest(),
            "classification_owner": (
                "nova.imas.mast_calibration_cohort.calibration_experiments"
            ),
            "scope": (
                "all TF-only shots in the pinned census; "
                "no claim of a fresh whole-store census"
            ),
        },
        "stores": {"level1": str(args.store), "level2": str(args.level2_store)},
        "observable": {
            "used": None,
            "candidate_product": "efm/bvac_r * efm/bvac_val [T m]",
            "reason": (
                "No admissible independent absolute R*Bphi measurement established "
                "in the searched stores. Reconstruction products have no independent "
                "acquisition ancestry; feed current has no sourced linked turn count; "
                "toroidal-probe orientation remains unresolved."
            ),
        },
        "nominal": {
            "law": "R*Bphi = mu0/(2*pi) * linked_turns * feed_current_A",
            "linked_turns": None,
            "source": (
                "nova/imas/mast_seed_parameters.py and "
                "nova/imas/mast_solve_inputs.py:toroidal_field_blocked"
            ),
            "fitted_corrections": [],
        },
        "scoring": {
            "shot_metric": (
                "RMS(nominal - measured) / RMS(measured) on aligned signed samples"
            ),
            "promotion": (
                "all cohort shots independently scored; median and p95 <= 0.02"
            ),
            "bounds": None,
            "orientation_assumed": False,
            "plasma_or_passive_correction": False,
        },
        "summary": summary,
        "inventory_control": {
            "shot": args.inventory_control_shot,
            "level2_toroidal_paths_found": control_paths,
            "scored": False,
        },
        "rejected_candidates": {
            key: value | {"shots_present": counts[key]}
            for key, value in sorted(catalog.items())
        },
        "detail": {
            "path": str(args.detail),
            "sha256": hashlib.sha256(detail_bytes).hexdigest(),
            "bytes": len(detail_bytes),
        },
        "rows": compact,
    }
    output = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    if len(output) > 300_000:
        raise ValueError(
            f"compact receipt exceeds repository size bound: {len(output)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print("SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    if (
        summary["finite_feed_current_shots"] != len(shots)
        or summary["inventory_errors"]
    ):
        raise ValueError(
            "inventory incomplete or positive control failed; inspect detail receipt"
        )


if __name__ == "__main__":
    main()
