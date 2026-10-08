"""Adjudicate directed toroidal probe axes against independent vacuum drives.

The shot cohort is fixed by the calibration owner. A signed reference field and
an independently acquired probe field are both required for inference; a feed
current alone cannot supply the reference without a sourced linked turn count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import fields
from pathlib import Path

import numpy as np

from nova.imas.mast_calibration_cohort import (
    ExperimentClass,
    calibration_experiments,
)
from nova.imas.mast_vacuum_cohort import SHOT_STORE, ShotSurvey

RELATIVE_TOLERANCE = 0.02
DEFAULT_CENSUS = Path.home() / ".cache/nova-mast/mast_vacuum_census.json"
LEVEL2_STORE = Path("/work/projects/imas_gpu/mast/level2/shots")
PROBE_CHANNELS = "magnetics/b_field_tor_probe_cc_geometry_channel"
SIGNAL_CHANNELS = "magnetics/b_field_tor_probe_cc_channel"
REFERENCE_SHOT = 30420


def infer_orientation(
    training: list[tuple[np.ndarray, np.ndarray]],
    held_out: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Infer axis sign on training shots and test absolute amplitude separately.

    Each pair is a signed, independently predicted B_phi and an aligned probe
    field in tesla. The signed reference fixes what positive e_phi means; neither
    its scale nor the probe response is fitted on held-out shots.
    """
    if not training or not held_out:
        return {
            "verdict": "unresolved",
            "reason": "training and held-out shots required",
        }
    arrays = []
    for reference, measured in training + held_out:
        reference = np.asarray(reference, dtype=float)
        measured = np.asarray(measured, dtype=float)
        if (
            reference.ndim != 1
            or reference.shape != measured.shape
            or not reference.size
            or not np.isfinite(reference).all()
            or not np.isfinite(measured).all()
        ):
            raise ValueError("aligned finite one-dimensional fields are required")
        arrays.append((reference, measured))
    n_training = len(training)
    signed_response = sum(
        float(np.dot(reference, measured))
        for reference, measured in arrays[:n_training]
    )
    if signed_response == 0:
        return {"verdict": "unresolved", "reason": "training sign is ambiguous"}
    sign = 1 if signed_response > 0 else -1
    differences = []
    for reference, measured in arrays[n_training:]:
        norm = float(np.linalg.norm(reference))
        if norm == 0:
            return {"verdict": "unresolved", "reason": "held-out field is zero"}
        differences.append(float(np.linalg.norm(measured - sign * reference) / norm))
    maximum = max(differences)
    promoted = maximum <= RELATIVE_TOLERANCE
    return {
        "verdict": "promoted" if promoted else "unresolved",
        "orientation_sign": sign if promoted else None,
        "field_direction": ("+e_phi" if sign > 0 else "-e_phi") if promoted else None,
        "candidate_sign": sign,
        "candidate_direction": "+e_phi" if sign > 0 else "-e_phi",
        "held_out_relative_errors": differences,
        "maximum_held_out_relative_error": maximum,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "training_shots": n_training,
        "held_out_shots": len(held_out),
    }


def cohort_shots(census: dict) -> list[int]:
    """Select only the calibration owner's plasma-free TF drive class."""
    names = {item.name for item in fields(ShotSurvey)}
    surveys = [
        ShotSurvey(**{key: value for key, value in row.items() if key in names})
        for row in census["surveys"]
    ]
    return [
        row.shot
        for row in calibration_experiments(surveys)
        if row.experiment == ExperimentClass.TOROIDAL_FIELD_ONLY
    ]


def score_observations(
    rows: list[dict], *, shots: list[int], probes: list[str]
) -> dict[str, dict]:
    """Score independent signed fields, reserving every fifth shot per probe."""
    admitted = set(shots)
    known = set(probes)
    grouped: dict[str, list[dict]] = {probe: [] for probe in probes}
    seen: set[tuple[str, int]] = set()
    for row in rows:
        probe, shot = str(row["probe"]), int(row["shot"])
        if probe not in known or shot not in admitted:
            raise ValueError("observation is outside the probe or TF-only cohort")
        identity = (probe, shot)
        if identity in seen:
            raise ValueError("duplicate probe and shot observation")
        seen.add(identity)
        if (
            not row.get("reference_source")
            or not row.get("measured_source")
            or row["reference_source"] == row["measured_source"]
        ):
            raise ValueError("independent reference and acquisition sources required")
        grouped[probe].append(row)
    outcomes = {}
    for probe, observations in grouped.items():
        ordered = sorted(observations, key=lambda row: int(row["shot"]))
        pairs = [
            (np.asarray(row["reference_T"]), np.asarray(row["measured_T"]))
            for row in ordered
        ]
        training = [pair for index, pair in enumerate(pairs) if index % 5 != 4]
        held_out = [pair for index, pair in enumerate(pairs) if index % 5 == 4]
        outcomes[probe] = infer_orientation(training, held_out)
        outcomes[probe]["training_shot_ids"] = [
            int(row["shot"]) for index, row in enumerate(ordered) if index % 5 != 4
        ]
        outcomes[probe]["held_out_shot_ids"] = [
            int(row["shot"]) for index, row in enumerate(ordered) if index % 5 == 4
        ]
    return outcomes


def inventory(
    census_path: Path, level2: Path, reference_shot: int, observations: Path | None
) -> dict:
    """Record cohort availability with a positive control for the probe reader."""
    import zarr

    census_bytes = census_path.read_bytes()
    shots = cohort_shots(json.loads(census_bytes))
    if not shots:
        raise ValueError("the calibration census has no TF-only shots")
    control_path = level2 / f"{reference_shot}.zarr"
    if not control_path.exists():
        raise ValueError("the known toroidal-probe inventory control is absent")
    control = zarr.open_group(str(control_path), mode="r")
    if PROBE_CHANNELS not in control or SIGNAL_CHANNELS not in control:
        raise ValueError("the inventory control cannot see known toroidal probes")
    identities = [str(value) for value in control[PROBE_CHANNELS][...]]
    signal_identities = [str(value) for value in control[SIGNAL_CHANNELS][...]]
    if not identities or not signal_identities:
        raise ValueError("the inventory control has no probe identities")
    available = [shot for shot in shots if (level2 / f"{shot}.zarr").exists()]
    reason = (
        "no level-2 toroidal probe shot in the TF-only cohort"
        if not available
        else "independent signed TF reference and probe calibration not established"
    )
    probes = [
        {
            "probe": identity,
            "verdict": "unresolved",
            "orientation_sign": None,
            "field_direction": None,
            "held_out_relative_error": None,
            "reason": reason,
        }
        for identity in identities
    ]
    if observations is not None:
        outcomes = score_observations(
            json.loads(observations.read_text()), shots=shots, probes=identities
        )
        for probe in probes:
            result = outcomes[probe["probe"]]
            probe.update(result)
            if result["verdict"] == "promoted":
                probe.pop("reason", None)
            elif "reason" not in result:
                probe["reason"] = "held-out amplitude exceeds tolerance"
            probe["held_out_relative_error"] = result.get(
                "maximum_held_out_relative_error"
            )
    return {
        "schema": "mast-toroidal-probe-orientation",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "census": {
            "path": str(census_path),
            "sha256": hashlib.sha256(census_bytes).hexdigest(),
            "classification_owner": (
                "nova.imas.mast_calibration_cohort.calibration_experiments"
            ),
            "toroidal_field_only_shots": len(shots),
        },
        "stores": {"level1": str(SHOT_STORE), "level2": str(level2)},
        "positive_control": {
            "shot": reference_shot,
            "probe_geometry_count": len(identities),
            "measured_channel_count": len(signal_identities),
            "measured_channels": signal_identities,
            "in_cohort": reference_shot in shots,
        },
        "level2_cohort_shots": available,
        "observation_source": str(observations) if observations else None,
        "scoring": {
            "training": "infer sign from independently predicted and measured B_phi",
            "held_out": "maximum RMS field error / RMS independent reference <= 0.02",
            "relative_tolerance": RELATIVE_TOLERANCE,
            "reference_required": (
                "sourced signed TF field; current alone is insufficient"
            ),
        },
        "summary": {
            "total_probes": len(probes),
            "promoted": sum(row["verdict"] == "promoted" for row in probes),
            "unresolved": sum(row["verdict"] == "unresolved" for row in probes),
            "level2_cohort_shots": len(available),
        },
        "probes": probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--level2-store", type=Path, default=LEVEL2_STORE)
    parser.add_argument("--reference-shot", type=int, default=REFERENCE_SHOT)
    parser.add_argument(
        "--observations",
        type=Path,
        help="independently sourced signed reference and probe fields per TF-only shot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/mast-toroidal-probe-orientation/receipt.json"),
    )
    args = parser.parse_args()
    print(
        f"Pre-registered held-out relative tolerance: {RELATIVE_TOLERANCE}",
        flush=True,
    )
    receipt = inventory(
        args.census, args.level2_store, args.reference_shot, args.observations
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
