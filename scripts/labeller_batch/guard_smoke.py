#!/usr/bin/env python3
"""Compare guarded and free-only labeller sessions from one smoke job."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from nova.equilibrium.steering_frames import SESSION_GROUP, frames_from_session


FRAME_FIELDS = (
    "p_prime_face",
    "ff_prime_face",
    "current_centroid_r",
    "current_centroid_z",
    "reference_centroid_z",
    "branch_guard_ok",
)
COMPANION_FIELDS = (
    "conditioned",
    "conditioning_target_source",
    "free_guard_evaluated",
    "free_branch_guard_ok",
    "conditioned_guard_evaluated",
    "conditioned_branch_guard_ok",
    "free_centroid_error_m",
    "conditioned_centroid_error_m",
)


def _fraction(numerator: int, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _load_arm(root: Path, shot: int, *, flag_enabled: bool) -> dict[str, Any]:
    """Read one arm and assert its frame and companion contracts."""
    if not root.is_absolute():
        raise ValueError(f"smoke session root must be absolute: {root}")
    manifest_path = root / f"{shot}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError(f"labeller arm is not complete: {manifest_path}")
    rows = [row for row in manifest["slices"] if row.get("written")]
    if bool(manifest["constraint"]["condition_on_guard_failure"]) != flag_enabled:
        raise ValueError(f"conditioning flag disagrees with arm: {manifest_path}")

    session_path = root / f"{shot}.nc"
    with xr.open_dataset(session_path, group=SESSION_GROUP, cache=True) as stored:
        dataset = stored.load()
    missing = sorted(set(FRAME_FIELDS) - set(dataset.variables))
    if missing:
        raise ValueError(f"session is missing frame fields: {missing}")
    frames = frames_from_session(dataset)
    if len(frames) != len(rows):
        raise ValueError("session frame count does not match written slice count")
    for frame, row in zip(frames, rows, strict=True):
        if not np.all(np.isfinite(np.asarray(frame.p_prime_face))):
            raise ValueError("p_prime_face must be finite for every admitted slice")
        if not np.all(np.isfinite(np.asarray(frame.ff_prime_face))):
            raise ValueError("ff_prime_face must be finite for every admitted slice")
        if not math.isfinite(float(frame.reference_centroid_z)):
            raise ValueError("reference_centroid_z must be finite for admitted slices")
        if not row.get("geometry_masked"):
            if not math.isfinite(float(frame.current_centroid_r)):
                raise ValueError("current_centroid_r must be finite on solved frames")
            if not math.isfinite(float(frame.current_centroid_z)):
                raise ValueError("current_centroid_z must be finite on solved frames")
        if bool(frame.branch_guard_ok) != bool(row["branch_guard_ok"]):
            raise ValueError("stored frame and slice manifest disagree on branch guard")

    companion_path = root / f"{shot}.npz"
    with np.load(companion_path, allow_pickle=False) as companion:
        companion_keys = sorted(companion.files)
        missing_companion = sorted(set(COMPANION_FIELDS) - set(companion.files))
        if missing_companion:
            raise ValueError(f"companion is missing fields: {missing_companion}")
        forbidden = {
            "psi_norm",
            "p_prime",
            "ff_prime",
            "current_centroid_r",
            "current_centroid_z",
            "reference_centroid_z",
            "branch_guard_ok",
        }
        retained = sorted(forbidden & set(companion.files))
        if retained:
            raise ValueError(f"frame-resident fields remain in companion: {retained}")

    warm_rows = rows[1:]
    warm_seconds = sum(float(row["wall_seconds"]) for row in warm_rows)
    free_guarded = [row for row in rows if row.get("free_branch_guard_ok") is not None]
    conditioned_guarded = [
        row for row in rows if row.get("conditioned_branch_guard_ok") is not None
    ]
    conditioned = [row for row in rows if row.get("conditioned")]
    return {
        "output_root": str(root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "session": str((root / f"{shot}.nc").resolve()),
        "companion": str(companion_path.resolve()),
        "admitted": len(rows),
        "converged": sum(bool(row.get("converged")) for row in rows),
        "qualified": sum(bool(row.get("qualified")) for row in rows),
        "guard_within_50mm_before": sum(
            bool(row["free_branch_guard_ok"]) for row in free_guarded
        ),
        "guard_evaluated_before": len(free_guarded),
        "guard_within_50mm_conditioned": sum(
            bool(row["conditioned_branch_guard_ok"]) for row in conditioned_guarded
        ),
        "guard_evaluated_conditioned": len(conditioned_guarded),
        "guard_within_50mm_after": sum(
            bool(row.get("branch_guard_ok")) for row in rows
        ),
        "conditioned": len(conditioned),
        "exceptions": sum(bool(row.get("exception")) for row in rows),
        "warm_slices": len(warm_rows),
        "warm_wall_seconds": warm_seconds,
        "slices_per_second_warm": _fraction(len(warm_rows), warm_seconds),
        "frame_fields": list(FRAME_FIELDS),
        "finite_p_prime_frames": sum(
            np.all(np.isfinite(np.asarray(frame.p_prime_face))) for frame in frames
        ),
        "finite_ff_prime_frames": sum(
            np.all(np.isfinite(np.asarray(frame.ff_prime_face))) for frame in frames
        ),
        "finite_current_centroid_frames": sum(
            math.isfinite(float(frame.current_centroid_r))
            and math.isfinite(float(frame.current_centroid_z))
            for frame in frames
        ),
        "finite_reference_centroid_frames": sum(
            math.isfinite(float(frame.reference_centroid_z)) for frame in frames
        ),
        "companion_fields": companion_keys,
        "companion_fields_without_frame_home": manifest[
            "companion_fields_without_frame_home"
        ],
    }


def _plot_receipt(receipt: dict[str, Any], figure: Path) -> None:
    """Plot the two arms' coverage, guard, and conditioning counts."""
    arms = receipt["arms"]
    labels = ("free only", "condition on guard failure")
    rows = (arms["free_only"], arms["condition_on_guard_failure"])
    metrics = (
        ("admitted", "admitted"),
        ("converged", "converged"),
        ("qualified", "qualified"),
        ("guard before", "guard_within_50mm_before"),
        ("guard after", "guard_within_50mm_after"),
        ("conditioned", "conditioned"),
        ("exceptions", "exceptions"),
    )
    x = np.arange(len(labels), dtype=float)
    width = 0.11
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for index, (name, key) in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2.0) * width
        values = [int(row[key]) for row in rows]
        bars = axis.bar(x + offset, values, width=width, label=name)
        axis.bar_label(bars, padding=2, fontsize=8)
    warm = [row["slices_per_second_warm"] for row in rows]
    axis.set_title(
        f"Shot {receipt['shot']} labeller guard conditioning\n"
        f"warm slices/s: free {warm[0]:.3f}, conditioned {warm[1]:.3f}"
    )
    axis.set_ylabel("slice count")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, fontsize=8, loc="upper center")
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180)
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioned-root", type=Path, required=True)
    parser.add_argument("--free-root", type=Path, required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write one quantitative comparison receipt."""
    arguments = _parser().parse_args(argv)
    for name in ("conditioned_root", "free_root", "output", "figure"):
        path = getattr(arguments, name)
        if not path.is_absolute():
            raise ValueError(f"{name} must be an absolute path: {path}")
    receipt = {
        "schema": "nova-forward-labeller-guard-conditioning-smoke",
        "shot": arguments.shot,
        "conditioning_trigger": (
            "free solve raised, did not converge, or missed the 0.05 m branch guard"
        ),
        "arms": {
            "condition_on_guard_failure": _load_arm(
                arguments.conditioned_root,
                arguments.shot,
                flag_enabled=True,
            ),
            "free_only": _load_arm(
                arguments.free_root,
                arguments.shot,
                flag_enabled=False,
            ),
        },
    }
    if receipt["arms"]["condition_on_guard_failure"]["conditioned"] == 0:
        raise ValueError("conditioned arm did not exercise its guarded re-solve")
    if receipt["arms"]["free_only"]["conditioned"] != 0:
        raise ValueError("free-only arm unexpectedly conditioned a slice")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_receipt(receipt, arguments.figure)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
