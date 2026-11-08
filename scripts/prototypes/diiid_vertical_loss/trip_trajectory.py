"""Persist the per-active-set-trip trajectory of one DIII-D frame's diverted branch.

Each frame is re-solved once per trip count: ``active_set_steps=k`` stops the
solve after exactly k promotions, so the state at trip k is a real solve state
rather than a post-hoc read of a terminal.  For every k the driver writes the
axis position, the confined current against the target, the boundary-set size,
the promotion step norm and the flux grid, as one npz per trip.

The governed DIII-D limiter is read from the frame's archived terminal grid,
because ``_prepare_frame`` returns the machine wall.  Containment is judged
against that limiter, and the step norm of trip k is the norm of the difference
between the trip k and trip k-1 flux states, so a re-solve reports its own trip
rather than the first one.

Arm and presence fields are read through ``build_receipt._arm_summary``.

Usage:
    python trip_trajectory.py --frames 146 137 --out-dir <dir> --artifacts <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "diiid_forward_convergence"),
)

import measure_branches as measure
from build_receipt import _arm_summary
from nova.equilibrium.topology import TopologyClass
from nova.media.poloidal import inside_wall_units

EXPANSION = 0.02
MAX_TRIPS = 8


def _limiter(artifacts: Path, frame: int) -> np.ndarray:
    """Return the governed limiter ring archived for this frame's diverted arm."""
    archive = artifacts / f"frame-{frame}-diverted-default.npz"
    with np.load(archive) as data:
        return np.asarray(data["limiter"], dtype=float)


def _trip_state(
    k: int, arm: dict[str, Any], limiter: np.ndarray, step_norm: float | None
) -> dict[str, Any]:
    summary = _arm_summary(arm)
    presence = summary.get("presence") or {}
    masks = list(summary.get("mask_history") or [])
    axis = presence.get("axis_rz_m")
    return {
        "trip": k,
        "axis_rz_m": axis,
        "axis_inside_limiter": bool(
            inside_wall_units(np.asarray(axis, dtype=float)[None, :], limiter)[0]
        )
        if axis is not None
        else None,
        "confined_current_a": presence.get("confined_current_a"),
        "target_current_a": presence.get("target_current_a"),
        "confined_to_target_ratio": presence.get("confined_to_target_ratio"),
        "closed_core_cells": presence.get("closed_core_cells"),
        "changed_mask_entries": masks[k - 1]
        if k - 1 < len(masks)
        else (masks[-1] if masks else None),
        "promotion_step_norm_wb": step_norm,
        "residual": summary.get("terminal_residual"),
        "achieved_class": summary.get("achieved_class"),
        "status": summary.get("status"),
    }


def run(frame: int, out_dir: Path, artifacts: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    limiter = _limiter(artifacts, frame)
    prepared = measure._prepare_frame(frame, EXPANSION)
    profile = prepared["profile"]
    seed = np.asarray(
        prepared["cold"].branches.flux[int(TopologyClass.DIVERTED)], dtype=float
    )
    carrier = f"diiid-vertical-loss:{EXPANSION}:DIVERTED:frame{frame}"
    label_axis = np.asarray(prepared["axis"], dtype=float)
    record: dict[str, Any] = {
        "frame": frame,
        "branch": "diverted",
        "seed_l2_distance_from_label_wb": float(
            np.linalg.norm(seed - prepared["label_seed"])
        ),
        "target_current_a": prepared["target_current_a"],
        "label_axis_rz_m": label_axis.tolist(),
        "label_axis_inside_limiter": bool(
            inside_wall_units(label_axis[None, :], limiter)[0]
        ),
        "limiter_rz_bounds": [
            [float(limiter[:, 0].min()), float(limiter[:, 0].max())],
            [float(limiter[:, 1].min()), float(limiter[:, 1].max())],
        ],
        "trips": [],
    }
    base_arrays = {
        "wall": prepared["wall"],
        "limiter": limiter,
        "label": prepared["label"],
        "radius": np.asarray(profile.lattice.radius),
        "height": np.asarray(profile.lattice.height),
        "label_boundary": prepared["contour"],
    }
    previous: np.ndarray | None = None
    for k in range(1, MAX_TRIPS + 1):
        started = time.perf_counter()
        try:
            receipt, seconds = measure._solve(
                profile,
                seed,
                prepared["current"],
                prepared["target_current_a"],
                f"{carrier}:{k}",
                active_set_steps=k,
            )
        except Exception as error:
            record["trips"].append(
                {
                    "trip": k,
                    "status": "refused",
                    "refusal_type": type(error).__name__,
                    "refusal_message": str(error),
                    "traceback": traceback.format_exc()[-2000:],
                }
            )
            break
        arm = measure._record_receipt(
            receipt, seconds, prepared, TopologyClass.DIVERTED
        )
        flux = np.asarray(
            receipt.equilibrium.flux[: profile.lattice.node_count], dtype=float
        ).reshape(profile.lattice.shape)
        step_norm = (
            float(np.linalg.norm(flux - previous)) if previous is not None else None
        )
        state = _trip_state(k, arm, limiter, step_norm)
        state["wall_seconds"] = round(time.perf_counter() - started, 1)
        record["trips"].append(state)
        arrays = dict(base_arrays)
        arrays["flux"] = flux
        np.savez_compressed(out_dir / f"frame-{frame}-trip-{k}.npz", **arrays)
        print(json.dumps(state, sort_keys=True, default=float), flush=True)
        (out_dir / "trajectory.json").write_text(json.dumps(record, indent=2) + "\n")
        previous = flux
        if (
            state["changed_mask_entries"] in (0, -1)
            and arm["active_set_iterations"] <= k
        ):
            break
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    arguments = parser.parse_args()
    for frame in arguments.frames:
        print(
            json.dumps({"frame": frame, "phase": "start"}, sort_keys=True), flush=True
        )
        run(frame, arguments.out_dir, arguments.artifacts)


if __name__ == "__main__":
    main()
