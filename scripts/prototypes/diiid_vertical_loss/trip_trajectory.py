"""Persist the per-active-set-trip trajectory of one DIII-D frame's diverted branch.

Each frame is re-solved once per trip count: ``active_set_steps=k`` stops the
solve after exactly k promotions, so the state at trip k is a real solve state
rather than a post-hoc read of a terminal.  For every k the driver writes the
axis position, the confined current against the target, the boundary-set size,
the promotion step norm (the flux amplitude of the accepted update), and the
flux grid, as one npz per trip.

Usage:
    python trip_trajectory.py --frame 146 --out-dir <dir> --receipt <receipt.json>
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
from nova.equilibrium.topology import TopologyClass

EXPANSION = 0.02
MAX_TRIPS = 8


def _trip_state(
    k: int, arm: dict[str, Any], prepared: dict[str, Any]
) -> dict[str, Any]:
    presence = arm["presence"] or {}
    masks = arm.get("mask_history") or []
    amplitudes = arm.get("amplitude_history") or []
    return {
        "trip": k,
        "axis_rz_m": presence.get("axis_rz_m"),
        "axis_inside_limiter": presence.get("axis_inside_limiter"),
        "confined_current_a": presence.get("confined_current_a"),
        "target_current_a": presence.get("target_current_a"),
        "confined_to_target_ratio": presence.get("confined_to_target_ratio"),
        "closed_core_cells": presence.get("closed_core_cells"),
        "boundary_set_size": masks[k]
        if k < len(masks)
        else (masks[-1] if masks else None),
        "promotion_step_norm_wb": amplitudes[0] if amplitudes else None,
        "residual": arm.get("terminal_residual"),
        "status": arm.get("status"),
    }


def run(frame: int, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = measure._prepare_frame(frame, EXPANSION)
    profile = prepared["profile"]
    seed = np.asarray(
        prepared["cold"].branches.flux[int(TopologyClass.DIVERTED)], dtype=float
    )
    carrier = f"diiid-vertical-loss:{EXPANSION}:DIVERTED:frame{frame}"
    record: dict[str, Any] = {
        "frame": frame,
        "branch": "diverted",
        "seed_l2_distance_from_label_wb": float(
            np.linalg.norm(seed - prepared["label_seed"])
        ),
        "target_current_a": prepared["target_current_a"],
        "label_axis_rz_m": [float(v) for v in prepared["axis"]],
        "label_saddle_rz_m": [float(v) for v in prepared["saddle"]],
        "label_axis_inside_limiter": None,
        "trips": [],
    }
    base_arrays = {
        "wall": prepared["wall"],
        "label": prepared["label"],
        "radius": np.asarray(profile.lattice.radius),
        "height": np.asarray(profile.lattice.height),
        "label_boundary": prepared["contour"],
        "limiter": prepared["wall"],
    }
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
        state = _trip_state(k, arm, prepared)
        state["wall_seconds"] = round(time.perf_counter() - started, 1)
        record["trips"].append(state)
        arrays = dict(base_arrays)
        arrays["flux"] = np.asarray(
            receipt.equilibrium.flux[: profile.lattice.node_count], dtype=float
        ).reshape(profile.lattice.shape)
        np.savez_compressed(out_dir / f"frame-{frame}-trip-{k}.npz", **arrays)
        print(json.dumps(state, sort_keys=True, default=float), flush=True)
        (out_dir / "trajectory.json").write_text(json.dumps(record, indent=2) + "\n")
        if state["boundary_set_size"] in (0, -1) and arm["active_set_iterations"] <= k:
            break
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    arguments = parser.parse_args()
    for frame in arguments.frames:
        print(
            json.dumps({"frame": frame, "phase": "start"}, sort_keys=True), flush=True
        )
        run(frame, arguments.out_dir)


if __name__ == "__main__":
    main()
