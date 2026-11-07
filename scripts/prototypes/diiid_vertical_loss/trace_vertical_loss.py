"""Per-trip trajectory comparison of DIII-D frames 146 (loses plasma) and 137 (holds).

Reads the per-arm records written by the forward-convergence probe and tabulates,
for each active-set trip, the boundary-set size, the residual, the promotion
amplitude and the terminal plasma presence.  These four columns are what separate
a branch that walks to a held diverted equilibrium from one that walks out of the
vessel.

Usage:
    python trace_vertical_loss.py --probe <probe.json> --out <receipt.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FRAMES = ("146", "137")
BRANCHES = ("diverted", "limited")
ARMS = ("default", "continued")


def _trips(arm: dict) -> list[dict]:
    masks = arm["mask_history"]
    residuals = arm["residual_history"]
    return [
        {
            "trip": i,
            "active_cells": mask,
            "residual": residual,
        }
        for i, (mask, residual) in enumerate(zip(masks, residuals, strict=False))
    ]


def _presence(arm: dict) -> dict:
    presence = arm["presence"] or {}
    return {
        "axis_rz_m": presence.get("axis_rz_m"),
        "axis_inside_limiter": presence.get("axis_inside_limiter"),
        "closed_core_cells": presence.get("closed_core_cells"),
        "confined_current_a": presence.get("confined_current_a"),
        "confined_to_target_ratio": presence.get("confined_to_target_ratio"),
        "target_current_a": presence.get("target_current_a"),
        "plasma_area_m2": presence.get("plasma_area_m2"),
    }


def _arm(arm: dict) -> dict:
    return {
        "achieved_class": arm["achieved_class"],
        "terminal_residual": arm["terminal_residual"],
        "termination_reason": arm["termination_reason"],
        "terminal": arm["terminal"],
        "wall_contact": arm["wall_contact"],
        "promotion_amplitude_wb": (arm["amplitude_history"] or [None])[0],
        "trips": _trips(arm),
        "presence": _presence(arm),
    }


def tabulate(probe: dict) -> dict:
    receipt: dict = {"tolerance": probe["tolerance"], "frames": {}}
    for frame in FRAMES:
        record = probe["frames"][frame]
        entry = {
            "time_ms": record["time_ms"],
            "target_current_a": record["target_current_a"],
            "label_axis_rz_m": record["label_axis_rz_m"],
            "label_axis_inside_limiter": record["label_axis_inside_limiter"],
            "label_saddle_rz_m": record["label_saddle_rz_m"],
            "limiter_ring_point_count": record["limiter_ring_points"],
            "branches": {},
        }
        for branch in BRANCHES:
            container = record["branches"][branch]
            entry["branches"][branch] = {
                "requested_class": container["requested_class"],
                "seed_l2_distance_from_label_wb": container[
                    "seed_l2_distance_from_label"
                ],
                "seed_identity_with_label": container["seed_identity_with_label"],
                "arms": {
                    name: _arm(container["arms"][name])
                    for name in ARMS
                    if name in container["arms"]
                },
            }
        receipt["frames"][frame] = entry
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = tabulate(json.loads(args.probe.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    for frame, entry in receipt["frames"].items():
        diverted = entry["branches"]["diverted"]
        limited = entry["branches"]["limited"]
        print(
            f"frame {frame}: seed_l2 diverted "
            f"{diverted['seed_l2_distance_from_label_wb']:.2f} "
            f"Wb / limited {limited['seed_l2_distance_from_label_wb']:.2f} Wb"
        )
        for name, arm in diverted["arms"].items():
            print(
                f"  diverted/{name}: amp {arm['promotion_amplitude_wb']:.4g} Wb "
                f"axis Z {arm['presence']['axis_rz_m'][1]:+.4f} m "
                f"inside {arm['presence']['axis_inside_limiter']} "
                f"ratio {arm['presence']['confined_to_target_ratio']:.4g} "
                f"mask {[t['active_cells'] for t in arm['trips']][:6]}"
            )


if __name__ == "__main__":
    main()
