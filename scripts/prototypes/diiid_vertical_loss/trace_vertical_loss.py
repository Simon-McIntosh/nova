"""Per-trip trajectory comparison of DIII-D frames 146 (loses plasma) and 137 (holds).

Reads the per-arm records written by the forward-convergence probe and tabulates,
for each active-set trip, the boundary-set size and the residual, beside the
promotion amplitude and the terminal plasma presence.  Those four columns are
what separate a branch that walks to a held diverted equilibrium from one that
walks out of the vessel.

The arm and presence fields are read through ``build_receipt._arm_summary``, the
receipt-facing extraction the convergence record already uses, so this tabulation
adds only the per-trip pairing and the promotion amplitude and duplicates no
field list of its own.

Usage:
    python trace_vertical_loss.py --probe <probe.json> --out <receipt.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "diiid_forward_convergence"),
)

from build_receipt import _arm_summary

FRAMES = (146, 137)
BRANCHES = ("diverted", "limited")
ARM_NAMES = ("default", "continued")


def _trips(arm: dict[str, Any]) -> list[dict[str, Any]]:
    """Pair each active-set trip's boundary-set size with its residual."""
    summary = _arm_summary(arm)
    if summary.get("status") != "solved":
        return []
    masks = list(summary.get("mask_history") or [])
    residuals = list(summary.get("residual_history") or [])
    return [
        {"trip": trip, "active_cells": mask, "residual": residual}
        for trip, (mask, residual) in enumerate(zip(masks, residuals, strict=False))
    ]


def _arm(arm: dict[str, Any]) -> dict[str, Any]:
    summary = _arm_summary(arm)
    return {
        "status": summary.get("status"),
        "achieved_class": summary.get("achieved_class"),
        "terminal_residual": summary.get("terminal_residual"),
        "termination_reason": summary.get("termination_reason"),
        "terminal": summary.get("terminal"),
        "wall_contact": summary.get("wall_contact"),
        "presence": summary.get("presence") or {},
        # The first promotion amplitude is the discriminator between the two
        # frames; the receipt summary does not carry it, so it is read here.
        "promotion_amplitude_wb": (arm.get("amplitude_history") or [None])[0],
        "trips": _trips(arm),
    }


def tabulate(probe: dict[str, Any]) -> dict[str, Any]:
    receipt: dict[str, Any] = {"tolerance": probe["tolerance"], "frames": {}}
    for frame in FRAMES:
        record = probe["frames"][str(frame)]
        entry: dict[str, Any] = {
            "time_ms": record["time_ms"],
            "target_current_a": record["target_current_a"],
            "label_axis_rz_m": record["label_axis_rz_m"],
            "label_axis_inside_limiter": record["label_axis_inside_limiter"],
            "label_saddle_rz_m": record["label_saddle_rz_m"],
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
                    for name in ARM_NAMES
                    if name in container["arms"]
                },
            }
        receipt["frames"][str(frame)] = entry
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = tabulate(json.loads(arguments.probe.read_text()))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(receipt, indent=2) + "\n")
    for frame, entry in receipt["frames"].items():
        diverted = entry["branches"]["diverted"]
        limited = entry["branches"]["limited"]
        print(
            f"frame {frame}: seed_l2 diverted "
            f"{diverted['seed_l2_distance_from_label_wb']:.2f} Wb / limited "
            f"{limited['seed_l2_distance_from_label_wb']:.2f} Wb"
        )
        for name, arm in diverted["arms"].items():
            presence = arm["presence"]
            print(
                f"  diverted/{name}: amp {arm['promotion_amplitude_wb']:.4g} Wb "
                f"axis Z {presence['axis_rz_m'][1]:+.4f} m "
                f"inside {presence['axis_inside_limiter']} "
                f"ratio {presence['confined_to_target_ratio']:.4g} "
                f"mask {[trip['active_cells'] for trip in arm['trips']][:6]}"
            )


if __name__ == "__main__":
    main()
