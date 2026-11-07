"""Build the per-frame convergence receipt.

Two sources are read and nothing is re-solved here.  The banked partial receipt
carries, for every gate frame, the executed branch trips of the branch the bank
kept, both seeds' terminal residuals and each terminal's achieved topology.  The
measurement file, when one is given, carries the both-branch descent this node
ran: the per-promotion relative residual and mask difference, the executed
inner-step tally of each arm (the solver's own promotion counts are published on
its fixed point result and are not in the stored record), and the
plasma-presence read of each arm.

Every number in the output is quoted with the path it came from, and a field the
sources do not carry is left absent rather than filled with a plausible value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from terminal_class import DEFAULT_TOLERANCE, classify_terminal

FRAMES = (179, 44, 144, 146, 137)
ARM_NAMES = ("default", "continued")


def _arm_summary(arm: dict[str, Any]) -> dict[str, Any]:
    """Return the receipt-facing summary of one measured solve arm."""

    if arm.get("status") != "solved":
        return {
            "status": arm.get("status"),
            "refusal_type": arm.get("refusal_type"),
            "refusal_message": arm.get("refusal_message"),
        }
    return {
        "status": "solved",
        "terminal_residual": arm.get("terminal_residual"),
        "residual_history": arm.get("residual_history"),
        "mask_history": arm.get("mask_history"),
        "attempted_promotions": arm.get("attempted_promotions"),
        "accepted_promotions": arm.get("accepted_promotions"),
        "promotion_counts_measured": arm.get("promotion_counts_measured"),
        "inner_step_trace": arm.get("inner_step_trace"),
        "globalisation_factors": arm.get("globalisation_factors"),
        "termination_reason": arm.get("termination_reason"),
        "active_set_iterations": arm.get("active_set_iterations"),
        "achieved_class": arm.get("achieved_class"),
        "topology_matches_request": arm.get("topology_matches_request"),
        # The class is derived data: recompute it from the recorded history so a
        # corrected classifier reaches every receipt without a re-solve.
        "terminal": classify_terminal(
            list(arm.get("residual_history") or []),
            arm.get("terminal_residual"),
        )
        if arm.get("residual_history")
        else arm.get("terminal"),
        "presence": arm.get("presence"),
        "x_point_rz_m": arm.get("x_point_rz_m"),
        "x_point_inside_wall": arm.get("x_point_inside_wall"),
        "wall_contact": arm.get("wall_contact"),
        "wall_seconds": arm.get("wall_seconds"),
    }


def _banked_branch(record: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one branch's banked terminal and the trips the bank executed."""

    selection = record.get("branch_selection") or {}
    branch = dict((selection.get("branches") or {}).get(name) or {})
    if name == "diverted":
        residuals = list(record.get("active_set_residuals") or [])
        masks = list(record.get("active_set_mask_differences") or [])
        damping = list(record.get("active_set_cycle_damping_activations") or [])
        branch["active_set_trips"] = int(record.get("active_set_iterations") or 0)
    else:
        residuals, masks, damping = [], [], []
        branch.setdefault("active_set_trips", branch.get("iterations"))
    branch["promotion_trajectory"] = residuals
    branch["mask_differences"] = masks
    branch["cycle_damping_activations"] = damping
    if residuals or branch.get("residual") is None:
        branch["terminal"] = classify_terminal(
            residuals, branch.get("residual"), DEFAULT_TOLERANCE
        )
    else:
        branch["terminal"] = {
            "class": "not measured",
            "evidence": (
                "the bank executes the limited seed and returns only the diverted "
                "receipt, so this branch has a terminal residual and no trips"
            ),
        }
    return branch


def build(
    partial_receipt: Path, receipt: Path, measurement: Path | None
) -> dict[str, Any]:
    partial = json.loads(partial_receipt.read_text())
    json.loads(receipt.read_text())
    records = {int(record["frame"]): record for record in partial["frame_records"]}
    probe: dict[int, Any] = {}
    if measurement is not None and measurement.exists():
        raw = json.loads(measurement.read_text())
        probe = {int(key): value for key, value in raw.get("frames", {}).items()}
    frames: dict[str, Any] = {}
    for frame in FRAMES:
        record = records[frame]
        entry: dict[str, Any] = {
            "frame": frame,
            "shot": record["shot"],
            "time_ms": record.get("time_ms"),
            "target_current_a": record.get("target_current_a"),
            "pseudo_wall_expansion": record.get("pseudo_wall_expansion"),
            "converged": record.get("converged"),
            "solve_exception_class": record.get("solve_exception_class"),
            "solver_termination": record.get("solver_termination"),
            "residual_tolerance": record.get("residual_tolerance", DEFAULT_TOLERANCE),
            "banked_branches": {},
            "measured_branches": {},
        }
        for name in ("diverted", "limited"):
            entry["banked_branches"][name] = _banked_branch(record, name)
        measured = probe.get(frame)
        if measured:
            entry["measured"] = {
                "shot": measured.get("shot"),
                "lattice_shape": measured.get("lattice_shape"),
                "label_axis_rz_m": measured.get("label_axis_rz_m"),
                "target_current_a": measured.get("target_current_a"),
                "branches": {},
            }
            for branch_name, branch in (measured.get("branches") or {}).items():
                arms = branch.get("arms") or {}
                staged = {
                    key: _arm_summary(value)
                    for key, value in (arms.get("staged") or {}).items()
                }
                entry["measured"]["branches"][branch_name] = {
                    "status": branch.get("status"),
                    "requested_class": branch.get("requested_class"),
                    "refusal_type": branch.get("refusal_type"),
                    "refusal_message": branch.get("refusal_message"),
                    "arms": {
                        name: _arm_summary(arms.get(name) or {}) for name in ARM_NAMES
                    },
                    "staged": staged,
                }
        frames[str(frame)] = entry
    diverting_rule = {
        "sites": [
            "benchmarks/diiid_forward_gs_match.py:2381",
            "benchmarks/diiid_forward_gs_match.py:2406",
            "benchmarks/diiid_forward_gs_match.py:2437",
        ],
        "statement": (
            "line 2381 binds `diverted = receipts[TopologyClass.DIVERTED]`; line "
            "2406 accepts a branch only when `branches['diverted']['converged'] "
            "and branches['diverted']['topology_consistent']`; line 2437 returns "
            "`diverted.equilibrium` unconditionally.  The limited seed's receipt "
            "is built and discarded, so no limited root can be selected however "
            "low its residual."
        ),
    }
    measured_note = (
        "The banked branch field is what the bank executed; the measured field is "
        "this node's own run of both seeds.  A frame with no measured block was "
        "not solved here."
        if probe
        else "No measurement file was supplied, so only the banked terminals are "
        "recorded; the both-branch descent this node ran is absent."
    )
    return {
        "measurement": (
            "terminal class and promotion trajectory per branch and frame, with the "
            "plasma-presence read of each terminal state"
        ),
        "sources": {
            "partial_receipt": str(partial_receipt),
            "receipt": str(receipt),
            "measurement": None if measurement is None else str(measurement),
            "measurement_present": bool(probe),
        },
        "notes": measured_note,
        "tolerance": DEFAULT_TOLERANCE,
        "diverted_branch_rule": diverting_rule,
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partial-receipt",
        type=Path,
        default=Path(
            "docs/figures/diiid-forward-onboarding/forward-gs-2026-11/"
            "forward_gs_partial_receipt.json"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "docs/figures/diiid-forward-onboarding/forward-gs-2026-11/"
            "forward_gs_receipt.json"
        ),
    )
    parser.add_argument("--measurement", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/diiid-forward-convergence/convergence-receipt.json"),
    )
    arguments = parser.parse_args()
    receipt = build(arguments.partial_receipt, arguments.receipt, arguments.measurement)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"written": str(arguments.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
