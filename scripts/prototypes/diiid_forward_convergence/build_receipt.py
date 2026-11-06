"""Build the per-frame convergence receipt from the banked forward receipts.

The banked partial receipt already carries, for every gate frame, the executed
active-set trips of the branch the bank ran, its per-trip relative residual and
mask difference, and the terminal residual of the other branch's seed.  This
script reads that record, classes each terminal, and states the branch rule
that kept the diverted state.  It does not re-run a solve: the numbers are the
bank's own, quoted with the path they came from.

Where a live measurement exists (``--measurement``), its per-branch promotion
trajectory, step-acceptance factors and mask history are merged in, so a
reader sees the frozen-mask descent the bank recorded and the both-branch
descent where one was measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TOLERANCE_DEFAULT = 1.0e-5
FRAMES = (179, 44, 144, 146, 137)


def _classify(
    residuals: list[float | None], mask_differences: list[int], terminal: float | None
) -> dict[str, Any]:
    """Class a branch terminal from its executed active-set trips."""

    finite = [value for value in residuals if value is not None]
    if terminal is None or not finite:
        return {
            "terminal_class": "refusal",
            "evidence": "no finite promotion was executed (no qualified axis)",
        }
    if terminal <= TOLERANCE_DEFAULT:
        return {
            "terminal_class": "contracted",
            "evidence": {"terminal_residual": terminal},
        }
    if len(finite) >= 4:
        tail = finite[-4:]
        if abs(tail[2] - tail[0]) <= 1.0e-12 * max(abs(tail[2]), 1.0) and abs(
            tail[3] - tail[1]
        ) <= 1.0e-12 * max(abs(tail[3]), 1.0):
            return {
                "terminal_class": "period-2 cycle",
                "evidence": {"alternating_levels": tail},
            }
    if len(finite) >= 2 and finite[-1] == finite[-2]:
        return {
            "terminal_class": "contraction stall",
            "subtype": "active-set stagnation at a frozen mask",
            "evidence": {
                "terminal_residual": terminal,
                "residual_change_over_last_promotion": 0.0,
                "mask_difference_at_terminal_promotion": (
                    mask_differences[-1] if mask_differences else None
                ),
                "note": (
                    "the active set stops changing (mask difference 0) while the "
                    "relative residual does not move at all, so the loop ends on "
                    "mask stability rather than on the residual criterion"
                ),
            },
        }
    return {
        "terminal_class": "contraction stall",
        "subtype": "residual plateau above tolerance",
        "evidence": {"terminal_residual": terminal, "trajectory": finite},
    }


def build(
    partial_receipt: Path, receipt: Path, measurement: Path | None
) -> dict[str, Any]:
    partial = json.loads(partial_receipt.read_text())
    banked = json.loads(receipt.read_text())
    records = {int(record["frame"]): record for record in partial["frame_records"]}
    tolerance = TOLERANCE_DEFAULT
    probe: dict[str, Any] = {}
    if measurement is not None and measurement.exists():
        raw = json.loads(measurement.read_text())
        probe = {int(key): value for key, value in raw.get("frames", {}).items()}
    bank_run = banked.get("bank_run", {})
    frames: dict[str, Any] = {}
    for frame in FRAMES:
        record = records[frame]
        selection = record.get("branch_selection") or {}
        branches = selection.get("branches") or {}
        entry: dict[str, Any] = {
            "frame": frame,
            "shot": record["shot"],
            "time_ms": record.get("time_ms"),
            "achieved_current_a": record.get("achieved_current_a"),
            "target_current_a": record.get("target_current_a"),
            "pseudo_wall_expansion": record.get("pseudo_wall_expansion"),
            "cold_seed": selection.get("cold_seed"),
            "selected_class": selection.get("selected_class"),
            "selection_reason": selection.get("reason"),
            "converged": record.get("converged"),
            "finite": record.get("finite"),
            "solve_exception_class": record.get("solve_exception_class"),
            "solver_termination": record.get("solver_termination"),
            "residual_tolerance": record.get("residual_tolerance", tolerance),
            "branches": {},
        }
        for name in ("diverted", "limited"):
            branch = dict(branches.get(name, {}))
            if name == "diverted":
                residuals = list(record.get("active_set_residuals") or [])
                masks = list(record.get("active_set_mask_differences") or [])
                damping = list(record.get("active_set_cycle_damping_activations") or [])
                trips = int(record.get("active_set_iterations") or 0)
            else:
                # The bank runs both seeds but returns only the diverted
                # receipt, so the limited seed's trips are not on the record;
                # only its terminal residual is.
                residuals = []
                masks = []
                damping = []
                trips = branch.get("iterations")
            terminal = branch.get("residual")
            branch["promotion_trajectory"] = residuals
            branch["mask_differences"] = masks
            branch["cycle_damping_activations"] = damping
            branch["active_set_trips"] = trips
            branch["terminal"] = (
                _classify(residuals, masks, terminal)
                if residuals
                else (
                    {
                        "terminal_class": "refusal",
                        "evidence": record.get("solver_termination"),
                    }
                    if terminal is None
                    else {
                        "terminal_class": (
                            "not measured: the bank returns the "
                            "diverted receipt"
                        ),
                        "evidence": {
                            "terminal_residual": terminal,
                            "achieved_class": branch.get("achieved_class"),
                        },
                    }
                )
            )
            entry["branches"][name] = branch
        if frame in probe:
            entry["measured"] = probe[frame]
        frames[str(frame)] = entry
    diverted_rule = {
        "sites": [
            "benchmarks/diiid_forward_gs_match.py:2381",
            "benchmarks/diiid_forward_gs_match.py:2406",
            "benchmarks/diiid_forward_gs_match.py:2437",
        ],
        "statement": (
            "line 2381 binds `diverted = receipts[TopologyClass.DIVERTED]`; line "
            "2406 accepts a branch only when `branches['diverted']['converged'] "
            "and branches['diverted']['topology_consistent']`; line 2437 returns "
            "`diverted.equilibrium` unconditionally. The limited seed's receipt is "
            "built and discarded, so no limited root can be selected however low "
            "its residual."
        ),
    }
    return {
        "measurement": (
            "per-promotion forward descent on both topology branches from the "
            "bank's cold seed, with the branch rule that keeps the diverted state"
        ),
        "sources": {
            "partial_receipt": str(partial_receipt),
            "receipt": str(receipt),
            "measurement": None if measurement is None else str(measurement),
            "bank_run_elapsed_seconds": bank_run.get("elapsed_seconds"),
        },
        "tolerance": tolerance,
        "diverted_branch_rule": diverted_rule,
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
