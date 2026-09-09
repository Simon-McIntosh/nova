"""Discriminate the weak static fixed-point regression from exact current placement.

Three arms on the weak -110 static row, all at the held stack tip, each a
fresh process persisting its receipt and poloidal flux contour panel as it
lands:

* ``control`` - the committed chord clip (positive control; must reproduce
  the banked terminal residual).
* ``b`` - exact participation with the two named boundary cells reverted to
  their chord-moment values.
* ``c`` - exact participation under a sweep of fixed-point step relaxation
  values; the best terminal residual is reported.

The clip mode and step relaxation are selected through explicit setters on
the solve machinery, each defaulting to the committed behaviour, so this
benchmark changes no production default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import time

import benchmarks.solovev_certificate as certificate
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.reduced_newton import set_step_relaxation


CASE_NAME = "weak-rotation-reactor-static"
REQUESTED_CELLS = -110
BANKED_WEAK_110_RESIDUAL = 0.007447526861632619
OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "docs/figures/uniform-cell-clip-and-coupling/exact-participation/discriminator"
)
PART_ROOT = OUTPUT_ROOT / "parts"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
PROGRESS = OUTPUT_ROOT / "progress.json"
RECEIPT = OUTPUT_ROOT / "discriminator-receipt.json"

ARMS = {
    "control": {"clip_mode": "chord", "relaxation": 1.0},
    "b": {"clip_mode": "chord_cells", "relaxation": 1.0},
    "c": {"clip_mode": "exact", "relaxation": 1.0},
}


def arm_label(arm: str, relaxation: float) -> str:
    tag = f"{int(relaxation * 100):03d}" if relaxation != 1.0 else "100"
    return arm if arm != "c" else f"c-r{tag}"


def _read_terminal_row(label: str) -> dict[str, object]:
    part = PART_ROOT / label / (f"{CASE_NAME}-production-route-reduced.json")
    row = json.loads(part.read_text(encoding="utf-8"))
    solver = row["solver"]
    geometry = row["geometry"]
    figure = row["figure"]
    return {
        "arm": label,
        "clip_mode": ARMS[label.split("-")[0] if "-" in label else label]["clip_mode"],
        "case": row["case"],
        "requested_cells": row["requested_cells"],
        "terminal_fixed_point_residual": solver["terminal_fixed_point_residual"],
        "qualification": solver["qualification"],
        "converged": solver["production_telemetry"]["converged"],
        "residual_ratio_to_baseline": (
            solver["terminal_fixed_point_residual"] / BANKED_WEAK_110_RESIDUAL
            if solver["terminal_fixed_point_residual"]
            else None
        ),
        "magnetic_axis_position_error_m": geometry["magnetic_axis_position_error_m"],
        "x_point_position_error_m": geometry["x_point_position_error_m"],
        "axis_rz_m": geometry["root_topology"]["axis_rz_m"],
        "x_point_rz_m": geometry["root_topology"]["x_point_rz_m"],
        "figure_filesystem_path": figure["filesystem_path"],
        "figure_sha256": figure["sha256"],
        "part": _repo_relative(part, Path(__file__).resolve().parents[1]),
    }


def _repo_relative(path: Path, root: Path) -> str:
    """Return the path repo-relative when it lives under the repo, else absolute."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def measure_one(label: str, clip_mode: str, relaxation: float) -> dict[str, object]:
    """Solve one arm in this process and persist its receipt and panel."""
    part_root = PART_ROOT / label
    figure_root = FIGURE_ROOT / label
    part_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    certificate.PART_ROOT = part_root
    certificate.FIGURE_ROOT = figure_root
    set_support_clip_mode(clip_mode)
    set_step_relaxation(relaxation)
    print(
        f"DISCRIMINATOR_ARM_BEGIN arm={label} clip_mode={clip_mode} "
        f"relaxation={relaxation}",
        flush=True,
    )
    certificate._measure(CASE_NAME, REQUESTED_CELLS)
    summary = _read_terminal_row(label)
    summary["completed_at_unix_seconds"] = time()
    summary["exit_status"] = 0
    receipt_dir = OUTPUT_ROOT / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{label}.json"
    receipt_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "DISCRIMINATOR_ARM_END " + json.dumps(summary, sort_keys=True),
        flush=True,
    )
    return summary


def relaxations_for(arm: str) -> list[float]:
    if arm == "c":
        return (1.0, 0.5, 0.25, 0.125)
    return (ARMS[arm]["relaxation"],)


def run_all() -> int:
    """Run every arm sequentially in fresh processes, persisting each row."""
    landed: list[dict[str, object]] = []
    write_progress(landed)
    for arm in ("control", "b", "c"):
        for relaxation in relaxations_for(arm):
            label = arm_label(arm, relaxation)
            clip_mode = ARMS[arm]["clip_mode"]
            print(
                f"DISCRIMINATOR_BEGIN arm={arm} label={label} relax={relaxation}",
                flush=True,
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--one",
                    "--label",
                    label,
                    "--clip-mode",
                    clip_mode,
                    "--relaxation",
                    str(relaxation),
                ],
                check=False,
            )
            record: dict[str, object] = {
                "arm": arm,
                "label": label,
                "clip_mode": clip_mode,
                "relaxation": relaxation,
                "exit_status": completed.returncode,
                "completed_at_unix_seconds": time(),
            }
            summary_path = OUTPUT_ROOT / "receipts" / f"{label}.json"
            if summary_path.exists():
                record.update(json.loads(summary_path.read_text(encoding="utf-8")))
            landed.append(record)
            write_progress(landed)
            print("DISCRIMINATOR_END " + json.dumps(record, sort_keys=True), flush=True)
    best = best_of(landed)
    best["banked_weak_110_residual"] = BANKED_WEAK_110_RESIDUAL
    RECEIPT.write_text(
        json.dumps(best, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("DISCRIMINATOR_FINAL " + json.dumps(best, sort_keys=True), flush=True)
    return 0 if all(row["exit_status"] == 0 for row in landed) else 1


def best_of(landed: list[dict[str, object]]) -> dict[str, object]:
    """Name the mechanism the numbers support among the three candidates."""
    residuals = {
        row["label"]: row.get("terminal_fixed_point_residual") for row in landed
    }
    banked = BANKED_WEAK_110_RESIDUAL
    control = residuals.get("control")
    b = residuals.get("b")
    c_residuals = {
        label: value for label, value in residuals.items() if label.startswith("c-")
    }
    c_best = min(c_residuals, key=lambda k: c_residuals[k]) if c_residuals else None
    verdict: list[str] = []
    if control is not None and abs(control - banked) <= 1.0e-9:
        verdict.append("control reproduced the banked chord residual to 1e-9")
    elif control is not None:
        verdict.append(
            f"control deviates from the banked residual "
            f"({abs(control - banked):.3e}) - chord reconstruction is imperfect"
        )

    def recovered(value: float) -> str:
        return (
            "recovers"
            if control is not None and value <= control * 2.0
            else "does not recover"
        )

    if b is not None and control is not None:
        verdict.append(
            f"arm b (two cells reverted) residual {b:.4e} vs control {control:.4e}"
            f" ({recovered(b)})"
        )
    if c_best is not None and control is not None:
        best_value = c_residuals[c_best]
        verdict.append(
            f"arm c best relaxation {c_best} residual {best_value:.4e} "
            f"vs control {control:.4e} ({recovered(best_value)})"
        )
    mechanism: str
    if control is not None and b is not None and recovered(b) == "recovers":
        mechanism = "two-cell contamination (b)"
    elif (
        control is not None
        and c_best is not None
        and recovered(c_residuals[c_best]) == "recovers"
    ):
        mechanism = "damping tuned to the chord map (c)"
    elif control is not None and b is not None and c_best is not None:
        mechanism = "neither - the flux-read inconsistency remains"
    else:
        mechanism = "inconclusive"
    return {
        "banked_weak_110_residual": banked,
        "verdict": verdict,
        "mechanism": mechanism,
        "residuals": residuals,
    }


def write_progress(rows: list[dict[str, object]]) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROGRESS.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(PROGRESS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--clip-mode", choices=("exact", "chord", "chord_cells"))
    parser.add_argument("--relaxation", type=float, default=1.0)
    arguments = parser.parse_args()
    if arguments.all:
        return run_all()
    if arguments.one:
        if not arguments.label or arguments.clip_mode is None:
            parser.error("--one requires --label and --clip-mode")
        measure_one(arguments.label, arguments.clip_mode, arguments.relaxation)
        return 0
    parser.error("specify --all or --one")


if __name__ == "__main__":
    raise SystemExit(main())
