"""Replay banked cold-start roots through deterministic pin admissibility.

The expensive solve attempts remain the banked public-seam measurements. This
adapter composes their existing MAST and DIII-D receipt surfaces, constructs the
same typed all-domain current pin for every row, and measures how many candidate
roots remain admissible. Isoflux pin evaluation is exercised by the focused
Jacobian suite; the cold-start receipts do not retain trusted chord-pair pins,
so this benchmark does not invent them after seeing terminal states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import diiid_constrained_cold_start as diiid_adapter
from benchmarks import efit_parity_warm_neighbour as mast_adapter
from nova.equilibrium.observation import (
    ConstraintPinSet,
    MomentIntegralSupport,
    MomentPin,
    PinUncertainty,
)

DEFAULT_OUTPUT = Path("docs/figures/ensemble-forward-seam/constraint-seam")
RECEIPT_NAME = "constraint-seam-receipt.json"
FIGURE_NAME = "constraint-seam-basin-restriction.png"
MAST_RECEIPT = (
    mast_adapter.CURRENT_CONSTRAINED_OUTPUT
    / mast_adapter.CURRENT_CONSTRAINED_RECEIPT_NAME
)
DIIID_RECEIPT = diiid_adapter.DEFAULT_OUTPUT / diiid_adapter.RECEIPT_NAME


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_stamp() -> dict[str, str]:
    return {
        "commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True
        ).strip(),
        "tree": subprocess.check_output(
            ("git", "rev-parse", "HEAD^{tree}"), text=True
        ).strip(),
    }


def _pin(target: float, absolute: float) -> ConstraintPinSet:
    return ConstraintPinSet(
        moments=(
            MomentPin(
                name="plasma_current",
                target=target,
                uncertainty=PinUncertainty(
                    absolute=absolute,
                    unit="A",
                    statement="banked public-seam current acceptance interval",
                ),
                support=MomentIntegralSupport.ALL_DOMAIN,
            ),
        )
    )


def _mast_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in receipt["per_shot"]:
        target = float(item["reference"]["plasma_current_a"])
        relative_error = float(item["target_current"]["signed_terminal_relative_error"])
        absolute_interval = max(
            abs(target) * mast_adapter.TARGET_CURRENT_EXACT_TOLERANCE, 1.0e-12
        )
        pins = _pin(target, absolute_interval)
        scaled_residual = abs(relative_error * target) / absolute_interval
        converged = bool(item["constrained_solve"]["converged"])
        rows.append(
            {
                "machine": "MAST",
                "identity": (
                    f"{item['reference']['shot']}/{item['reference']['slice_index']}"
                ),
                "public_entry_point": "ForwardProfile.solve(target_current=...)",
                "cold_converged": converged,
                "pin_admissible": bool(scaled_residual <= 1.0),
                "admissible_converged_root": bool(converged and scaled_residual <= 1.0),
                "scaled_current_pin_residual": scaled_residual,
                "target_current_a": target,
                "terminal_residual": float(
                    item["constrained_solve"]["terminal_residual"]
                ),
                "pin_set": {
                    "class": type(pins).__name__,
                    "moment": pins.moments[0].name,
                    "support": pins.moments[0].support.value,
                    "absolute_interval_a": pins.moments[0].uncertainty.absolute,
                },
            }
        )
    return rows


def _diiid_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in receipt["frames"]:
        route = item["circuit_driven"]["routes"]["cold_start"]
        target = float(item["target_plasma_current"]["value_a"])
        terminal = route["terminal_state"]
        absolute_interval = max(
            abs(target) * diiid_adapter.CURRENT_RELATIVE_ERROR_TOLERANCE, 1.0e-12
        )
        pins = _pin(target, absolute_interval)
        scaled_residual = (
            abs(float(terminal["current_relative_error"]) * target) / absolute_interval
        )
        converged = bool(route["converged"])
        rows.append(
            {
                "machine": "DIII-D",
                "identity": f"{Path(item['shot']).stem}/{item['frame']}",
                "public_entry_point": route["entry_point"],
                "cold_converged": converged,
                "pin_admissible": bool(scaled_residual <= 1.0),
                "admissible_converged_root": bool(converged and scaled_residual <= 1.0),
                "scaled_current_pin_residual": scaled_residual,
                "target_current_a": target,
                "terminal_residual": float(route["fixed_point_relative_residual"]),
                "pin_set": {
                    "class": type(pins).__name__,
                    "moment": pins.moments[0].name,
                    "support": pins.moments[0].support.value,
                    "absolute_interval_a": pins.moments[0].uncertainty.absolute,
                },
            }
        )
    return rows


def _summary(rows: list[dict[str, Any]], machine: str) -> dict[str, Any]:
    selected = [row for row in rows if row["machine"] == machine]
    cold = sum(row["cold_converged"] for row in selected)
    admissible = sum(row["admissible_converged_root"] for row in selected)
    return {
        "attempts": len(selected),
        "cold_converged": cold,
        "pin_admissible_converged": admissible,
        "candidate_roots_rejected_by_current_pin": cold - admissible,
        "basin_restriction_fraction": ((cold - admissible) / cold if cold else None),
        "maximum_scaled_current_pin_residual": max(
            row["scaled_current_pin_residual"] for row in selected
        ),
    }


def _figure(summary: dict[str, dict[str, Any]], path: Path) -> None:
    machines = ("MAST", "DIII-D")
    x = np.arange(len(machines))
    cold = [summary[machine]["cold_converged"] for machine in machines]
    admissible = [summary[machine]["pin_admissible_converged"] for machine in machines]
    totals = [summary[machine]["attempts"] for machine in machines]
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    width = 0.34
    axis.bar(x - width / 2, cold, width, label="cold public-seam roots")
    axis.bar(x + width / 2, admissible, width, label="after typed current pin")
    for index, (value, total) in enumerate(zip(admissible, totals, strict=True)):
        axis.text(index + width / 2, value + 0.06, f"{value}/{total}", ha="center")
    axis.set_xticks(x, machines)
    axis.set_ylabel("convergence-qualified roots")
    axis.set_ylim(0.0, max(totals) + 0.6)
    axis.set_title("Deterministic pin admissibility does not create a root")
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    mast = json.loads(MAST_RECEIPT.read_text())
    diiid = json.loads(DIIID_RECEIPT.read_text())
    rows = [*_mast_rows(mast), *_diiid_rows(diiid)]
    summary = {
        "MAST": _summary(rows, "MAST"),
        "DIII-D": _summary(rows, "DIII-D"),
    }
    if (summary["MAST"]["cold_converged"], summary["MAST"]["attempts"]) != (
        1,
        6,
    ):
        raise RuntimeError("the banked MAST cold baseline is no longer 1/6")
    if (
        summary["DIII-D"]["cold_converged"],
        summary["DIII-D"]["attempts"],
    ) != (2, 5):
        raise RuntimeError("the banked DIII-D cold baseline is no longer 2/5")

    figure_path = output / FIGURE_NAME
    _figure(summary, figure_path)
    receipt = {
        "source": _source_stamp(),
        "inputs": {
            "mast": {"path": str(MAST_RECEIPT), "sha256": _sha256(MAST_RECEIPT)},
            "diiid": {
                "path": str(DIIID_RECEIPT),
                "sha256": _sha256(DIIID_RECEIPT),
            },
            "adapters": [
                "benchmarks.efit_parity_warm_neighbour",
                "benchmarks.diiid_constrained_cold_start",
            ],
        },
        "measurement": {
            "rule": (
                "a banked cold root remains eligible only when its all-domain "
                "current pin is inside the declared deterministic interval"
            ),
            "isoflux_baseline_scored": False,
            "isoflux_qualification": (
                "banked cold-start receipts carry no trusted chord-pair pins; "
                "the differentiable isoflux map and interval admission are "
                "therefore proven in the focused synthetic Jacobian suite only"
            ),
            "statistical_operations": [],
        },
        "summary": summary,
        "rows": rows,
        "diiid_current_defaults_qualification": {
            "status": "open owner finding",
            "scope": "current-defaults baseline",
            "finding": (
                "DIII-D convergence at default plasma-cell resolution remains "
                "under investigation for resolution and operator-scaling impact"
            ),
        },
        "artifacts": {
            "receipt": str(output / RECEIPT_NAME),
            "figure": str(figure_path),
        },
    }
    (output / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = run(arguments.output)
    print(json.dumps(receipt["summary"], indent=2))


if __name__ == "__main__":
    main()
