"""Measure passive-closure response over an explicit plasma-mesh ladder.

Each rung uses one machine carrier and solves the same absolute-source case
twice: with every declared passive current, then with only those currents set
to zero.  The comparison therefore changes neither geometry nor source
profiles.  It measures whether the anomalous closure response follows the
coarse suite carrier or persists when the plasma mesh reaches the reference
scale.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from time import perf_counter

import jax
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.measurement_provenance import measurement_stamp
from nova.jax.config import configure_dtypes
from tests import test_equilibrium_forward_reference as reference


DEFAULT_RECEIPT = Path(
    "docs/figures/forward-operator-refinement/passive-closure-mesh-response.json"
)
DEFAULT_FIGURE = Path(
    "docs/figures/forward-operator-refinement/passive-closure-mesh-response.png"
)
DEFAULT_CELL_REQUESTS = (-500, -1500, -2100)
REPORTED_COARSE_MEASUREMENT_POINTS = 0.6098
REGISTERED_CEILING_POINTS = 0.15
DOCUMENTED_DIRECT_RESPONSE_POINTS = 0.098


def _strict_json(value):
    """Return ordinary finite JSON values for the banked receipt."""
    if isinstance(value, dict):
        return {key: _strict_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("the measurement receipt contains a non-finite scalar")
    return value


def _solve_receipt(case, machine, model: str):
    """Solve one carrier and return the result with convergence accounting."""
    started = perf_counter()
    solved = reference.solve(case, machine)
    deviations = solved.deviations()
    residual = float(solved.fixed_point.residual)
    elapsed_seconds = perf_counter() - started
    trace = np.asarray(solved.fixed_point.trace, dtype=float)
    measured = trace[np.isfinite(trace)]
    if measured.size == 0:
        raise RuntimeError(f"{model} solve produced no measured residual")
    return solved, {
        "model": model,
        "elapsed_seconds": elapsed_seconds,
        "final_relative_fixed_point_residual": residual,
        "trace_slots": int(trace.size),
        "measured_residuals": int(measured.size),
        "first_measured_residual": float(measured[0]),
        "minimum_measured_residual": float(measured.min()),
        "deviations": deviations,
    }


def _measure_rung(case, requested_cells: int) -> dict[str, object]:
    """Measure both passive-current states on one explicit cell request."""
    machine_started = perf_counter()
    machine = reference.cached_machine(case, requested_cells, passive=True)
    machine_seconds = perf_counter() - machine_started
    if machine.passive_columns != len(case.passive):
        raise RuntimeError("the machine carrier omitted declared passive columns")

    structure, structure_receipt = _solve_receipt(
        case, machine, "declared_passive_currents"
    )
    without_current = machine.source_current.copy()
    without_current[-machine.passive_columns :] = 0.0
    without_machine = replace(
        machine,
        source_current=without_current,
        passive_columns=0,
        cache_receipt=None,
    )
    without, without_receipt = _solve_receipt(
        case, without_machine, "passive_currents_zeroed"
    )

    structure_deviation = structure_receipt["deviations"]
    without_deviation = without_receipt["deviations"]
    flux_move = (
        without_deviation["flux sup-norm"] - structure_deviation["flux sup-norm"]
    )
    inductance_move = (
        structure_deviation["internal inductance"]
        - without_deviation["internal inductance"]
    )
    passive_flux = 100.0 * float(
        np.max(np.abs(machine.passive_flux)) / abs(case.flux_span)
    )
    cache = machine.cache_receipt
    if cache is None:
        raise RuntimeError("the machine carrier has no persistent-cache receipt")

    return {
        "requested_cells": requested_cells,
        "realised_plasma_cells": len(machine.node),
        "cell_pitch_m": structure.pitch,
        "passive_columns": machine.passive_columns,
        "passive_flux_percent_of_reference_span": passive_flux,
        "machine_request_seconds": machine_seconds,
        "machine_cache": asdict(cache),
        "solves": [structure_receipt, without_receipt],
        "closure": {
            "flux_reproduction_move_points": flux_move,
            "flux_move_over_without_passive_deviation": (
                flux_move / without_deviation["flux sup-norm"]
            ),
            "internal_inductance_move_points": inductance_move,
            "axis_radius_move_mm": (
                structure_deviation["axis radius"] - without_deviation["axis radius"]
            ),
            "axis_height_move_mm": (
                structure_deviation["axis height"] - without_deviation["axis height"]
            ),
        },
    }


def _classify(rows: list[dict[str, object]]) -> dict[str, object]:
    """Classify the dense response using the registered physical alternatives."""
    coarse = float(rows[0]["closure"]["flux_reproduction_move_points"])
    dense = float(rows[-1]["closure"]["flux_reproduction_move_points"])
    coarse_distance = abs(coarse - DOCUMENTED_DIRECT_RESPONSE_POINTS)
    dense_distance = abs(dense - DOCUMENTED_DIRECT_RESPONSE_POINTS)
    collapsed = (
        0.0 < dense <= REGISTERED_CEILING_POINTS and dense_distance < coarse_distance
    )
    if collapsed:
        classification = "coarse_carrier_artifact"
        ruling = (
            "The reference-scale rung falls inside the 0.15-point ceiling and "
            "moves closer to the documented 0.098-point direct response; the "
            "coarse suite carrier is responsible."
        )
        follow_on = "raise_the_suite_cell_carrier_to_the_consistent_scale"
    else:
        classification = "mesh_independent_closure_defect"
        ruling = (
            "The reference-scale rung does not collapse inside the 0.15-point "
            "ceiling toward the documented 0.098-point direct response; the "
            "closure path, rather than the coarse carrier, is responsible."
        )
        follow_on = "trace_the_passive_closure_path_on_a_fixed_dense_carrier"
    return {
        "classification": classification,
        "ruling": ruling,
        "follow_on": follow_on,
        "collapse_rule": (
            "dense move is positive, no greater than 0.15 points, and closer "
            "to 0.098 points than the -500 move"
        ),
        "dense_inside_registered_ceiling": dense <= REGISTERED_CEILING_POINTS,
        "dense_closer_to_documented_direct_response": dense_distance < coarse_distance,
        "coarse_move_points": coarse,
        "dense_move_points": dense,
        "coarse_distance_from_direct_points": coarse_distance,
        "dense_distance_from_direct_points": dense_distance,
        "dense_over_coarse_move": dense / coarse,
    }


def _plot(receipt: dict[str, object], path: Path) -> None:
    """Plot the response ladder and the two decision comparators."""
    rows = receipt["rungs"]
    cells = np.asarray([row["realised_plasma_cells"] for row in rows])
    moves = np.asarray(
        [row["closure"]["flux_reproduction_move_points"] for row in rows]
    )

    figure, axes = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    axes.plot(cells, moves, color="#2a6099", marker="o", linewidth=1.8)
    axes.axhline(
        REGISTERED_CEILING_POINTS,
        color="#b24c3d",
        linestyle="--",
        linewidth=1.2,
    )
    axes.axhline(
        DOCUMENTED_DIRECT_RESPONSE_POINTS,
        color="0.35",
        linestyle=":",
        linewidth=1.2,
    )
    axes.text(
        cells[-1],
        REGISTERED_CEILING_POINTS,
        " 0.15-point ceiling ",
        color="#b24c3d",
        fontsize=9,
        ha="right",
        va="bottom",
    )
    axes.text(
        cells[-1],
        DOCUMENTED_DIRECT_RESPONSE_POINTS,
        " documented direct response 0.098 points ",
        color="0.35",
        fontsize=9,
        ha="right",
        va="top",
    )
    for row, x, y in zip(rows, cells, moves, strict=True):
        axes.annotate(
            f"{row['requested_cells']}: {y:.4f}",
            (x, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#2a6099",
        )
    axes.set_xlabel("realised plasma cells")
    axes.set_ylabel("passive-closure flux move [percentage points]")
    axes.set_title(receipt["verdict"]["classification"].replace("_", " "))
    axes.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, transparent=True)
    plt.close(figure)


def run(cell_requests: tuple[int, ...]) -> dict[str, object]:
    """Run the frozen ladder and return its attributable receipt."""
    if len(cell_requests) != 3 or cell_requests[0] != -500:
        raise ValueError("the ladder requires -500 followed by two denser rungs")
    if any(value >= 0 for value in cell_requests):
        raise ValueError("plasma cell-count requests must be negative")
    if list(map(abs, cell_requests)) != sorted(map(abs, cell_requests)):
        raise ValueError("the ladder must progress from coarse to dense")

    source_commit = measurement_stamp(Path.cwd())
    configure_dtypes()
    case = reference.require_reference()
    rows = [_measure_rung(case, request) for request in cell_requests]
    verdict = _classify(rows)
    return {
        "receipt": {
            "kind": "passive_closure_mesh_response",
            "status": "complete",
            "source_commit": source_commit,
            "checkout_porcelain_empty_before_measurement": True,
            "device_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "solve_count": 2 * len(rows),
            "rung_count": len(rows),
        },
        "comparators": {
            "reported_coarse_measurement_points": (REPORTED_COARSE_MEASUREMENT_POINTS),
            "registered_ceiling_points": REGISTERED_CEILING_POINTS,
            "documented_direct_response_points": DOCUMENTED_DIRECT_RESPONSE_POINTS,
            "reported_measurement_over_ceiling": (
                REPORTED_COARSE_MEASUREMENT_POINTS / REGISTERED_CEILING_POINTS
            ),
            "reported_measurement_over_direct_response": (
                REPORTED_COARSE_MEASUREMENT_POINTS / DOCUMENTED_DIRECT_RESPONSE_POINTS
            ),
        },
        "method": {
            "cell_requests": list(cell_requests),
            "intermediate_request": cell_requests[1],
            "reference_scale_request": cell_requests[2],
            "controlled_change": (
                "zero only the declared passive-current columns on one shared "
                "machine carrier per rung"
            ),
            "measured_quantity": (
                "without-passive flux sup-norm deviation minus passive-inclusive "
                "flux sup-norm deviation, in percentage points of reference span"
            ),
        },
        "rungs": rows,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs=3, default=DEFAULT_CELL_REQUESTS)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    arguments = parser.parse_args()

    receipt = _strict_json(run(tuple(arguments.cells)))
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot(receipt, arguments.figure)
    print(
        json.dumps(
            {
                "receipt": str(arguments.receipt),
                "figure": str(arguments.figure),
                "moves_points": [
                    row["closure"]["flux_reproduction_move_points"]
                    for row in receipt["rungs"]
                ],
                "realised_cells": [
                    row["realised_plasma_cells"] for row in receipt["rungs"]
                ],
                "verdict": receipt["verdict"]["classification"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
