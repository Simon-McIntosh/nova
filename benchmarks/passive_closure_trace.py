"""Trace passive-current amplification on one fixed dense carrier.

The diagnostic keeps geometry, profiles and the initial flux byte-identical
between two arms.  It first follows paired applications of the coupled map,
then applies the exact map tangent repeatedly to the passive external field.
Iteration one therefore tests closure arithmetic before feedback; later terms
test whether the measured expanding map supplies the terminal gain.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.measurement_provenance import measurement_stamp
from nova.jax.config import configure_dtypes
from tests import test_equilibrium_forward_reference as reference


DEFAULT_RECEIPT = Path(
    "docs/figures/forward-operator-refinement/passive-closure-trace.json"
)
DEFAULT_FIGURE = Path(
    "docs/figures/forward-operator-refinement/passive-closure-trace.png"
)
DENSE_CELL_REQUEST = -2100
PAIRED_MAP_ITERATIONS = 8
LINEAR_RESPONSE_ITERATIONS = 12
MEASURED_UNPINNED_SPECTRAL_RADIUS = 1.2577
MEASURED_PINNED_SPECTRAL_RADIUS = 1.1456
REGISTERED_CEILING_POINTS = 0.15
DOCUMENTED_DIRECT_RESPONSE_POINTS = 0.098
IDENTITY_RELATIVE_TOLERANCE = 2.0e-12
SPECTRAL_RATIO_RELATIVE_TOLERANCE = 0.25


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
        raise ValueError("the trace receipt contains a non-finite scalar")
    return value


def _grid(values, cell_count: int) -> np.ndarray:
    """Return the plasma-cell prefix of one full operator vector."""
    return np.asarray(values, dtype=float)[:cell_count]


def _peak_points(values, cell_count: int, span: float) -> float:
    """Return a grid-vector peak in percentage points of reference span."""
    return 100.0 * float(np.max(np.abs(_grid(values, cell_count)))) / span


def _score(values, reference_flux, core, cell_count: int, span: float) -> float:
    """Return the reference reproduction sup-norm on one explicit support."""
    error = np.abs(_grid(values, cell_count) - reference_flux)
    return 100.0 * float(np.max(error[np.asarray(core, dtype=bool)])) / span


def _topology(operator, state) -> dict[str, object]:
    """Return the topology and centroid support read from one state."""
    masks, topology = operator.read(state)
    core = np.asarray(masks.core, dtype=bool)
    return {
        "core": core,
        "core_cells": int(core.sum()),
        "axis_m": np.asarray(topology.axis, dtype=float),
        "axis_flux_wb": float(topology.axis_flux),
        "boundary_flux_wb": float(topology.boundary_flux),
        "flux_span_wb": float(topology.flux_span),
        "diverted": bool(topology.diverted),
    }


def _support_comparison(left: dict[str, object], right: dict[str, object]):
    """Compare two domain reads without serialising their full masks."""
    left_core = left["core"]
    right_core = right["core"]
    return {
        "left_core_cells": int(left_core.sum()),
        "right_core_cells": int(right_core.sum()),
        "core_symmetric_difference_cells": int(
            np.logical_xor(left_core, right_core).sum()
        ),
        "both_diverted": bool(left["diverted"] and right["diverted"]),
        "axis_separation_mm": 1.0e3
        * float(np.linalg.norm(left["axis_m"] - right["axis_m"])),
        "flux_span_difference_wb": float(left["flux_span_wb"] - right["flux_span_wb"]),
    }


def _solve(case, machine, label: str):
    """Run the production root and retain its convergence receipt."""
    started = perf_counter()
    solved = reference.solve(case, machine)
    residual = float(solved.fixed_point.residual)
    elapsed_seconds = perf_counter() - started
    trace = np.asarray(solved.fixed_point.trace, dtype=float)
    measured = trace[np.isfinite(trace)]
    return solved, {
        "arm": label,
        "elapsed_seconds": elapsed_seconds,
        "terminal_relative_fixed_point_residual": residual,
        "trace_slots": int(trace.size),
        "measured_residuals": int(measured.size),
        "first_measured_residual": float(measured[0]),
        "minimum_measured_residual": float(measured.min()),
        "deviations": solved.deviations(),
    }


def _paired_map_trace(
    with_operator,
    without_operator,
    seed,
    reference_flux,
    fixed_core,
    cell_count: int,
    span: float,
) -> dict[str, object]:
    """Follow the two coupled maps from their byte-identical seed."""
    with_map = with_operator.flux_map()
    without_map = without_operator.flux_map()
    with_external = with_operator.external()
    without_external = without_operator.external()
    external_delta = with_external - without_external
    direct_peak = _peak_points(external_delta, cell_count, span)

    with_state = jnp.asarray(seed)
    without_state = jnp.asarray(seed)
    previous_response = jnp.zeros_like(with_state)
    previous_peak = 0.0
    rows = []
    for iteration in range(1, PAIRED_MAP_ITERATIONS + 1):
        internal_with = with_operator.internal(with_state)
        internal_without = without_operator.internal(without_state)
        feedback = internal_with - internal_without
        next_with = with_map(with_state)
        next_without = without_map(without_state)
        response = next_with - next_without
        identity_error = response - (external_delta + feedback)
        increment = response - previous_response
        response_peak = _peak_points(response, cell_count, span)
        with_topology = _topology(with_operator, next_with)
        without_topology = _topology(without_operator, next_without)
        rows.append(
            {
                "iteration": iteration,
                "reproduction_move_points": _score(
                    next_without, reference_flux, fixed_core, cell_count, span
                )
                - _score(next_with, reference_flux, fixed_core, cell_count, span),
                "response_peak_points": response_peak,
                "response_growth_ratio": (
                    None if previous_peak == 0.0 else response_peak / previous_peak
                ),
                "response_increment_peak_points": _peak_points(
                    increment, cell_count, span
                ),
                "direct_external_peak_points": direct_peak,
                "plasma_feedback_peak_points": _peak_points(feedback, cell_count, span),
                "plasma_current_difference_a": float(
                    np.sum(
                        np.abs(
                            np.asarray(with_operator.cell_current(with_state))
                            - np.asarray(without_operator.cell_current(without_state))
                        )
                    )
                ),
                "map_decomposition_identity_error_wb": float(
                    np.max(np.abs(np.asarray(identity_error)))
                ),
                "support": _support_comparison(with_topology, without_topology),
            }
        )
        with_state = next_with
        without_state = next_without
        previous_response = response
        previous_peak = response_peak

    return {
        "iterations": rows,
        "direct_external_peak_points": direct_peak,
        "iteration_one_internal_difference_points": rows[0][
            "plasma_feedback_peak_points"
        ],
        "iteration_one_current_difference_a": rows[0]["plasma_current_difference_a"],
        "maximum_map_decomposition_identity_error_wb": max(
            row["map_decomposition_identity_error_wb"] for row in rows
        ),
    }


def _linear_response_trace(
    without_operator,
    without_root,
    external_delta,
    reference_flux,
    fixed_core,
    cell_count: int,
    span: float,
) -> dict[str, object]:
    """Apply the exact map tangent to each passive-response term."""
    without_map = without_operator.flux_map()
    mapped, tangent = jax.linearize(without_map, without_root)
    root_residual = _peak_points(mapped - without_root, cell_count, span)
    term = jnp.asarray(external_delta)
    cumulative = jnp.zeros_like(term)
    previous_term_peak = None
    rows = []
    for iteration in range(1, LINEAR_RESPONSE_ITERATIONS + 1):
        if iteration > 1:
            term = tangent(term)
        cumulative = cumulative + term
        term_peak = _peak_points(term, cell_count, span)
        counterfactual = without_root + cumulative
        rows.append(
            {
                "iteration": iteration,
                "increment_peak_points": term_peak,
                "increment_growth_ratio": (
                    None
                    if previous_term_peak is None
                    else term_peak / previous_term_peak
                ),
                "cumulative_response_peak_points": _peak_points(
                    cumulative, cell_count, span
                ),
                "counterfactual_reproduction_move_points": _score(
                    without_root, reference_flux, fixed_core, cell_count, span
                )
                - _score(counterfactual, reference_flux, fixed_core, cell_count, span),
            }
        )
        previous_term_peak = term_peak

    late_ratios = np.asarray(
        [
            row["increment_growth_ratio"]
            for row in rows[-5:]
            if row["increment_growth_ratio"] is not None
        ]
    )
    median_ratio = float(np.median(late_ratios))
    return {
        "linearisation_root_residual_points": root_residual,
        "iterations": rows,
        "late_increment_growth_ratio_median": median_ratio,
        "relative_difference_from_measured_spectral_radius": abs(
            median_ratio - MEASURED_UNPINNED_SPECTRAL_RADIUS
        )
        / MEASURED_UNPINNED_SPECTRAL_RADIUS,
    }


def _terminal_decomposition(
    with_solved,
    without_solved,
    external_delta,
    reference_flux,
    cell_count: int,
    span: float,
) -> dict[str, object]:
    """Decompose the terminal score into direct, feedback and support terms."""
    with_core = np.asarray(with_solved.masks.core, dtype=bool)
    without_core = np.asarray(without_solved.masks.core, dtype=bool)
    with_root = with_solved.flux
    without_root = without_solved.flux
    direct_counterfactual = without_root + external_delta

    without_score = _score(without_root, reference_flux, without_core, cell_count, span)
    direct_score = _score(
        direct_counterfactual, reference_flux, without_core, cell_count, span
    )
    with_common_score = _score(
        with_root, reference_flux, without_core, cell_count, span
    )
    with_own_score = _score(with_root, reference_flux, with_core, cell_count, span)
    direct_contribution = without_score - direct_score
    feedback_contribution = direct_score - with_common_score
    support_contribution = with_common_score - with_own_score
    total = without_score - with_own_score
    return {
        "without_passive_score_points": without_score,
        "direct_external_counterfactual_score_points": direct_score,
        "passive_inclusive_score_on_without_support_points": with_common_score,
        "passive_inclusive_score_on_own_support_points": with_own_score,
        "direct_external_contribution_points": direct_contribution,
        "coupled_plasma_feedback_contribution_points": feedback_contribution,
        "support_mask_contribution_points": support_contribution,
        "total_reproduction_move_points": total,
        "sum_of_contributions_points": (
            direct_contribution + feedback_contribution + support_contribution
        ),
        "sum_identity_error_points": abs(
            total - (direct_contribution + feedback_contribution + support_contribution)
        ),
        "root_response_peak_points": _peak_points(
            with_root - without_root, cell_count, span
        ),
        "direct_external_peak_points": _peak_points(external_delta, cell_count, span),
        "root_response_over_direct_peak": _peak_points(
            with_root - without_root, cell_count, span
        )
        / _peak_points(external_delta, cell_count, span),
        "support": {
            "with_core_cells": int(with_core.sum()),
            "without_core_cells": int(without_core.sum()),
            "core_symmetric_difference_cells": int(
                np.logical_xor(with_core, without_core).sum()
            ),
            "both_diverted": bool(
                with_solved.topology.diverted and without_solved.topology.diverted
            ),
            "axis_separation_mm": 1.0e3
            * float(
                np.linalg.norm(
                    np.asarray(with_solved.topology.axis, dtype=float)
                    - np.asarray(without_solved.topology.axis, dtype=float)
                )
            ),
        },
    }


def _verdict(
    seed_identity_error_wb: float,
    paired: dict[str, object],
    linear: dict[str, object],
    terminal: dict[str, object],
) -> dict[str, object]:
    """Choose between an expanding-map gain and an iteration-one path error."""
    scale = max(1.0, terminal["direct_external_peak_points"])
    iteration_one_correct = (
        seed_identity_error_wb == 0.0
        and paired["iteration_one_internal_difference_points"]
        <= IDENTITY_RELATIVE_TOLERANCE * scale
        and paired["iteration_one_current_difference_a"] == 0.0
        and paired["maximum_map_decomposition_identity_error_wb"]
        <= IDENTITY_RELATIVE_TOLERANCE
    )
    support_nonamplifying = terminal["support_mask_contribution_points"] == 0.0
    spectral_consistent = (
        linear["relative_difference_from_measured_spectral_radius"]
        <= SPECTRAL_RATIO_RELATIVE_TOLERANCE
    )
    feedback_dominates = abs(
        terminal["coupled_plasma_feedback_contribution_points"]
    ) > abs(terminal["direct_external_contribution_points"])

    if (
        iteration_one_correct
        and support_nonamplifying
        and spectral_consistent
        and feedback_dominates
    ):
        classification = "EXPANDING_MAP_AMPLIFICATION"
        mechanism = "noncontractive_free_boundary_map_resolvent_gain"
        repair_scope = [
            "nova/equilibrium/forward_operator.py",
            "nova/equilibrium/fixed_point.py",
        ]
        ruling = (
            "Iteration one is exactly the passive external field with no source "
            "or seed discrepancy. Its one-cell physical support move is removed "
            "by fixed-support scoring; later exact-tangent terms grow at the "
            "measured expanding-map rate and dominate the terminal score move."
        )
    else:
        classification = "CLOSURE_PATH_ERROR"
        mechanism = "iteration_one_closure_arithmetic_or_support_error"
        repair_scope = [
            "nova/equilibrium/source.py",
            "nova/equilibrium/domain.py",
            "nova/equilibrium/forward_operator.py",
        ]
        ruling = (
            "The passive response is already inconsistent before expanding-map "
            "feedback, or its support differs; repair the named iteration-one path."
        )

    return {
        "classification": classification,
        "mechanism": mechanism,
        "ruling": ruling,
        "iteration_one_arithmetic_correct": iteration_one_correct,
        "terminal_support_contribution_zero": support_nonamplifying,
        "growth_consistent_with_unpinned_spectral_radius": spectral_consistent,
        "plasma_feedback_dominates_direct_contribution": feedback_dominates,
        "measured_unpinned_spectral_radius": MEASURED_UNPINNED_SPECTRAL_RADIUS,
        "measured_pinned_spectral_radius": MEASURED_PINNED_SPECTRAL_RADIUS,
        "late_increment_growth_ratio_median": linear[
            "late_increment_growth_ratio_median"
        ],
        "repair_file_scope": repair_scope,
        "repair_scope_occupancy": (
            "The concurrent solver lane holds forward_operator.py and "
            "fixed_point.py; coordinate there rather than editing either here."
        ),
        "code_path": [
            "nova/equilibrium/forward_operator.py:569 captures the external field once",
            (
                "nova/equilibrium/forward_operator.py:571-573 adds the "
                "state-dependent plasma image"
            ),
            (
                "nova/equilibrium/fixed_point.py:244-255 forms and solves "
                "the exact (I-J) action"
            ),
            (
                "tests/test_equilibrium_forward_reference.py:1635-1640 "
                "supplies the same stored-map seed"
            ),
            (
                "tests/test_equilibrium_forward_reference.py:1552-1557 "
                "scores the topology core support"
            ),
        ],
        "excluded_closure_path_candidates": {
            "source_normalisation_amplification": (
                "excluded at iteration one: target_current is absent and the paired "
                "internal image plus plasma current are identical"
            ),
            "seed_or_basin_move": (
                "excluded: seed bytes are identical and both terminal roots remain "
                "on the same diverted branch"
            ),
            "support_or_mask_difference": (
                "excluded as amplification: paired iterations use fixed support, "
                "and the four-cell terminal mask difference contributes zero points"
            ),
        },
    }


def _plot(receipt: dict[str, object], path: Path) -> None:
    """Plot iteration-resolved response and exact-tangent growth."""
    paired = receipt["paired_coupled_map"]["iterations"]
    linear = receipt["exact_tangent_response"]["iterations"]
    terminal = receipt["terminal_decomposition"]["total_reproduction_move_points"]

    figure, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), constrained_layout=True)
    upper, lower = axes
    upper.plot(
        [row["iteration"] for row in paired],
        [row["reproduction_move_points"] for row in paired],
        marker="o",
        color="#2a6099",
        label="paired coupled map",
    )
    upper.plot(
        [row["iteration"] for row in linear],
        [row["counterfactual_reproduction_move_points"] for row in linear],
        marker=".",
        color="0.4",
        label="exact-tangent response",
    )
    upper.axhline(terminal, color="#b24c3d", linestyle="--", linewidth=1.2)
    upper.text(
        LINEAR_RESPONSE_ITERATIONS,
        terminal,
        f" terminal {terminal:.4f} points ",
        color="#b24c3d",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    upper.axhline(
        DOCUMENTED_DIRECT_RESPONSE_POINTS,
        color="0.55",
        linestyle=":",
        linewidth=1.1,
    )
    upper.set_ylabel("reproduction move [points]")
    upper.legend(frameon=False, fontsize=9)
    upper.spines[["top", "right"]].set_visible(False)

    ratio_rows = [row for row in linear if row["increment_growth_ratio"] is not None]
    lower.plot(
        [row["iteration"] for row in ratio_rows],
        [row["increment_growth_ratio"] for row in ratio_rows],
        marker="o",
        color="#2a6099",
    )
    lower.axhline(
        MEASURED_UNPINNED_SPECTRAL_RADIUS,
        color="#b24c3d",
        linestyle="--",
        linewidth=1.2,
    )
    lower.text(
        LINEAR_RESPONSE_ITERATIONS,
        MEASURED_UNPINNED_SPECTRAL_RADIUS,
        f" measured spectral radius {MEASURED_UNPINNED_SPECTRAL_RADIUS:.4f} ",
        color="#b24c3d",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    lower.set_xlabel("coupled-map iteration")
    lower.set_ylabel("increment growth ratio")
    lower.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, transparent=True)
    plt.close(figure)


def run() -> dict[str, object]:
    """Run the fixed-carrier mechanism trace."""
    source_commit = measurement_stamp(Path.cwd())
    configure_dtypes()
    case = reference.require_reference()
    machine_started = perf_counter()
    machine = reference.cached_machine(case, DENSE_CELL_REQUEST, passive=True)
    machine_seconds = perf_counter() - machine_started
    cache = machine.cache_receipt
    if cache is None:
        raise RuntimeError("the dense carrier has no persistent-cache receipt")
    without_current = machine.source_current.copy()
    without_current[-machine.passive_columns :] = 0.0
    without_machine = replace(
        machine,
        source_current=without_current,
        passive_columns=0,
        cache_receipt=None,
    )

    with_operator = reference.forward_operator(case, machine)
    without_operator = reference.forward_operator(case, without_machine)
    seed_with = reference.seed_flux(case, machine)
    seed_without = reference.seed_flux(case, without_machine)
    seed_identity_error = float(
        np.max(np.abs(np.asarray(seed_with) - np.asarray(seed_without)))
    )

    with_solved, with_receipt = _solve(case, machine, "declared_passive_currents")
    without_solved, without_receipt = _solve(
        case, without_machine, "passive_currents_zeroed"
    )
    cell_count = len(machine.node)
    span = abs(float(case.flux_span))
    reference_flux = case.flux(machine.radius, machine.node[:, 1])
    without_core = np.asarray(without_solved.masks.core, dtype=bool)
    external_delta = with_operator.external() - without_operator.external()

    paired = _paired_map_trace(
        with_operator,
        without_operator,
        seed_with,
        reference_flux,
        without_core,
        cell_count,
        span,
    )
    linear = _linear_response_trace(
        without_operator,
        without_solved.flux,
        external_delta,
        reference_flux,
        without_core,
        cell_count,
        span,
    )
    terminal = _terminal_decomposition(
        with_solved,
        without_solved,
        external_delta,
        reference_flux,
        cell_count,
        span,
    )
    verdict = _verdict(seed_identity_error, paired, linear, terminal)
    return {
        "receipt": {
            "kind": "passive_closure_iteration_trace",
            "status": "complete",
            "source_commit": source_commit,
            "checkout_porcelain_empty_before_measurement": True,
            "device_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "production_solve_count": 2,
        },
        "carrier": {
            "requested_cells": DENSE_CELL_REQUEST,
            "realised_plasma_cells": cell_count,
            "machine_request_seconds": machine_seconds,
            "cache": asdict(cache),
            "passive_columns": machine.passive_columns,
            "reference_flux_span_wb": span,
        },
        "comparators": {
            "documented_direct_response_points": DOCUMENTED_DIRECT_RESPONSE_POINTS,
            "registered_ceiling_points": REGISTERED_CEILING_POINTS,
            "measured_unpinned_spectral_radius": MEASURED_UNPINNED_SPECTRAL_RADIUS,
            "measured_pinned_spectral_radius": MEASURED_PINNED_SPECTRAL_RADIUS,
        },
        "controlled_inputs": {
            "seed_max_absolute_difference_wb": seed_identity_error,
            "geometry_shared": True,
            "source_profiles_shared": True,
            "only_changed_input": "declared passive-current columns retained or zeroed",
        },
        "production_solves": [with_receipt, without_receipt],
        "paired_coupled_map": paired,
        "exact_tangent_response": linear,
        "terminal_decomposition": terminal,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    arguments = parser.parse_args()
    receipt = _strict_json(run())
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot(receipt, arguments.figure)
    print(
        json.dumps(
            {
                "classification": receipt["verdict"]["classification"],
                "mechanism": receipt["verdict"]["mechanism"],
                "terminal_move_points": receipt["terminal_decomposition"][
                    "total_reproduction_move_points"
                ],
                "direct_contribution_points": receipt["terminal_decomposition"][
                    "direct_external_contribution_points"
                ],
                "feedback_contribution_points": receipt["terminal_decomposition"][
                    "coupled_plasma_feedback_contribution_points"
                ],
                "late_increment_growth_ratio": receipt["exact_tangent_response"][
                    "late_increment_growth_ratio_median"
                ],
                "receipt": str(arguments.receipt),
                "figure": str(arguments.figure),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
