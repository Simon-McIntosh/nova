# ruff: noqa: E501
"""Measure whether same-shot warm-neighbour seeding lifts the five bounded
MAST frozen-six current-constrained stalls.

Every reference is first cold-solved from its own EFIT-derived seed through
the public target-current-constrained ``ForwardProfile.solve`` seam, driving all 101 fitted
active and passive/vessel circuits through the passive-inclusive prescribed-
current policy and eliminating the plasma amplitude against the row's own
declared current. When that cold arm does not reach a converged, current-
consistent plasma root, a same-shot neighbour on the declared symmetric
time-offset ladder is cold-solved in turn; the first ladder entry whose own
cold solve converges supplies its terminal flux as the warm seed for one
more solve of the target row's own operator. The warm source is never
selected by how well it scores against the EFIT label - only by whether its
own cold solve converges - and earlier offsets are tried before later ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from benchmarks.diiid_constrained_cold_start import (
    CURRENT_RELATIVE_ERROR_TOLERANCE,
    NEIGHBOUR_FRAME_OFFSETS,
    RELATIVE_RESIDUAL_TOLERANCE,
    _neighbour_candidates,
    _solve_public_seam,
)
from benchmarks.efit_forward_parity_slice import (
    CURRENT_CONSTRAINED_OUTPUT,
    CURRENT_CONSTRAINED_RECEIPT_NAME,
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    _mast_case_from_selection,
    _metric_qualification,
    _passive_inclusive_case,
    _pinned_metrics,
    select_slices_by_shot,
)
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes

DEFAULT_OUTPUT = Path("docs/figures/efit-forward-parity")
RECEIPT_NAME = "warm-neighbour-stall-lift.json"
FIGURE_NAME = "warm-neighbour-stall-lift.png"
BANKED_CONSTRAINED_RECEIPT = (
    CURRENT_CONSTRAINED_OUTPUT / CURRENT_CONSTRAINED_RECEIPT_NAME
)
CURRENT_MATCH_TOLERANCE = 1.0e-9
TARGET_CURRENT_EXACT_TOLERANCE = 1.0e-9
DECLARED_NEIGHBOUR_FRAME_OFFSETS = (
    -1,
    1,
    -2,
    2,
    -4,
    4,
    -8,
    8,
    -16,
    16,
    -32,
    32,
)


@dataclass(frozen=True)
class _MastSelection:
    """Selected-row surface consumed by the imported shared helpers."""

    frame: int
    recorded_plasma_current_a: float
    path: Path


@dataclass(frozen=True)
class _MastFrame:
    """MAST adapter exposing exactly the shared constrained-helper surface."""

    selected: _MastSelection
    row: dict[str, Any]
    profile: Any
    current: np.ndarray
    seed: np.ndarray


def _prepare_frame(
    store: Path, shot: int, row: int, cache_box: list[Any]
) -> tuple[_MastFrame, dict[str, Any], dict[str, Any]]:
    """Build one MAST row through the passive-inclusive 101-circuit seam.

    ``cache_box`` is a one-element mutable container holding the shared
    response-matrix cache once the first row bootstraps it; the matrix is
    geometry-only and reused across every row and shot, matching the
    reuse already proven by the landed constrained frozen-six scorecard.
    """
    mast_case, context = _mast_case_from_selection(
        store, {"shot": shot, "slice_index": row}, qualification=None
    )
    passive_case, profile, policy = _passive_inclusive_case(
        mast_case, context, cache_box[0]
    )
    if cache_box[0] is None:
        prescribed = profile.operator.prescribed_current_field
        cache_box[0] = {
            "response": np.asarray(prescribed.response, dtype=np.float64),
            "input_digests": policy["response_input_digests"],
            "audit": {
                name: policy[name]
                for name in (
                    "stored_circuit_count",
                    "active_circuit_count",
                    "passive_or_vessel_circuit_count",
                    "section_kernel_evaluations",
                    "passive_registry_minimum_overlap_fraction",
                    "passive_registry_maximum_separation_m",
                )
            },
        }
    target_current = abs(float(mast_case["reference"]["plasma_current_a"]))
    group = context["group"]
    frame = _MastFrame(
        selected=_MastSelection(
            frame=row,
            recorded_plasma_current_a=target_current,
            path=store / f"{shot}.zarr",
        ),
        row={"efit_times": np.asarray(group["time"], dtype=np.float64)},
        profile=profile,
        current=np.asarray(profile.operator.external_current, dtype=np.float64),
        seed=np.asarray(passive_case["state"], dtype=np.float64),
    )
    return frame, mast_case, context


def _candidate_rows(frame: _MastFrame) -> list[int]:
    """Apply the imported declared ladder to the MAST frame adapter."""
    return _neighbour_candidates(frame)


def _record_outcome(
    frame: _MastFrame,
    context: dict[str, Any],
    outcome: Any,
) -> tuple[dict[str, Any], Any]:
    """Classify and score one imported-seam terminal without label selection."""
    state = np.asarray(outcome.state, dtype=np.float64)
    target_current = frame.selected.recorded_plasma_current_a
    current_error = abs(outcome.achieved_current_a / target_current - 1.0)
    _masks, topology = frame.profile.operator.read(jnp.asarray(state))
    topology_consistent = bool(topology.diverted)
    nonzero_current = bool(abs(outcome.achieved_current_a) >= 0.01 * target_current)
    converged = bool(
        np.all(np.isfinite(state))
        and topology_consistent
        and np.isfinite(outcome.residual)
        and outcome.residual <= RELATIVE_RESIDUAL_TOLERANCE
        and np.isfinite(current_error)
        and current_error <= CURRENT_RELATIVE_ERROR_TOLERANCE
    )
    if converged and nonzero_current:
        outcome_class = "converged_plasma_root"
    elif not nonzero_current:
        outcome_class = "vacuum_collapse"
    else:
        outcome_class = "bounded_non_convergence"
    equilibrium = frame.profile.observe(
        jnp.asarray(state),
        current=jnp.asarray(frame.current),
        target_current=target_current,
    )
    metrics = _pinned_metrics(
        context["group"],
        context["row"],
        frame.profile,
        context["reference_flux"],
        equilibrium,
    )
    trajectory = list(outcome.residual_trajectory)
    if not trajectory or trajectory[-1] != outcome.residual:
        trajectory.append(float(outcome.residual))
    return (
        {
            "forward_branch_receipt": {
                "converged": converged,
                "residual": float(outcome.residual),
                "achieved_class": "diverted" if topology_consistent else "limited",
                "topology_consistent": topology_consistent,
            },
            "terminal_state": {
                "plasma_current_a": float(outcome.achieved_current_a),
                "nonzero_current": nonzero_current,
                "profile_amplitude": float(outcome.amplitude),
            },
            "registered_parity_metrics": metrics,
            "residual_trajectory": trajectory,
            "iterations": int(outcome.iterations),
            "termination": outcome.termination,
            "outcome_class": outcome_class,
            "target_current_relative_error": current_error,
        },
        equilibrium,
    )


def _find_mast_warm_source(
    store: Path,
    shot: int,
    target_frame: _MastFrame,
    cache_box: list[Any],
) -> tuple[
    list[dict[str, Any]],
    tuple[int, _MastFrame, dict[str, Any], Any] | None,
]:
    """Walk the declared ladder and return the first cold-converged source.

    Candidate order comes only from the imported enumerator. Qualification uses
    the candidate's own constrained solve and never any EFIT parity metric.
    """
    checks: list[dict[str, Any]] = []
    for candidate_row in _candidate_rows(target_frame):
        candidate, mast_case, context = _prepare_frame(
            store, shot, candidate_row, cache_box
        )
        outcome = _solve_public_seam(candidate, candidate.seed)
        record, _equilibrium = _record_outcome(candidate, context, outcome)
        outcome_class = record["outcome_class"]
        branch_receipt = record["forward_branch_receipt"]
        checks.append(
            {
                "row": candidate_row,
                "time_s": mast_case["reference"]["time_s"],
                "outcome_class": outcome_class,
                "converged": bool(outcome_class == "converged_plasma_root"),
                "fixed_point_residual": branch_receipt["residual"],
                "achieved_class": branch_receipt["achieved_class"],
                "topology_consistent": branch_receipt["topology_consistent"],
                "terminal_plasma_current_a": record["terminal_state"][
                    "plasma_current_a"
                ],
            }
        )
        if outcome_class == "converged_plasma_root":
            return checks, (candidate_row, candidate, record, outcome)
    return checks, None


def measure_reference(
    store: Path,
    shot: int,
    row: int,
    cache_box: list[Any],
    banked_control: dict[str, Any],
) -> dict[str, Any]:
    """Measure one reference's cold control, ladder search, and warm solve."""
    frame, mast_case, context = _prepare_frame(store, shot, row, cache_box)
    target_current = frame.selected.recorded_plasma_current_a
    cold_solve = _solve_public_seam(frame, frame.seed)
    cold_record, _cold_equilibrium = _record_outcome(frame, context, cold_solve)
    cold_outcome = cold_record["outcome_class"]
    cold_residual = cold_record["forward_branch_receipt"]["residual"]
    banked_residual = banked_control["fixed_point_residual"]
    cold_current = cold_record["terminal_state"]["plasma_current_a"]
    cold_metrics = cold_record["registered_parity_metrics"]
    cold_control = {
        "outcome_class": cold_outcome,
        "terminal_fixed_point_residual": cold_residual,
        "terminal_plasma_current_a": cold_current,
        "target_current_relative_error": cold_record["target_current_relative_error"],
        "banked_terminal_fixed_point_residual": banked_residual,
        "reproduces_banked_control": bool(
            cold_residual is not None
            and abs(cold_residual - banked_residual)
            <= CURRENT_MATCH_TOLERANCE * max(abs(banked_residual), 1.0)
        ),
        "registered_parity_metrics": cold_metrics,
        "per_metric_qualification": _metric_qualification(cold_metrics, cold_residual),
        "iterations": cold_record["iterations"],
        "termination": cold_record["termination"],
        "residual_trajectory": cold_record["residual_trajectory"],
    }
    warm_search: dict[str, Any]
    warm_solve: dict[str, Any] | None = None
    if cold_outcome == "converged_plasma_root":
        warm_search = {
            "triggered": False,
            "reason": "cold entry already reaches a converged plasma root",
            "candidate_offsets": list(NEIGHBOUR_FRAME_OFFSETS),
            "checks": [],
            "qualified_source_found": False,
        }
    else:
        checks, warm_source = _find_mast_warm_source(store, shot, frame, cache_box)
        warm_search = {
            "triggered": True,
            "candidate_offsets": list(NEIGHBOUR_FRAME_OFFSETS),
            "selection_rule": (
                "declared symmetric time-offset ladder, earlier before later; "
                "first same-shot cold-converged candidate wins; never ranked "
                "against the EFIT label"
            ),
            "checks": checks,
            "qualified_source_found": warm_source is not None,
        }
        if warm_source is not None:
            source_row, _source_frame, source_record, source_outcome = warm_source
            source_time_s = next(
                item["time_s"] for item in checks if item["row"] == source_row
            )
            warm_search["selected_source"] = {
                "row": source_row,
                "time_s": source_time_s,
                "time_offset_s": source_time_s - mast_case["reference"]["time_s"],
                "own_cold_fixed_point_residual": source_record[
                    "forward_branch_receipt"
                ]["residual"],
            }
            warm_outcome = _solve_public_seam(
                frame, np.asarray(source_outcome.state, dtype=np.float64)
            )
            warm_record, _warm_equilibrium = _record_outcome(
                frame, context, warm_outcome
            )
            warm_outcome_class = warm_record["outcome_class"]
            warm_residual = warm_record["forward_branch_receipt"]["residual"]
            warm_current = warm_record["terminal_state"]["plasma_current_a"]
            metrics = warm_record["registered_parity_metrics"]
            warm_solve = {
                "outcome_class": warm_outcome_class,
                "terminal_fixed_point_residual": warm_residual,
                "terminal_plasma_current_a": warm_current,
                "target_current_relative_error": warm_record[
                    "target_current_relative_error"
                ],
                "lifted_to_converged_plasma_root": bool(
                    warm_outcome_class == "converged_plasma_root"
                ),
                "registered_parity_metrics": metrics,
                "per_metric_qualification": _metric_qualification(
                    metrics, warm_residual
                ),
                "iterations": warm_record["iterations"],
                "termination": warm_record["termination"],
                "residual_trajectory": warm_record["residual_trajectory"],
            }
    return {
        "reference": {
            "shot": shot,
            "slice_index": row,
            "time_s": mast_case["reference"]["time_s"],
            "target_current_a": target_current,
            "target_current_source": "abs(efm/plasma_current_c) on the selected row",
        },
        "cold_control": cold_control,
        "warm_neighbour_search": warm_search,
        "warm_solve": warm_solve,
    }


def render_figure(references: list[dict[str, Any]], path: Path) -> None:
    """Plot cold vs. warm residual trajectories for every bounded reference."""
    figure, axes = plt.subplots(
        1,
        len(references),
        figsize=(max(4.5, 3.2 * len(references)), 3.6),
        constrained_layout=True,
    )
    if len(references) == 1:
        axes = [axes]
    for axis, reference in zip(axes, references):
        cold = [
            value
            for value in reference["cold_control"]["residual_trajectory"]
            if value is not None
        ]
        warm = reference["warm_solve"]
        warm_trajectory = []
        if warm is not None:
            warm_trajectory = [
                value for value in warm["residual_trajectory"] if value is not None
            ]
        if len(cold) == 1 and len(warm_trajectory) <= 1:
            values = cold + warm_trajectory
            labels = ["cold"] + (["warm"] if warm_trajectory else [])
            axis.semilogy(
                np.arange(len(values)),
                np.maximum(values, np.finfo(float).tiny),
                marker="o",
                ls="none",
            )
            axis.set_xticks(np.arange(len(values)), labels)
            axis.set_xlabel("Initial state")
        else:
            axis.semilogy(
                np.arange(len(cold)),
                np.maximum(cold, np.finfo(float).tiny),
                marker="o",
                ms=3,
                lw=1.0,
                label="cold",
            )
            if warm_trajectory:
                axis.semilogy(
                    np.arange(len(warm_trajectory)),
                    np.maximum(warm_trajectory, np.finfo(float).tiny),
                    marker="s",
                    ms=3,
                    lw=1.0,
                    label="warm neighbour",
                )
            axis.set_xlabel("Recorded residual evaluation")
            axis.legend(fontsize=7)
        axis.axhline(FIXED_POINT_CRITERION, color="black", ls="--", lw=0.8)
        shot = reference["reference"]["shot"]
        row = reference["reference"]["slice_index"]
        axis.set_title(f"{shot}/{row}")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Fixed-point residual")
    figure.suptitle("Cold and warm-neighbour residuals")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _banked_artifact_digests(output: Path) -> dict[str, str]:
    excluded = {RECEIPT_NAME, FIGURE_NAME}
    return {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name not in excluded
    }


def run(
    store: Path = SHOT_STORE,
    output: Path = DEFAULT_OUTPUT,
    shots: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Measure warm-neighbour stall lift on the five bounded frozen-six rows."""
    configure_dtypes()
    if NEIGHBOUR_FRAME_OFFSETS != DECLARED_NEIGHBOUR_FRAME_OFFSETS:
        raise RuntimeError("the imported warm-neighbour offset ladder changed")
    output.mkdir(parents=True, exist_ok=True)
    banked_artifacts_before = _banked_artifact_digests(output)
    if len(banked_artifacts_before) != 23:
        raise RuntimeError(
            "the immutable parity bank no longer contains exactly 23 files"
        )
    banked_receipt = json.loads(BANKED_CONSTRAINED_RECEIPT.read_text())
    banked_by_shot = {int(row["shot"]): row for row in banked_receipt["per_shot_table"]}
    bounded_keys = {
        (int(row["shot"]), int(row["slice_index"]))
        for row in banked_receipt["per_shot_table"]
        if row["constrained_outcome"] != "converged_plasma_root"
    }
    requested_shots = None if shots is None else set(shots)
    if requested_shots is not None:
        unknown = requested_shots - {shot for shot, _row in bounded_keys}
        if unknown:
            raise ValueError(
                f"requested shots are not bounded frozen-six references: {sorted(unknown)}"
            )
    selected = select_slices_by_shot(DECOMPOSITION_BANK)
    cache_box: list[Any] = [None]
    references = []
    already_converged = None
    for selected_row, _qualification in selected:
        shot = int(selected_row["shot"])
        row = int(selected_row["slice_index"])
        banked_control = banked_by_shot[shot]
        if int(banked_control["slice_index"]) != row:
            raise RuntimeError("the current selection differs from the banked row")
        if banked_control["constrained_outcome"] == "converged_plasma_root":
            already_converged = {
                "shot": shot,
                "slice_index": row,
                "banked_fixed_point_residual": banked_control["fixed_point_residual"],
                "note": (
                    "the banked converged plasma root; outside the five "
                    "bounded stalls this measure targets"
                ),
            }
            continue
        if requested_shots is not None and shot not in requested_shots:
            continue
        references.append(
            measure_reference(store, shot, row, cache_box, banked_control)
        )
    expected_reference_count = (
        len(bounded_keys) if requested_shots is None else len(requested_shots)
    )
    if len(references) != expected_reference_count:
        raise RuntimeError("the requested bounded references were not all measured")
    figure_path = output / FIGURE_NAME
    render_figure(references, figure_path)
    banked_artifacts_after = _banked_artifact_digests(output)
    if banked_artifacts_after != banked_artifacts_before:
        raise RuntimeError("an immutable parity-bank artifact changed during the run")
    banked_converged_count = sum(
        1
        for row in banked_receipt["per_shot_table"]
        if row["reaches_nonzero_plasma_root"]
    )
    lifted = [
        reference
        for reference in references
        if reference["warm_solve"] is not None
        and reference["warm_solve"]["lifted_to_converged_plasma_root"]
    ]
    unrecovered = [
        reference
        for reference in references
        if not reference["warm_neighbour_search"]["qualified_source_found"]
    ]
    measured_keys = {
        (
            reference["reference"]["shot"],
            reference["reference"]["slice_index"],
        )
        for reference in references
    }
    unmeasured_keys = sorted(bounded_keys - measured_keys)
    all_targets_exact = all(
        reference["cold_control"]["target_current_relative_error"]
        <= TARGET_CURRENT_EXACT_TOLERANCE
        and (
            reference["warm_solve"] is None
            or reference["warm_solve"]["target_current_relative_error"]
            <= TARGET_CURRENT_EXACT_TOLERANCE
        )
        for reference in references
    )
    aggregate = {
        "bounded_reference_count": len(bounded_keys),
        "measured_reference_count": len(references),
        "measured_references": [
            {
                "shot": reference["reference"]["shot"],
                "slice_index": reference["reference"]["slice_index"],
            }
            for reference in references
        ],
        "unmeasured_references": [
            {"shot": shot, "slice_index": row} for shot, row in unmeasured_keys
        ],
        "banked_converged_plasma_roots": banked_converged_count,
        "cold_converged_plasma_roots_among_bounded": sum(
            1
            for reference in references
            if reference["cold_control"]["outcome_class"] == "converged_plasma_root"
        ),
        "warm_neighbour_source_found_count": sum(
            1
            for reference in references
            if reference["warm_neighbour_search"]["qualified_source_found"]
        ),
        "warm_lifted_to_converged_plasma_root_count": len(lifted),
        "warm_source_found_but_target_not_lifted_count": sum(
            1
            for reference in references
            if reference["warm_neighbour_search"]["qualified_source_found"]
            and (
                reference["warm_solve"] is None
                or not reference["warm_solve"]["lifted_to_converged_plasma_root"]
            )
        ),
        "unrecovered_count": len(unrecovered),
        "unrecovered_shots": [
            {
                "shot": reference["reference"]["shot"],
                "slice_index": reference["reference"]["slice_index"],
                "rows_tried": [
                    item["row"] for item in reference["warm_neighbour_search"]["checks"]
                ],
            }
            for reference in unrecovered
        ],
        "total_converged_plasma_roots_after_warm_lift": banked_converged_count
        + len(lifted),
        "all_terminal_currents_exact_at_target": all_targets_exact,
        "cold_control_reproduction_count": sum(
            reference["cold_control"]["reproduces_banked_control"]
            for reference in references
        ),
        "all_cold_controls_reproduce_banked": all(
            reference["cold_control"]["reproduces_banked_control"]
            for reference in references
        ),
        "verdict": (
            "PARTIAL_WARM_NEIGHBOUR_MEASUREMENT"
            if unmeasured_keys
            else (
                "PASS_WARM_NEIGHBOUR_LIFTS_AT_LEAST_ONE_BOUNDED_STALL"
                if lifted
                else "FAIL_WARM_NEIGHBOUR_LIFTS_NO_BOUNDED_STALL"
            )
        ),
    }
    receipt = {
        "measurement": (
            "same-shot warm-neighbour seeding measured against the five bounded "
            "MAST frozen-six current-constrained stalls, with the cold arm "
            "reproduced beside it as the control"
        ),
        "banked_control_receipt": str(BANKED_CONSTRAINED_RECEIPT),
        "banked_control_receipt_sha256": hashlib.sha256(
            BANKED_CONSTRAINED_RECEIPT.read_bytes()
        ).hexdigest(),
        "reuse": {
            "imported_unchanged_from_benchmarks.diiid_constrained_cold_start": [
                "NEIGHBOUR_FRAME_OFFSETS",
                "_neighbour_candidates",
                "_solve_public_seam",
            ],
            "imported_unchanged_from_benchmarks.efit_forward_parity_slice": [
                "DECOMPOSITION_BANK",
                "FIXED_POINT_CRITERION",
                "select_slices_by_shot",
                "_mast_case_from_selection",
                "_passive_inclusive_case",
                "_pinned_metrics",
                "_metric_qualification",
            ],
            "written_mast_analogue": [
                "_MastFrame/_MastSelection (duck-typed PreparedFrame adapter)",
                "_prepare_frame (MAST frame preparation analogue)",
                "_find_mast_warm_source (local earlier-first walk replacing only "
                "the DIII-D-global _find_warm_source)",
            ],
        },
        "solver": {
            "entry_point": "nova.equilibrium.forward.ForwardProfile.solve",
            "shared_helper": (
                "benchmarks.diiid_constrained_cold_start._solve_public_seam"
            ),
            "route": "host_krylov",
            "target_current_policy": (
                "abs(efm/plasma_current_c) on the row being solved; declared "
                "current-elimination target, never a label fit"
            ),
            "prescribed_current_policy": (
                "all 101 fitted EFIT circuits (13 active + 88 passive/vessel) "
                "through one explicit response-matrix policy, as the current-"
                "constrained frozen-six scorecard drives"
            ),
            "registered_fixed_point_criterion": FIXED_POINT_CRITERION,
        },
        "offset_ladder": {
            "offsets": list(NEIGHBOUR_FRAME_OFFSETS),
            "repr": repr(NEIGHBOUR_FRAME_OFFSETS),
            "source": "benchmarks.diiid_constrained_cold_start.NEIGHBOUR_FRAME_OFFSETS",
            "rule": (
                "earlier offsets first; the warm source is never selected by how "
                "well it scores against EFIT, only by whether its own cold solve "
                "converges; a reference whose ladder finds no converged source "
                "within the shot's available rows is reported unrecovered with "
                "the rows tried, not dropped"
            ),
        },
        "already_converged_baseline": already_converged,
        "references": references,
        "aggregate": aggregate,
        "banked_artifact_integrity": {
            "verified_unchanged_count": len(banked_artifacts_after),
            "unchanged": banked_artifacts_after == banked_artifacts_before,
            "sha256_by_name": banked_artifacts_after,
        },
        "artifacts": {
            "receipt": str(output / RECEIPT_NAME),
            "figure": str(figure_path),
        },
    }
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shots", nargs="+", type=int)
    arguments = parser.parse_args()
    receipt = run(
        arguments.store,
        arguments.output,
        shots=None if arguments.shots is None else tuple(arguments.shots),
    )
    print(json.dumps(receipt["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
