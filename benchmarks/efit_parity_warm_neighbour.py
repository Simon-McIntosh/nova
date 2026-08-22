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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from benchmarks.diiid_constrained_cold_start import (
    CURRENT_RELATIVE_ERROR_TOLERANCE,
    MOMENT_SEEDED_CRITERION,
    MOMENT_SEEDED_OUTPUT,
    NEIGHBOUR_FRAME_OFFSETS,
    RELATIVE_RESIDUAL_TOLERANCE,
    _neighbour_candidates,
    _solve_public_seam,
    _source_stamp,
    run_moment_seed as run_diiid_moment_seed,
)
from benchmarks.diiid_forward_gs_match import DEFAULT_DATA as DIIID_DATA
from benchmarks.efit_forward_parity_slice import (
    CURRENT_CONSTRAINED_OUTPUT,
    CURRENT_CONSTRAINED_RECEIPT_NAME,
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    GMRES_ITERATIONS,
    NEWTON_STEPS,
    RELAXATION,
    STEP_CAP,
    WARMUP_SWEEPS,
    _mast_case_from_selection,
    _metric_qualification,
    _passive_inclusive_case,
    _pinned_metrics,
    select_slices_by_shot,
)
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes

DEFAULT_OUTPUT = Path("docs/figures/efit-forward-parity")
NEWTON_REPLAY_OUTPUT = Path("docs/figures/dual-basin-solve")
RECEIPT_NAME = "warm-neighbour-stall-lift.json"
FIGURE_NAME = "warm-neighbour-stall-lift.png"
NEWTON_REPLAY_RECEIPT_NAME = "newton-warm-ladder-replay.json"
NEWTON_REPLAY_FIGURE_NAME = "newton-warm-ladder-replay.png"
NEWTON_REPLAY_ATTEMPT_PREFIX = "newton-warm-ladder"
MOMENT_SEEDED_AGGREGATE_NAME = "moment-seeded-cold-start.json"
MOMENT_SEEDED_FIGURE_NAME = "moment-seeded-cold-start.png"
MOMENT_SEEDED_ATTEMPT_PREFIX = "mast-moment-seeded-attempt"
BANKED_CONSTRAINED_RECEIPT = (
    CURRENT_CONSTRAINED_OUTPUT / CURRENT_CONSTRAINED_RECEIPT_NAME
)
BANKED_WARM_RECEIPT = DEFAULT_OUTPUT / RECEIPT_NAME
TARED_TOPOLOGY_RECEIPT = DEFAULT_OUTPUT / "tared-plasma-support-solve.json"
BALANCED_REFERENCE_RECEIPT = DEFAULT_OUTPUT / "converged-root-geometry-attribution.json"
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


@dataclass(frozen=True)
class _NewtonBranchProfileAdapter:
    """Route the imported public-seam helper through the pinned Newton branch.

    The imported helper owns the constrained map preflight, solve invocation,
    failure handling, and terminal observation. This adapter changes only the
    public solve method selected by that invocation: its host-shaped call is
    translated to the banked fixed-shape diverted-branch contract.
    """

    wrapped: Any

    @property
    def operator(self) -> Any:
        return self.wrapped.operator

    @property
    def lattice(self) -> Any:
        return self.wrapped.lattice

    def flux_map(
        self,
        current: Any = None,
        requested_class: Any = None,
        target_current: Any = None,
    ) -> Any:
        return self.wrapped.flux_map(
            current,
            requested_class=requested_class,
            target_current=target_current,
        )

    def observe(
        self,
        flux: Any,
        current: Any = None,
        target_current: Any = None,
    ) -> Any:
        return self.wrapped.observe(
            flux,
            current=current,
            target_current=target_current,
        )

    def solve(
        self,
        initial_flux: Any,
        *,
        route: str,
        current: Any = None,
        target_current: Any = None,
        **host_options: Any,
    ) -> Any:
        expected_host_options = {
            "method",
            "inner_maxiter",
            "maxiter",
            "f_tol",
            "line_search",
        }
        if route != "host_krylov" or set(host_options) != expected_host_options:
            raise RuntimeError("the imported constrained helper call shape changed")
        branch = self.wrapped.solve_branch(
            initial_flux,
            TopologyClass.DIVERTED,
            route="newton_krylov",
            current=current,
            target_current=target_current,
            tolerance=FIXED_POINT_CRITERION,
            newton_steps=NEWTON_STEPS,
            gmres_iterations=GMRES_ITERATIONS,
            warmup=WARMUP_SWEEPS,
            relaxation=RELAXATION,
            step_cap=STEP_CAP,
        )
        return branch.equilibrium


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


def _prepare_newton_frame(
    store: Path, shot: int, row: int, cache_box: list[Any]
) -> tuple[_MastFrame, dict[str, Any], dict[str, Any]]:
    """Wrap a MAST row so the imported seam selects the pinned Newton solve."""
    frame, mast_case, context = _prepare_frame(store, shot, row, cache_box)
    return (
        replace(frame, profile=_NewtonBranchProfileAdapter(frame.profile)),
        mast_case,
        context,
    )


def _candidate_rows(frame: _MastFrame) -> list[int]:
    """Apply the imported declared ladder to the MAST frame adapter."""
    return _neighbour_candidates(frame)


def _classify_outcome(
    frame: _MastFrame,
    outcome: Any,
    residual_tolerance: float = RELATIVE_RESIDUAL_TOLERANCE,
) -> dict[str, Any]:
    """Qualify a constrained terminal without consulting its EFIT metrics."""
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
        and outcome.residual <= residual_tolerance
        and np.isfinite(current_error)
        and current_error <= CURRENT_RELATIVE_ERROR_TOLERANCE
    )
    if converged and nonzero_current:
        outcome_class = "converged_plasma_root"
    elif not nonzero_current:
        outcome_class = "vacuum_collapse"
    else:
        outcome_class = "bounded_non_convergence"
    return {
        "outcome_class": outcome_class,
        "converged": converged,
        "achieved_class": "diverted" if topology_consistent else "limited",
        "topology_consistent": topology_consistent,
        "nonzero_current": nonzero_current,
        "target_current_relative_error": current_error,
    }


def _record_outcome(
    frame: _MastFrame,
    context: dict[str, Any],
    outcome: Any,
    residual_tolerance: float = RELATIVE_RESIDUAL_TOLERANCE,
) -> tuple[dict[str, Any], Any]:
    """Classify and score one imported-seam terminal without label selection."""
    state = np.asarray(outcome.state, dtype=np.float64)
    target_current = frame.selected.recorded_plasma_current_a
    classification = _classify_outcome(frame, outcome, residual_tolerance)
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
                "converged": classification["converged"],
                "residual": float(outcome.residual),
                "achieved_class": classification["achieved_class"],
                "topology_consistent": classification["topology_consistent"],
            },
            "terminal_state": {
                "plasma_current_a": float(outcome.achieved_current_a),
                "nonzero_current": classification["nonzero_current"],
                "profile_amplitude": float(outcome.amplitude),
            },
            "registered_parity_metrics": metrics,
            "residual_trajectory": trajectory,
            "iterations": int(outcome.iterations),
            "termination": outcome.termination,
            "outcome_class": classification["outcome_class"],
            "target_current_relative_error": classification[
                "target_current_relative_error"
            ],
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


def _find_mast_newton_warm_source(
    store: Path,
    shot: int,
    target_frame: _MastFrame,
    cache_box: list[Any],
) -> tuple[list[dict[str, Any]], tuple[int, _MastFrame, Any] | None]:
    """Return the first own-converged Newton source on the imported ladder."""
    checks: list[dict[str, Any]] = []
    for candidate_row in _candidate_rows(target_frame):
        candidate, mast_case, _context = _prepare_newton_frame(
            store, shot, candidate_row, cache_box
        )
        outcome = _solve_public_seam(candidate, candidate.seed)
        classification = _classify_outcome(candidate, outcome)
        checks.append(
            {
                "row": candidate_row,
                "time_s": mast_case["reference"]["time_s"],
                "outcome_class": classification["outcome_class"],
                "converged": classification["converged"],
                "fixed_point_residual": float(outcome.residual),
                "achieved_class": classification["achieved_class"],
                "topology_consistent": classification["topology_consistent"],
                "terminal_plasma_current_a": float(outcome.achieved_current_a),
                "target_current_relative_error": classification[
                    "target_current_relative_error"
                ],
            }
        )
        if classification["outcome_class"] == "converged_plasma_root":
            return checks, (candidate_row, candidate, outcome)
    return checks, None


def _newton_arm_record(
    frame: _MastFrame,
    context: dict[str, Any],
    outcome: Any,
    residual_tolerance: float = RELATIVE_RESIDUAL_TOLERANCE,
) -> dict[str, Any]:
    """Serialize one Newton terminal in the registered MAST metric shape."""
    record, _equilibrium = _record_outcome(frame, context, outcome, residual_tolerance)
    residual = record["forward_branch_receipt"]["residual"]
    metrics = record["registered_parity_metrics"]
    return {
        "outcome_class": record["outcome_class"],
        "converged": record["outcome_class"] == "converged_plasma_root",
        "terminal_fixed_point_residual": residual,
        "terminal_plasma_current_a": record["terminal_state"]["plasma_current_a"],
        "target_current_relative_error": record["target_current_relative_error"],
        "registered_parity_metrics": metrics,
        "per_metric_qualification": _metric_qualification(metrics, residual),
        "iterations": record["iterations"],
        "termination": record["termination"],
        "residual_trajectory": record["residual_trajectory"],
    }


def measure_newton_reference(
    store: Path,
    shot: int,
    row: int,
    cache_box: list[Any],
    banked_control: dict[str, Any],
    scoreability: dict[str, Any],
) -> dict[str, Any]:
    """Replay one frozen reference through cold and eligible warm Newton arms."""
    frame, mast_case, context = _prepare_newton_frame(store, shot, row, cache_box)
    cold_outcome = _solve_public_seam(frame, frame.seed)
    cold = _newton_arm_record(frame, context, cold_outcome)
    cold["banked_terminal_fixed_point_residual"] = banked_control[
        "fixed_point_residual"
    ]
    cold["reproduces_banked_newton_control"] = bool(
        abs(
            cold["terminal_fixed_point_residual"]
            - banked_control["fixed_point_residual"]
        )
        <= CURRENT_MATCH_TOLERANCE
        * max(abs(banked_control["fixed_point_residual"]), 1.0)
    )

    warm: dict[str, Any] | None = None
    if cold["converged"]:
        search = {
            "triggered": False,
            "reason": "cold Newton entry already reaches a converged plasma root",
            "candidate_offsets": list(NEIGHBOUR_FRAME_OFFSETS),
            "checks": [],
            "qualified_source_found": False,
        }
    else:
        checks, source = _find_mast_newton_warm_source(store, shot, frame, cache_box)
        search = {
            "triggered": True,
            "candidate_offsets": list(NEIGHBOUR_FRAME_OFFSETS),
            "selection_rule": (
                "declared symmetric row-offset ladder, earlier before later; "
                "the first same-shot candidate whose own Newton solve converges "
                "wins, without consulting any EFIT score"
            ),
            "checks": checks,
            "qualified_source_found": source is not None,
        }
        if source is not None:
            source_row, _source_frame, source_outcome = source
            source_check = next(item for item in checks if item["row"] == source_row)
            search["selected_source"] = {
                "row": source_row,
                "time_s": source_check["time_s"],
                "time_offset_s": (
                    source_check["time_s"] - mast_case["reference"]["time_s"]
                ),
                "own_cold_fixed_point_residual": source_check["fixed_point_residual"],
            }
            warm_outcome = _solve_public_seam(
                frame, np.asarray(source_outcome.state, dtype=np.float64)
            )
            warm = _newton_arm_record(frame, context, warm_outcome)
            warm["lifted_to_converged_plasma_root"] = bool(warm["converged"])

    terminal = warm if warm is not None else cold
    return {
        "reference": {
            "shot": shot,
            "slice_index": row,
            "time_s": mast_case["reference"]["time_s"],
            "target_current_a": frame.selected.recorded_plasma_current_a,
            "target_current_source": "abs(efm/plasma_current_c) on the selected row",
        },
        "preregistered_scoreability": scoreability,
        "cold_newton_control": cold,
        "warm_neighbour_search": search,
        "warm_newton_solve": warm,
        "reported_terminal_arm": "warm_newton" if warm is not None else "cold_newton",
        "reported_terminal": {
            "converged": terminal["converged"],
            "outcome_class": terminal["outcome_class"],
            "terminal_fixed_point_residual": terminal["terminal_fixed_point_residual"],
            "terminal_plasma_current_a": terminal["terminal_plasma_current_a"],
            "target_current_relative_error": terminal["target_current_relative_error"],
        },
    }


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


def render_newton_figure(references: list[dict[str, Any]], path: Path) -> None:
    """Plot the cold and eligible warm Newton trajectories by scoreability."""
    figure, axes = plt.subplots(
        1,
        len(references),
        figsize=(max(4.5, 3.2 * len(references)), 3.6),
        constrained_layout=True,
    )
    if len(references) == 1:
        axes = [axes]
    for axis, reference in zip(axes, references, strict=True):
        cold = reference["cold_newton_control"]["residual_trajectory"]
        warm_record = reference["warm_newton_solve"]
        warm = [] if warm_record is None else warm_record["residual_trajectory"]
        axis.semilogy(
            np.arange(len(cold)),
            np.maximum(cold, np.finfo(float).tiny),
            marker="o",
            ms=3,
            lw=1.0,
            label="cold Newton",
        )
        if warm:
            axis.semilogy(
                np.arange(len(warm)),
                np.maximum(warm, np.finfo(float).tiny),
                marker="s",
                ms=3,
                lw=1.0,
                label="warm Newton",
            )
        axis.axhline(FIXED_POINT_CRITERION, color="black", ls="--", lw=0.8)
        shot = reference["reference"]["shot"]
        row = reference["reference"]["slice_index"]
        score_status = reference["preregistered_scoreability"]["status"]
        status = "closed LCFS" if score_status == "scoreable" else "no closed LCFS"
        axis.set_title(f"{shot}/{row}\n{status}")
        axis.set_xlabel("Residual evaluation")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("Fixed-point residual")
    figure.suptitle("Same-shot warm-neighbour replay on the pinned Newton branch")
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


def _protected_dual_basin_digests(output: Path) -> dict[str, str]:
    """Digest the landed dual-basin artifacts outside this replay's namespace."""
    return {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and not path.name.startswith(NEWTON_REPLAY_ATTEMPT_PREFIX)
    }


def _preregistered_scoreability() -> tuple[dict[tuple[int, int], dict[str, Any]], str]:
    """Load and validate the declared two/four closed-LCFS split."""
    receipt = json.loads(TARED_TOPOLOGY_RECEIPT.read_text())
    by_reference = {
        (int(row["reference"]["shot"]), int(row["reference"]["slice_index"])): row[
            "instrument_controlled_rows"
        ]["lcfs_closed_branch"]
        for row in receipt["per_shot"]
    }
    scoreable = {
        key for key, value in by_reference.items() if value["status"] == "scoreable"
    }
    unscoreable = {
        key
        for key, value in by_reference.items()
        if value["status"] == "unscoreable_no_closed_axis_branch"
    }
    if scoreable != {(21983, 35), (21985, 51)}:
        raise RuntimeError("the preregistered closed-axis scoreable pair changed")
    if unscoreable != {
        (21978, 35),
        (21986, 46),
        (21989, 55),
        (22086, 43),
    }:
        raise RuntimeError("the preregistered no-closed-axis quartet changed")
    return by_reference, _sha256(TARED_TOPOLOGY_RECEIPT)


def _existing_newton_replay(output: Path, resume: bool) -> list[dict[str, Any]]:
    """Load prior replay attempts when a later foreground segment resumes it."""
    receipt_path = output / NEWTON_REPLAY_RECEIPT_NAME
    if not resume:
        return []
    if not receipt_path.is_file():
        raise FileNotFoundError("Newton replay resume requested without a receipt")
    return list(json.loads(receipt_path.read_text())["references"])


def _existing_measurement(
    output: Path,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load committed partial references when extending a measurement."""
    receipt_path = output / RECEIPT_NAME
    if not resume:
        return [], []
    if not receipt_path.is_file():
        raise FileNotFoundError("resume requested without an existing receipt")
    receipt_digest = _sha256(receipt_path)
    prior = json.loads(receipt_path.read_text())
    segments = list(prior.get("run_segments", []))
    if not segments:
        segments.append(
            {
                "measured_references": prior["aggregate"]["measured_references"],
                "source_receipt_sha256": receipt_digest,
            }
        )
    return list(prior["references"]), segments


def run(
    store: Path = SHOT_STORE,
    output: Path = DEFAULT_OUTPUT,
    shots: tuple[int, ...] | None = None,
    resume: bool = False,
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
    existing_references, run_segments = _existing_measurement(output, resume)
    banked_receipt = json.loads(BANKED_CONSTRAINED_RECEIPT.read_text())
    banked_by_shot = {int(row["shot"]): row for row in banked_receipt["per_shot_table"]}
    bounded_keys = {
        (int(row["shot"]), int(row["slice_index"]))
        for row in banked_receipt["per_shot_table"]
        if row["constrained_outcome"] != "converged_plasma_root"
    }
    requested_shots = None if shots is None else set(shots)
    existing_keys = {
        (
            int(reference["reference"]["shot"]),
            int(reference["reference"]["slice_index"]),
        )
        for reference in existing_references
    }
    if not existing_keys <= bounded_keys:
        raise RuntimeError("the existing receipt contains an unexpected reference")
    if requested_shots is not None:
        unknown = requested_shots - {shot for shot, _row in bounded_keys}
        if unknown:
            raise ValueError(
                f"requested shots are not bounded frozen-six references: {sorted(unknown)}"
            )
        overlap = requested_shots & {shot for shot, _row in existing_keys}
        if overlap:
            raise ValueError(
                f"requested shots already exist in the resumed receipt: {sorted(overlap)}"
            )
    selected = select_slices_by_shot(DECOMPOSITION_BANK)
    cache_box: list[Any] = [None]
    references = list(existing_references)
    newly_measured = []
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
        measured = measure_reference(store, shot, row, cache_box, banked_control)
        references.append(measured)
        newly_measured.append(measured)
    expected_reference_count = (
        len(bounded_keys) - len(existing_keys)
        if requested_shots is None
        else len(requested_shots)
    )
    if len(newly_measured) != expected_reference_count:
        raise RuntimeError("the requested bounded references were not all measured")
    reference_order = {
        int(row["shot"]): position
        for position, row in enumerate(banked_receipt["per_shot_table"])
    }
    references.sort(
        key=lambda reference: reference_order[reference["reference"]["shot"]]
    )
    run_segments.append(
        {
            "measured_references": [
                {
                    "shot": reference["reference"]["shot"],
                    "slice_index": reference["reference"]["slice_index"],
                }
                for reference in newly_measured
            ]
        }
    )
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
        "run_segments": run_segments,
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


def _newton_solver_contract() -> dict[str, Any]:
    return {
        "public_entry_point": "ForwardProfile.solve_branch(target_current=...)",
        "route": "newton_krylov",
        "requested_class": "diverted",
        "registered_fixed_point_criterion": FIXED_POINT_CRITERION,
        "newton_promotions": NEWTON_STEPS,
        "gmres_iterations_per_promotion": GMRES_ITERATIONS,
        "warmup_sweeps": WARMUP_SWEEPS,
        "relaxation": RELAXATION,
        "step_cap": STEP_CAP,
        "target_current_policy": "abs(efm/plasma_current_c) on the row being solved",
        "prescribed_current_policy": (
            "all 101 fitted EFIT circuits through the passive-inclusive "
            "prescribed-current response policy"
        ),
        "shared_call_path": (
            "imported benchmarks.diiid_constrained_cold_start._solve_public_seam "
            "calling a duck-typed profile adapter whose solve delegates to the "
            "public pinned Newton branch"
        ),
    }


def _newton_replay_split(
    references: list[dict[str, Any]],
    expected_keys: set[tuple[int, int]],
) -> dict[str, Any]:
    """Summarize convergence only inside the declared topology strata."""
    measured_by_key = {
        (
            int(reference["reference"]["shot"]),
            int(reference["reference"]["slice_index"]),
        ): reference
        for reference in references
    }
    strata = {
        "closed_axis_branch": {
            "status": "scoreable",
            "expected_references": [(21983, 35), (21985, 51)],
        },
        "no_closed_axis_branch": {
            "status": "unscoreable_no_closed_axis_branch",
            "expected_references": [
                (21978, 35),
                (21986, 46),
                (21989, 55),
                (22086, 43),
            ],
        },
    }
    result: dict[str, Any] = {}
    for name, stratum in strata.items():
        keys = set(stratum["expected_references"])
        measured = [measured_by_key[key] for key in keys if key in measured_by_key]
        result[name] = {
            "score_status": stratum["status"],
            "expected_reference_count": len(keys),
            "measured_reference_count": len(measured),
            "unmeasured_references": [
                {"shot": shot, "slice_index": row}
                for shot, row in sorted(keys - measured_by_key.keys())
            ],
            "terminal_converged_count": sum(
                reference["reported_terminal"]["converged"] for reference in measured
            ),
            "cold_converged_count": sum(
                reference["cold_newton_control"]["converged"] for reference in measured
            ),
            "warm_attempted_count": sum(
                reference["warm_newton_solve"] is not None for reference in measured
            ),
            "warm_source_found_count": sum(
                reference["warm_neighbour_search"]["qualified_source_found"]
                for reference in measured
            ),
            "warm_lifted_count": sum(
                reference["warm_newton_solve"] is not None
                and reference["warm_newton_solve"]["converged"]
                for reference in measured
            ),
            "per_reference": [
                {
                    "shot": reference["reference"]["shot"],
                    "slice_index": reference["reference"]["slice_index"],
                    "terminal_arm": reference["reported_terminal_arm"],
                    "converged": reference["reported_terminal"]["converged"],
                    "terminal_fixed_point_residual": reference["reported_terminal"][
                        "terminal_fixed_point_residual"
                    ],
                    "terminal_plasma_current_a": reference["reported_terminal"][
                        "terminal_plasma_current_a"
                    ],
                }
                for reference in sorted(
                    measured,
                    key=lambda item: (
                        item["reference"]["shot"],
                        item["reference"]["slice_index"],
                    ),
                )
            ],
            "verdict": (
                "PARTIAL"
                if len(measured) != len(keys)
                else (
                    "WARM_LIFTS_AT_LEAST_ONE"
                    if any(
                        reference["warm_newton_solve"] is not None
                        and reference["warm_newton_solve"]["converged"]
                        for reference in measured
                    )
                    else "WARM_LIFTS_NONE"
                )
            ),
        }
    if set(measured_by_key) - expected_keys:
        raise RuntimeError("the replay contains a reference outside the frozen six")
    return result


def _write_newton_attempt(
    output: Path,
    reference: dict[str, Any],
    topology_receipt_digest: str,
) -> Path:
    shot = reference["reference"]["shot"]
    row = reference["reference"]["slice_index"]
    path = output / f"{NEWTON_REPLAY_ATTEMPT_PREFIX}-{shot}-{row}.json"
    attempt = {
        "receipt": "MAST same-shot warm-neighbour Newton replay attempt",
        "reference": reference,
        "solver": _newton_solver_contract(),
        "offset_ladder": {
            "offsets": list(NEIGHBOUR_FRAME_OFFSETS),
            "repr": repr(NEIGHBOUR_FRAME_OFFSETS),
            "source": (
                "benchmarks.diiid_constrained_cold_start.NEIGHBOUR_FRAME_OFFSETS"
            ),
            "selection_rule": (
                "earlier offsets first; the first candidate whose own constrained "
                "Newton solve converges supplies the seed; no EFIT score enters "
                "source selection"
            ),
        },
        "preregistered_scoreability_source": {
            "path": str(TARED_TOPOLOGY_RECEIPT),
            "sha256": topology_receipt_digest,
        },
    }
    path.write_text(
        json.dumps(attempt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return path


def run_newton_replay(
    store: Path = SHOT_STORE,
    output: Path = NEWTON_REPLAY_OUTPUT,
    shots: tuple[int, ...] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Replay the same-shot ladder through the public constrained Newton branch."""
    configure_dtypes()
    if NEIGHBOUR_FRAME_OFFSETS != DECLARED_NEIGHBOUR_FRAME_OFFSETS:
        raise RuntimeError("the imported warm-neighbour offset ladder changed")
    if RELATIVE_RESIDUAL_TOLERANCE != FIXED_POINT_CRITERION:
        raise RuntimeError("the imported and banked convergence criteria differ")
    output.mkdir(parents=True, exist_ok=True)
    protected_before = _protected_dual_basin_digests(output)
    existing = _existing_newton_replay(output, resume)
    banked_control_receipt = json.loads(BANKED_CONSTRAINED_RECEIPT.read_text())
    banked_by_key = {
        (int(row["shot"]), int(row["slice_index"])): row
        for row in banked_control_receipt["per_shot_table"]
    }
    scoreability, topology_receipt_digest = _preregistered_scoreability()
    expected_keys = set(banked_by_key)
    if expected_keys != set(scoreability):
        raise RuntimeError("the frozen-six controls and scoreability split disagree")
    requested_shots = (
        {shot for shot, _row in expected_keys} if shots is None else set(shots)
    )
    unknown = requested_shots - {shot for shot, _row in expected_keys}
    if unknown:
        raise ValueError(
            f"requested shots are not frozen-six references: {sorted(unknown)}"
        )
    existing_keys = {
        (
            int(reference["reference"]["shot"]),
            int(reference["reference"]["slice_index"]),
        )
        for reference in existing
    }
    overlap = requested_shots & {shot for shot, _row in existing_keys}
    if overlap:
        raise ValueError(
            f"requested shots already exist in the replay: {sorted(overlap)}"
        )

    cache_box: list[Any] = [None]
    references = list(existing)
    newly_measured: list[dict[str, Any]] = []
    for selected_row, _qualification in select_slices_by_shot(DECOMPOSITION_BANK):
        shot = int(selected_row["shot"])
        row = int(selected_row["slice_index"])
        if shot not in requested_shots:
            continue
        key = (shot, row)
        if key not in expected_keys:
            raise RuntimeError("the selected row differs from the frozen-six control")
        reference = measure_newton_reference(
            store,
            shot,
            row,
            cache_box,
            banked_by_key[key],
            scoreability[key],
        )
        references.append(reference)
        newly_measured.append(reference)
        _write_newton_attempt(output, reference, topology_receipt_digest)
    if len(newly_measured) != len(requested_shots):
        raise RuntimeError("the requested frozen-six references were not all measured")
    references.sort(
        key=lambda item: (item["reference"]["shot"], item["reference"]["slice_index"])
    )
    figure_path = output / NEWTON_REPLAY_FIGURE_NAME
    render_newton_figure(references, figure_path)
    protected_after = _protected_dual_basin_digests(output)
    if protected_after != protected_before:
        raise RuntimeError("a landed dual-basin artifact changed during the replay")

    measured_keys = {
        (reference["reference"]["shot"], reference["reference"]["slice_index"])
        for reference in references
    }
    terminal_currents_exact = all(
        reference["reported_terminal"]["target_current_relative_error"]
        <= TARGET_CURRENT_EXACT_TOLERANCE
        for reference in references
    )
    receipt = {
        "receipt": "MAST same-shot warm-neighbour pinned-Newton replay",
        "measurement_status": (
            "complete" if measured_keys == expected_keys else "partial"
        ),
        "measurement_rule": (
            "report the preregistered two-reference closed-axis stratum and "
            "four-reference no-closed-axis stratum separately; no pooled "
            "convergence rate is defined"
        ),
        "solver": _newton_solver_contract(),
        "reuse": {
            "imported_unchanged": [
                "benchmarks.diiid_constrained_cold_start.NEIGHBOUR_FRAME_OFFSETS",
                "benchmarks.diiid_constrained_cold_start._neighbour_candidates",
                "benchmarks.diiid_constrained_cold_start._solve_public_seam",
            ],
            "duck_typed_adapter": (
                "_MastFrame plus _NewtonBranchProfileAdapter; helper-visible "
                "selected, row, profile, current, and seed surfaces are preserved"
            ),
            "not_reused": (
                "benchmarks.diiid_constrained_cold_start._find_warm_source; "
                "the MAST walk is local and retains earlier-first ordering"
            ),
        },
        "offset_ladder": {
            "offsets": list(NEIGHBOUR_FRAME_OFFSETS),
            "repr": repr(NEIGHBOUR_FRAME_OFFSETS),
        },
        "baseline_inputs": {
            "banked_warm_arm": {
                "path": str(BANKED_WARM_RECEIPT),
                "sha256": _sha256(BANKED_WARM_RECEIPT),
            },
            "banked_newton_control": {
                "path": str(BANKED_CONSTRAINED_RECEIPT),
                "sha256": _sha256(BANKED_CONSTRAINED_RECEIPT),
            },
            "preregistered_scoreability": {
                "path": str(TARED_TOPOLOGY_RECEIPT),
                "sha256": topology_receipt_digest,
            },
        },
        "references": references,
        "split_results": _newton_replay_split(references, expected_keys),
        "integrity": {
            "all_reported_terminal_currents_exact_at_target": terminal_currents_exact,
            "all_cold_newton_controls_reproduce_banked": all(
                reference["cold_newton_control"]["reproduces_banked_newton_control"]
                for reference in references
            ),
            "protected_dual_basin_artifact_count": len(protected_after),
            "protected_dual_basin_artifacts_unchanged": True,
            "protected_sha256_by_name": protected_after,
        },
        "unmeasured_references": [
            {"shot": shot, "slice_index": row}
            for shot, row in sorted(expected_keys - measured_keys)
        ],
        "attempt_receipts": [
            str(output / f"{NEWTON_REPLAY_ATTEMPT_PREFIX}-{shot}-{row}.json")
            for shot, row in sorted(measured_keys)
        ],
        "figure": str(figure_path),
    }
    (output / NEWTON_REPLAY_RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def _balanced_reference() -> dict[str, Any]:
    """Load the independently banked near-fixed-point reference identity."""

    receipt = json.loads(BALANCED_REFERENCE_RECEIPT.read_text())
    reference = receipt["reference"]
    residual = receipt["constrained_map_at_reference_flux"]["residual"]
    defect = residual["all_physical_nodes"]["rms_fraction_of_span"]
    if not np.isclose(defect, 0.001350589126280159, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("the banked balanced fixed-point proximity changed")
    return {
        "shot": int(reference["shot"]),
        "slice_index": int(reference["slice_index"]),
        "reference_map_defect_fraction_of_span": float(defect),
        "reference_map_defect_percent_of_span": 100.0 * float(defect),
        "interior_rms_fraction_of_span": residual["interior_grid_inside_stored_lcfs"][
            "rms_fraction_of_span"
        ],
        "exterior_rms_fraction_of_span": residual["exterior_grid_and_limiter_wall"][
            "rms_fraction_of_span"
        ],
        "source": str(BALANCED_REFERENCE_RECEIPT),
        "source_sha256": _sha256(BALANCED_REFERENCE_RECEIPT),
    }


def _measure_moment_seed_reference(
    store: Path,
    shot: int,
    row: int,
    cache_box: list[Any],
    scoreability: dict[str, Any],
    balanced_reference: dict[str, Any],
) -> dict[str, Any]:
    """Run one frozen MAST row from its flux-functions-only moment seed."""

    frame, mast_case, context = _prepare_frame(store, shot, row, cache_box)
    seed = frame.profile.moment_seed(
        mast_case["boundary"],
        frame.selected.recorded_plasma_current_a,
        current=frame.current,
    )
    newton_frame = replace(
        frame,
        profile=_NewtonBranchProfileAdapter(frame.profile),
        seed=np.asarray(seed.flux, dtype=np.float64),
    )
    outcome = _solve_public_seam(
        newton_frame,
        newton_frame.seed,
        relative_tolerance=MOMENT_SEEDED_CRITERION,
    )
    terminal = _newton_arm_record(
        newton_frame,
        context,
        outcome,
        MOMENT_SEEDED_CRITERION,
    )
    is_balanced_reference = (shot, row) == (
        balanced_reference["shot"],
        balanced_reference["slice_index"],
    )
    return {
        "machine": "MAST",
        "reference": {
            "shot": shot,
            "slice_index": row,
            "time_s": mast_case["reference"]["time_s"],
            "target_current_a": frame.selected.recorded_plasma_current_a,
        },
        "selection": (
            "member of the frozen-six receipt; no solve or EFIT score changes "
            "cohort membership"
        ),
        "preregistered_scoreability": scoreability,
        "seed": {
            "constructor": "ForwardProfile.moment_seed",
            "boundary_hypothesis": "target reference own LCFS",
            "predicted_current_a": seed.moments.plasma_current,
            "predicted_centroid_r_m": seed.moments.centroid_r,
            "predicted_centroid_z_m": seed.moments.centroid_z,
            "prediction_current_support": seed.moments.current_support.value,
            "prediction_centroid_support": seed.moments.centroid_support.value,
            "representation_support": seed.support.value,
            "representation_supported_cells": seed.supported_cells,
            "representation_radius_m": seed.radius,
        },
        "terminal": terminal,
        "balanced_fixed_point_reference": {
            "applies_to_this_row": is_balanced_reference,
            "source": balanced_reference["source"],
            "reference_map_defect_percent_of_span": balanced_reference[
                "reference_map_defect_percent_of_span"
            ],
            "interior_rms_fraction_of_span": balanced_reference[
                "interior_rms_fraction_of_span"
            ],
            "exterior_rms_fraction_of_span": balanced_reference[
                "exterior_rms_fraction_of_span"
            ],
            "moment_seed_converges": terminal["converged"]
            if is_balanced_reference
            else None,
            "discretisation_escape_hatch_condition": (
                "NOT_APPLICABLE"
                if not is_balanced_reference
                else ("NOT_MET" if terminal["converged"] else "MET")
            ),
        },
    }


def _moment_seed_split(references: list[dict[str, Any]]) -> dict[str, Any]:
    """Report the registered MAST strata separately, never as a pooled rate."""

    strata = {
        "closed_axis_branch": "scoreable",
        "no_closed_axis_branch": "unscoreable_no_closed_axis_branch",
    }
    result = {}
    for name, status in strata.items():
        rows = [
            reference
            for reference in references
            if reference["preregistered_scoreability"]["status"] == status
        ]
        result[name] = {
            "status": status,
            "converged": sum(row["terminal"]["converged"] for row in rows),
            "attempted": len(rows),
            "baseline": (
                {"cold": "0/2", "host_warm": "0/2", "newton_warm": "0/2"}
                if name == "closed_axis_branch"
                else {
                    "cold": "1/4",
                    "host_warm": "0/3 among cold stalls",
                    "newton_warm": "0/3 among cold stalls",
                }
            ),
            "rows": [
                {
                    "shot": row["reference"]["shot"],
                    "slice_index": row["reference"]["slice_index"],
                    "converged": row["terminal"]["converged"],
                    "terminal_residual": row["terminal"][
                        "terminal_fixed_point_residual"
                    ],
                }
                for row in rows
            ],
        }
    if (
        len(result["closed_axis_branch"]["rows"]) != 2
        or len(result["no_closed_axis_branch"]["rows"]) != 4
    ):
        raise RuntimeError("the preregistered MAST two/four split changed")
    return result


def _render_moment_seed_figure(
    mast: list[dict[str, Any]],
    diiid: list[dict[str, Any]],
    path: Path,
) -> None:
    """Plot per-attempt residuals and convergence counts against baselines."""

    figure, (residual_axis, count_axis) = plt.subplots(
        2, 1, figsize=(10.5, 7.2), constrained_layout=True
    )
    labels = [
        f"M {row['reference']['shot']}\n{row['reference']['slice_index']}"
        for row in mast
    ] + [f"D {Path(row['shot']).stem}\n{row['frame']}" for row in diiid]
    residuals = [row["terminal"]["terminal_fixed_point_residual"] for row in mast] + [
        row["route"]["fixed_point_relative_residual"] for row in diiid
    ]
    converged = [row["terminal"]["converged"] for row in mast] + [
        row["route"]["converged"] for row in diiid
    ]
    colours = ["#2a9d8f" if passed else "#e76f51" for passed in converged]
    x = np.arange(len(labels))
    residual_axis.bar(x, np.maximum(residuals, 1.0e-16), color=colours)
    residual_axis.axhline(
        MOMENT_SEEDED_CRITERION,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="registered 1e-8 criterion",
    )
    residual_axis.set_yscale("log")
    residual_axis.set_ylabel("terminal relative residual")
    residual_axis.set_xticks(x, labels, fontsize=7)
    residual_axis.legend(fontsize=8)

    names = ["MAST closed", "MAST construct", "DIII-D"]
    moment_counts = [
        sum(
            row["terminal"]["converged"]
            for row in mast
            if row["preregistered_scoreability"]["status"] == "scoreable"
        ),
        sum(
            row["terminal"]["converged"]
            for row in mast
            if row["preregistered_scoreability"]["status"]
            == "unscoreable_no_closed_axis_branch"
        ),
        sum(row["route"]["converged"] for row in diiid),
    ]
    totals = [2, 4, 5]
    baseline_cold = [0, 1, 2]
    baseline_warm = [0, 1, 4]
    positions = np.arange(3)
    width = 0.24
    count_axis.bar(positions - width, baseline_cold, width, label="cold baseline")
    count_axis.bar(positions, baseline_warm, width, label="best warm baseline")
    count_axis.bar(positions + width, moment_counts, width, label="moment seed")
    count_axis.set_xticks(
        positions, [f"{name}\n(n={total})" for name, total in zip(names, totals)]
    )
    count_axis.set_ylabel("converged frames")
    count_axis.set_ylim(0, 5.5)
    count_axis.legend(fontsize=8)
    figure.suptitle("Moment-seeded constrained cold starts on frozen cohorts")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_moment_seed_driver(
    store: Path = SHOT_STORE,
    diiid_data: Path = DIIID_DATA,
    output: Path = MOMENT_SEEDED_OUTPUT,
) -> dict[str, Any]:
    """Measure the public moment seed on both frozen machine cohorts."""

    configure_dtypes()
    output.mkdir(parents=True, exist_ok=True)
    source = _source_stamp()
    scoreability, topology_digest = _preregistered_scoreability()
    balanced = _balanced_reference()
    selected = [
        (int(row["shot"]), int(row["slice_index"]))
        for row, _qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    ]
    if set(selected) != set(scoreability) or len(selected) != 6:
        raise RuntimeError("the selected MAST cohort differs from the frozen six")

    cache_box: list[Any] = [None]
    mast = []
    attempt_paths = []
    for shot, row in selected:
        reference = _measure_moment_seed_reference(
            store,
            shot,
            row,
            cache_box,
            scoreability[(shot, row)],
            balanced,
        )
        attempt_path = output / f"{MOMENT_SEEDED_ATTEMPT_PREFIX}-{shot}-{row}.json"
        attempt = {
            "receipt": "MAST moment-seeded constrained cold-start attempt",
            "source": source,
            "solver": {
                "entry_point": "ForwardProfile.solve_branch",
                "route": "newton_krylov",
                "relative_residual_criterion": MOMENT_SEEDED_CRITERION,
                "newton_promotions": NEWTON_STEPS,
                "gmres_iterations_per_promotion": GMRES_ITERATIONS,
                "warmup_sweeps": WARMUP_SWEEPS,
                "relaxation": RELAXATION,
                "step_cap": STEP_CAP,
            },
            "attempt": reference,
        }
        attempt_path.write_text(
            json.dumps(attempt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        reference["receipt_path"] = str(attempt_path)
        attempt_paths.append(str(attempt_path))
        mast.append(reference)

    diiid = run_diiid_moment_seed(diiid_data, output)
    attempt_paths.extend(row["receipt_path"] for row in diiid["attempts"])
    split = _moment_seed_split(mast)
    balanced_row = next(
        row
        for row in mast
        if row["balanced_fixed_point_reference"]["applies_to_this_row"]
    )
    escape_hatch = balanced_row["balanced_fixed_point_reference"][
        "discretisation_escape_hatch_condition"
    ]
    figure_path = output / MOMENT_SEEDED_FIGURE_NAME
    _render_moment_seed_figure(mast, diiid["attempts"], figure_path)
    receipt = {
        "receipt": "moment-seeded constrained cold-start aggregate",
        "source": source,
        "measurement_rule": (
            "fixed frozen cohorts; no selection keys on a known-good frame identity; "
            "MAST is reported only as the preregistered two/four split"
        ),
        "solver": {
            "seed_constructor": "ForwardProfile.moment_seed",
            "constrained_entry_points": [
                "ForwardProfile.solve_branch",
                "ForwardProfile.solve",
            ],
            "registered_relative_residual_criterion": MOMENT_SEEDED_CRITERION,
            "current_relative_error_tolerance": CURRENT_RELATIVE_ERROR_TOLERANCE,
        },
        "mast": {
            "split_results": split,
            "baseline_statement": (
                "cold 1/6 and both warm arms 0/5 are contextual controls only; "
                "the moment-seed result is never pooled and is stated as 2/4"
            ),
            "references": mast,
        },
        "diiid": diiid,
        "balanced_fixed_point_reference": {
            **balanced,
            "moment_seed_converges": balanced_row["terminal"]["converged"],
            "terminal_residual": balanced_row["terminal"][
                "terminal_fixed_point_residual"
            ],
            "discretisation_escape_hatch_condition": escape_hatch,
            "orchestrator_fact": (
                "discretisation escape hatch condition MET"
                if escape_hatch == "MET"
                else "discretisation escape hatch condition NOT MET"
            ),
        },
        "inputs": {
            "scoreability_receipt": str(TARED_TOPOLOGY_RECEIPT),
            "scoreability_receipt_sha256": topology_digest,
            "balanced_reference_receipt": str(BALANCED_REFERENCE_RECEIPT),
            "balanced_reference_receipt_sha256": balanced["source_sha256"],
        },
        "artifacts": {
            "attempt_receipts": attempt_paths,
            "aggregate": str(output / MOMENT_SEEDED_AGGREGATE_NAME),
            "figure": str(figure_path),
        },
    }
    (output / MOMENT_SEEDED_AGGREGATE_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shots", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--newton-replay", action="store_true")
    parser.add_argument("--newton-output", type=Path, default=NEWTON_REPLAY_OUTPUT)
    parser.add_argument("--moment-seeded-driver", action="store_true")
    parser.add_argument("--diiid-data", type=Path, default=DIIID_DATA)
    parser.add_argument("--moment-output", type=Path, default=MOMENT_SEEDED_OUTPUT)
    arguments = parser.parse_args()
    selected_shots = None if arguments.shots is None else tuple(arguments.shots)
    if arguments.moment_seeded_driver:
        receipt = run_moment_seed_driver(
            arguments.store,
            arguments.diiid_data,
            arguments.moment_output,
        )
        summary = {
            "mast": receipt["mast"]["split_results"],
            "diiid": receipt["diiid"]["moment_seeded"],
            "balanced_fixed_point_reference": receipt["balanced_fixed_point_reference"],
        }
    elif arguments.newton_replay:
        receipt = run_newton_replay(
            arguments.store,
            arguments.newton_output,
            shots=selected_shots,
            resume=arguments.resume,
        )
        summary = {
            "measurement_status": receipt["measurement_status"],
            "split_results": receipt["split_results"],
            "unmeasured_references": receipt["unmeasured_references"],
        }
    else:
        receipt = run(
            arguments.store,
            arguments.output,
            shots=selected_shots,
            resume=arguments.resume,
        )
        summary = receipt["aggregate"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
