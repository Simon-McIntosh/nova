"""Measure topology-pinned portfolio economy from neighbouring flux states.

The benchmark reconstructs the banked, well-separated diverted case, finds the
coexisting limited fixed point with the production map, and times both roots
together through :meth:`ForwardProfile.solve_portfolio`.  Seed distance,
nonlinear budget, and batch width are declared as module constants so the
catalog-regime projection cannot be selected after observing the timings.

The JSON receipt is the primary artifact.  It retains terminal branch receipts,
fixed input/output shapes, independent phase probes, the earlier cold-solve
controls, and the arithmetic used to compare the catalog rung with one
millisecond.  Compilation and fixture construction are excluded from warm wall
times and reported separately.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium import ForwardProfile, SaddleSeedGeometry, fixed_point
from nova.equilibrium.convention import toroidal_current_density
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures.measure import (
    FIXTURE_REQUESTS,
    WALL_POINT_COUNT,
    analytic_case,
    cached_machine,
    forward_operator,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/figures/dual-basin-solve"
FIXTURE_BANK = ROOT / "scripts/dual_basin_fixtures"

# Declared independently of the measurement.  The 1e-3 rung is the catalog
# projection: neighbouring reconstructed slices differ by one part per
# thousand of the branch's axis-to-boundary flux span.
SEED_DISTANCES = (0.0, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2)
BATCH_SIZES = (1, 4, 16)
NEWTON_BUDGETS = (1, 2, 10)
CATALOG_SEED_DISTANCE = 1.0e-3
CATALOG_BATCH_SIZE = 16
GMRES_ITERATIONS = 30
CONVERGENCE_TOLERANCE = 1.0e-10
ROOT_PARITY_TOLERANCE = 1.0e-10
TIMING_REPEATS = 5
TIMING_WARMUPS = 2
LIMITED_ROOT_STEPS = 20
TARGET_SECONDS = 1.0e-3
DIVERTED_STATE_DIGEST = (
    "11a7e9d00556e91a6d76a69212107592501e1e8cedae60fd17e9e8032ff14801"
)

# These are controls, not values this benchmark attempts to refit.  They are
# the cold ten-by-thirty coarse solve measured on the same H200 class.
BANKED_COLD_CONTROLS_MS = {
    1: 198.5446671023965,
    4: 56.311405496671796,
    16: 30.24408535566181,
}


def _strict(value: Any) -> Any:
    """Return a JSON-safe tree without non-finite numeric spellings."""
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable strict JSON receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _digest(values: np.ndarray) -> str:
    """Return the binary64 array digest used by the fixture bank."""
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _flat(value: float) -> Callable[[jax.Array], jax.Array]:
    """Return one constant flux function."""

    def profile(psi_norm: jax.Array) -> jax.Array:
        return jnp.full_like(jnp.asarray(psi_norm), value)

    return profile


def _problem() -> tuple[ForwardProfile, Any, np.ndarray]:
    """Reconstruct the diverted fixture and geometry-derived cold portfolio."""
    case = analytic_case()
    machine = cached_machine(
        case,
        FIXTURE_REQUESTS["fine"],
        wall_nodes=WALL_POINT_COUNT,
    )
    fixture = json.loads(
        (FIXTURE_BANK / "diverted-receipt.json").read_text(encoding="utf-8")
    )
    bank = np.load(FIXTURE_BANK / "diverted-state.npz")
    diverted_root = np.asarray(bank["state"], dtype=np.float64)
    if _digest(diverted_root) != DIVERTED_STATE_DIGEST:
        raise RuntimeError(
            "the diverted fixture state does not match its banked digest"
        )

    gradients = fixture["closed_form"]["constant_flux_functions"]
    source = ForwardSource(
        core=DomainProfile(
            p_prime=_flat(gradients["p_prime_pa_per_wb"]),
            ff_prime=_flat(gradients["ff_prime_t2_m2_per_wb"]),
        ),
        boundary_pressure=0.0,
        boundary_field_function=5.0,
    )
    empty = replace(forward_operator(case, machine), source=source)
    exterior = diverted_root - np.asarray(
        empty.internal(diverted_root, TopologyClass.DIVERTED)
    )
    operator = replace(forward_operator(case, machine, exterior), source=source)
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=10,
    )

    axis = np.asarray(fixture["analytic_stationary_points"]["axis"]["coordinate_m"])
    saddle = np.asarray(
        fixture["analytic_stationary_points"]["x_point"]["coordinate_m"]
    )
    geometry = SaddleSeedGeometry(tuple(axis), tuple(saddle))
    seed_radius = 0.9 * np.linalg.norm(saddle - axis)
    supported = np.linalg.norm(machine.node - axis, axis=1) < seed_radius
    cell_current = (
        toroidal_current_density(
            machine.node[:, 0],
            gradients["p_prime_pa_per_wb"],
            gradients["ff_prime_t2_m2_per_wb"],
        )
        * machine.area
        * supported
    )
    total_current = float(cell_current.sum())
    centroid = np.sum(machine.node * cell_current[:, None], axis=0) / total_current
    cold = profile.cold_seed_portfolio(
        total_current,
        centroid,
        diverted_geometry=geometry,
    )
    return profile, cold, diverted_root


def _time_call(compiled: Callable[..., Any], *arguments: Any) -> dict[str, Any]:
    """Return synchronized warm timings for one compiled device call."""
    for _ in range(TIMING_WARMUPS):
        jax.block_until_ready(compiled(*arguments))
    samples = []
    for _ in range(TIMING_REPEATS):
        started = time.perf_counter()
        jax.block_until_ready(compiled(*arguments))
        samples.append(time.perf_counter() - started)
    return {
        "samples_seconds": samples,
        "minimum_seconds": float(np.min(samples)),
        "median_seconds": float(np.median(samples)),
        "maximum_seconds": float(np.max(samples)),
    }


def _limited_root(
    profile: ForwardProfile, cold_flux: jax.Array
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find and qualify the limited root coexisting with the diverted fixture."""
    requested = jnp.asarray(int(TopologyClass.LIMITED), dtype=jnp.int8)
    map_fn = profile.operator.flux_map(requested_class=requested)
    solve = jax.jit(
        lambda seed: fixed_point.newton_krylov(
            map_fn,
            seed,
            newton_steps=LIMITED_ROOT_STEPS,
            gmres_iterations=GMRES_ITERATIONS,
            warmup=0,
        )
    )
    compiled = solve.lower(cold_flux).compile()
    started = time.perf_counter()
    result = compiled(cold_flux)
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - started
    state = np.asarray(result.state)
    _, topology = profile.operator.read(result.state)
    achieved = int(np.asarray(topology.diverted))
    residual = float(np.asarray(result.residual))
    if achieved != int(TopologyClass.LIMITED) or residual > CONVERGENCE_TOLERANCE:
        raise RuntimeError(
            "limited root preparation failed: "
            f"achieved={achieved}, residual={residual:.6e}"
        )
    return state, {
        "newton_steps": LIMITED_ROOT_STEPS,
        "gmres_iterations": GMRES_ITERATIONS,
        "requested_class": int(TopologyClass.LIMITED),
        "achieved_class": achieved,
        "relative_residual": residual,
        "first_execution_seconds": elapsed,
        "state_sha256": _digest(state),
    }


def _seed_ladder(
    profile: ForwardProfile,
    cold_flux: np.ndarray,
    roots: np.ndarray,
) -> tuple[dict[float, jax.Array], dict[str, Any]]:
    """Return two-branch seeds at each declared relative span distance."""
    requested = (TopologyClass.LIMITED, TopologyClass.DIVERTED)
    spans = []
    directions = []
    for index, branch_class in enumerate(requested):
        _, topology = profile.operator.read(roots[index], branch_class)
        span = abs(float(np.asarray(topology.flux_span)))
        direction = np.asarray(cold_flux[index]) - roots[index]
        scale = float(np.max(np.abs(direction)))
        if span <= 0.0 or scale <= 0.0:
            raise RuntimeError("a branch seed direction or flux span is degenerate")
        spans.append(span)
        directions.append(direction / scale)
    seeds = {
        distance: jnp.asarray(
            roots + distance * np.asarray(spans)[:, None] * np.asarray(directions)
        )
        for distance in SEED_DISTANCES
    }
    return seeds, {
        "relative_to": "each root's pinned axis-to-boundary flux span",
        "direction": "geometry-derived cold branch seed minus its terminal root",
        "distances": SEED_DISTANCES,
        "branch_flux_spans_wb": spans,
        "catalog_distance": CATALOG_SEED_DISTANCE,
    }


def _terminal_rows(result: Any, roots: np.ndarray) -> list[dict[str, Any]]:
    """Extract one terminal qualification row per portfolio branch."""
    rows = []
    for index, name in enumerate(("limited", "diverted")):
        branch = jax.tree.map(lambda value: value[0, index], result.branches)
        state = np.asarray(branch.equilibrium.flux)
        scale = max(float(np.max(np.abs(roots[index]))), np.finfo(float).tiny)
        rows.append(
            {
                "branch": name,
                "requested_class": int(np.asarray(branch.requested_class)),
                "achieved_class": int(np.asarray(branch.achieved_class)),
                "topology_consistent": bool(np.asarray(branch.topology_consistent)),
                "converged": bool(np.asarray(branch.converged)),
                "relative_residual": float(np.asarray(branch.residual)),
                "iterations": int(np.asarray(branch.iterations)),
                "finite": bool(np.asarray(branch.equilibrium.finite.passed)),
                "root_relative_error": float(
                    np.max(np.abs(state - roots[index])) / scale
                ),
            }
        )
    return rows


def _solve_grid(
    profile: ForwardProfile,
    seeds: dict[float, jax.Array],
    roots: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure the declared solve matrix and retain fixed-shape evidence."""
    measurements = []
    shapes = []
    for batch_size in BATCH_SIZES:
        for newton_steps in NEWTON_BUDGETS:
            first_row = len(measurements)

            def solve(states: jax.Array) -> Any:
                return jax.vmap(
                    lambda portfolio: profile.solve_portfolio(
                        portfolio,
                        route="newton_krylov",
                        tolerance=CONVERGENCE_TOLERANCE,
                        newton_steps=newton_steps,
                        gmres_iterations=GMRES_ITERATIONS,
                        warmup=0,
                    )
                )(states)

            example = jnp.broadcast_to(
                seeds[SEED_DISTANCES[0]],
                (batch_size, *seeds[SEED_DISTANCES[0]].shape),
            )
            compile_started = time.perf_counter()
            lowered = jax.jit(solve).lower(example)
            compiled = lowered.compile()
            compile_seconds = time.perf_counter() - compile_started
            example_result = compiled(example)
            jax.block_until_ready(example_result)
            output_shapes = jax.tree.map(lambda value: value.shape, example_result)
            shapes.append(
                {
                    "batch_size": batch_size,
                    "newton_steps": newton_steps,
                    "input_shape": tuple(example.shape),
                    "flux_output_shape": tuple(output_shapes.branches.equilibrium.flux),
                    "requested_class_shape": tuple(
                        output_shapes.branches.requested_class
                    ),
                    "jit": True,
                    "outer_vmap": True,
                    "portfolio_branch_axis": 2,
                }
            )
            for distance in SEED_DISTANCES:
                argument = jnp.broadcast_to(
                    seeds[distance],
                    (batch_size, *seeds[distance].shape),
                )
                timing = _time_call(compiled, argument)
                result = compiled(argument)
                jax.block_until_ready(result)
                median = timing["median_seconds"]
                measurements.append(
                    {
                        "seed_distance": distance,
                        "batch_size": batch_size,
                        "newton_steps": newton_steps,
                        "gmres_iterations": GMRES_ITERATIONS,
                        "compile_seconds": compile_seconds,
                        "timing": timing,
                        "wall_ms_per_portfolio_state": 1.0e3 * median / batch_size,
                        "wall_ms_per_branch_state": 1.0e3 * median / (2 * batch_size),
                        "branches": _terminal_rows(result, roots),
                    }
                )
            phase = _phase_profile(
                profile,
                seeds[CATALOG_SEED_DISTANCE],
                newton_steps,
                batch_size,
                CATALOG_SEED_DISTANCE,
            )
            for item in measurements[first_row:]:
                item["phase_decomposition"] = phase
            del compiled, lowered
            jax.clear_caches()
    return measurements, shapes


def _minimum_converged(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the smallest declared budget that qualifies each branch."""
    rows = []
    for distance in SEED_DISTANCES:
        for batch_size in BATCH_SIZES:
            matching = [
                item
                for item in measurements
                if item["seed_distance"] == distance
                and item["batch_size"] == batch_size
            ]
            for branch_index, branch_name in enumerate(("limited", "diverted")):
                passing = [
                    item
                    for item in matching
                    if item["branches"][branch_index]["converged"]
                    and item["branches"][branch_index]["root_relative_error"]
                    <= ROOT_PARITY_TOLERANCE
                ]
                selected = (
                    min(passing, key=lambda item: item["newton_steps"])
                    if passing
                    else None
                )
                rows.append(
                    {
                        "seed_distance": distance,
                        "batch_size": batch_size,
                        "branch": branch_name,
                        "minimum_converged_iterations": (
                            selected["newton_steps"] if selected else None
                        ),
                        "wall_ms_per_branch_state": (
                            selected["wall_ms_per_branch_state"] if selected else None
                        ),
                        "relative_residual": (
                            selected["branches"][branch_index]["relative_residual"]
                            if selected
                            else None
                        ),
                        "root_relative_error": (
                            selected["branches"][branch_index]["root_relative_error"]
                            if selected
                            else None
                        ),
                    }
                )
    return rows


def _phase_profile(
    profile: ForwardProfile,
    seed: jax.Array,
    selected_steps: int,
    batch_size: int,
    seed_distance: float,
) -> dict[str, Any]:
    """Profile additive portfolio phases at one fixed solve shape."""
    batch = jnp.broadcast_to(seed, (batch_size, *seed.shape))
    requested = jnp.asarray(
        (int(TopologyClass.LIMITED), int(TopologyClass.DIVERTED)), dtype=jnp.int8
    )

    def map_portfolio(states: jax.Array) -> jax.Array:
        return jax.vmap(
            lambda portfolio: jax.vmap(
                lambda state, branch_class: profile.operator(
                    state, requested_class=branch_class
                )
            )(portfolio, requested)
        )(states)

    def core(states: jax.Array, steps: int) -> Any:
        return jax.vmap(
            lambda portfolio: jax.vmap(
                lambda state, branch_class: fixed_point.newton_krylov(
                    profile.operator.flux_map(requested_class=branch_class),
                    state,
                    newton_steps=steps,
                    gmres_iterations=GMRES_ITERATIONS,
                    warmup=0,
                )
            )(portfolio, requested)
        )(states)

    def full(states: jax.Array) -> Any:
        return jax.vmap(
            lambda portfolio: profile.solve_portfolio(
                portfolio,
                route="newton_krylov",
                tolerance=CONVERGENCE_TOLERANCE,
                newton_steps=selected_steps,
                gmres_iterations=GMRES_ITERATIONS,
                warmup=0,
            )
        )(states)

    probes = {}
    for name, function in (
        ("map_evaluation", map_portfolio),
        ("one_newton_core", lambda states: core(states, 1)),
        ("selected_newton_core", lambda states: core(states, selected_steps)),
        ("full_portfolio_receipt", full),
    ):
        started = time.perf_counter()
        compiled = jax.jit(function).lower(batch).compile()
        compile_seconds = time.perf_counter() - started
        timing = _time_call(compiled, batch)
        probes[name] = {
            **timing,
            "compile_seconds": compile_seconds,
            "median_ms_per_branch_state": 1.0e3
            * timing["median_seconds"]
            / (2 * batch_size),
        }
        del compiled
        jax.clear_caches()

    map_ms = probes["map_evaluation"]["median_ms_per_branch_state"]
    one_ms = probes["one_newton_core"]["median_ms_per_branch_state"]
    core_ms = probes["selected_newton_core"]["median_ms_per_branch_state"]
    full_ms = probes["full_portfolio_receipt"]["median_ms_per_branch_state"]
    components = {
        "map_evaluation_ms": map_ms,
        "first_step_krylov_and_promotion_ms": max(one_ms - map_ms, 0.0),
        "additional_newton_steps_ms": max(core_ms - one_ms, 0.0),
        "terminal_topology_and_receipts_ms": max(full_ms - core_ms, 0.0),
    }
    accounted = sum(components.values())
    components["unattributed_fusion_or_timing_ms"] = full_ms - accounted
    return {
        "method": (
            "independent synchronized compiled probes; nested differences are "
            "additive at the catalog shape"
        ),
        "batch_size": batch_size,
        "seed_distance": seed_distance,
        "selected_newton_steps": selected_steps,
        "probes": probes,
        "components_per_branch_state": components,
        "full_ms_per_branch_state": full_ms,
    }


def _catalog_projection(
    minimum: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    phase: dict[str, Any],
) -> dict[str, Any]:
    """State the catalog operating point and its measured target gap."""
    branch_rows = [
        item
        for item in minimum
        if item["seed_distance"] == CATALOG_SEED_DISTANCE
        and item["batch_size"] == CATALOG_BATCH_SIZE
    ]
    iterations = [item["minimum_converged_iterations"] for item in branch_rows]
    selected_steps = (
        max(int(item) for item in iterations if item is not None)
        if all(item is not None for item in iterations)
        else max(NEWTON_BUDGETS)
    )
    measured = next(
        item
        for item in measurements
        if item["seed_distance"] == CATALOG_SEED_DISTANCE
        and item["batch_size"] == CATALOG_BATCH_SIZE
        and item["newton_steps"] == selected_steps
    )
    qualification = []
    for index, branch in enumerate(measured["branches"]):
        qualified = bool(branch["converged"]) and (
            branch["root_relative_error"] <= ROOT_PARITY_TOLERANCE
        )
        if qualified:
            cause = "qualified"
        elif (
            not branch["finite"]
            or not np.isfinite(branch["relative_residual"])
            or not np.isfinite(branch["root_relative_error"])
        ):
            cause = "non_finite_terminal_precludes_budget_distance_attribution"
        elif not branch["topology_consistent"]:
            cause = "terminal_topology_contradiction"
        else:
            smaller = [
                item
                for item in measurements
                if item["batch_size"] == CATALOG_BATCH_SIZE
                and item["newton_steps"] == selected_steps
                and item["seed_distance"] < CATALOG_SEED_DISTANCE
            ]
            smaller_passes = any(
                item["branches"][index]["converged"]
                and item["branches"][index]["root_relative_error"]
                <= ROOT_PARITY_TOLERANCE
                for item in smaller
            )
            cause = (
                "catalog_distance_exceeds_reach_at_bounded_budget"
                if smaller_passes
                else "bounded_budget_insufficient_at_smaller_declared_distances"
            )
        qualification.append(
            {
                **branch,
                "qualified": qualified,
                "miss_localisation": cause,
            }
        )
    value = measured["wall_ms_per_branch_state"]
    components = phase["components_per_branch_state"]
    target_credit = 1.0
    gap_attribution = {}
    for name, component in components.items():
        credited = min(component, target_credit)
        gap_attribution[name] = component - credited
        target_credit -= credited
    gap_attribution["measurement_vs_phase_profile_delta_ms"] = (
        value - phase["full_ms_per_branch_state"]
    )
    return {
        "definition": (
            "adjacent-slice seed at 1e-3 of each branch flux span, both branches "
            "batched together at production width 16"
        ),
        "batch_size": CATALOG_BATCH_SIZE,
        "seed_distance": CATALOG_SEED_DISTANCE,
        "branch_minimum_iterations": {
            row["branch"]: row["minimum_converged_iterations"] for row in branch_rows
        },
        "shared_fixed_shape_iterations": selected_steps,
        "branch_qualification": qualification,
        "portfolio_qualified": all(item["qualified"] for item in qualification),
        "qualification_interpretation": (
            "non-finite residual or parity precludes a bounded-budget versus "
            "seed-distance attribution; otherwise miss_localisation states which binds"
        ),
        "measured_ms_per_branch_state": value,
        "target_ms_per_branch_state": 1.0,
        "multiple_of_target": value,
        "remaining_gap_ms": value - 1.0,
        "full_phase_decomposition_ms": components,
        "remaining_gap_attribution_ms": gap_attribution,
        "target_met": value <= 1.0 and all(item["qualified"] for item in qualification),
    }


def _matrix_rows(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten every measured portfolio into durable branch-level rows."""
    rows = []
    for measurement in measurements:
        for branch in measurement["branches"]:
            rows.append(
                {
                    "branch": branch["branch"],
                    "seed_distance": measurement["seed_distance"],
                    "batch_size": measurement["batch_size"],
                    "newton_steps": measurement["newton_steps"],
                    "gmres_iterations": measurement["gmres_iterations"],
                    "requested_class": branch["requested_class"],
                    "achieved_class": branch["achieved_class"],
                    "topology_consistent": branch["topology_consistent"],
                    "converged": branch["converged"],
                    "relative_residual": branch["relative_residual"],
                    "root_relative_error": branch["root_relative_error"],
                    "iterations": branch["iterations"],
                    "finite": branch["finite"],
                    "wall_ms_per_branch_state": measurement["wall_ms_per_branch_state"],
                    "wall_time_samples_seconds": measurement["timing"][
                        "samples_seconds"
                    ],
                    "phase_decomposition": measurement["phase_decomposition"],
                }
            )
    return rows


def _figure(receipt: dict[str, Any], path: Path) -> None:
    """Render timing, iteration, and catalog phase evidence."""
    minimum = receipt["minimum_converged"]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    colors = {"limited": "#2563eb", "diverted": "#dc2626"}
    markers = {1: "o", 4: "s", 16: "^"}
    for branch in ("limited", "diverted"):
        for batch in BATCH_SIZES:
            rows = [
                item
                for item in minimum
                if item["branch"] == branch and item["batch_size"] == batch
            ]
            x = [max(item["seed_distance"], 3.0e-6) for item in rows]
            y = [item["wall_ms_per_branch_state"] for item in rows]
            axes[0].plot(
                x,
                y,
                marker=markers[batch],
                color=colors[branch],
                alpha=0.45 + 0.03 * batch,
                label=f"{branch}, batch {batch}",
            )
            axes[1].plot(
                x,
                [item["minimum_converged_iterations"] for item in rows],
                marker=markers[batch],
                color=colors[branch],
                alpha=0.45 + 0.03 * batch,
            )
    axes[0].axhline(
        1.0, color="black", linestyle="--", linewidth=1.0, label="1 ms target"
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("seed distance / branch flux span")
    axes[0].set_ylabel("warm wall time (ms / branch state)")
    axes[0].set_title("converged portfolio economy", loc="left")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("seed distance / branch flux span")
    axes[1].set_ylabel("minimum fixed Newton steps")
    axes[1].set_title("iteration economy", loc="left")

    components = receipt["phase_profile"]["components_per_branch_state"]
    labels = [
        "map",
        "Krylov + promote",
        "extra Newton",
        "terminal receipts",
        "fusion / timing",
    ]
    values = [
        components["map_evaluation_ms"],
        components["first_step_krylov_and_promotion_ms"],
        components["additional_newton_steps_ms"],
        components["terminal_topology_and_receipts_ms"],
        components["unattributed_fusion_or_timing_ms"],
    ]
    bottom = 0.0
    palette = ("#0f766e", "#7c3aed", "#c2410c", "#64748b", "#f59e0b")
    for label, value, color in zip(labels, values, palette, strict=True):
        axes[2].bar(["catalog"], [value], bottom=bottom, label=label, color=color)
        bottom += value
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("ms / branch state")
    axes[2].set_title("catalog gap by phase", loc="left")
    axes[2].legend(fontsize=7, loc="upper right")
    projection = receipt["catalog_projection"]
    axes[2].text(
        0.02,
        0.98,
        f"{projection['measured_ms_per_branch_state']:.3f} ms\n"
        f"{projection['multiple_of_target']:.2f}x target",
        transform=axes[2].transAxes,
        va="top",
        fontsize=9,
    )
    figure.suptitle("Topology-pinned warm-start ladder on NVIDIA H200 NVL", fontsize=12)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def measure(output: Path, figure: Path) -> dict[str, Any]:
    """Run the declared H200 ladder and write its receipt and figure."""
    configure_dtypes()
    device = jax.devices()[0]
    host = socket.gethostname()
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise RuntimeError(f"this receipt requires an H200 GPU, got {device}")
    if not host.startswith("98dci4-gpu-0003"):
        raise RuntimeError(f"this receipt requires the reserved H200 host, got {host}")

    profile, cold, diverted_root = _problem()
    limited_root, limited_preparation = _limited_root(
        profile, cold.branches.flux[int(TopologyClass.LIMITED)]
    )
    roots = np.stack((limited_root, diverted_root))
    seeds, seed_policy = _seed_ladder(
        profile,
        np.asarray(cold.branches.flux),
        roots,
    )
    jax.clear_caches()
    measurements, shapes = _solve_grid(profile, seeds, roots)
    minimum = _minimum_converged(measurements)
    matrix_rows = _matrix_rows(measurements)
    catalog_rows = [
        item
        for item in minimum
        if item["seed_distance"] == CATALOG_SEED_DISTANCE
        and item["batch_size"] == CATALOG_BATCH_SIZE
    ]
    selected_steps = (
        max(
            int(item["minimum_converged_iterations"])
            for item in catalog_rows
            if item["minimum_converged_iterations"] is not None
        )
        if all(
            item["minimum_converged_iterations"] is not None for item in catalog_rows
        )
        else max(NEWTON_BUDGETS)
    )
    catalog_measurement = next(
        item
        for item in measurements
        if item["seed_distance"] == CATALOG_SEED_DISTANCE
        and item["batch_size"] == CATALOG_BATCH_SIZE
        and item["newton_steps"] == selected_steps
    )
    phase = catalog_measurement["phase_decomposition"]

    receipt = {
        "schema": "nova.portfolio-warm-start-receipt",
        "environment": {
            "hostname": host,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            "tmpdir": os.environ.get("TMPDIR"),
            "device_kind": device.device_kind,
            "platform": device.platform,
            "jax_backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "x64_enabled": bool(jax.config.x64_enabled),
        },
        "solver_policy": {
            "route": "production topology-pinned ForwardProfile.solve_portfolio",
            "newton_budgets": NEWTON_BUDGETS,
            "gmres_iterations": GMRES_ITERATIONS,
            "warmup": 0,
            "convergence_tolerance": CONVERGENCE_TOLERANCE,
            "root_parity_tolerance": ROOT_PARITY_TOLERANCE,
            "timing_repeats": TIMING_REPEATS,
            "timing_warmups": TIMING_WARMUPS,
            "compilation_included_in_wall_time": False,
        },
        "fixture": {
            "diverted_state_sha256": _digest(diverted_root),
            "limited_root_preparation": limited_preparation,
            "root_shape": tuple(roots.shape),
            "cold_seed_stored_flux_samples_used": np.asarray(
                cold.branches.stored_flux_samples_used
            ),
        },
        "seed_policy": seed_policy,
        "fixed_shapes": shapes,
        "measurements": measurements,
        "matrix_rows": matrix_rows,
        "minimum_converged": minimum,
        "banked_cold_controls": {
            "source": "scripts/accuracy_cost_ladder/gpu-coarse.json",
            "definition": "coarse single-branch ten-Newton, thirty-GMRES solve",
            "wall_ms_per_state": BANKED_COLD_CONTROLS_MS,
        },
        "phase_profile": phase,
        "catalog_projection": {
            "status": "not_evaluated",
            "publication_order": (
                "all matrix and phase rows were serialized before qualification"
            ),
        },
        "verdict": {
            "target_met": None,
            "trajectory": (
                "cold control to adjacent-slice minimum fixed-shape budget at "
                "production batch width"
            ),
            "remaining_gap_is_attributed_by_phase": True,
        },
    }
    _write_json(output, receipt)
    projection = _catalog_projection(minimum, measurements, phase)
    receipt["catalog_projection"] = projection
    receipt["verdict"]["target_met"] = projection["target_met"]
    receipt["verdict"]["portfolio_qualified"] = projection["portfolio_qualified"]
    _write_json(output, receipt)
    _figure(receipt, figure)
    return receipt


def main() -> None:
    """Run the benchmark from its command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT / "portfolio-warm-start-receipt.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=OUTPUT / "portfolio-warm-start.png",
    )
    arguments = parser.parse_args()
    receipt = measure(arguments.output, arguments.figure)
    projection = receipt["catalog_projection"]
    print(
        "PORTFOLIO_WARM_START "
        f"catalog_ms={projection['measured_ms_per_branch_state']:.9g} "
        f"multiple={projection['multiple_of_target']:.9g} "
        f"target_met={projection['target_met']}"
    )


if __name__ == "__main__":
    main()
