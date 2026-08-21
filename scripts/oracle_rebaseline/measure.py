"""Solve and score closed-form oracle fixtures from production moment seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium import fixed_point
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.moment import (
    CurrentCells,
    MomentConfig,
    ReconstructMoment,
    limiter_radial_extent,
)
from nova.equilibrium.stencil_mesh import CellCurrentMoments, StencilMesh
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures.measure import (
    FIXTURE_REQUESTS,
    TOTAL_FLUX_FACTOR,
    WALL_POINT_COUNT,
    _internal_flux_image,
    analytic_case,
    cached_machine,
    exact_current_moments,
    exact_state,
    forward_operator,
)


OUTPUT = Path(__file__).resolve().parent
REPOSITORY_ROOT = OUTPUT.parents[1]
SOLVER_CRITERION = 1.0e-10
NEWTON_STEPS = 10
KRYLOV_ITERATIONS = 30
CONTINUATION_STRENGTHS = (0.5, 0.75, 1.0)
ORACLE_BASIN_FLUX_FRACTION = 1.0e-6
FLAT_EXCESS_RATIO = 0.8


def _digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _array_identity(values: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(values)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": _digest(array),
    }


def _json_write(path: Path, payload: dict[str, object]) -> None:
    def strict(value):
        if isinstance(value, dict):
            return {key: strict(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [strict(item) for item in value]
        if isinstance(value, np.generic):
            return strict(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.write_text(
        json.dumps(strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _relative_difference(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), np.finfo(float).tiny)


def _relative_map_residual(map_fn, state) -> float:
    mapped = map_fn(jnp.asarray(state))
    return float(
        jnp.max(jnp.abs(mapped - state))
        / jnp.maximum(jnp.max(jnp.abs(mapped)), 1.0e-30)
    )


def _topology_class(topology) -> str:
    return "diverted" if bool(topology.diverted) else "limited"


def _point_or_none(point) -> list[float] | None:
    values = np.asarray(point, dtype=float)
    if not np.all(np.isfinite(values)):
        return None
    return values.tolist()


def _aggregate_current_moment(case) -> tuple[float, np.ndarray, dict[str, object]]:
    """Return the closed-form zeroth moment and current centroid, never its flux."""
    radius, half_height, weight, _offset = case._surface_nodes(0.0, 512)
    density = case.toroidal_current_density(radius, np.zeros_like(radius))
    current_weight = 2.0 * half_height * density * weight
    integrated_current = float(np.sum(current_weight))
    total_current = float(case.plasma_current())
    centroid = np.array(
        [float(np.sum(current_weight * radius) / integrated_current), 0.0]
    )
    return (
        total_current,
        centroid,
        {
            "construction": (
                "closed-form current-density quadrature for I_p and its first radial "
                "moment; no closed-form flux value is evaluated"
            ),
            "quadrature_nodes": 512,
            "quadrature_current_a": integrated_current,
            "declared_current_a": total_current,
            "quadrature_relative_closure": _relative_difference(
                integrated_current, total_current
            ),
            "current_centroid_m": centroid.tolist(),
        },
    )


def _moment_seed(
    case, machine, operator
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return the production uniform-disc zeroth-current-moment basin seed."""
    total_current, centroid, aggregate = _aggregate_current_moment(case)
    inboard, outboard = limiter_radial_extent(
        machine.wall_node[:, 0], machine.wall_node[:, 1], centroid[1]
    )
    config = MomentConfig()
    supported_distance = min(centroid[0] - inboard, outboard - centroid[0])
    seed_radius = config.seed_radius_fraction * supported_distance
    reconstruction = ReconstructMoment(
        CurrentCells(machine.node[:, 0], machine.node[:, 1]),
        major_radius=centroid[0],
        config=config,
    )
    cell_current = reconstruction.uniform_disc(
        centroid[0], centroid[1], seed_radius, total_current
    )
    zero = jnp.zeros_like(jnp.asarray(cell_current))
    physical = CellCurrentMoments(jnp.asarray(cell_current), zero, zero)
    coefficients = operator.coupling_current_moments(physical)
    moment_image = _internal_flux_image(operator, coefficients)
    seed = np.asarray(operator.external()) + moment_image
    masks, topology = operator.read(jnp.asarray(seed))
    return (
        seed,
        moment_image,
        {
            "kind": "production-current-centroid-uniform-disc-zeroth-moment",
            "independent_of_closed_form_state": True,
            "closed_form_flux_samples_used": False,
            "closed_form_coupling_image_used": False,
            "fixture_exterior_boundary_condition_used": True,
            "aggregate_moment": aggregate,
            "limiter_midplane_extent_m": [inboard, outboard],
            "supported_minor_distance_m": supported_distance,
            "seed_radius_fraction": config.seed_radius_fraction,
            "seed_radius_m": seed_radius,
            "supported_cell_count": int(np.count_nonzero(cell_current)),
            "current_sum_a": float(cell_current.sum()),
            "initial_core_cell_count": int(np.count_nonzero(np.asarray(masks.core))),
            "initial_axis_m": _point_or_none(topology.axis),
            "initial_topology_class": _topology_class(topology),
            "state_sha256_binary64": _digest(np.asarray(seed, dtype=np.float64)),
        },
    )


def _solve(map_fn, seed) -> fixed_point.FixedPointResult:
    history = fixed_point.newton_krylov(
        map_fn,
        jnp.asarray(seed),
        newton_steps=NEWTON_STEPS,
        gmres_iterations=KRYLOV_ITERATIONS,
        warmup=0,
    )
    jax.block_until_ready(history.state)
    return history


def _history_receipt(history) -> dict[str, object]:
    trace = np.asarray(history.trace, dtype=float)
    finite_index = np.flatnonzero(np.isfinite(trace))
    finite_trace = trace[finite_index]
    stride = KRYLOV_ITERATIONS + 2
    promoted = trace[stride - 1 :: stride]
    reached = np.flatnonzero(np.isfinite(promoted) & (promoted < SOLVER_CRITERION))
    residual = float(history.residual)
    return {
        "terminal_residual": residual if np.isfinite(residual) else None,
        "criterion": SOLVER_CRITERION,
        "criterion_met": bool(np.isfinite(residual) and residual < SOLVER_CRITERION),
        "newton_steps_requested": NEWTON_STEPS,
        "gmres_iterations_per_step": KRYLOV_ITERATIONS,
        "map_evaluation_budget": NEWTON_STEPS * stride,
        "recorded_residual_evaluations": int(len(finite_trace)),
        "first_criterion_newton_step": None
        if not len(reached)
        else int(reached[0] + 1),
        "finite_trace": finite_trace.tolist(),
        "finite_trace_indices": finite_index.tolist(),
    }


def _state_flux_fraction(state, oracle, grid_count: int, span: float) -> float:
    difference = np.asarray(state)[:grid_count] - np.asarray(oracle)[:grid_count]
    return float(np.max(np.abs(difference)) / span)


def _continuation(
    operator,
    seed: np.ndarray,
    moment_image: np.ndarray,
    oracle: np.ndarray,
    span: float,
) -> tuple[fixed_point.FixedPointResult | None, list[dict[str, object]]]:
    """Follow the plasma-source-strength branch from a nonempty moment state."""
    full_map = operator.flux_map()
    exterior = np.asarray(operator.external())
    accepted_state = None
    accepted_history = None
    log: list[dict[str, object]] = []
    for strength in CONTINUATION_STRENGTHS:
        initial = (
            exterior + strength * moment_image
            if accepted_state is None
            else np.asarray(accepted_state)
        )
        masks, topology = operator.read(jnp.asarray(initial))
        initial_core = int(np.count_nonzero(np.asarray(masks.core)))
        row: dict[str, object] = {
            "source_strength": strength,
            "initial_core_cell_count": initial_core,
            "initial_topology_class": _topology_class(topology),
            "accepted": False,
        }
        if initial_core == 0:
            row["reason"] = "moment state has no axis-connected core"
            log.append(row)
            continue

        def scaled_map(state):
            return jnp.asarray(exterior) + strength * (
                full_map(state) - jnp.asarray(exterior)
            )

        started = perf_counter()
        history = _solve(scaled_map, initial)
        row["seconds"] = perf_counter() - started
        row["history"] = _history_receipt(history)
        terminal_masks, terminal_topology = operator.read(history.state)
        terminal_core = int(np.count_nonzero(np.asarray(terminal_masks.core)))
        row["terminal_core_cell_count"] = terminal_core
        row["terminal_topology_class"] = _topology_class(terminal_topology)
        row["oracle_flux_sup_fraction_of_span"] = _state_flux_fraction(
            history.state, oracle, operator.grid.node_number, span
        )
        accepted = bool(row["history"]["criterion_met"] and terminal_core > 0)
        row["accepted"] = accepted
        if not accepted:
            row["reason"] = "solver criterion or nonempty-core qualification failed"
            log.append(row)
            accepted_state = None
            accepted_history = None
            continue
        log.append(row)
        accepted_state = history.state
        accepted_history = history
    if not log or log[-1]["source_strength"] != 1.0 or not log[-1]["accepted"]:
        return None, log
    return accepted_history, log


def _conservation(receipt) -> dict[str, float]:
    conservation = receipt.conservation
    return {
        "grad_shafranov_relative": float(conservation.relative_grad_shafranov),
        "force_relative": float(conservation.relative_force),
        "divergence_b_relative": float(conservation.relative_divergence_b),
        "divergence_j_relative": float(conservation.relative_divergence_j),
        "checked_cells": int(conservation.checked_cells),
    }


def _observed_moments(receipt) -> dict[str, float]:
    moments = receipt.moments
    return {
        "plasma_current_a": float(moments.plasma_current),
        "poloidal_beta": float(moments.poloidal_beta),
        "internal_inductance": float(moments.internal_inductance),
        "poloidal_field_integral": float(moments.poloidal_field_integral),
        "volume_m3": float(moments.volume),
        "major_radius_m": float(moments.major_radius),
    }


def _metric(value: float, absolute: float, floor: float) -> dict[str, float]:
    return {
        "recovery_value": float(value),
        "absolute_reference_deviation": float(absolute),
        "representation_reference_floor": float(floor),
    }


def _measure_root(case, machine, operator, oracle, seed, history) -> dict[str, object]:
    span = TOTAL_FLUX_FACTOR * case.axis_flux
    lattice = StencilMesh(machine.node, machine.stencil, machine.area)
    profile = ForwardProfile(operator, lattice, newton_steps=NEWTON_STEPS)
    oracle_receipt = profile.observe(jnp.asarray(oracle))
    root_receipt = profile.observe(history.state)
    oracle_topology = oracle_receipt.topology
    root_topology = root_receipt.topology
    oracle_moments = _observed_moments(oracle_receipt)
    root_moments = _observed_moments(root_receipt)
    oracle_conservation = _conservation(oracle_receipt)
    root_conservation = _conservation(root_receipt)
    grid_count = len(machine.node)
    grid_difference = np.asarray(history.state)[:grid_count] - oracle[:grid_count]
    axis_difference = np.asarray(root_topology.axis) - np.asarray(oracle_topology.axis)
    root_axis_reference = float(
        np.linalg.norm(np.asarray(root_topology.axis) - np.asarray(case.magnetic_axis))
    )
    oracle_axis_floor = float(
        np.linalg.norm(
            np.asarray(oracle_topology.axis) - np.asarray(case.magnetic_axis)
        )
    )
    analytic = {
        "plasma_current_a": float(case.plasma_current()),
        "poloidal_beta": float(case.beta_poloidal),
        "internal_inductance": float(case.internal_inductance),
        "poloidal_field_integral": float(case.poloidal_field_volume_integral()),
    }
    metric = {
        "standing_forcing_sup_wb": _metric(
            float(
                np.max(
                    np.abs(
                        np.asarray(operator.flux_map()(jnp.asarray(oracle))) - oracle
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        np.asarray(operator.flux_map()(jnp.asarray(oracle))) - oracle
                    )
                )
            ),
            0.0,
        ),
        "fixed_point_residual": _metric(
            _relative_map_residual(operator.flux_map(), history.state),
            _relative_map_residual(operator.flux_map(), history.state),
            0.0,
        ),
        "axis_position_m": _metric(
            float(np.linalg.norm(axis_difference)),
            root_axis_reference,
            oracle_axis_floor,
        ),
        "flux_sup_fraction_of_span": _metric(
            float(np.max(np.abs(grid_difference)) / span),
            float(np.max(np.abs(grid_difference)) / span),
            0.0,
        ),
        "flux_rms_fraction_of_span": _metric(
            float(np.sqrt(np.mean(grid_difference**2)) / span),
            float(np.sqrt(np.mean(grid_difference**2)) / span),
            0.0,
        ),
    }
    for gate_name, observation_name, analytic_name in (
        ("plasma_current_fraction", "plasma_current_a", "plasma_current_a"),
        ("poloidal_beta_fraction", "poloidal_beta", "poloidal_beta"),
        ("internal_inductance_fraction", "internal_inductance", "internal_inductance"),
        (
            "field_integral_fraction",
            "poloidal_field_integral",
            "poloidal_field_integral",
        ),
    ):
        metric[gate_name] = _metric(
            _relative_difference(
                root_moments[observation_name], oracle_moments[observation_name]
            ),
            _relative_difference(
                root_moments[observation_name], analytic[analytic_name]
            ),
            _relative_difference(
                oracle_moments[observation_name], analytic[analytic_name]
            ),
        )
    for name in (
        "grad_shafranov_relative",
        "divergence_b_relative",
        "divergence_j_relative",
    ):
        metric[name] = _metric(
            abs(root_conservation[name] - oracle_conservation[name]),
            root_conservation[name],
            oracle_conservation[name],
        )
    root_grid = np.asarray(history.state)[:grid_count]
    oracle_grid = np.asarray(oracle)[:grid_count]
    root_psi_norm = (root_grid - float(root_topology.axis_flux)) / float(
        root_topology.flux_span
    )
    oracle_psi_norm = (oracle_grid - float(oracle_topology.axis_flux)) / float(
        oracle_topology.flux_span
    )
    arrays = {
        "root_state": np.asarray(history.state, dtype=np.float64),
        "oracle_state": np.asarray(oracle, dtype=np.float64),
        "seed_state": np.asarray(seed, dtype=np.float64),
        "residual_trace": np.asarray(history.trace, dtype=np.float64),
        "root_grid_psi_norm": np.asarray(root_psi_norm, dtype=np.float64),
        "oracle_grid_psi_norm": np.asarray(oracle_psi_norm, dtype=np.float64),
        "root_axis": np.asarray(root_topology.axis, dtype=np.float64),
        "oracle_axis": np.asarray(oracle_topology.axis, dtype=np.float64),
        "root_cell_current": np.asarray(root_receipt.cell_current, dtype=np.float64),
        "oracle_cell_current": np.asarray(
            oracle_receipt.cell_current, dtype=np.float64
        ),
    }
    return {
        "metric": metric,
        "arrays": arrays,
        "root_moments": root_moments,
        "closed_form_state_observed_moments": oracle_moments,
        "closed_form_analytic_invariants": analytic,
        "root_conservation": root_conservation,
        "closed_form_state_conservation_floor": oracle_conservation,
        "root_topology": {
            "class": _topology_class(root_topology),
            "axis_m": np.asarray(root_topology.axis).tolist(),
            "axis_flux_wb": float(root_topology.axis_flux),
            "boundary_flux_wb": float(root_topology.boundary_flux),
            "flux_span_wb": float(root_topology.flux_span),
            "x_point": (
                _point_or_none(root_topology.x_point)
                if bool(root_topology.diverted)
                else None
            ),
        },
        "oracle_topology": {
            "class": _topology_class(oracle_topology),
            "axis_m": np.asarray(oracle_topology.axis).tolist(),
            "axis_flux_wb": float(oracle_topology.axis_flux),
            "boundary_flux_wb": float(oracle_topology.boundary_flux),
            "flux_span_wb": float(oracle_topology.flux_span),
            "x_point": (
                _point_or_none(oracle_topology.x_point)
                if bool(oracle_topology.diverted)
                else None
            ),
        },
        "gauge_receipt": {
            "raw_flux_comparison_gauge": "shared_exact_exterior",
            "raw_flux_difference": (
                "root_state - independently evaluated closed_form_state"
            ),
            "reference_gauge_constant_used": False,
            "psi_norm_root_anchors_from": "root_field",
            "psi_norm_oracle_anchors_from": "closed_form_field",
            "root_anchors_wb": [
                float(root_topology.axis_flux),
                float(root_topology.boundary_flux),
            ],
            "oracle_anchors_wb": [
                float(oracle_topology.axis_flux),
                float(oracle_topology.boundary_flux),
            ],
            "psi_norm_sup_difference": float(
                np.max(np.abs(root_psi_norm - oracle_psi_norm))
            ),
        },
    }


def measure_fixture(name: str) -> dict[str, object]:
    """Solve one warm oracle carrier and bank its recovery receipt."""
    configure_dtypes()
    case = analytic_case()
    requested_cells = FIXTURE_REQUESTS[name]
    print(f"CACHE_REQUEST fixture={name} requested_cells={requested_cells}", flush=True)
    machine = cached_machine(case, requested_cells, wall_nodes=WALL_POINT_COUNT)
    print(
        f"CACHE_RESULT fixture={name} cells={len(machine.node)} "
        f"hit={machine.cache['hit']} key={machine.cache['semantic_key']}",
        flush=True,
    )
    coordinates = np.vstack(
        [machine.node, machine.wall_node, machine.sample_coordinates]
    )
    oracle = exact_state(case, coordinates)
    empty_operator = forward_operator(case, machine)
    exact_physical = exact_current_moments(case, empty_operator, oracle)
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = _internal_flux_image(empty_operator, exact_coefficients)
    prescribed_exterior = oracle - exact_internal
    operator = forward_operator(case, machine, prescribed_exterior)
    seed, moment_image, seed_receipt = _moment_seed(case, machine, operator)
    seed_receipt["oracle_state_sup_difference_fraction_of_span"] = _state_flux_fraction(
        seed, oracle, operator.grid.node_number, TOTAL_FLUX_FACTOR * case.axis_flux
    )
    print(
        f"SEED fixture={name} core={seed_receipt['initial_core_cell_count']} "
        f"centroid={seed_receipt['aggregate_moment']['current_centroid_m']} "
        f"radius={seed_receipt['seed_radius_m']:.17g} "
        f"oracle_flux_fraction="
        f"{seed_receipt['oracle_state_sup_difference_fraction_of_span']:.17g}",
        flush=True,
    )
    started = perf_counter()
    direct = _solve(operator.flux_map(), seed)
    direct_seconds = perf_counter() - started
    span = TOTAL_FLUX_FACTOR * case.axis_flux
    direct_flux_fraction = _state_flux_fraction(
        direct.state, oracle, operator.grid.node_number, span
    )
    direct_receipt = {
        **_history_receipt(direct),
        "seconds": direct_seconds,
        "oracle_flux_sup_fraction_of_span": direct_flux_fraction,
    }
    print(
        f"DIRECT fixture={name} residual={direct.residual:.17g} "
        f"oracle_flux_fraction={direct_flux_fraction:.17g} "
        f"seconds={direct_seconds:.9g}",
        flush=True,
    )
    continuation_history = None
    continuation_log: list[dict[str, object]] = []
    if direct_flux_fraction > ORACLE_BASIN_FLUX_FRACTION:
        continuation_history, continuation_log = _continuation(
            operator, seed, moment_image, oracle, span
        )
        for row in continuation_log:
            residual = row.get("history", {}).get("terminal_residual")
            print(
                f"CONTINUATION fixture={name} strength={row['source_strength']:.3f} "
                f"accepted={row['accepted']} residual={residual} "
                f"core={row.get('terminal_core_cell_count')}",
                flush=True,
            )
    _json_write(
        OUTPUT / f"attempts-{name}.json",
        {
            "fixture": name,
            "seed": seed_receipt,
            "direct_attempt": direct_receipt,
            "continuation": continuation_log,
        },
    )
    candidates = [("direct_moment_seed", direct)]
    if continuation_history is not None:
        candidates.append(("source_strength_continuation", continuation_history))
    qualified = [
        (route, history)
        for route, history in candidates
        if _history_receipt(history)["criterion_met"]
    ]
    if not qualified:
        route, selected = candidates[0]
    else:
        route, selected = min(
            qualified,
            key=lambda item: _state_flux_fraction(
                item[1].state, oracle, operator.grid.node_number, span
            ),
        )
    measured = _measure_root(case, machine, operator, oracle, seed, selected)
    root_path = OUTPUT / f"root-{name}.npz"
    np.savez(root_path, **measured.pop("arrays"))
    with np.load(root_path, allow_pickle=False) as stored:
        artifact_arrays = {
            key: _array_identity(np.asarray(stored[key])) for key in stored.files
        }
    history_receipt = _history_receipt(selected)
    terminal_topology = measured["root_topology"]
    receipt = {
        "fixture": name,
        "requested_cells": requested_cells,
        "realised_cells": len(machine.node),
        "state_size": len(oracle),
        "cache": machine.cache,
        "solver": {
            "route": "undamped_newton_krylov",
            "criterion_unchanged": True,
            "criterion": SOLVER_CRITERION,
            "newton_steps": NEWTON_STEPS,
            "gmres_iterations": KRYLOV_ITERATIONS,
            "selection": (
                "lowest closed-form flux drift among independently seeded, "
                "criterion-qualified direct and continuation roots"
            ),
        },
        "seed": seed_receipt,
        "direct_attempt": direct_receipt,
        "continuation": {
            "triggered": bool(continuation_log),
            "trigger_flux_fraction": ORACLE_BASIN_FLUX_FRACTION,
            "map_definition": ("external + alpha*(production_map(state)-external)"),
            "accepted_steps": continuation_log,
        },
        "terminal_root": {
            **history_receipt,
            "selected_route": route,
            "topology_class": terminal_topology["class"],
            "axis_m": terminal_topology["axis_m"],
            "x_point": terminal_topology["x_point"],
        },
        "metric": measured.pop("metric"),
        **measured,
        "root_artifact": {
            "path": str(root_path.relative_to(REPOSITORY_ROOT)),
            "bytes": root_path.stat().st_size,
            "arrays": artifact_arrays,
        },
    }
    _json_write(OUTPUT / f"receipt-{name}.json", receipt)
    return receipt


def _numeric_gate(
    name: str,
    fixtures: dict[str, dict[str, object]],
    base_bound: float,
    units: str,
) -> dict[str, object]:
    values = {
        fixture_name: float(fixture["metric"][name]["recovery_value"])
        for fixture_name, fixture in fixtures.items()
    }
    absolute = {
        fixture_name: float(fixture["metric"][name]["absolute_reference_deviation"])
        for fixture_name, fixture in fixtures.items()
    }
    floors = {
        fixture_name: float(fixture["metric"][name]["representation_reference_floor"])
        for fixture_name, fixture in fixtures.items()
    }
    measured = max(values.values())
    floor_maximum = max(floors.values())
    proposed = max(base_bound, 8.0 * floor_maximum)
    slack = {
        fixture_name: max(floors[fixture_name] * 0.05, base_bound)
        for fixture_name in fixtures
    }
    excess = {
        fixture_name: max(
            0.0, absolute[fixture_name] - floors[fixture_name] - slack[fixture_name]
        )
        for fixture_name in fixtures
    }
    coarse_excess = excess["coarse"]
    fine_excess = excess["fine"]
    ratio = (
        fine_excess / coarse_excess
        if coarse_excess > 0.0
        else (0.0 if fine_excess == 0.0 else None)
    )
    flat_above = bool(
        coarse_excess > base_bound
        and fine_excess > base_bound
        and ratio is not None
        and ratio >= FLAT_EXCESS_RATIO
    )
    return {
        "status": "proposed",
        "owner_lock_required": True,
        "units": units,
        "measured_floor": measured,
        "measured_by_fixture": values,
        "proposed_bound": proposed,
        "headroom": proposed / max(floor_maximum, base_bound),
        "proposal_basis": (
            "eightfold headroom over the measured representation/reference floor, "
            "never widened to admit an alternate-root recovery deviation"
        ),
        "fixture_pass": {
            fixture_name: value <= proposed for fixture_name, value in values.items()
        },
        "gauge": "shared_exact_exterior" if "flux" in name else "not_applicable",
        "psi_norm_anchor": "local" if "flux" in name else "not_applicable",
        "convergence_clause": {
            "rejects_flat_above_floor": True,
            "absolute_reference_deviation_by_fixture": absolute,
            "representation_reference_floor_by_fixture": floors,
            "roundoff_slack_by_fixture": slack,
            "excess_above_floor_by_fixture": excess,
            "fine_to_coarse_excess_ratio": ratio,
            "h_independent_ratio_threshold": FLAT_EXCESS_RATIO,
            "flat_above_floor": flat_above,
            "passed": not flat_above,
        },
    }


def _discrete_gate(
    fixtures: dict[str, dict[str, object]], *, x_point: bool = False
) -> dict[str, object]:
    expected = "absent" if x_point else "limited"
    values = {
        name: (
            ("absent" if fixture["terminal_root"]["x_point"] is None else "present")
            if x_point
            else fixture["terminal_root"]["topology_class"]
        )
        for name, fixture in fixtures.items()
    }
    passed = {name: value == expected for name, value in values.items()}
    return {
        "status": "proposed",
        "owner_lock_required": True,
        "units": "discrete",
        "measured_floor": values,
        "proposed_bound": expected,
        "headroom": "exact discrete agreement",
        "measured_by_fixture": values,
        "fixture_pass": passed,
        "gauge": "not_applicable",
        "psi_norm_anchor": "not_applicable",
        "convergence_clause": {
            "rejects_flat_above_floor": True,
            "expected": expected,
            "passed": all(passed.values()),
        },
    }


def _gate_registry(fixtures: dict[str, dict[str, object]]) -> dict[str, object]:
    specifications = {
        "standing_forcing_sup_wb": (1.0e-13, "Wb"),
        "fixed_point_residual": (1.0e-12, "relative"),
        "axis_position_m": (1.0e-10, "m"),
        "flux_sup_fraction_of_span": (1.0e-12, "fraction"),
        "flux_rms_fraction_of_span": (1.0e-12, "fraction"),
        "plasma_current_fraction": (1.0e-12, "fraction"),
        "poloidal_beta_fraction": (1.0e-12, "fraction"),
        "internal_inductance_fraction": (1.0e-12, "fraction"),
        "field_integral_fraction": (1.0e-12, "fraction"),
        "grad_shafranov_relative": (1.0e-12, "relative"),
        "divergence_b_relative": (1.0e-12, "relative"),
        "divergence_j_relative": (1.0e-12, "relative"),
    }
    registry = {
        name: _numeric_gate(name, fixtures, bound, units)
        for name, (bound, units) in specifications.items()
    }
    registry["topology_class"] = _discrete_gate(fixtures)
    registry["x_point_absence"] = _discrete_gate(fixtures, x_point=True)
    return registry


def _draw(report: dict[str, object]) -> Path:
    fixtures = report["fixtures"]
    registry = report["gate_registry"]
    numeric = [
        name
        for name, gate in registry.items()
        if isinstance(gate["proposed_bound"], (int, float))
    ]
    figure, axes = plt.subplots(
        1, 2, figsize=(13.0, 7.2), gridspec_kw={"width_ratios": (1.15, 1.0)}
    )
    positions = np.arange(len(numeric))
    for offset, fixture_name in ((-0.18, "coarse"), (0.18, "fine")):
        ratio = [
            registry[name]["measured_by_fixture"][fixture_name]
            / max(registry[name]["proposed_bound"], np.finfo(float).tiny)
            for name in numeric
        ]
        axes[0].barh(positions + offset, ratio, height=0.34, label=fixture_name)
    axes[0].axvline(1.0, color="0.2", linewidth=1.0)
    axes[0].set_xscale("log")
    axes[0].set_yticks(positions, [name.replace("_", " ") for name in numeric])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("measured recovery / proposed bound")
    axes[0].legend(frameon=False)
    selected = (
        "axis_position_m",
        "plasma_current_fraction",
        "grad_shafranov_relative",
        "divergence_b_relative",
        "divergence_j_relative",
    )
    for name in selected:
        values = [
            fixtures[fixture]["metric"][name]["absolute_reference_deviation"]
            for fixture in ("coarse", "fine")
        ]
        floors = [
            fixtures[fixture]["metric"][name]["representation_reference_floor"]
            for fixture in ("coarse", "fine")
        ]
        axes[1].plot((0, 1), values, "o-", label=name.replace("_", " "))
        axes[1].plot((0, 1), floors, ":", color=axes[1].lines[-1].get_color())
    axes[1].set_xticks((0, 1), ("coarse", "fine"))
    axes[1].set_yscale("log")
    axes[1].set_ylabel("absolute deviation; dotted = reference floor")
    axes[1].legend(
        frameon=False,
        fontsize="small",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    figure.tight_layout()
    path = OUTPUT / "recovery-floors.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def merge() -> dict[str, object]:
    """Merge both ordered fixture receipts and register proposed gates."""
    fixtures = {
        name: json.loads((OUTPUT / f"receipt-{name}.json").read_text(encoding="utf-8"))
        for name in ("coarse", "fine")
    }
    for fixture in fixtures.values():
        if fixture["terminal_root"]["topology_class"] == "limited":
            fixture["terminal_root"]["x_point"] = None
            fixture["root_topology"]["x_point"] = None
    registry = _gate_registry(fixtures)
    for fixture_name, fixture in fixtures.items():
        fixture["gate_results"] = {
            "passed": {
                name: bool(gate["fixture_pass"][fixture_name])
                for name, gate in registry.items()
            },
        }
        fixture["gate_results"]["all_pass"] = all(
            fixture["gate_results"]["passed"].values()
        )
    report: dict[str, object] = {
        "schema": "nova.solovev-recovery-gates",
        "analytic_case": "moderate-rotation-conventional",
        "independent_seed_contract": (
            "production current-centroid uniform-disc zeroth moment; exact "
            "exterior boundary supply is retained, but no closed-form flux sample "
            "or closed-form coupling image constructs the plasma seed"
        ),
        "solver_contract": {
            "criterion": SOLVER_CRITERION,
            "newton_steps": NEWTON_STEPS,
            "gmres_iterations": KRYLOV_ITERATIONS,
            "damping": "none",
            "coarse_before_fine": True,
        },
        "fixtures": fixtures,
        "gate_registry": registry,
        "bounds_policy": (
            "All bounds are measured proposals with explicit headroom; owner "
            "review is required before any bound becomes normative."
        ),
    }
    report["verdict"] = {
        "all_roots_meet_solver_criterion": all(
            fixture["terminal_root"]["criterion_met"] for fixture in fixtures.values()
        ),
        "all_proposed_gates_pass": all(
            fixture["gate_results"]["all_pass"] for fixture in fixtures.values()
        ),
        "all_convergence_clauses_pass": all(
            gate["convergence_clause"]["passed"] for gate in registry.values()
        ),
        "roundoff_class_recovery": all(
            fixture["metric"]["flux_sup_fraction_of_span"]["recovery_value"]
            <= ORACLE_BASIN_FLUX_FRACTION
            for fixture in fixtures.values()
        ),
    }
    report["recovery_finding"] = {
        "status": (
            "recovered"
            if report["verdict"]["roundoff_class_recovery"]
            else "alternate-root-hold"
        ),
        "interpretation": (
            "The independently constructed production moment seed reached the "
            "closed-form fixed point."
            if report["verdict"]["roundoff_class_recovery"]
            else "The production moment seed reached criterion-qualified physical "
            "roots outside the closed-form basin; failed source-strength steps and "
            "the full residual trajectories are retained, so small terminal "
            "residuals are not treated as recovery."
        ),
        "topology_qualification": (
            "This closed-form fixture is limited and has no X-point; topology is "
            "read from each terminal field rather than inferred from the seed."
        ),
    }
    figure = _draw(report)
    report["artifacts"] = {
        "figure": str(figure.relative_to(REPOSITORY_ROOT)),
        "figure_bytes": figure.stat().st_size,
        "terminal_roots": {
            name: fixture["root_artifact"]["path"] for name, fixture in fixtures.items()
        },
    }
    _json_write(OUTPUT / "results.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", choices=("coarse", "fine"))
    parser.add_argument("--merge", action="store_true")
    arguments = parser.parse_args()
    if arguments.merge:
        report = merge()
        print(
            "MERGED "
            f"roots={report['verdict']['all_roots_meet_solver_criterion']} "
            f"gates={report['verdict']['all_proposed_gates_pass']} "
            f"convergence={report['verdict']['all_convergence_clauses_pass']} "
            f"roundoff={report['verdict']['roundoff_class_recovery']}",
            flush=True,
        )
        return
    if arguments.fixture is None:
        parser.error("select --fixture or --merge")
    receipt = measure_fixture(arguments.fixture)
    print(
        f"BANKED fixture={arguments.fixture} "
        f"residual={receipt['terminal_root']['terminal_residual']:.17g} "
        f"flux_fraction="
        f"{receipt['metric']['flux_sup_fraction_of_span']['recovery_value']:.17g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
