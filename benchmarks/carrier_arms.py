"""Measure exact-value and plasma-only coefficient-carrier iterations.

The closed-form coarse fixture supplies one warm semantic machine, one banked
root, and one ordinary production map.  The exact-value route is the banked
shipped solve.  Both coefficient routes reuse the same fixed spline expansion;
the higher-support route replaces the three-function cell density projection
with a six-function quadratic projection while retaining exact Green output
for residual and topology reads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.second_moment_kernel import flux_density_columns
from nova.equilibrium.coefficient_carrier import (
    CoefficientCarrier,
    IterateRoute,
    coefficient_fixed_point_map,
    dense_newton,
    select_fixed_point_map,
)
from nova.equilibrium.observation import clipped_support_quadrature
from nova.equilibrium.topology import boundary_mode
from nova.jax.config import configure_dtypes


OUTPUT = Path("docs/figures/coefficient-space-newton/plasma-only-carrier.json")
TOTAL_CARRIER_RECEIPT = Path("docs/figures/coefficient-space-newton/carrier-arms.json")
FIXTURE_MODULE = Path("scripts/analytic_oracle_fixtures/measure.py")
ROOT_BANK = Path("scripts/oracle_rebaseline/root-coarse.npz")
ROOT_RECEIPT = Path("scripts/oracle_rebaseline/results.json")
DEGREE_RECEIPT = Path(
    "docs/figures/coefficient-space-newton/representation-degree.json"
)
SUPPORT_RECEIPT = Path("docs/figures/coefficient-space-newton/second-order-kernel.json")
KNOTS_PER_AXIS = 6
KNOT_SCAN = tuple(range(4, 21, 2))
NEWTON_STEPS = 4
BATCH_SIZE = 4
SEPARATRIX_HALF_WIDTH = 0.05
TIMING_REPEATS = 3
SUPPORT_COLUMNS = 6
SUPPORT_BUILD_QUADRATURE_ORDER = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _cgroup_peak_memory() -> int | None:
    """Return this allocation's resident-memory high-water mark when exposed."""
    try:
        rows = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        relative = next(row.split("::", 1)[1] for row in rows if "::" in row)
        peak = Path("/sys/fs/cgroup") / relative.lstrip("/") / "memory.peak"
        return int(peak.read_text(encoding="utf-8").strip())
    except FileNotFoundError, PermissionError, StopIteration, ValueError:
        pass

    job = os.environ.get("SLURM_JOB_ID")
    step = os.environ.get("SLURM_STEP_ID", "batch")
    if job is None:
        return None
    try:
        value = subprocess.check_output(
            [
                "sstat",
                "-j",
                f"{job}.{step}",
                "--noheader",
                "--parsable2",
                "-o",
                "MaxRSS",
            ],
            text=True,
        ).strip()
        suffix_scale = {"K": 1024, "M": 1024**2, "G": 1024**3}
        return int(float(value[:-1]) * suffix_scale[value[-1]])
    except KeyError, OSError, subprocess.CalledProcessError, ValueError:
        return None


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _timed(callable_, argument) -> dict[str, float]:
    compiled = jax.jit(callable_)
    started = perf_counter()
    compiled(argument).block_until_ready()
    compile_and_warm = perf_counter() - started
    samples = []
    for _ in range(TIMING_REPEATS):
        started = perf_counter()
        compiled(argument).block_until_ready()
        samples.append(perf_counter() - started)
    return {
        "compile_and_warm_seconds": compile_and_warm,
        "median_seconds": float(np.median(samples)),
        "minimum_seconds": float(np.min(samples)),
        "repeats": TIMING_REPEATS,
    }


def _basis(local):
    radial = local[..., 0]
    vertical = local[..., 1]
    return jnp.stack(
        (
            jnp.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        ),
        axis=-1,
    )


def _polygon_rule(vertices: np.ndarray, order: int = 6):
    node, weight = np.polynomial.legendre.leggauss(order)
    node = 0.5 * (node + 1.0)
    weight = 0.5 * weight
    points = []
    weights = []
    for index in range(1, len(vertices) - 1):
        origin, second, third = vertices[[0, index, index + 1]]
        first_edge = second - origin
        second_edge = third - origin
        determinant = abs(np.linalg.det(np.stack((first_edge, second_edge))))
        for first, first_weight in zip(node, weight, strict=True):
            for second_node, second_weight in zip(node, weight, strict=True):
                points.append(
                    origin
                    + first * first_edge
                    + (1.0 - first) * second_node * second_edge
                )
                weights.append(
                    determinant * (1.0 - first) * first_weight * second_weight
                )
    return np.asarray(points), np.asarray(weights)


def _support_geometry(machine, target_coordinate):
    """Build quadratic density Gram inverses and exact weighted Green columns."""
    polygons = machine.cell_polygons
    centres = np.asarray(machine.moment_geometry.atomic_mesh.centroids)
    spread = np.asarray(
        [
            np.max(np.abs(vertices - centre), axis=0)
            for vertices, centre in zip(polygons, centres, strict=True)
        ]
    )
    if np.any(spread <= 0.0):
        raise ValueError("every support must span both coordinate axes")

    inverse_gram = []
    area = []
    for vertices, centre, scale in zip(polygons, centres, spread, strict=True):
        point, weight = _polygon_rule(vertices)
        basis = np.asarray(_basis(jnp.asarray((point - centre) / scale)))
        gram = np.einsum("qi,qj,q->ij", basis, basis, weight)
        inverse_gram.append(np.linalg.inv(gram))
        area.append(float(np.sum(weight)))

    target_r = jnp.asarray(target_coordinate[:, 0])
    target_z = jnp.asarray(target_coordinate[:, 1])
    columns = np.empty(
        (len(polygons), len(target_coordinate), SUPPORT_COLUMNS), dtype=np.float64
    )
    powers = np.asarray([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]], dtype=int)
    for vertex_count in sorted({len(vertices) for vertices in polygons}):
        indices = np.asarray(
            [
                index
                for index, vertices in enumerate(polygons)
                if len(vertices) == vertex_count
            ]
        )
        vertices = jnp.asarray(np.stack([polygons[index] for index in indices]))
        expansion_points = jnp.asarray(centres[indices])

        def one(cell_vertices, expansion_point):
            return flux_density_columns(
                jnp,
                target_r,
                target_z,
                cell_vertices,
                expansion_point=expansion_point,
                order=SUPPORT_BUILD_QUADRATURE_ORDER,
                columns=SUPPORT_COLUMNS,
            )

        values = np.asarray(jax.jit(jax.vmap(one))(vertices, expansion_points))
        scale_factor = np.prod(spread[indices, None, :] ** powers[None, :, :], axis=2)
        columns[indices] = values / scale_factor[:, None, :]
    return (
        jnp.asarray(columns),
        jnp.asarray(np.stack(inverse_gram)),
        jnp.asarray(area),
        jnp.asarray(centres),
        jnp.asarray(spread),
    )


def _quadratic_support_map(operator, geometry):
    columns, inverse_gram, area, centres, spread = geometry
    external = operator.external()

    def mapped(state):
        masks, _topology, sample_flux, support, _common_support = (
            operator._support_partition(state)
        )
        selection = masks.core | masks.common_sol
        points, weights = clipped_support_quadrature(support, selection)
        psi_norm, _radial, _vertical = operator.sample_flux_field(
            masks.psi_norm, sample_flux, points
        )
        density = operator.source.core.current_density(points[..., 0], psi_norm)
        basis = _basis((points - centres[:, None, :]) / spread[:, None, :])
        moments = jnp.einsum("nq,nq,nqk->nk", density, weights, basis)
        coefficients = jnp.einsum("nij,nj->ni", inverse_gram, moments)
        internal = jnp.einsum("ntk,nk,n->t", columns, coefficients, area)
        return external + internal

    return mapped


def _region_masks(psi_norm):
    lower = 1.0 - SEPARATRIX_HALF_WIDTH
    upper = 1.0 + SEPARATRIX_HALF_WIDTH
    return {
        "closed_flux": psi_norm < lower,
        "separatrix_band": (psi_norm >= lower) & (psi_norm <= upper),
        "scrape_off_layer": psi_norm > upper,
    }


def _regional_error(target, measured, psi_norm):
    result = {}
    for name, mask in _region_masks(psi_norm).items():
        count = int(np.count_nonzero(mask))
        scale = float(np.max(np.abs(target[mask]))) if count else float("nan")
        result[name] = {
            "count": count,
            "relative_sup": (
                float(np.max(np.abs(measured[mask] - target[mask])) / scale)
                if count and scale > 0.0
                else None
            ),
        }
    return result


def _topology(operator, state):
    _masks, topology = operator.read(jnp.asarray(state))
    x_point = np.asarray(topology.x_point, dtype=float)
    return {
        "class": boundary_mode(topology).value,
        "axis_m": np.asarray(topology.axis, dtype=float).tolist(),
        "x_point_m": x_point.tolist() if np.all(np.isfinite(x_point)) else None,
        "x_point_finite": bool(np.all(np.isfinite(x_point))),
    }


def _residual_normalisations(exact_output, exact_state, external):
    output = np.asarray(exact_output)
    state = np.asarray(exact_state)
    known_external = np.asarray(external)
    absolute_sup = float(np.max(np.abs(output - state)))
    total_peak = float(np.max(np.abs(output)))
    plasma_peak = float(np.max(np.abs(output - known_external)))
    return {
        "absolute_sup_wb": absolute_sup,
        "total_flux_peak_wb": total_peak,
        "plasma_flux_peak_wb": plasma_peak,
        "normalised_by_total_flux_peak": absolute_sup / total_peak,
        "normalised_by_plasma_flux_peak": absolute_sup / plasma_peak,
        "reported_terminal_residual_normalisation": "total_flux_peak",
    }


def _projection_ladder(
    coordinate, plasma_flux, grid_count, psi_norm, terminal_residual
):
    rows = []
    first_reaching = None
    for knots in KNOT_SCAN:
        candidate = CoefficientCarrier.from_coordinates(
            coordinate,
            radial_knots=knots,
            vertical_knots=knots,
        )
        represented = np.asarray(candidate.expand(candidate.project(plasma_flux)))
        regional = _regional_error(
            plasma_flux[:grid_count], represented[:grid_count], psi_norm
        )
        worst = max(
            value["relative_sup"]
            for value in regional.values()
            if value["relative_sup"] is not None
        )
        row = {
            "knots_per_axis": knots,
            "coefficient_count": candidate.coefficient_count,
            "regional_projection_floor": regional,
            "worst_regional_relative_sup": worst,
        }
        rows.append(row)
        if first_reaching is None and worst <= terminal_residual:
            first_reaching = {
                "knots_per_axis": knots,
                "coefficient_count": candidate.coefficient_count,
                "worst_regional_relative_sup": worst,
            }
            break
    return rows, first_reaching


def _solve_arm(exact_map, operator, carrier, initial, external):
    def admissible(output):
        _masks, topology = operator.read(output)
        return jnp.all(jnp.isfinite(topology.axis)) & (~topology.diverted)

    started = perf_counter()
    result = dense_newton(
        exact_map,
        carrier,
        initial,
        steps=NEWTON_STEPS,
        admissible=admissible,
        external=external,
    )
    result.exact_output.block_until_ready()
    wall_seconds = perf_counter() - started
    return result, wall_seconds


def measure() -> dict[str, object]:
    configure_dtypes()
    fixture = _load_module(FIXTURE_MODULE, "carrier_arms_fixture")
    source_revision = _source_revision()
    case = fixture.analytic_case()

    machine_started = perf_counter()
    machine = fixture.cached_machine(
        case,
        fixture.FIXTURE_REQUESTS["coarse"],
        wall_nodes=fixture.WALL_POINT_COUNT,
    )
    machine_seconds = perf_counter() - machine_started
    coordinate = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    exact_analytic = fixture.exact_state(case, coordinate)
    zero_exterior = fixture.forward_operator(case, machine)
    exact_physical = fixture.exact_current_moments(case, zero_exterior, exact_analytic)
    exact_coefficients = zero_exterior.coupling_current_moments(exact_physical)
    exact_internal = fixture._internal_flux_image(zero_exterior, exact_coefficients)
    operator = fixture.forward_operator(case, machine, exact_analytic - exact_internal)
    first_support_map = operator.flux_map()
    known_external = np.asarray(operator.external(), dtype=np.float64)

    with np.load(ROOT_BANK, allow_pickle=False) as bank:
        root = np.asarray(bank["root_state"], dtype=np.float64)
        seed = np.asarray(bank["seed_state"], dtype=np.float64)
        root_psi_norm = np.asarray(bank["root_grid_psi_norm"], dtype=np.float64)
    root_receipt = json.loads(ROOT_RECEIPT.read_text(encoding="utf-8"))["fixtures"][
        "coarse"
    ]
    total_carrier_receipt = json.loads(
        TOTAL_CARRIER_RECEIPT.read_text(encoding="utf-8")
    )
    total_carrier_arm = total_carrier_receipt["arms"]["C"]

    carrier_started = perf_counter()
    carrier = CoefficientCarrier.from_coordinates(
        coordinate,
        radial_knots=KNOTS_PER_AXIS,
        vertical_knots=KNOTS_PER_AXIS,
    )
    carrier_seconds = perf_counter() - carrier_started
    initial_exact = root + 0.02 * (seed - root)
    initial_plasma = initial_exact - known_external
    root_plasma = root - known_external
    initial_coefficients = carrier.project(initial_plasma)

    support_started = perf_counter()
    support_geometry = _support_geometry(machine, coordinate)
    support_seconds = perf_counter() - support_started
    second_support_map = _quadratic_support_map(operator, support_geometry)

    first_result, first_wall = _solve_arm(
        first_support_map,
        operator,
        carrier,
        initial_coefficients,
        known_external,
    )
    second_result, second_wall = _solve_arm(
        second_support_map,
        operator,
        carrier,
        initial_coefficients,
        known_external,
    )

    exact_route = select_fixed_point_map(
        IterateRoute.EXACT_VALUES,
        first_support_map,
        carrier=carrier,
        external=known_external,
    )
    np.testing.assert_array_equal(
        np.asarray(exact_route(jnp.asarray(root))),
        np.asarray(first_support_map(jnp.asarray(root))),
    )
    first_coefficient_map = coefficient_fixed_point_map(
        first_support_map, carrier, external=known_external
    )[0]
    second_coefficient_map = coefficient_fixed_point_map(
        second_support_map, carrier, external=known_external
    )[0]
    exact_per_frame = _timed(exact_route, jnp.asarray(root))
    root_coefficients = carrier.project(root_plasma)
    first_per_frame = _timed(first_coefficient_map, root_coefficients)
    second_per_frame = _timed(second_coefficient_map, root_coefficients)

    exact_batch = jax.vmap(exact_route)
    first_batch = jax.vmap(first_coefficient_map)
    second_batch = jax.vmap(second_coefficient_map)
    exact_per_batch = _timed(
        exact_batch, jnp.broadcast_to(jnp.asarray(root), (BATCH_SIZE, root.size))
    )
    coefficient_batch = jnp.broadcast_to(
        root_coefficients, (BATCH_SIZE, carrier.coefficient_count)
    )
    first_per_batch = _timed(first_batch, coefficient_batch)
    second_per_batch = _timed(second_batch, coefficient_batch)
    allocation_peak_memory = _cgroup_peak_memory()

    represented_plasma = np.asarray(carrier.expand(root_coefficients))
    grid_count = len(machine.node)
    projection_floor = _regional_error(
        root_plasma[:grid_count], represented_plasma[:grid_count], root_psi_norm
    )
    banked_total_projection_floor = total_carrier_arm["projection_floor"]
    banked_total_carrier_residual = float(
        total_carrier_arm["terminal_exact_field_residual"]
    )
    projection_ladder, first_reaching_banked_carrier = _projection_ladder(
        coordinate,
        root_plasma,
        grid_count,
        root_psi_norm,
        banked_total_carrier_residual,
    )

    banked_residual = float(
        root_receipt["metric"]["fixed_point_residual"]["recovery_value"]
    )
    banked_history = root_receipt["direct_attempt"]
    baseline_output = np.asarray(first_support_map(jnp.asarray(root)))
    baseline = {
        "iterate": "exact psi values",
        "support_order": "monopole plus two first moments",
        "terminal_exact_field_residual": banked_residual,
        "admitted_advance_count": int(banked_history["newton_steps_requested"]),
        "newton_step_equivalents": float(banked_history["first_criterion_newton_step"]),
        "terminal_topology": _topology(operator, root),
        "timing": {
            "banked_full_solve_seconds": float(banked_history["seconds"]),
            "per_iterate": exact_per_frame,
            "batched_frames": exact_per_batch,
            "batch_size": BATCH_SIZE,
            "one_time_build_seconds": 0.0,
            "jacobian_formation_seconds": 0.0,
            "linear_solve_seconds": float(banked_history["seconds"])
            / int(banked_history["newton_steps_requested"]),
            "linear_solve_qualification": (
                "upper bound per Newton step from the banked full solve; includes "
                "map evaluations and admission beside thirty-GMRES work"
            ),
        },
        "peak_device_memory_bytes": allocation_peak_memory,
        "peak_device_memory_qualification": (
            "shared CPU-allocation cgroup high-water mark after all three warm arms; "
            "an upper bound rather than an isolated-arm counter"
        ),
        "accuracy_against_banked_root": {
            name: {"count": value["count"], "relative_sup": 0.0}
            for name, value in projection_floor.items()
        },
        "projection_floor": None,
        "terminal_residual_normalisations": _residual_normalisations(
            baseline_output, root, known_external
        ),
    }

    def carrier_record(result, wall, per_frame, per_batch, order):
        exact_output = np.asarray(result.exact_output)
        residual_normalisations = _residual_normalisations(
            exact_output, result.exact_state, known_external
        )
        return {
            "iterate": "six-by-six plasma-only spline knot values",
            "coefficient_count": carrier.coefficient_count,
            "support_order": order,
            "terminal_exact_field_residual": residual_normalisations[
                "normalised_by_total_flux_peak"
            ],
            "terminal_residual_normalisations": residual_normalisations,
            "admitted_advance_count": result.admitted_advances,
            "newton_step_equivalents": result.newton_step_equivalents,
            "terminal_topology": _topology(operator, exact_output),
            "timing": {
                "full_solve_seconds": wall,
                "per_iterate": per_frame,
                "batched_frames": per_batch,
                "batch_size": BATCH_SIZE,
                "one_time_build_seconds": carrier_seconds,
                "jacobian_formation_seconds": result.jacobian_seconds,
                "linear_solve_seconds": result.solve_seconds,
            },
            "peak_device_memory_bytes": allocation_peak_memory,
            "peak_device_memory_qualification": (
                "shared CPU-allocation cgroup high-water mark after all three warm "
                "arms; an upper bound rather than an isolated-arm counter"
            ),
            "accuracy_against_banked_root": _regional_error(
                root[:grid_count], exact_output[:grid_count], root_psi_norm
            ),
            "projection_floor": projection_floor,
            "banked_total_flux_projection_floor": banked_total_projection_floor,
            "residual_trace": np.asarray(result.trace).tolist(),
        }

    first_record = carrier_record(
        first_result,
        first_wall,
        first_per_frame,
        first_per_batch,
        "monopole plus two first moments",
    )
    second_record = carrier_record(
        second_result,
        second_wall,
        second_per_frame,
        second_per_batch,
        "adds three quadratic moments",
    )
    second_record["timing"]["support_operator_one_time_build_seconds"] = support_seconds
    second_record["support_operator_qualification"] = (
        "six fixed density functions; degree-fifteen moving-support moments; "
        f"order-{SUPPORT_BUILD_QUADRATURE_ORDER} parent-cell weighted Green build"
    )

    arms = {"A": baseline, "C": first_record, "D": second_record}
    for name in ("C", "D"):
        arm = arms[name]
        accuracy = max(
            value["relative_sup"]
            for value in arm["accuracy_against_banked_root"].values()
            if value["relative_sup"] is not None
        )
        single_speed_improves = (
            arm["timing"]["per_iterate"]["median_seconds"]
            < exact_per_frame["median_seconds"]
        )
        batch_speed_improves = (
            arm["timing"]["batched_frames"]["median_seconds"] / BATCH_SIZE
            < exact_per_batch["median_seconds"] / BATCH_SIZE
        )
        arm["verdict_against_A"] = {
            "convergence": {
                "improves": arm["terminal_exact_field_residual"] < banked_residual,
                "statement": (
                    "lower terminal exact-field residual"
                    if arm["terminal_exact_field_residual"] < banked_residual
                    else "does not beat the banked exact-value terminal residual"
                ),
            },
            "speed": {
                "improves": single_speed_improves and batch_speed_improves,
                "single_frame_improves": single_speed_improves,
                "batched_frame_improves": batch_speed_improves,
                "statement": (
                    "improves both warm single-frame and per-batched-frame wall time"
                    if single_speed_improves and batch_speed_improves
                    else (
                        "improves warm per-batched-frame wall time but not the "
                        "single-frame wall time"
                        if batch_speed_improves
                        else (
                            "does not beat exact-value warm single or batched wall time"
                        )
                    )
                ),
            },
            "accuracy": {
                "improves": accuracy < 0.0,
                "statement": (
                    "cannot improve on the banked root's zero self-error; "
                    f"worst regional relative-sup error is {accuracy:.6g}"
                ),
            },
        }

    return {
        "artifact": str(OUTPUT),
        "schema": "plasma-only-carrier-comparison-1",
        "source_revision": source_revision,
        "measurement_scope": {
            "frame_set": ["closed-form-oracle-coarse"],
            "common_warm_cache": {
                "semantic_key": machine.cache["semantic_key"],
                "hit": bool(machine.cache["hit"]),
                "load_seconds": machine_seconds,
                "realised_plasma_cells": grid_count,
                "state_values": root.size,
            },
            "platform": jax.devices()[0].platform,
            "precision": "float64",
            "exact_output_contract": (
                "all residual, topology, domain and profile reads use the ordinary "
                "total-flux Green output; coefficients carry only plasma flux"
            ),
            "carrier_state_contract": (
                "u = psi_total - psi_external; u_next = "
                "map(u + psi_external) - psi_external"
            ),
            "banked_root": str(ROOT_BANK),
            "banked_root_sha256": _sha256(ROOT_BANK),
        },
        "build_costs_seconds": {
            "warm_machine_load": machine_seconds,
            "coefficient_carrier": carrier_seconds,
            "quadratic_support_operator": support_seconds,
        },
        "evidence_inputs": {
            "representation_degree_receipt": str(DEGREE_RECEIPT),
            "representation_degree_sha256": _sha256(DEGREE_RECEIPT),
            "second_order_kernel_receipt": str(SUPPORT_RECEIPT),
            "second_order_kernel_sha256": _sha256(SUPPORT_RECEIPT),
            "root_receipt": str(ROOT_RECEIPT),
            "root_receipt_sha256": _sha256(ROOT_RECEIPT),
            "banked_total_carrier_receipt": str(TOTAL_CARRIER_RECEIPT),
            "banked_total_carrier_receipt_sha256": _sha256(TOTAL_CARRIER_RECEIPT),
        },
        "exact_value_route_retained": {
            "route": IterateRoute.EXACT_VALUES.value,
            "call_site_selection": True,
            "coefficient_route": IterateRoute.COEFFICIENT_CARRIER.value,
            "asserted_equal_to_direct_map": True,
            "global_switch_introduced": False,
        },
        "normalisation_contract": {
            "reported_terminal_residual": (
                "absolute residual sup divided by total-flux output peak"
            ),
            "secondary_terminal_residual": (
                "absolute residual sup divided by plasma-only output peak"
            ),
            "both_denominators_reported_per_arm": True,
        },
        "projection_floor_comparison": {
            "unchanged_knots_per_axis": KNOTS_PER_AXIS,
            "coefficient_count": carrier.coefficient_count,
            "plasma_only": projection_floor,
            "banked_total_flux": banked_total_projection_floor,
        },
        "knot_threshold": {
            "criterion": (
                "worst regional plasma-only projection floor no greater than "
                "the banked total-flux carrier terminal residual"
            ),
            "banked_total_flux_carrier_terminal_residual": (
                banked_total_carrier_residual
            ),
            "first_reaching": first_reaching_banked_carrier,
            "ladder": projection_ladder,
        },
        "arms": arms,
        "interaction": {
            "statement": (
                "Only the carrier lever and its combination with quadratic support "
                "are present in this receipt; the independently measured exact-value "
                "support-order arm is required before additive interaction can be "
                "adjudicated."
            ),
            "classification": "pending independent support-order receipt",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    receipt = measure()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "arms": receipt["arms"]}, indent=2))


if __name__ == "__main__":
    main()
