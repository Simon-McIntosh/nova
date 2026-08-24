"""Compare first- and second-order in-cell current support.

The receipt uses the following output-data labels: ``A`` is the shipped
constant-plus-linear support and ``B`` adds the three quadratic support rows.
Both retain the exact flux value at every solve node as the iterated state.

Each measurement runs in a fresh process.  That makes accelerator peak-memory
statistics independent while keeping both routes on the same semantic machine
cache, exact exterior field, source, seed, topology admission rule and Newton
budget.  One-time carrier, response and compilation costs are reported apart
from compiled solve time and warmed map-evaluation time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/coefficient-space-newton/support-order-arms.json"
ROOT_BANK = ROOT / "scripts/oracle_rebaseline"
CARRIERS = ("coarse", "fine")
SUPPORT_COLUMNS = {"A": 3, "B": 6}
REGIONS = ("closed_flux_region", "separatrix_band", "scrape_off_layer")
SEPARATRIX_HALF_WIDTH = 0.05
NEWTON_STEPS = 10
KRYLOV_ITERATIONS = 18
NONMONOTONE_ALLOWANCE = 0.05
MAP_TIMING_REPEATS = 7
RESPONSE_QUADRATURE_ORDER = 8
CURRENT_QUADRATURE_ORDER = 8
RESPONSE_CELL_CHUNK = 8


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_fixture_module():
    from scripts.analytic_oracle_fixtures import measure

    return measure


def _carrier_state(carrier: str) -> dict[str, np.ndarray]:
    path = ROOT_BANK / f"root-{carrier}.npz"
    with np.load(path, allow_pickle=False) as bank:
        return {name: np.asarray(bank[name]) for name in bank.files}


def _full_coordinates(machine) -> np.ndarray:
    return np.vstack((machine.node, machine.wall_node, machine.sample_coordinates))


def _padded_polygons(polygons: tuple[np.ndarray, ...]) -> np.ndarray:
    capacity = max(len(polygon) for polygon in polygons)
    padded = np.empty((len(polygons), capacity, 2), dtype=np.float64)
    for index, polygon in enumerate(polygons):
        count = len(polygon)
        padded[index, :count] = polygon
        padded[index, count:] = polygon[-1]
    return padded


def _cell_scales(polygons: tuple[np.ndarray, ...], centres: np.ndarray) -> np.ndarray:
    scales = np.asarray(
        [
            np.max(np.abs(np.asarray(polygon) - centre), axis=0)
            for polygon, centre in zip(polygons, centres, strict=True)
        ],
        dtype=np.float64,
    )
    if np.any(scales <= 0.0):
        raise RuntimeError("a source cell has a zero coordinate scale")
    return scales


def _normalised_basis(local: np.ndarray) -> np.ndarray:
    radial = local[..., 0]
    vertical = local[..., 1]
    return np.stack(
        (
            np.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        ),
        axis=-1,
    )


def _polygon_rule(vertices: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    node, weight = np.polynomial.legendre.leggauss(order)
    node = 0.5 * (node + 1.0)
    weight = 0.5 * weight
    points: list[np.ndarray] = []
    weights: list[float] = []
    for index in range(1, len(vertices) - 1):
        first, second, third = vertices[[0, index, index + 1]]
        first_edge = second - first
        second_edge = third - first
        determinant = abs(
            first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
        )
        for radial, radial_weight in zip(node, weight, strict=True):
            for vertical, vertical_weight in zip(node, weight, strict=True):
                points.append(
                    first
                    + radial * first_edge
                    + (1.0 - radial) * vertical * second_edge
                )
                weights.append(
                    determinant * (1.0 - radial) * radial_weight * vertical_weight
                )
    return np.asarray(points), np.asarray(weights)


def _basis_gram_inverse(
    polygons: tuple[np.ndarray, ...], centres: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    inverses = []
    for polygon, centre, scale in zip(polygons, centres, scales, strict=True):
        points, weights = _polygon_rule(polygon, RESPONSE_QUADRATURE_ORDER)
        basis = _normalised_basis((points - centre) / scale)
        gram = np.einsum("qi,qj,q->ij", basis, basis, weights)
        inverses.append(np.linalg.inv(gram))
    return np.asarray(inverses)


def _quadratic_response(machine, targets: np.ndarray) -> tuple[np.ndarray, float]:
    import jax
    import jax.numpy as jnp

    from nova.biot.second_moment_kernel import flux_density_columns

    started = perf_counter()
    polygons = _padded_polygons(machine.cell_polygons)
    centres = np.asarray(machine.node, dtype=np.float64)
    scales = _cell_scales(machine.cell_polygons, centres)
    target_r = jnp.asarray(targets[:, 0])
    target_z = jnp.asarray(targets[:, 1])

    def one_cell(vertices, centre):
        return flux_density_columns(
            jnp,
            target_r,
            target_z,
            vertices,
            expansion_point=centre,
            order=RESPONSE_QUADRATURE_ORDER,
            columns=6,
        )

    compiled = jax.jit(jax.vmap(one_cell))
    response = np.empty((len(targets), len(polygons), 6), dtype=np.float64)
    for start in range(0, len(polygons), RESPONSE_CELL_CHUNK):
        stop = min(start + RESPONSE_CELL_CHUNK, len(polygons))
        count = stop - start
        polygon_chunk = polygons[start:stop]
        centre_chunk = centres[start:stop]
        if count < RESPONSE_CELL_CHUNK:
            pad = RESPONSE_CELL_CHUNK - count
            polygon_chunk = np.concatenate(
                (polygon_chunk, np.repeat(polygon_chunk[-1:], pad, axis=0))
            )
            centre_chunk = np.concatenate(
                (centre_chunk, np.repeat(centre_chunk[-1:], pad, axis=0))
            )
        columns = np.asarray(
            jax.block_until_ready(
                compiled(jnp.asarray(polygon_chunk), jnp.asarray(centre_chunk))
            )
        )[:count]
        response[:, start:stop] = np.transpose(columns, (1, 0, 2))
    divisor = np.stack(
        (
            np.ones(len(scales)),
            scales[:, 0],
            scales[:, 1],
            scales[:, 0] ** 2,
            scales[:, 0] * scales[:, 1],
            scales[:, 1] ** 2,
        ),
        axis=1,
    )
    response /= divisor[None, :, :]
    response *= np.asarray(machine.area)[None, :, None]
    return response, perf_counter() - started


def _support_quadrature(support, order: int):
    import jax.numpy as jnp

    node, weight = np.polynomial.legendre.leggauss(order)
    node = jnp.asarray(0.5 * (node + 1.0))
    weight = jnp.asarray(0.5 * weight)
    vertices = jnp.asarray(support.support_vertices)
    count = jnp.asarray(support.vertex_count)
    capacity = vertices.shape[1]
    triangle_slot = jnp.arange(1, capacity - 1)
    first = jnp.broadcast_to(vertices[:, :1], (len(vertices), capacity - 2, 2))
    second = vertices[:, triangle_slot]
    third = vertices[:, triangle_slot + 1]
    radial, vertical = jnp.meshgrid(node, node, indexing="ij")
    radial_weight, vertical_weight = jnp.meshgrid(weight, weight, indexing="ij")
    radial = radial.reshape(-1)
    vertical = vertical.reshape(-1)
    rule_weight = (radial_weight * vertical_weight).reshape(-1)
    first_edge = second - first
    second_edge = third - first
    points = (
        first[:, :, None, :]
        + radial[None, None, :, None] * first_edge[:, :, None, :]
        + (1.0 - radial)[None, None, :, None]
        * vertical[None, None, :, None]
        * second_edge[:, :, None, :]
    )
    determinant = jnp.abs(
        first_edge[..., 0] * second_edge[..., 1]
        - first_edge[..., 1] * second_edge[..., 0]
    )
    live = (triangle_slot[None, :] + 1 < count[:, None]) & (count[:, None] >= 3)
    weights = (
        determinant[:, :, None]
        * (1.0 - radial)[None, None, :]
        * rule_weight[None, None, :]
    )
    weights = jnp.where(live[:, :, None], weights, 0.0)
    points = points.reshape(len(vertices), -1, 2)
    weights = weights.reshape(len(vertices), -1)
    points = jnp.where(
        (weights > 0.0)[..., None], points, jnp.asarray(support.centroids)[:, None]
    )
    return points, weights


def _quadratic_map(operator, machine, response: np.ndarray) -> Callable:
    import jax.numpy as jnp

    centres = jnp.asarray(machine.node)
    scales = jnp.asarray(_cell_scales(machine.cell_polygons, np.asarray(machine.node)))
    inverse = jnp.asarray(
        _basis_gram_inverse(
            machine.cell_polygons,
            np.asarray(machine.node),
            np.asarray(scales),
        )
    )
    response_array = jnp.asarray(response)
    exterior = jnp.asarray(operator.external())

    def mapped(state):
        partition = operator._support_partition(state)
        masks, _topology, sample_flux, core_support, _common_support = partition
        points, weights = _support_quadrature(core_support, CURRENT_QUADRATURE_ORDER)
        psi_norm, _radial, _vertical = operator.sample_flux_field(
            masks.psi_norm, sample_flux, points
        )
        density = operator.source.core.current_density(points[..., 0], psi_norm)
        local = (points - centres[:, None, :]) / scales[:, None, :]
        radial = local[..., 0]
        vertical = local[..., 1]
        basis = jnp.stack(
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
        physical_moments = jnp.einsum("cq,cqk,cq->ck", density, basis, weights)
        coefficients = jnp.einsum("cij,cj->ci", inverse, physical_moments)
        return exterior + jnp.einsum("tck,ck->t", response_array, coefficients)

    return mapped


def _relative_residual(mapped: np.ndarray, state: np.ndarray) -> float:
    return float(
        np.max(np.abs(mapped - state))
        / max(np.max(np.abs(mapped)), np.finfo(float).tiny)
    )


def _regional_errors(
    state: np.ndarray, reference: np.ndarray, psi_norm: np.ndarray, cell_count: int
) -> dict[str, dict[str, float | int]]:
    state = state[:cell_count]
    reference = reference[:cell_count]
    span = float(np.ptp(reference))
    masks = {
        "closed_flux_region": psi_norm < 1.0 - SEPARATRIX_HALF_WIDTH,
        "separatrix_band": np.abs(psi_norm - 1.0) <= SEPARATRIX_HALF_WIDTH,
        "scrape_off_layer": psi_norm > 1.0 + SEPARATRIX_HALF_WIDTH,
    }
    absolute = np.abs(state - reference)
    result = {}
    for name, mask in masks.items():
        if not np.any(mask):
            raise RuntimeError(f"the {name} region has no carrier cells")
        result[name] = {
            "cell_count": int(np.count_nonzero(mask)),
            "relative_sup_error": float(np.max(absolute[mask]) / span),
            "relative_rms_error": float(np.sqrt(np.mean(absolute[mask] ** 2)) / span),
        }
    return result


def _topology_record(operator, state: np.ndarray) -> dict[str, Any]:
    import jax.numpy as jnp

    _masks, topology = operator.read(jnp.asarray(state))
    x_point = np.asarray(topology.x_point, dtype=float)
    return {
        "class": "diverted" if bool(topology.diverted) else "limited",
        "x_point_finite": bool(np.all(np.isfinite(x_point))),
        "x_point_rz_m": x_point.tolist() if np.all(np.isfinite(x_point)) else None,
    }


def _time_map(compiled, state) -> dict[str, Any]:
    import jax

    compiled(state).block_until_ready()
    samples = []
    for _ in range(MAP_TIMING_REPEATS):
        started = perf_counter()
        compiled(state).block_until_ready()
        samples.append(perf_counter() - started)
    samples.sort()
    return {
        "median_seconds": samples[len(samples) // 2],
        "minimum_seconds": samples[0],
        "maximum_seconds": samples[-1],
        "repeats": MAP_TIMING_REPEATS,
        "synchronisation": "every device call blocked to completion",
        "device": str(jax.devices()[0]),
    }


def _measure(carrier: str, label: str) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.fixed_point import kink_aware_newton_krylov
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    fixture = _load_fixture_module()
    bank_path = ROOT_BANK / f"root-{carrier}.npz"
    bank = _carrier_state(carrier)
    case = fixture.analytic_case()

    carrier_started = perf_counter()
    machine = fixture.cached_machine(
        case, fixture.FIXTURE_REQUESTS[carrier], wall_nodes=fixture.WALL_POINT_COUNT
    )
    carrier_seconds = perf_counter() - carrier_started
    coordinates = _full_coordinates(machine)
    oracle = fixture.exact_state(case, coordinates)
    empty_operator = fixture.forward_operator(case, machine)
    exact_physical = fixture.exact_current_moments(case, empty_operator, oracle)
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = fixture._internal_flux_image(empty_operator, exact_coefficients)
    prescribed_exterior = oracle - exact_internal
    operator = fixture.forward_operator(case, machine, prescribed_exterior)

    response_seconds = 0.0
    if SUPPORT_COLUMNS[label] == 3:
        map_fn = operator.flux_map()
    else:
        response, response_seconds = _quadratic_response(machine, coordinates)
        map_fn = _quadratic_map(operator, machine, response)

    seed = jnp.asarray(bank["seed_state"])
    reference = np.asarray(bank["root_state"], dtype=np.float64)
    expected_topology = _topology_record(operator, reference)["class"]

    def admissible(candidate):
        _masks, topology = operator.read(candidate)
        class_matches = (
            topology.diverted
            if expected_topology == "diverted"
            else jnp.logical_not(topology.diverted)
        )
        return jnp.all(jnp.isfinite(candidate)) & class_matches

    map_compile_started = perf_counter()
    compiled_map = jax.jit(map_fn).lower(seed).compile()
    map_compile_seconds = perf_counter() - map_compile_started
    map_timing = _time_map(compiled_map, seed)

    def solve(initial):
        return kink_aware_newton_krylov(
            map_fn,
            initial,
            strategy="nonmonotone",
            newton_steps=NEWTON_STEPS,
            gmres_iterations=KRYLOV_ITERATIONS,
            warmup=0,
            admissibility_fn=admissible,
            nonmonotone_allowance=NONMONOTONE_ALLOWANCE,
        )

    solve_compile_started = perf_counter()
    compiled_solve = jax.jit(solve).lower(seed).compile()
    solve_compile_seconds = perf_counter() - solve_compile_started
    solve_started = perf_counter()
    solved = compiled_solve(seed)
    jax.block_until_ready(solved)
    solve_seconds = perf_counter() - solve_started
    terminal = np.asarray(solved.state, dtype=np.float64)
    mapped = np.asarray(compiled_map(jnp.asarray(terminal)), dtype=np.float64)
    accepted_factors = np.asarray(solved.accepted_factors, dtype=np.float64)
    effective_fractions = np.asarray(
        solved.effective_newton_fractions, dtype=np.float64
    )
    memory = jax.devices()[0].memory_stats() or {}
    topology = _topology_record(operator, terminal)
    peak = int(memory.get("peak_bytes_in_use", memory.get("bytes_in_use", 0)))
    return {
        "label": label,
        "support": {
            "order": "first" if SUPPORT_COLUMNS[label] == 3 else "second",
            "columns_per_cell": SUPPORT_COLUMNS[label],
            "iterate_carrier": "exact_flux_values",
        },
        "carrier": {
            "name": carrier,
            "requested_cells": int(fixture.FIXTURE_REQUESTS[carrier]),
            "realised_cells": len(machine.node),
            "state_dimension": len(seed),
            "semantic_cache_key": machine.cache["semantic_key"],
            "cache_hit": bool(machine.cache["hit"]),
            "banked_root": str(bank_path.relative_to(ROOT)),
            "banked_root_sha256": _sha256(bank_path),
        },
        "convergence": {
            "terminal_exact_field_relative_residual": _relative_residual(
                mapped, terminal
            ),
            "admitted_advance_count": int(np.count_nonzero(accepted_factors > 0.0)),
            "achieved_newton_step_equivalents": float(np.sum(effective_fractions)),
            "accepted_factors": accepted_factors.tolist(),
            "newton_steps": NEWTON_STEPS,
            "gmres_iterations": KRYLOV_ITERATIONS,
            "terminal_topology_class": topology["class"],
            "terminal_x_point_finite": topology["x_point_finite"],
            "terminal_x_point_rz_m": topology["x_point_rz_m"],
        },
        "speed": {
            "one_time_build_seconds": {
                "warm_carrier_load_and_operator": carrier_seconds,
                "support_response": response_seconds,
                "map_compile": map_compile_seconds,
                "solver_compile": solve_compile_seconds,
                "total": (
                    carrier_seconds
                    + response_seconds
                    + map_compile_seconds
                    + solve_compile_seconds
                ),
            },
            "compiled_solve_wall_clock_seconds_per_frame": solve_seconds,
            "warmed_map_wall_clock_seconds_per_iterate": map_timing,
            "peak_device_memory_bytes": peak,
            "peak_device_memory_gib": peak / 2**30,
        },
        "accuracy_against_banked_converged_root": {
            "normalisation": "full banked grid-root flux span",
            "region_partition": {
                "closed_flux_region": "banked psi_N < 0.95",
                "separatrix_band": "0.95 <= banked psi_N <= 1.05",
                "scrape_off_layer": "banked psi_N > 1.05",
            },
            "regions": _regional_errors(
                terminal,
                reference,
                np.asarray(bank["root_grid_psi_norm"], dtype=np.float64),
                len(machine.node),
            ),
        },
    }


def _axis_verdict(
    baseline: list[dict[str, Any]], measured: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    convergence_pairs = [
        (
            left["convergence"]["terminal_exact_field_relative_residual"],
            right["convergence"]["terminal_exact_field_relative_residual"],
        )
        for left, right in zip(baseline, measured, strict=True)
    ]
    topology_equal = all(
        left["convergence"]["terminal_topology_class"]
        == right["convergence"]["terminal_topology_class"]
        and left["convergence"]["terminal_x_point_finite"]
        == right["convergence"]["terminal_x_point_finite"]
        for left, right in zip(baseline, measured, strict=True)
    )
    convergence_better = all(right < left for left, right in convergence_pairs)
    convergence_mixed = any(right < left for left, right in convergence_pairs) and any(
        right >= left for left, right in convergence_pairs
    )

    speed_pairs = [
        (
            left["speed"]["warmed_map_wall_clock_seconds_per_iterate"][
                "median_seconds"
            ],
            right["speed"]["warmed_map_wall_clock_seconds_per_iterate"][
                "median_seconds"
            ],
        )
        for left, right in zip(baseline, measured, strict=True)
    ]
    memory_pairs = [
        (
            left["speed"]["peak_device_memory_bytes"],
            right["speed"]["peak_device_memory_bytes"],
        )
        for left, right in zip(baseline, measured, strict=True)
    ]
    speed_better = all(right < left for left, right in speed_pairs) and all(
        right <= left for left, right in memory_pairs
    )
    speed_mixed = any(right < left for left, right in speed_pairs) or any(
        right < left for left, right in memory_pairs
    )

    accuracy_pairs = []
    for left, right in zip(baseline, measured, strict=True):
        for region in REGIONS:
            accuracy_pairs.append(
                (
                    left["accuracy_against_banked_converged_root"]["regions"][region][
                        "relative_sup_error"
                    ],
                    right["accuracy_against_banked_converged_root"]["regions"][region][
                        "relative_sup_error"
                    ],
                )
            )
    accuracy_better = all(right < left for left, right in accuracy_pairs)
    accuracy_mixed = any(right < left for left, right in accuracy_pairs) and any(
        right >= left for left, right in accuracy_pairs
    )

    def record(improves: bool, mixed: bool, basis: str) -> dict[str, Any]:
        return {
            "arm_B_improves_on_arm_A": improves,
            "classification": (
                "improves" if improves else ("trade" if mixed else "does_not_improve")
            ),
            "basis": basis,
        }

    return {
        "convergence": record(
            convergence_better and topology_equal,
            convergence_mixed or not topology_equal,
            "all carrier residuals must fall strictly while terminal topology "
            "class and X-point finiteness remain unchanged",
        ),
        "speed": record(
            speed_better,
            speed_mixed,
            "all warmed per-iterate medians must fall and no carrier may use "
            "more peak device memory",
        ),
        "accuracy": record(
            accuracy_better,
            accuracy_mixed,
            "every carrier-region relative-sup error against its banked "
            "converged root must fall",
        ),
    }


def _aggregate(parts: list[Path], output: Path) -> dict[str, Any]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in parts]
    by_label = {
        label: [
            next(
                row
                for row in rows
                if row["label"] == label and row["carrier"]["name"] == carrier
            )
            for carrier in CARRIERS
        ]
        for label in SUPPORT_COLUMNS
    }
    verdict = _axis_verdict(by_label["A"], by_label["B"])
    receipt = {
        "schema": "nova.support-order-arms",
        "source_revision": _source_revision(),
        "comparison_contract": {
            "common_carrier_set": list(CARRIERS),
            "common_initial_state": "banked seed_state for each carrier",
            "common_banked_accuracy_root": (
                "banked converged root_state for each carrier"
            ),
            "common_source": "moderate-rotation-conventional analytic source",
            "common_exterior": (
                "exact field minus the shipped first-order exact-state current image"
            ),
            "common_solver": {
                "route": "topology-qualified nonmonotone Newton-Krylov",
                "newton_steps": NEWTON_STEPS,
                "gmres_iterations": KRYLOV_ITERATIONS,
                "nonmonotone_allowance": NONMONOTONE_ALLOWANCE,
            },
            "only_differing_mechanism": (
                "three first-order current-support columns versus six columns "
                "including RR, RZ and ZZ"
            ),
            "warm_cache_required": True,
            "all_cache_hits": all(row["carrier"]["cache_hit"] for row in rows),
        },
        "arms": by_label,
        "per_axis_verdict": verdict,
        "overall_statement": (
            "Arm B improves on all three axes."
            if all(item["arm_B_improves_on_arm_A"] for item in verdict.values())
            else (
                "Arm B is not an unqualified win; the per-axis verdicts retain "
                "every measured trade or loss."
            )
        ),
    }
    _write_json(output, receipt)
    return receipt


def run(output: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="nova-support-order-") as directory:
        work = Path(directory)
        parts = []
        for carrier in CARRIERS:
            for label in SUPPORT_COLUMNS:
                part = work / f"{carrier}-{label}.json"
                log = work / f"{carrier}-{label}.log"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "measure",
                    "--carrier",
                    carrier,
                    "--label",
                    label,
                    "--output",
                    str(part),
                ]
                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(ROOT)
                with log.open("w", encoding="utf-8") as stream:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{carrier} {label} failed; child log: "
                        f"{log.read_text(encoding='utf-8')}"
                    )
                parts.append(part)
        return _aggregate(parts, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--carrier", choices=CARRIERS, required=True)
    measure_parser.add_argument(
        "--label", choices=tuple(SUPPORT_COLUMNS), required=True
    )
    measure_parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.command == "measure":
        payload = _measure(arguments.carrier, arguments.label)
        _write_json(arguments.output, payload)
        print(
            json.dumps(
                {
                    "carrier": arguments.carrier,
                    "label": arguments.label,
                    "residual": payload["convergence"][
                        "terminal_exact_field_relative_residual"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    receipt = run(arguments.output)
    print(json.dumps(receipt["per_axis_verdict"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
