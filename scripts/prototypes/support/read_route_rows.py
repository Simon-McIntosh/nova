"""Measure the support read route: booked-moment and map-flux identity, step cost.

One row reads a closed-form state on a realised hex carrier through
``nova.equilibrium.support_read``: the batched fixed-point solve, Bell-only
membership at those points and ``current.book``. The booked moments are
compared with the exact analytic support, and their flux image at every
carrier node with the closed form. Cost rows time each part of one solve step
on production-shaped operands (one source per cell, a dense nodal row array
and a 512-node row window), whose values are placeholders because the
contraction and the solve are data independent in cost.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, fixed_points as fixed, support_read, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures.measure import analytic_profile
from scripts.prototypes.support.geometry import analytic_moments, analytic_polygons
from tests.equilibrium import test_topology_read as fixture

WINDOW = 512
CALLS = 20


def _median_seconds(function, *operands):
    jax.block_until_ready(function(*operands))
    seconds = []
    for _ in range(CALLS):
        start = perf_counter()
        jax.block_until_ready(function(*operands))
        seconds.append(perf_counter() - start)
    return float(np.median(seconds)), float(np.percentile(seconds, 99))


def _window(nodes, mesh, jets, source_cells, previous, margin):
    """Row window around the previous points with placeholder rows."""
    triangles = np.asarray(mesh.triangles)
    generators = np.asarray(mesh.generators)
    distance = np.min(
        np.linalg.norm(generators[:, None] - np.asarray(previous)[None], axis=-1),
        axis=1,
    )
    radius = np.max(
        np.linalg.norm(
            nodes[triangles] - generators[:, None, None],
            axis=-1,
        ),
        axis=(1, 2),
    )
    chosen = np.unique(triangles[distance <= margin + radius])
    assert 0 < len(chosen) <= WINDOW, len(chosen)
    lookup = np.full(len(nodes), -1, dtype=np.int32)
    lookup[chosen] = np.arange(len(chosen))
    exterior = np.zeros((WINDOW, 6))
    exterior[: len(chosen)] = np.asarray(jets)[chosen]
    return fixed.RowCache(
        mesh,
        jnp.asarray(nodes),
        jnp.asarray(lookup),
        jnp.zeros((WINDOW, source_cells, 3, 6)),
        jnp.asarray(exterior),
    )


@dataclass
class Case:
    kind: str
    cells: int
    geometry: object
    roots: jax.Array
    pitch: jax.Array
    analytic: object
    nodes: np.ndarray
    sample_index: np.ndarray
    jets: jax.Array
    convention: object
    policy: object
    profile: object
    quadrature: object


def _case(kind, cells):
    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs(kind)
    geometry = fixture._realised_hex_geometry(wall, cells)
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    jets = fixed.stack_jet(jax.jit(jax.vmap(analytic.evaluate))(jnp.asarray(nodes)))
    name = (
        "diverted-single-null" if kind == "diverted" else "weak-rotation-reactor-static"
    )
    _, source, exact = certificate._case(name, clip_mode="exact")
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = (
        TopologyPolicy(normal_form_radius=0.054806712567833996)
        if kind == "diverted"
        else TopologyPolicy()
    )
    case = Case(
        kind,
        cells,
        geometry,
        jnp.asarray([axis] if saddle is None else [axis, saddle]),
        jnp.median(geometry.pitch),
        analytic,
        nodes,
        inverse.reshape(samples.shape[:-1]),
        jets,
        convention,
        policy,
        analytic_profile(source),
        current.Quadrature(sigma=convention.sigma, segments=64),
    )
    return case, source, exact


def _identity(case, source, exact):
    geometry = case.geometry
    operands = support_read.SupportOperands.from_rows(
        geometry,
        jnp.zeros((len(case.nodes), 1, 3, 6)),
        case.jets,
        case.nodes,
        case.sample_index,
    )
    cache = _window(
        case.nodes, operands.mesh, case.jets, 1, case.roots, 1.6 * float(case.pitch)
    )
    moments = (jnp.zeros(1), jnp.zeros(1), jnp.zeros(1))
    psi = jax.vmap(jax.vmap(case.analytic.value))(geometry.sample_points)

    @jax.jit
    def step(operands, cache, seeds, previous, psi):
        return support_read.read_and_book(
            operands,
            cache,
            moments,
            seeds,
            previous,
            case.convention,
            case.policy,
            case.profile,
            case.quadrature,
            psi=psi,
        )

    seeds = case.roots + 0.03 * case.pitch
    start = perf_counter()
    support, booked = step(operands, cache, seeds, case.roots, psi)
    jax.block_until_ready(booked)
    compile_seconds = perf_counter() - start
    polygons = [
        np.asarray(v)[: int(n)]
        for v, n in zip(geometry.vertices, geometry.vertex_count, strict=True)
    ]
    reference = analytic_polygons(
        exact,
        polygons,
        points=4097 if case.kind == "diverted" else 16385,
    )
    truth = analytic_moments(source, reference, np.asarray(geometry.centre))
    booked_values = np.stack([np.asarray(value) for value in booked[:3]])
    truth_values = np.stack([np.asarray(value) for value in truth])
    scale = float(np.max(np.abs(truth_values)))
    relative_max = float(np.max(np.abs(booked_values - truth_values)) / scale)
    coupling = topology.BiotMomentCoupling.from_polygons(polygons)
    delta = tuple(jnp.asarray(booked_values[i] - truth_values[i]) for i in range(3))
    flux = jax.jit(
        lambda points, coupling, delta: jax.lax.map(
            lambda point: coupling.value_gradient(point, delta)[0],
            points,
            batch_size=64,
        )
    )(jnp.asarray(case.nodes), coupling, delta)
    span = abs(float(support.reading.axis_flux) - float(support.reading.boundary_flux))
    map_flux_error = float(np.max(np.abs(np.asarray(flux))) / span)
    reading = support.reading
    return dict(
        realised_cells=len(geometry.centre),
        nodes=len(case.nodes),
        read_valid=bool(reading.valid),
        qualified=bool(reading.qualified),
        support_valid=bool(support.valid),
        booked_valid=bool(booked.valid),
        status=np.asarray(support.stationary.status).tolist(),
        iterations=np.asarray(support.stationary.iterations).tolist(),
        retried=np.asarray(support.stationary.retried).tolist(),
        boundary_class=int(reading.boundary_class),
        axis=np.asarray(reading.axis).tolist(),
        axis_flux=float(reading.axis_flux),
        boundary_flux=float(reading.boundary_flux),
        x_points=np.asarray(reading.x_points)[
            np.asarray(reading.x_point_valid)
        ].tolist(),
        booked_moment_relative_max=relative_max,
        map_flux_error=map_flux_error,
        booked_current=float(np.sum(booked_values[0])),
        analytic_current=float(np.sum(truth_values[0])),
        first_step_seconds=compile_seconds,
    ), (operands, cache, support, booked, psi, step)


def _cost(case, operands_identity):
    geometry = case.geometry
    cells = len(geometry.centre)
    nodes = len(case.nodes)
    operands = support_read.SupportOperands.from_rows(
        geometry,
        jnp.zeros((nodes, cells, 3, 6)),
        case.jets,
        case.nodes,
        case.sample_index,
    )
    cache = _window(
        case.nodes, operands.mesh, case.jets, cells, case.roots, 1.6 * float(case.pitch)
    )
    moments = tuple(jnp.full(cells, 1e-3 * (index + 1)) for index in range(3))
    psi = jax.vmap(jax.vmap(case.analytic.value))(geometry.sample_points)
    seeds = case.roots + 0.03 * case.pitch

    fixed_program = jax.jit(
        lambda cache, moments, seeds, previous: fixed.solve(
            cache,
            jnp.stack(tuple(moments), axis=-1),
            seeds,
            previous,
            case.pitch,
        )
    )
    membership_program = jax.jit(
        lambda operands, moments, solved: topology.read_at_stationary_points(
            fixed.field_from_jets(
                support_read.nodal_jets(operands, moments),
                operands.sample_index,
                operands.mesh,
            ),
            operands.geometry,
            case.convention,
            case.policy,
            solved,
        )
    )
    book_program = jax.jit(
        lambda operands, reading, psi: current.book(
            psi, reading, operands.geometry, case.profile, case.quadrature
        )
    )
    whole_program = jax.jit(
        lambda operands, cache, moments, seeds, previous, psi: (
            support_read.read_and_book(
                operands,
                cache,
                moments,
                seeds,
                previous,
                case.convention,
                case.policy,
                case.profile,
                case.quadrature,
                psi=psi,
            )
        )
    )
    solved = fixed_program(cache, moments, seeds, case.roots)
    reading = membership_program(operands, moments, solved)
    jax.block_until_ready(reading)
    fixed_median, fixed_p99 = _median_seconds(
        fixed_program, cache, moments, seeds, case.roots
    )
    member_median, member_p99 = _median_seconds(
        membership_program, operands, moments, solved
    )
    book_median, book_p99 = _median_seconds(book_program, operands, reading, psi)
    whole_median, whole_p99 = _median_seconds(
        whole_program, operands, cache, moments, seeds, case.roots, psi
    )
    return dict(
        cost_cells=cells,
        cost_nodes=nodes,
        dense_row_bytes=int(nodes * cells * 18 * 8),
        window_row_bytes=int(WINDOW * cells * 18 * 8),
        fixed_points_median_seconds=fixed_median,
        fixed_points_p99_seconds=fixed_p99,
        membership_median_seconds=member_median,
        membership_p99_seconds=member_p99,
        booking_median_seconds=book_median,
        booking_p99_seconds=book_p99,
        whole_step_median_seconds=whole_median,
        whole_step_p99_seconds=whole_p99,
        calls=CALLS,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cost-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"REVISION={revision} TREE={Path.cwd().resolve()} COMMAND={sys.executable} "
        + " ".join(sys.argv),
        flush=True,
    )
    print(
        f"NOVA_MODULE={nova.__file__} SUPPORT_READ={support_read.__file__} "
        f"CWD={Path.cwd().resolve()} DEVICES={jax.devices()}",
        flush=True,
    )
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    case, source, exact = _case(args.kind, args.cells)
    row = dict(kind=args.kind, requested_cells=args.cells, revision=revision)
    identity = None
    if not args.cost_only:
        identity, _ = _identity(case, source, exact)
        row.update(identity)
        print("IDENTITY " + json.dumps(row, sort_keys=True), flush=True)
        (args.out / f"{args.kind}-{args.cells}.json").write_text(
            json.dumps(row, indent=2) + "\n"
        )
    row.update(_cost(case, identity))
    row["completed"] = True
    (args.out / f"{args.kind}-{args.cells}.json").write_text(
        json.dumps(row, indent=2) + "\n"
    )
    print("ROW " + json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
