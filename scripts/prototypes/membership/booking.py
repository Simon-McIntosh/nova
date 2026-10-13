"""Compare level-set and local-curve membership through the current booking."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from functools import partial
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import nova
from shapely.geometry import GeometryCollection, LineString, Polygon

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, topology
from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures.measure import analytic_profile
from scripts.prototypes.fixed_points.solve import (
    CachedField,
    build_rows,
    select_cache_nodes,
    solve,
)
from scripts.prototypes.membership.hybrid import connected_support, neighbour_mask
from scripts.prototypes.read_route.bell import build_mesh, field_from_jets
from scripts.prototypes.read_route.check_element import stack_jet
from scripts.prototypes.support.geometry import analytic_moments, polygon_rings
from tests.equilibrium import test_topology_read as fixture

assert jax.config.jax_enable_x64 is True


class MembershipRead(NamedTuple):
    axis_flux: object
    boundary_flux: object
    membership: object
    fragment_area: object
    fragment_selected: object
    edge_interval: object
    field_coefficients: object
    saddle_form: object
    normal_form_cells: object
    valid: object
    qualified: object


def classify(jets, indices, mesh, geometry, fixed, level, sigma, *, curved, diverted):
    """Use the same Bell-sampled conic fragments, optionally replacing seven cells."""
    field = field_from_jets(jets, indices, mesh)
    values = jets[indices, 0]
    values = values.at[:, 0].set(
        jax.vmap(field.cell_value)(jnp.arange(len(values)), geometry.centre)
    )
    coefficient = jnp.einsum("nij,nj->ni", geometry.fit_inverse, values)
    form = None
    represented = jnp.zeros(len(values), dtype=bool)
    if curved and diverted:
        point = fixed.position[-1]
        third = jax.jacfwd(lambda target: field.evaluate(target).hessian)(point)
        form = topology.saddle_normal_form(
            point, sigma * fixed.jet.hessian[-1], sigma * third, 1e-12
        )
        represented = neighbour_mask(geometry, point)
    fractions, valid, selected, areas, intervals = connected_support(
        geometry, coefficient, sigma, level, fixed.position[0], form, represented, True
    )
    if form is None:
        form = topology.SaddleNormalForm(
            jnp.zeros(2),
            jnp.zeros((4, 2)),
            jnp.zeros((4, 2)),
            jnp.zeros((4, 2)),
            jnp.zeros(4, dtype=bool),
            jnp.asarray(False),
            jnp.asarray(0),
        )
    valid &= jnp.all(fixed.valid)
    return MembershipRead(
        fixed.jet.value[0],
        level,
        fractions,
        areas,
        selected,
        intervals,
        coefficient,
        form,
        represented,
        valid,
        valid,
    )


def reference_control():
    """A collection containing a polygon must retain its known nonzero integral."""

    class Density:
        @staticmethod
        def toroidal_current_density(radius, height):
            return np.ones_like(radius)

    polygon = Polygon(((1, 0), (2, 0), (2, 1), (1, 1)))
    collection = GeometryCollection([LineString(((0, 0), (0, 1))), polygon])
    nested = GeometryCollection([collection])
    moments = np.asarray(
        analytic_moments(Density(), [nested], np.asarray([[1.5, 0.5]]))
    )
    np.testing.assert_allclose(moments[:, 0], (1, 0, 0), atol=2e-15)
    skipped = sum(part.area for part in [nested] if part.geom_type == "Polygon")
    assert skipped == 0 and len(polygon_rings(nested)) == 1
    print(
        "REFERENCE_CONTROL",
        json.dumps(
            dict(
                collection_current=float(moments[0, 0]),
                shallow_current=skipped,
                nested_polygon_count=1,
            )
        ),
        flush=True,
    )


def exact_reference(kind, oracle, source, geometry):
    """Integrate the exact support, recursively retaining every polygon and hole."""
    if kind == "limited":
        theta = np.linspace(0, 2 * np.pi, 4097)
        core = Polygon(
            np.column_stack(
                (
                    np.sqrt(
                        oracle.major_radius**2
                        + np.sqrt(2 * oracle.axis_flux / oracle.pressure_coefficient)
                        * np.cos(theta)
                    ),
                    np.sqrt(oracle.axis_flux / oracle.field_coefficient)
                    * np.sin(theta),
                )
            )
        )
    else:
        core = Polygon(oracle.separatrix(4097))
    cells = [
        np.asarray(p)[: int(n)]
        for p, n in zip(geometry.vertices, geometry.vertex_count, strict=True)
    ]
    polygons = [Polygon(cell).intersection(core) for cell in cells]
    moments = ClippedCurrentMoments(
        *analytic_moments(source, polygons, np.asarray(geometry.centre))
    )
    values = np.asarray(moments)
    areas = np.asarray([p.area for p in polygons])
    assert np.max(np.abs(values)) > 0 and np.isfinite(values).all()
    assert np.all(values[0, areas > 1e-15] > 0)
    collections = [
        i for i, p in enumerate(polygons) if p.geom_type == "GeometryCollection"
    ]
    missing = [i for i in collections if areas[i] > 1e-15]
    receipt = dict(
        collection_cells=collections,
        recovered_polygon_cells=missing,
        recovered_current=float(values[0, missing].sum()),
        exact_scale=float(np.max(np.abs(values))),
    )
    print("REFERENCE_INTEGRATION", json.dumps(receipt), flush=True)
    return cells, moments, receipt


@jax.jit
def nodal_jets(field, points):
    padding = -len(points) % 8
    padded = jnp.concatenate((points, jnp.repeat(points[-1:], padding, axis=0)))
    jets = jax.lax.map(field.evaluate, padded, batch_size=8)
    return stack_jet(jets)[: len(points)]


@jax.jit
def map_delta(coupling, difference, points):
    return topology._point_values(
        topology.TotalField(difference, coupling, ZeroExterior()), points
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ZeroExterior:
    """Zero exterior for the linear booking-error image."""

    def value(self, point):
        return jnp.zeros_like(point[0])

    def evaluate(self, point):
        return topology.FieldJet(
            jnp.zeros_like(point[0]), jnp.zeros(2), jnp.zeros((2, 2))
        )


def relative(left, right):
    left, right = np.asarray(left), np.asarray(right)
    scale = np.max(np.abs(right))
    assert np.isfinite(scale) and scale > 0
    return float(np.max(np.abs(left - right)) / scale)


def persist(path, receipt):
    path.write_text(json.dumps(receipt, indent=2) + "\n")


def run(args):
    reference_control()
    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs(args.kind)
    geometry = fixture._realised_hex_geometry(wall, args.cells)
    _, source, _ = certificate._case(
        "diverted-single-null"
        if saddle is not None
        else "weak-rotation-reactor-static",
        clip_mode="exact",
    )
    cells, exact, reference_receipt = exact_reference(
        args.kind, oracle, source, geometry
    )
    coupling = topology.BiotMomentCoupling.from_polygons(cells)
    np.testing.assert_allclose(coupling.centre, geometry.centre, rtol=0, atol=1e-12)
    profile = analytic_profile(source)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996 if saddle is not None else 0
    )
    positions = jnp.asarray([axis] if saddle is None else [axis, saddle])
    pitch = jnp.median(geometry.pitch)
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    nodes = jnp.asarray(nodes)
    indices = jnp.asarray(inverse.reshape(samples.shape[:-1]))
    mesh = build_mesh(np.asarray(nodes), indices, geometry)
    analytic_jets = stack_jet(jax.jit(jax.vmap(analytic.evaluate))(nodes))
    chosen = select_cache_nodes(mesh, np.asarray(positions), float(pitch))
    rows = build_rows(np.asarray(nodes)[chosen], coupling)
    lookup = np.full(len(nodes), -1, dtype=np.int32)
    lookup[chosen] = np.arange(len(chosen))
    exact_array = jnp.stack(tuple(exact), -1)
    exterior = analytic_jets[chosen] - jnp.einsum("ncmq,cm->nq", rows, exact_array)
    book = jax.jit(
        lambda psi, reading: current.book(
            psi,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma, segments=64),
        )
    )
    print("REFERENCE_READ_START", flush=True)
    analytic_read = jax.jit(topology.read)(analytic, geometry, convention, policy)
    analytic_samples = jax.jit(jax.vmap(jax.vmap(analytic.value)))(
        geometry.sample_points
    )
    seed = book(analytic_samples, analytic_read)
    jax.block_until_ready(seed)
    assert bool(seed.valid)
    seed_difference = ClippedCurrentMoments(
        *(a - b for a, b in zip(seed[:3], exact, strict=True))
    )
    kernel = topology.TotalField(seed_difference, coupling, analytic)
    output = args.out / f"{args.kind}-{args.cells}.json"
    receipt = dict(
        kind=args.kind,
        cells=args.cells,
        reference=reference_receipt,
        rows=[],
        complete=False,
        state_seed="analytic read current.book image",
        material_fraction=0.20,
        maximum_added_wall_fraction=0.25,
    )
    persist(output, receipt)
    programs = {}
    for state in ("analytic", "picard"):
        print("STATE_START", state, flush=True)
        jets = analytic_jets if state == "analytic" else nodal_jets(kernel, nodes)
        jax.block_until_ready(jets)
        moments = exact_array if state == "analytic" else jnp.stack(tuple(seed[:3]), -1)
        cached = CachedField(
            mesh, jnp.asarray(lookup), rows, moments, exterior, jnp.asarray(1.0)
        )
        fixed = jax.jit(solve)(cached, positions, pitch)
        jax.block_until_ready(fixed)
        assert bool(jnp.all(fixed.valid))
        if state == "analytic":
            reference_booking = exact
            level = (
                fixed.jet.value[-1]
                if saddle is not None
                else analytic_read.boundary_flux
            )
            psi = analytic_samples
        else:
            print("LIVE_REFERENCE_START", flush=True)
            live = jax.jit(topology.read)(kernel, geometry, convention, policy)
            jax.block_until_ready(live)
            assert bool(live.valid), live.reason
            psi = jax.jit(topology._point_values)(kernel, geometry.sample_points)
            reference_booking = book(psi, live)
            jax.block_until_ready(reference_booking)
            assert bool(reference_booking.valid)
            level = fixed.jet.value[-1] if saddle is not None else live.boundary_flux
            print("LIVE_REFERENCE_READY", flush=True)
        for method, curved in (("A", False), ("B", True)):
            operands = (jets, indices, mesh, geometry, fixed, level, convention.sigma)
            if method not in programs:
                start = perf_counter()
                program = (
                    jax.jit(
                        partial(classify, curved=curved, diverted=saddle is not None)
                    )
                    .lower(*operands)
                    .compile()
                )
                cold = perf_counter() - start
                programs[method] = program, cold
            program, cold = programs[method]
            reading = program(*operands)
            jax.block_until_ready(reading)
            walls = []
            for _ in range(20):
                start = perf_counter()
                reading = program(*operands)
                jax.block_until_ready(reading)
                walls.append(perf_counter() - start)
            booked = book(psi, reading)
            jax.block_until_ready(booked)
            assert bool(booked.valid), (state, method)
            target = reference_booking[:3]
            row = dict(
                state=state,
                method=method,
                valid=bool(booked.valid),
                booked_moment_error=relative(booked[:3], target),
                warm_seconds=float(np.median(walls)),
                compile_seconds=cold,
                patch_cells=int(jnp.sum(reading.normal_form_cells)),
            )
            if state == "analytic":
                difference = ClippedCurrentMoments(
                    *(a - b for a, b in zip(booked[:3], exact, strict=True))
                )
                delta = map_delta(coupling, difference, geometry.centre)
                truth = jax.vmap(analytic.value)(geometry.centre)
                row["map_flux_error"] = float(
                    jnp.max(jnp.abs(delta)) / jnp.max(jnp.abs(truth))
                )
            receipt["rows"].append(row)
            persist(output, receipt)
            np.savez_compressed(
                args.out / f"{args.kind}-{args.cells}-{state}-{method}.npz",
                booked=np.asarray(booked[:3]),
                reference=np.asarray(target),
                membership=np.asarray(reading.membership),
            )
            print("BOOKING_EFFECT_ROW", json.dumps(row), flush=True)
    receipt["complete"] = True
    persist(output, receipt)
    print("ROW_COMPLETE", output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"revision={revision} tree={Path.cwd()} command={sys.executable} "
        "-m scripts.prototypes.membership.booking " + " ".join(sys.argv[1:]),
        flush=True,
    )
    print(
        f"NOVA_MODULE={nova.__file__} CWD={Path.cwd()} DEVICES={jax.devices()}",
        flush=True,
    )
    assert jax.default_backend() == "gpu"
    args.out.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
