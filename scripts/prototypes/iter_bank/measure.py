"""Build and qualify a version-pinned ITER nodal operator bank."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import time

from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

configure_dtypes()

import imas
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon

from nova.equilibrium import support_read, topology
from nova.imas.machine import CrossSection, ImasGeometryReader
from scripts.prototypes.iter_bank.inspect import DD, SOURCE
from scripts.prototypes.read_route.jets import _source_jets, build_jet_matrix

assert jax.config.jax_enable_x64 is True

# Source-cell corners can coincide with sample nodes where the vertical Hessian
# is discontinuous. Moment and direct-kernel Hessians agree near 2e-9 relative;
# these derivatives guide null finding, while the residual retains its 1e-10
# budget. Coil fields retain the stricter scalar budget in assert_accuracy.
PLASMA_JET_TOLERANCE = (1e-10, 1e-10, 1e-10, 1e-8, 1e-8, 1e-8)


def emit(label, **values):
    print(label + " " + json.dumps(values, sort_keys=True), flush=True)


def pack(jet):
    return jnp.stack(
        (
            jet.value,
            jet.gradient[..., 0],
            jet.gradient[..., 1],
            jet.hessian[..., 0, 0],
            jet.hessian[..., 0, 1],
            jet.hessian[..., 1, 1],
        ),
        axis=-1,
    )


@dataclass(frozen=True)
class ZeroField:
    def evaluate(self, point):
        return topology.FieldJet(jnp.asarray(0.0), jnp.zeros(2), jnp.zeros((2, 2)))


def read_scenario():
    """Read native geometry, signed winding turns, and homogeneous time bases."""
    with imas.DBEntry(f"imas:hdf5?path={SOURCE}", "r", dd_version=DD) as entry:
        names = ("pf_active", "wall", "equilibrium")
        for name in names:
            if 0 not in entry.list_all_occurrences(name):
                raise ValueError(f"{name} occurrence zero is absent from {SOURCE}")
        active, wall, eq = (
            entry.get(name, 0, lazy=False, autoconvert=False) for name in names
        )
    for ids in (active, wall, eq):
        assert str(ids.ids_properties.version_put.data_dictionary) == DD
    if int(eq.ids_properties.homogeneous_time) != 1:
        raise ValueError("equilibrium requires its own per-slice time policy")
    wall_units = tuple(
        np.column_stack((np.asarray(u.outline.r), np.asarray(u.outline.z)))
        for u in wall.description_2d[0].limiter.unit
    )
    if len(wall_units) != 1:
        raise ValueError("mesh requires one connected limiter enclosure")
    names, polygons, turns, owners = [], [], [], []
    for coil in active.coil:
        name = str(coil.name)
        # The stabilization winding also changes per slice, so retain it.
        names.append(name)
        if not len(coil.element):
            raise ValueError(f"{name} has no geometry in the scenario")
        for element in coil.element:
            kind = CrossSection.transform[int(element.geometry.geometry_type)]
            section = ImasGeometryReader(element.geometry).section(kind)
            p = np.asarray(section.poly.exterior.coords[:-1], dtype=float)
            if not np.all(np.isfinite(p)) or np.max(np.abs(p)) > 100:
                raise ValueError(f"{name} has incomplete geometry in the scenario")
            if not Polygon(p).exterior.is_ccw:
                p = p[::-1]
            polygons.append(p)
            turns.append(float(element.turns_with_sign))
            owners.append(len(names) - 1)
    currents = np.asarray(
        [
            float(eq.time_slice[i].global_quantities.ip)
            for i in range(len(eq.time_slice))
        ]
    )
    flat = np.flatnonzero(np.abs(currents) >= 0.99 * np.max(np.abs(currents)))
    indices = flat[np.linspace(0, len(flat) - 1, 3).astype(int)]
    states = []
    for i in indices:
        slice_ = eq.time_slice[int(i)]
        current = []
        for coil in active.coil:
            base = (
                np.asarray(active.time)
                if int(active.ids_properties.homogeneous_time) == 1
                else np.asarray(coil.current.time)
            )
            current.append(
                float(np.interp(float(eq.time[i]), base, np.asarray(coil.current.data)))
            )
        states.append((int(i), float(eq.time[i]), slice_, np.asarray(current)))
    return (
        wall_units,
        tuple(polygons),
        np.asarray(turns),
        np.asarray(owners),
        names,
        states,
    )


def build_geometry(wall, target=550):
    """Clip a hexagonal carrier, retaining every disconnected wall fragment.

    A reentrant divertor wall can cut one generator into multiple components.
    Each component is an independent source cell with the generator's stencil.
    """
    vessel = Polygon(wall)
    origin = np.asarray(vessel.centroid.coords[0])
    bounds = np.asarray(vessel.bounds)

    def generate(radius):
        lower = np.floor((bounds[:2] - origin) / (1.5 * radius)).astype(int) - 3
        upper = np.ceil((bounds[2:] - origin) / (1.5 * radius)).astype(int) + 3
        angle = np.arange(6) * np.pi / 3
        offset = radius * np.column_stack((np.cos(angle), np.sin(angle)))
        cells, sampling = [], []
        for radial in range(lower[0], upper[0] + 1):
            for vertical in range(2 * lower[1], 2 * upper[1] + 1):
                centre = origin + radius * np.asarray(
                    (
                        1.5 * (radial + 0.137),
                        np.sqrt(3) * (vertical + 0.5 * (radial % 2) + 0.219),
                    )
                )
                stencil = centre + offset
                clipped = Polygon(stencil).intersection(vessel)
                if clipped.is_empty:
                    continue
                parts = (clipped,) if clipped.geom_type == "Polygon" else clipped.geoms
                for part in parts:
                    if part.geom_type != "Polygon" or part.area < 1e-14 * radius**2:
                        continue
                    points = np.asarray(part.exterior.coords[:-1])
                    cells.append(points if part.exterior.is_ccw else points[::-1])
                    sampling.append(stencil)
        return cells, sampling

    estimate = np.sqrt(vessel.area / target / (3 * np.sqrt(3) / 2))
    low, high = 0.5 * estimate, 2 * estimate
    best = None
    for _ in range(40):
        radius = 0.5 * (low + high)
        cells, sampling = generate(radius)
        distance = abs(len(cells) - target)
        if best is None or distance < best[0]:
            best = distance, cells, sampling
        if distance == 0:
            break
        if len(cells) > target:
            low = radius
        else:
            high = radius
    _, cells, sampling = best
    geometry = topology.TopologyGeometry.from_cells(
        tuple(cells), np.asarray(sampling), (wall,)
    )
    emit(
        "MESH",
        cells=len(cells),
        target=target,
        area_m2=float(np.sum(np.asarray(geometry.full_area))),
        wall_area_m2=vessel.area,
    )
    return geometry


def sampled_moments(geometry, slice_):
    """Integrate the stored toroidal current density over the clipped cells."""
    surface = slice_.profiles_2d[0]
    density = RectBivariateSpline(
        np.asarray(surface.grid.dim1),
        np.asarray(surface.grid.dim2),
        np.asarray(surface.j_tor),
    )
    boundary = Polygon(
        np.column_stack(
            (
                np.asarray(slice_.boundary.outline.r),
                np.asarray(slice_.boundary.outline.z),
            )
        )
    )
    node, weight = np.polynomial.legendre.leggauss(4)
    u, v = np.meshgrid((node + 1) / 2, (node + 1) / 2, indexing="ij")
    weight = weight[:, None] * weight[None, :] / 4
    result = np.zeros((len(geometry.centre), 3))
    for cell, count in enumerate(np.asarray(geometry.vertex_count)):
        polygon = Polygon(np.asarray(geometry.vertices[cell, :count])).intersection(
            boundary
        )
        if polygon.is_empty:
            continue
        parts = [polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms)
        for part in parts:
            points = np.asarray(part.exterior.coords[:-1])
            if not part.exterior.is_ccw:
                points = points[::-1]
            for i in range(1, len(points) - 1):
                a, b = points[i] - points[0], points[i + 1] - points[0]
                q = points[0] + u[..., None] * (
                    (1 - v[..., None]) * a + v[..., None] * b
                )
                weighted = (
                    weight
                    * u
                    * np.linalg.det(np.stack((a, b)))
                    * density.ev(q[..., 0], q[..., 1])
                )
                delta = q - np.asarray(geometry.centre[cell])
                result[cell, 0] += np.sum(weighted)
                result[cell, 1:] += np.sum(weighted[..., None] * delta, axis=(0, 1))
    if not np.all(np.isfinite(result)) or not np.count_nonzero(result[:, 0]):
        raise ValueError("DINA current-density integration yielded no finite plasma")
    return result


def assert_accuracy(actual, expected, tolerance=1e-10):
    """Require finite nodal jets to agree in every jet entry's relative norm."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    scale = np.linalg.norm(expected, axis=0)
    if not np.all(np.isfinite(expected)) or np.any(scale == 0):
        raise AssertionError("analytic reference has no finite measurable scale")
    errors = np.linalg.norm(actual - expected, axis=0) / scale
    if not np.all(np.isfinite(errors)) or np.any(errors > tolerance):
        raise AssertionError(
            f"analytic-map relative jet errors {errors.tolist()} exceed {tolerance}"
        )
    return errors.tolist()


def direct_map(points, plasma, moments, coils, coil_moments, batch=4):
    """Evaluate the analytic kernels directly, without using bank rows."""

    def point_map(point, plasma, moments, coils, coil_moments):
        p = plasma.evaluate(point, tuple(moments.T))
        c = coils.evaluate(point, tuple(coil_moments.T))
        return pack(jax.tree.map(jnp.add, p, c))

    function = jax.jit(jax.vmap(point_map, in_axes=(0, None, None, None, None)))
    output = []
    for first in range(0, len(points), batch):
        p = points[first : first + batch]
        size = len(p)
        p = np.pad(p, ((0, batch - size), (0, 0)), mode="edge")
        output.append(
            np.asarray(
                function(
                    jnp.asarray(p),
                    plasma,
                    jnp.asarray(moments),
                    coils,
                    jnp.asarray(coil_moments),
                )
            )[:size]
        )
    return np.concatenate(output)


def capacity(bank):
    """Find allocator admission for shared bank plus live batched states and jets.

    This probes bank application storage, not Newton factorization workspace.
    The bank is shared; every lane holds moments, currents and six nodal jets.
    """
    stats = jax.devices()[0].memory_stats()
    base = int(stats["bytes_in_use"])
    per_lane = (
        bank.rows.shape[1] * 3 + bank.rows.shape[0] * 6 + int(bank.coil_count)
    ) * 8
    largest, refused = 0, None
    trials = []

    def probe(width):
        arrays = []
        try:
            arrays = [jnp.zeros((width, per_lane // 8), dtype=jnp.float64)]
            jax.block_until_ready(arrays)
            record = dict(
                width=width,
                fits=True,
                live_bytes=int(jax.devices()[0].memory_stats()["bytes_in_use"]),
            )
            good = True
        except jax.errors.JaxRuntimeError as error:
            if (
                "RESOURCE_EXHAUSTED" not in str(error)
                and "out of memory" not in str(error).lower()
            ):
                raise
            record = dict(width=width, fits=False, refusal=str(error)[:500])
            good = False
        arrays.clear()
        gc.collect()
        trials.append(record)
        emit("WIDTH_PROBE", **record)
        return good

    width = 16
    while probe(width):
        largest = width
        width *= 2
    refused = width
    while refused - largest > 1:
        width = (largest + refused) // 2
        if probe(width):
            largest = width
        else:
            refused = width
    return dict(
        base_device_bytes=base,
        bytes_per_lane=per_lane,
        allocator_limit_bytes=int(stats["bytes_limit"]),
        largest_fitting_width=largest,
        first_refused_width=refused,
        scope=(
            "shared bank plus moments, coil currents, nodal jets; "
            "excludes solver workspace"
        ),
        trials=trials,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.prepare_only:
        assert jax.default_backend() == "gpu"
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    emit(
        "RUNTIME",
        devices=[str(x) for x in jax.devices()],
        cache=str(cache.directory),
        x64=True,
    )
    wall, polygons, turns, owners, names, states = read_scenario()
    geometry = build_geometry(wall[0], 550)
    plasma_polygons = tuple(
        np.asarray(geometry.vertices[i, :n])
        for i, n in enumerate(np.asarray(geometry.vertex_count))
    )
    coupling = topology.BiotMomentCoupling.from_polygons(plasma_polygons)
    coil_coupling = topology.BiotMomentCoupling.from_polygons(polygons)
    moments = [sampled_moments(geometry, s[2]) for s in states]
    np.savez(
        args.output / "geometry.npz",
        vertices=np.asarray(geometry.vertices),
        vertex_count=np.asarray(geometry.vertex_count),
        sample_points=np.asarray(geometry.sample_points),
        centre=np.asarray(geometry.centre),
        wall=wall[0],
        turns=turns,
        owners=owners,
        **{f"coil_{i}": p for i, p in enumerate(polygons)},
    )
    source = dict(
        path=SOURCE,
        dd_version=DD,
        geometry="scenario pf_active and wall; no fallback needed",
        coil_names=names,
        signed_turns=turns.tolist(),
        element_owner=owners.tolist(),
        cells=len(geometry.centre),
        pitch_m=float(jnp.median(geometry.pitch)),
        states=[
            dict(
                index=s[0],
                time_s=s[1],
                ip_A=float(s[2].global_quantities.ip),
                integrated_ip_A=float(m[:, 0].sum()),
                coil_currents_A=s[3].tolist(),
            )
            for s, m in zip(states, moments, strict=True)
        ],
    )
    (args.output / "source.json").write_text(json.dumps(source, indent=2))
    emit("SOURCE", **source)
    if args.prepare_only:
        return
    started = time.perf_counter()
    row_receipt = {}
    rows, exterior, nodes, sample_index, _ = build_jet_matrix(
        coupling,
        geometry,
        ZeroField(),
        tuple(jnp.zeros(len(geometry.centre)) for _ in range(3)),
        persist=True,
        receipt=row_receipt,
    )
    point_batch = jax.jit(jax.vmap(_source_jets, in_axes=(0, None)))
    coil_rows = []
    node_host = np.asarray(nodes)
    for first in range(0, len(nodes), 4):
        p = node_host[first : first + 4]
        size = len(p)
        p = np.pad(p, ((0, 4 - size), (0, 0)), mode="edge")
        coil_rows.append(
            np.asarray(point_batch(jnp.asarray(p), coil_coupling))[:size, :, 0, :]
        )
    section_rows = np.concatenate(coil_rows) * turns[None, :, None]
    coil_exterior = np.zeros((len(nodes), len(names), 6))
    for i, owner in enumerate(owners):
        coil_exterior[:, owner] += section_rows[:, i]
    bank = support_read.SupportOperands.from_rows(
        geometry,
        rows,
        exterior,
        nodes,
        sample_index,
        bucket=len(nodes),
        coil_exterior=coil_exterior,
    )
    jax.block_until_ready(bank)
    build_seconds = time.perf_counter() - started
    store = args.output / "store"
    started = time.perf_counter()
    digest = support_read.write_bank(bank, store)
    publish_seconds = time.perf_counter() - started
    started = time.perf_counter()
    loaded = support_read.read_bank(store, digest, geometry, bank.mesh)
    jax.block_until_ready(loaded)
    load_seconds = time.perf_counter() - started
    for name in (
        "rows",
        "exterior",
        "coil_exterior",
        "nodes",
        "sample_index",
        "coil_count",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(bank, name)), np.asarray(getattr(loaded, name))
        )
    result = dict(
        source=source,
        bank=dict(
            digest=digest,
            store=str(store),
            bucket=bank.bucket,
            active_nodes=int(bank.active_nodes),
            coil_count=int(bank.coil_count),
            dense_bytes=sum(
                int(x.nbytes)
                for x in (
                    bank.rows,
                    bank.exterior,
                    bank.coil_exterior,
                    bank.nodes,
                    bank.sample_index,
                )
            ),
            build_seconds=build_seconds,
            publish_seconds=publish_seconds,
            load_seconds=load_seconds,
            rows_receipt=row_receipt,
        ),
        accuracy=[],
    )
    emit("BANK", **result["bank"])
    for state, moment in zip(states, moments, strict=True):
        section_moments = np.zeros((len(polygons), 3))
        section_moments[:, 0] = turns * state[3][owners]
        expected = direct_map(
            node_host, coupling, moment, coil_coupling, section_moments
        )
        actual = (
            support_read.nodal_jets(loaded, tuple(jnp.asarray(moment.T)))
            + jnp.einsum("nkq,k->nq", loaded.coil_exterior, jnp.asarray(state[3]))
        )[: len(nodes)]
        errors = assert_accuracy(actual, expected, PLASMA_JET_TOLERANCE)
        dropped = names.index("PF1")
        mutated = np.asarray(actual) - coil_exterior[:, dropped] * state[3][dropped]
        try:
            assert_accuracy(mutated, expected, PLASMA_JET_TOLERANCE)
        except AssertionError as error:
            negative = str(error)
        else:
            raise AssertionError(
                "dropped PF1 was not detected by the analytic-map check"
            )
        item = dict(
            index=state[0],
            time_s=state[1],
            relative_errors=errors,
            dropped_coil="PF1",
            negative_control=negative,
        )
        result["accuracy"].append(item)
        np.savez(
            args.output / f"accuracy-{state[0]}.npz",
            expected=expected,
            actual=np.asarray(actual),
            dropped=mutated,
        )
        emit("ACCURACY", **item)
        (args.output / "receipt.json").write_text(json.dumps(result, indent=2))
    del rows, bank, section_rows, coil_rows
    gc.collect()
    result["capacity"] = capacity(loaded)
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2))
    emit(
        "SUMMARY",
        bucket=loaded.bucket,
        cells=len(geometry.centre),
        load_seconds=load_seconds,
        largest_fitting_width=result["capacity"]["largest_fitting_width"],
        maximum_relative_error=max(
            max(x["relative_errors"]) for x in result["accuracy"]
        ),
    )


if __name__ == "__main__":
    main()
