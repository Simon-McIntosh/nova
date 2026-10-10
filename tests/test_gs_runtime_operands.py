"""Runtime geometry ownership in the forward flux map."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import gs
from nova.jax.config import configure_dtypes
from tests.test_forward_operator_arguments import _operator
from tests.test_forward_profile_arguments import _source


def _resident_kib():
    """Read current process RSS without confusing it with prior high water."""
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") // 1024


def _sample_peak(call):
    """Sample live RSS while a compiled phase is running."""
    stop = threading.Event()
    samples = [_resident_kib()]

    def sample():
        while not stop.wait(0.01):
            samples.append(_resident_kib())

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        result = call()
    finally:
        stop.set()
        worker.join()
        samples.append(_resident_kib())
    return result, max(samples)


def _report_phase(name, **values):
    print(
        "PHASE_RSS " + json.dumps({"phase": name, **values}, sort_keys=True), flush=True
    )


def _geometry_pair():
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    first = _operator()
    second = _operator()
    first.source = _source([2.0], [0.5])
    second.source = _source([2.0], [0.5])
    second.area = np.asarray(first.area) * 1.01
    second.grid.plasma_target = first.grid.plasma_target * 1.1
    second._geometry_identity = second._compute_geometry_identity()
    assert first.geometry_identity != second.geometry_identity
    return first, second


def test_map_matches_the_existing_book_and_image():
    first, second = _geometry_pair()
    state = jnp.linspace(-1.0, 1.0, first.node_number)
    for operator in (first, second):
        external = operator.external()
        image = external + operator.internal(state)
        expected = operator._exclude_shadow_residual(state, image, None)
        np.testing.assert_allclose(
            gs.apply(state, external, operator), expected, rtol=1e-13, atol=1e-13
        )


def test_one_compilation_serves_distinct_geometry_values():
    first, second = _geometry_pair()
    assert first.program_identity == second.program_identity
    state = jnp.linspace(-1.0, 1.0, first.node_number)
    compiled = jax.jit(gs.apply)
    first_result = compiled(state, first.external(), first)
    assert compiled._cache_size() == 1
    second_result = compiled(state, second.external(), second)
    assert compiled._cache_size() == 1
    assert not np.array_equal(np.asarray(first_result), np.asarray(second_result))


def test_captured_coupling_is_a_program_constant():
    first, second = _geometry_pair()
    state = jnp.ones(first.grid.node_number)
    traces = []

    def bound(block):
        def image(current):
            traces.append(1)
            return block @ current

        return image

    first_bound = bound(first.grid.plasma_target)
    second_bound = bound(second.grid.plasma_target)
    first_jaxpr = jax.make_jaxpr(first_bound)(state)
    second_jaxpr = jax.make_jaxpr(second_bound)(state)
    assert any(
        tuple(np.shape(value)) == tuple(first.grid.plasma_target.shape)
        for value in first_jaxpr.consts
    )
    assert any(
        tuple(np.shape(value)) == tuple(second.grid.plasma_target.shape)
        for value in second_jaxpr.consts
    )
    jax.jit(first_bound)(state).block_until_ready()
    jax.jit(second_bound)(state).block_until_ready()
    assert len(traces) == 2


def _measure_cached_map(
    case_name: str, requested_cells: int, cache_root: Path, group: str
):
    """Measure one cached carrier without rebuilding or writing its machine."""
    from benchmarks import solovev_certificate as certificate
    from nova.database.zarrstore import ZarrStore
    from scripts.analytic_oracle_fixtures import measure as fixture
    import zarr

    _carrier, source, exact = certificate._case(case_name)
    filename = f"{fixture.CACHE_FILENAME}_{abs(requested_cells)}"
    store = ZarrStore(filename=filename, dirname=cache_root, group=group)
    if not store.is_store():
        raise FileNotFoundError(store.filepath)
    cached = zarr.open_group(store.filepath, mode="r")[group]
    identity = json.loads(cached.attrs["semantic_identity"])
    if identity["analytic_case"] not in (
        case_name,
        case_name.removesuffix("-static"),
    ):
        raise ValueError("cached carrier does not match the requested case")
    if identity["discretisation"]["requested_cells"] != requested_cells:
        raise ValueError("cached carrier does not match the requested cell count")
    store.load()
    machine = fixture._from_dataset(store.data, identity, store.group)
    operator = fixture.forward_operator(source, machine)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    state = jnp.asarray(certificate._exact_state(case_name, exact, coordinates))
    external = jnp.zeros_like(state)
    old = operator._exclude_shadow_residual(
        state, external + operator.internal(state), None
    )
    new = gs.apply(state, external, operator)
    relative = float(
        jnp.max(jnp.abs(old - new)) / jnp.maximum(jnp.max(jnp.abs(old)), 1.0)
    )
    lowered = jax.jit(gs.apply).lower(state, external, operator)
    hlo_constant_text_bytes = sum(
        len(line.encode())
        for line in lowered.as_text().splitlines()
        if "stablehlo.constant" in line
    )
    started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started
    executable_bytes = len(executable.runtime_executable().serialize())
    print(
        "MAP_ROW "
        + json.dumps(
            {
                "case": case_name,
                "requested_cells": requested_cells,
                "realised_cells": len(machine.node),
                "map_relative_difference": relative,
                "compile_seconds": compile_seconds,
                "serialized_executable_bytes": executable_bytes,
                "hlo_constant_text_bytes": hlo_constant_text_bytes,
                "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "cache_path": str(store.filepath),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    assert relative <= 1e-13
    assert executable_bytes <= 50_000_000


def _measure_placeholder_map(case_name: str, requested_cells: int):
    """Build the production mesh and pass zero-valued coupling blocks as data."""
    from benchmarks import solovev_certificate as certificate
    from nova.biot.plasmagrid import PlasmaGrid
    from nova.equilibrium.stencil_mesh import (
        MomentGeometry,
        StencilMesh,
        ring_condition,
    )
    from nova.frame.coilset import CoilSet
    from scipy.spatial import Delaunay
    from shapely.geometry import LineString
    from scripts.analytic_oracle_fixtures import measure as fixture

    carrier, source, exact = certificate._case(case_name)
    wall = fixture._resolved_wall(
        carrier,
        wall_nodes=fixture.WALL_POINT_COUNT,
        wall=(
            certificate._diverted_wall(exact)
            if certificate._is_diverted_case(case_name)
            else None
        ),
    )
    coilset = CoilSet(dplasma=requested_cells, tplasma="hex")
    coilset.firstwall.insert(wall, turn="hex")
    plasma = np.asarray(coilset.subframe.loc[:, "plasma"], dtype=bool)
    material = np.asarray(coilset.subframe.loc[:, "poly"], dtype=object)[plasma]
    centres = np.c_[
        np.asarray(coilset.subframe.loc[plasma, "x"], dtype=np.float64),
        np.asarray(coilset.subframe.loc[plasma, "z"], dtype=np.float64),
    ]
    polygons = tuple(
        fixture._clean_vertices(np.asarray(item.poly.exterior.coords)[:-1, :2])
        for item in material
    )
    area = np.asarray([item.poly.area for item in material], dtype=np.float64)
    triangulation = Delaunay(centres)
    boundary_cells = np.asarray(
        [
            index
            for index, item in enumerate(material)
            if item.poly.intersects(LineString(wall))
        ]
    )
    stencil, _ = PlasmaGrid.loop_neighbour_vertices(
        centres, triangulation.vertex_neighbor_vertices, boundary_cells
    )
    mesh = StencilMesh(centres, stencil, area)
    sections = np.asarray(coilset.aloc["plasma", "section"], dtype=object).astype(str)
    full = np.flatnonzero(sections == "hexagon")
    dimensions = np.c_[
        np.asarray(coilset.aloc["plasma", "dl"], dtype=float)[full],
        np.asarray(coilset.aloc["plasma", "dt"], dtype=float)[full],
    ]
    width, height = dimensions[0]
    radius = min(width / 2.0, height / np.sqrt(3.0))
    angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    sampling = (
        centres[:, None, :]
        + radius * np.column_stack([np.cos(angles), np.sin(angles)])[None, :, :]
    )
    geometry = MomentGeometry.from_cells(mesh, polygons, sampling_vertices=sampling)
    condition = ring_condition(centres, stencil)
    regular = np.asarray([len(polygon) == 6 for polygon in polygons])
    interior = stencil[(condition < 1.0e3) & regular[stencil].all(axis=1)]
    sample = geometry.sample_node_coordinates
    zero = np.zeros((1, 1), dtype=np.float64)

    def block(rows):
        return np.broadcast_to(zero, (rows, len(centres)))

    machine = fixture.OracleMachine(
        node=centres,
        area=area,
        cell_polygons=polygons,
        stencil=stencil,
        interior_stencil=interior,
        wall_node=wall,
        sampling_vertices=sampling,
        sample_coordinates=sample,
        plasma_to_grid=block(len(centres)),
        plasma_to_grid_r=block(len(centres)),
        plasma_to_grid_z=block(len(centres)),
        plasma_to_wall=block(len(wall)),
        plasma_to_wall_r=block(len(wall)),
        plasma_to_wall_z=block(len(wall)),
        plasma_to_sample=block(len(sample)),
        plasma_to_sample_r=block(len(sample)),
        plasma_to_sample_z=block(len(sample)),
    )
    geometry_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    geometry_live_rss_kib = _resident_kib()
    _report_phase(
        "geometry", high_water_kib=geometry_rss_kib, live_kib=geometry_live_rss_kib
    )
    operator = fixture.forward_operator(source, machine)
    operand_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    operand_live_rss_kib = _resident_kib()
    _report_phase(
        "operands", high_water_kib=operand_rss_kib, live_kib=operand_live_rss_kib
    )
    coordinates = np.vstack((centres, wall, sample))
    state = jnp.asarray(certificate._exact_state(case_name, exact, coordinates))
    external = jnp.zeros_like(state)
    lowered = jax.jit(gs.apply).lower(state, external, operator)
    lowered_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    lowered_live_rss_kib = _resident_kib()
    _report_phase(
        "lowering", high_water_kib=lowered_rss_kib, live_kib=lowered_live_rss_kib
    )
    hlo_constant_text_bytes = sum(
        len(line.encode())
        for line in lowered.as_text().splitlines()
        if "stablehlo.constant" in line
    )
    started = time.perf_counter()
    executable, compile_sampled_peak_rss_kib = _sample_peak(lowered.compile)
    compile_seconds = time.perf_counter() - started
    compiled_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    compiled_live_rss_kib = _resident_kib()
    _report_phase(
        "compile",
        high_water_kib=compiled_rss_kib,
        live_kib=compiled_live_rss_kib,
        sampled_peak_kib=compile_sampled_peak_rss_kib,
    )
    executable_bytes = len(executable.runtime_executable().serialize())
    _output, execution_sampled_peak_rss_kib = _sample_peak(
        lambda: jax.block_until_ready(executable(state, external, operator))
    )
    execution_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    execution_live_rss_kib = _resident_kib()
    _report_phase(
        "execution",
        high_water_kib=execution_rss_kib,
        live_kib=execution_live_rss_kib,
        sampled_peak_kib=execution_sampled_peak_rss_kib,
    )
    print(
        "MAP_ROW "
        + json.dumps(
            {
                "case": case_name,
                "requested_cells": requested_cells,
                "realised_cells": len(centres),
                "operands": "zero-valued production-shape blocks",
                "compile_seconds": compile_seconds,
                "serialized_executable_bytes": executable_bytes,
                "hlo_constant_text_bytes": hlo_constant_text_bytes,
                "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "geometry_rss_kib": geometry_rss_kib,
                "geometry_live_rss_kib": geometry_live_rss_kib,
                "operand_rss_kib": operand_rss_kib,
                "operand_live_rss_kib": operand_live_rss_kib,
                "lowered_rss_kib": lowered_rss_kib,
                "lowered_live_rss_kib": lowered_live_rss_kib,
                "compiled_rss_kib": compiled_rss_kib,
                "compiled_live_rss_kib": compiled_live_rss_kib,
                "compile_sampled_peak_rss_kib": compile_sampled_peak_rss_kib,
                "execution_rss_kib": execution_rss_kib,
                "execution_live_rss_kib": execution_live_rss_kib,
                "execution_sampled_peak_rss_kib": execution_sampled_peak_rss_kib,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    assert executable_bytes <= 50_000_000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--requested-cells", type=int, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--group")
    arguments = parser.parse_args()
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    if arguments.cache_root is None:
        _measure_placeholder_map(arguments.case, arguments.requested_cells)
    else:
        _measure_cached_map(
            arguments.case,
            arguments.requested_cells,
            arguments.cache_root,
            arguments.group,
        )
