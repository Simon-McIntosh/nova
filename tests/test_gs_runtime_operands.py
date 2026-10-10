"""Runtime geometry ownership in the forward flux map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import gs
from nova.jax.config import configure_dtypes
from tests.test_forward_operator_arguments import _operator
from tests.test_forward_profile_arguments import _source


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--requested-cells", type=int, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--group", required=True)
    arguments = parser.parse_args()
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    _measure_cached_map(
        arguments.case, arguments.requested_cells, arguments.cache_root, arguments.group
    )
