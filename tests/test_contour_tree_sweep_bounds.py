"""Keep contour-tree receipts stable and sweep parent walks shallow."""

from __future__ import annotations

import os
import subprocess
import sys
from math import ceil, log2
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp

from benchmarks.contour_tree_brute_force import (
    certificate_fixtures,
    certificate_rung_fixtures,
    mast_fixtures,
)
from nova.equilibrium import contour_tree
from nova.jax.config import configure_dtypes
from scripts.prototypes.contour_tree_sweep_profile.profile import (
    assert_identity,
    inputs,
    instrumented_sweep,
    load_builder,
)


configure_dtypes()
assert jax.config.jax_enable_x64 is True


def _receipt_reference():
    """Load the source that recorded the fixture receipts."""

    reference_path = os.environ.get("REFERENCE_CONTOUR_TREE_MODULE")
    if reference_path is not None:
        return load_builder(Path(reference_path))

    root = Path(__file__).resolve().parents[1]
    source = subprocess.run(
        ["git", "show", "511a22e06^:nova/equilibrium/contour_tree.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = ModuleType("contour_tree_receipt_reference")
    module.__file__ = str(root / "nova/equilibrium/contour_tree.py")
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def test_brute_force_fixture_receipts_match_recorded_fields() -> None:
    """Each stored carrier retains all seven receipt fields bit for bit."""

    reference = _receipt_reference()
    fixtures = (
        *certificate_fixtures(),
        *certificate_rung_fixtures(),
        *mast_fixtures(),
    )
    assert fixtures
    for fixture in fixtures:
        mesh = fixture.mesh
        arguments = (
            mesh.vertex_psi,
            mesh.vertex_valid,
            mesh.vertex_is_wall,
            mesh.edges,
            mesh.edge_valid,
            jnp.asarray(1, dtype=jnp.int32),
        )
        expected = reference.build_contour_tree(*arguments)
        actual = contour_tree.build_contour_tree(*arguments)
        jax.block_until_ready((expected, actual))
        try:
            assert_identity(expected, actual)
        except AssertionError as error:
            raise AssertionError(f"{fixture.name}: {error}") from error
    print(f"IDENTICAL_FIXTURES={len(fixtures)}")


def test_largest_profile_fixture_bounds_root_hops() -> None:
    """Both sweep directions stay within the rank-balanced tree depth."""

    arguments, realised_cells = inputs("limited", 10_000)
    vertex_psi, vertex_valid, vertex_is_wall, edges, edge_valid, _ = arguments
    vertex_count = int(vertex_psi.size)
    neighbours, neighbour_valid, _ = contour_tree._carrier_adjacency(
        edges, edge_valid, vertex_count
    )
    bound = ceil(log2(vertex_count)) + 1

    for name, descending in (("join", True), ("split", False)):
        sweep_arguments = (
            jnp.asarray(vertex_psi),
            jnp.asarray(vertex_valid),
            jnp.asarray(vertex_is_wall),
            neighbours,
            neighbour_valid,
            descending,
            vertex_count,
            int(edges.shape[0]),
        )
        _, depth = instrumented_sweep(contour_tree, sweep_arguments)
        assert depth["visits"] == vertex_count
        assert depth["roots_max"] <= bound, (
            f"{name} sweep root-hop depth {depth['roots_max']} exceeds "
            f"ceil(log2({vertex_count})) + 1 = {bound} "
            f"for {realised_cells} realised cells"
        )
        print(
            f"ROOT_HOPS={name}:{depth['roots_max']}/{bound} "
            f"VERTICES={vertex_count} CELLS={realised_cells}"
        )
