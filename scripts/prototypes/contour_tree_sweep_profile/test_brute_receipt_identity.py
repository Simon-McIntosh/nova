"""Compare complete contour receipts on the independent carrier fixtures."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.prototypes.contour_tree_sweep_profile.profile import (
    assert_identity,
    installed_contour_tree,
    load_builder,
)

import jax
import jax.numpy as jnp

from benchmarks.contour_tree_brute_force import (
    certificate_fixtures,
    certificate_rung_fixtures,
    mast_fixtures,
)


def test_independent_carrier_receipts_match_base() -> None:
    baseline = load_builder(Path(os.environ["REFERENCE_CONTOUR_TREE_MODULE"]))
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
        expected = baseline.build_contour_tree(*arguments)
        actual = installed_contour_tree.build_contour_tree(*arguments)
        jax.block_until_ready((expected, actual))
        try:
            assert_identity(expected, actual)
        except AssertionError as error:
            raise AssertionError(f"{fixture.name}: {error}") from error
    print(f"IDENTICAL_FIXTURES={len(fixtures)}")


def test_monotone_carrier_receipt_matches_base() -> None:
    baseline = load_builder(Path(os.environ["REFERENCE_CONTOUR_TREE_MODULE"]))
    count = 257
    arguments = (
        jnp.arange(count, 0, -1, dtype=jnp.float64),
        jnp.ones(count, dtype=bool),
        jnp.zeros(count, dtype=bool),
        jnp.stack((jnp.arange(count - 1), jnp.arange(1, count)), axis=1).astype(
            jnp.int32
        ),
        jnp.ones(count - 1, dtype=bool),
        jnp.asarray(1, dtype=jnp.int32),
    )
    expected = baseline.build_contour_tree(*arguments)
    actual = installed_contour_tree.build_contour_tree(*arguments)
    jax.block_until_ready((expected, actual))
    assert_identity(expected, actual)
