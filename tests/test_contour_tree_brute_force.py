"""Independent component-count positive controls for contour-tree receipts."""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.contour_tree_brute_force import (
    batched_identical,
    certificate_fixtures,
    compare,
)
from nova.equilibrium.contour_tree import build_contour_tree
from nova.jax.config import configure_dtypes


configure_dtypes()


@pytest.fixture(scope="module")
def fixtures():
    """Load the compact persisted certificate terminal fields once."""

    result = certificate_fixtures()
    assert result
    return result


def test_certificate_tree_matches_independent_superlevel_components(fixtures):
    """Every stored certificate field agrees at every vertex-separated level."""

    corrupt = os.environ.get("CONTOUR_TREE_CORRUPT") == "drop-wall-joins"
    for fixture in fixtures:
        result = compare(fixture.mesh, corrupt=corrupt)
        assert result["node_count"] - result["edge_count"] == 1
        assert all(row["tree"] == row["brute_force"] for row in result["rows"])


def test_batched_certificate_trees_match_independent_builds(fixtures):
    """The fixed-capacity batch result is identical to per-field receipts."""

    assert batched_identical(fixtures)


def test_capacity_refusal_remains_visible():
    """An over-capacity carrier is refused instead of returning a prefix."""

    count = 257
    result = build_contour_tree(
        jnp.arange(count, 0, -1, dtype=jnp.float64),
        jnp.ones(count, dtype=bool),
        jnp.zeros(count, dtype=bool),
        jnp.stack((jnp.arange(count - 1), jnp.arange(1, count)), axis=1).astype(
            jnp.int32
        ),
        jnp.ones(count - 1, dtype=bool),
        jnp.asarray(1, dtype=jnp.int32),
    )
    assert bool(np.asarray(result.overflow))
