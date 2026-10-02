"""Independent component-count positive controls for contour-tree receipts.

The brute-force side of every comparison is a plain host graph search over the
carrier's own vertices and edges: it shares no union-find, no merge sweep and no
node or edge of the tree's code, only the mesh and the field values. The tree
arm reads the receipt's arc set and counts arcs crossing a sampled level.

``CONTOUR_TREE_CORRUPT=drop-wall-joins`` applies the declared mutation: the
wall-contact joins are dropped from the computed tree, so the per-level
comparison must stop agreeing. That run is the negative-control log.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.contour_tree_brute_force import (
    batched_identical,
    certificate_fixtures,
    compare,
    mast_fixtures,
    standalone_rows,
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


@pytest.fixture(scope="module")
def mast():
    """Load the MAST carriers named in the plan's primary-selection section."""

    result = mast_fixtures()
    assert result
    return result


def _corrupt() -> bool:
    """Whether the declared tree mutation is selected for this run."""

    return os.environ.get("CONTOUR_TREE_CORRUPT") == "drop-wall-joins"


def test_certificate_tree_matches_independent_superlevel_components(fixtures):
    """Every stored certificate field agrees at every vertex-separated level."""

    for fixture in fixtures:
        result = compare(fixture.mesh, corrupt=_corrupt())
        assert result["node_count"] - result["edge_count"] == 1
        assert all(row["tree"] == row["brute_force"] for row in result["rows"])


def test_batched_certificate_trees_match_independent_builds(fixtures):
    """The fixed-capacity batch result is identical to per-field receipts."""

    assert batched_identical(fixtures)


def test_mast_standalone_superlevel_counts_are_well_formed(mast):
    """The MAST carriers' independent superlevel counts are well formed.

    This is the standalone half of the end-to-end control: the host graph
    search alone resolves every critical level of each MAST row. The tree arm
    is compared once the receipt accepts these carriers' fixed capacity.
    """

    for fixture in mast:
        assert not fixture.mesh.overflow
        rows = standalone_rows(fixture.mesh)
        counts = [row["brute_force"] for row in rows]
        values = np.asarray(fixture.mesh.vertex_psi)[
            np.asarray(fixture.mesh.vertex_valid)
        ]
        assert len(rows) == np.unique(values).size
        assert counts
        assert min(counts) >= 1
        # The highest sampled level holds exactly the single global maximum.
        assert counts[0] == 1
        # The deepest sampled level is the whole connected carrier, one region.
        assert counts[-1] == 1
        # The count never exceeds the number of live vertices at its level.
        for row in rows:
            assert row["brute_force"] <= int(np.count_nonzero(values > row["level"]))
        # The MAST equilibrium splits into more than one superlevel region.
        assert max(counts) >= 2


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
