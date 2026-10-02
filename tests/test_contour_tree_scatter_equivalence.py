"""The carrier neighbour combine is a duplicate-index maximum scatter.

The carrier sweep folds one contribution per edge endpoint into a per-vertex
carrier key, and a vertex named by several edges supplies that key more than
once.  A last-write-wins combine would make the reduction depend on the order
the carrier happens to store its edges, so that a receipt could change without
the graph changing.  These tests pin the duplicate-index maximum scatter the
sweep relies on against ``jax.ops.segment_max`` and a NumPy reference, bit for
bit, on carrier graphs that carry a saturated vertex, an isolated vertex and
repeated identical edges.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.contour_tree import _carrier_adjacency
from nova.jax.config import configure_dtypes


configure_dtypes()
assert jax.config.jax_enable_x64 is True

_INT32_MIN = int(np.iinfo(np.int32).min)

# The neighbour width the carrier adjacency reserves once a mesh edge count
# exceeds twice its vertex count: a denser carrier has bounded vertex degree.
_SATURATED_DEGREE = 16


def _endpoint_keys(edges) -> np.ndarray:
    """Every edge endpoint in carrier storage order, one key per occurrence."""

    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    return np.concatenate((edges[:, 0], edges[:, 1])).astype(np.int32)


def _max_scatter(keys, values, num_segments):
    """Fold values into per-key maxima with a duplicate-index scatter."""

    identity = jnp.full((num_segments,), _INT32_MIN, dtype=jnp.int32)
    return identity.at[keys].max(values)


def _numpy_reference(keys, values, num_segments) -> np.ndarray:
    """A NumPy reduction of the same keys and values."""

    reference = np.full((num_segments,), _INT32_MIN, dtype=np.int32)
    np.maximum.at(reference, np.asarray(keys), np.asarray(values))
    return reference


def _assert_agrees(keys, values, num_segments) -> np.ndarray:
    """The scatter equals segment_max and the NumPy reference, bit for bit."""

    keys = jnp.asarray(keys, dtype=jnp.int32)
    values = jnp.asarray(values, dtype=jnp.int32)
    scatter = _max_scatter(keys, values, num_segments)
    segmented = jax.ops.segment_max(values, keys, num_segments=num_segments)
    reference = _numpy_reference(keys, values, num_segments)

    assert scatter.dtype == jnp.int32
    assert segmented.dtype == jnp.int32
    assert np.array_equal(np.asarray(scatter), np.asarray(segmented))
    assert np.asarray(scatter).tobytes() == np.asarray(segmented).tobytes()
    assert np.array_equal(np.asarray(segmented), reference)
    return np.asarray(scatter)


def test_saturated_vertex_combine_matches_segment_max():
    """A vertex at the carrier's full neighbour width reduces as a maximum.

    The hub is named by sixteen edges, so the carrier adjacency fills its whole
    neighbour row and the scatter sees the same key sixteen times.  The values
    fall along the storage order, so the reduction keeps the first occurrence's
    large value rather than the last occurrence's small one.
    """

    leaves = _SATURATED_DEGREE
    hub = np.zeros(leaves, dtype=np.int32)
    spokes = np.arange(1, leaves + 1, dtype=np.int32)
    edges = np.stack((hub, spokes), axis=1)

    neighbours, neighbour_valid, overflow = _carrier_adjacency(
        jnp.asarray(edges), jnp.ones(leaves, dtype=bool), leaves + 1
    )
    assert not bool(overflow)
    assert int(np.sum(np.asarray(neighbour_valid)[0])) == _SATURATED_DEGREE
    assert np.asarray(neighbours)[0].tolist() == spokes.tolist()

    keys = _endpoint_keys(edges)
    values = np.arange(2 * leaves, 0, -1, dtype=np.int32)

    result = _assert_agrees(keys, values, leaves + 1)

    assert result[0] == 2 * leaves


def test_isolated_vertex_segment_keeps_the_identity():
    """A vertex no edge names reduces to the identity in every reduction."""

    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int32)

    _, neighbour_valid, _ = _carrier_adjacency(
        jnp.asarray(edges), jnp.ones(2, dtype=bool), 4
    )
    assert int(np.sum(np.asarray(neighbour_valid)[3])) == 0

    keys = _endpoint_keys(edges)
    values = np.asarray([3, -7, 5, 11], dtype=np.int32)

    result = _assert_agrees(keys, values, 4)

    assert 3 not in keys.tolist()
    assert result[3] == _INT32_MIN


def test_repeated_identical_edges_keep_one_maximum_per_vertex():
    """An edge stored three times still yields one maximum per endpoint."""

    edges = np.asarray([[0, 1], [0, 1], [0, 1], [1, 2]], dtype=np.int32)
    keys = _endpoint_keys(edges)
    values = np.asarray([5, 4, 9, 2, 3, 1, 7, 6], dtype=np.int32)

    result = _assert_agrees(keys, values, 3)

    assert keys.tolist() == [0, 0, 0, 1, 1, 1, 1, 2]
    assert result[0] == 9
    assert result[1] == 7
    assert result[2] == 6


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_carrier_endpoint_scatter_matches_segment_max(seed):
    """Small random carriers with repeated endpoints reduce as a maximum."""

    rng = np.random.default_rng(seed)
    vertex_count = int(rng.integers(3, 7))
    edge_count = int(rng.integers(4, 13))
    edges = rng.integers(0, vertex_count, size=(edge_count, 2), dtype=np.int32)

    keys = _endpoint_keys(edges)
    values = rng.integers(-500, 500, size=keys.size, dtype=np.int32)

    result = _assert_agrees(keys, values, vertex_count)

    assert result.shape == (vertex_count,)
