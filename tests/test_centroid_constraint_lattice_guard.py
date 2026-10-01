"""The lattice-order guard on the cell-carried point read.

A flux vector is longer than its cells.  The operator assembles the pool it
reads as the carried cells first and the direct sampling nodes after them,
indexing the second block from the cell count, while the physical prefix adds
the wall nodes between the two.  The cell-carried point read therefore hands
the operator the state's first ``node_count`` values, and a call site that
sizes that slice from the physical prefix moves every sampled value by the
difference and answers with a neighbouring cell's polynomial -- a shift a
uniform offset read passes through unchanged, which is why neither the
unit-leverage assertion nor an offset assertion catches it.

These cases pin the guard, not the arithmetic it protects: the pool slice is
compared against the cell count the stencil convention states, and the pre-fix
prefix indexing is refused rather than silently answered.  Reverting the guard
to the prefix slice reddens :func:`test_pre_fix_prefix_indexing_is_refused`.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from nova.equilibrium.constraint import _mesh_carried_point_flux, _sampled_cell_pool
from nova.jax.config import configure_dtypes

# one carried value per cell, one extra physical node between the carried cells
# and the sampling nodes, then one sampling node per cell
_CELL_MESH_VALUES = (0.1, 0.2, 0.3, 0.9, 10.0, 20.0, 30.0)
_PHYSICAL_NODE_NUMBER = 4


class _CellMeshOperator:
    """Operator stub whose point read mixes the cell and its sampling nodes.

    The read is a weighted combination -- one weight on the owning cell's own
    value, the rest on its direct sampling nodes -- so the weights sum to one
    and a constant added to every flux value moves the read by exactly that
    constant.  The pool is assembled the way the mesh assembles it, carried
    cells first and sampling nodes after them indexed from the cell count, so
    a caller that hands the read a longer carried block than the cells shifts
    every sampled value and reads a neighbouring cell's neighbourhood.
    """

    physical_node_number = _PHYSICAL_NODE_NUMBER

    def __init__(self, node_number: int) -> None:
        self.grid = SimpleNamespace(node_number=node_number)

    def sample_node_flux(self, state):
        return jnp.asarray(state)[self.physical_node_number :]

    def sample_flux_field(self, centroid_flux, sample_flux, points):
        pool = jnp.concatenate([jnp.asarray(centroid_flux), jnp.asarray(sample_flux)])
        count = self.grid.node_number
        query = jnp.asarray(points)[:, :, 0]
        carried = pool[:count]
        sampled = pool[count + jnp.arange(count)]
        mixed = 0.5 * carried + 0.5 * sampled
        values = jnp.broadcast_to(mixed[:, None], query.shape) + query
        zeros = jnp.zeros_like(values)
        return values, zeros, zeros


def _cell_mesh_profile() -> SimpleNamespace:
    """Return a three-cell carrier whose centroids sit on the inboard axis."""
    mesh = SimpleNamespace(
        coordinate=np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]), node_count=3
    )
    return SimpleNamespace(lattice=mesh, operator=_CellMeshOperator(node_number=3))


def _cell_mesh_state():
    """Return the test state, built only after extended precision is enabled."""
    return jnp.asarray(_CELL_MESH_VALUES)


def _owning_cell_mix(owner: int) -> float:
    count = 3
    carried = _CELL_MESH_VALUES[:count]
    sampled = _CELL_MESH_VALUES[_PHYSICAL_NODE_NUMBER:][:count]
    return 0.5 * carried[owner] + 0.5 * sampled[owner]


def test_the_pool_slice_is_the_cell_count_the_stencil_states() -> None:
    """The guard's positive: the carried block is the stencil's cell count.

    The three cells the stencil convention states are the slice that is taken,
    whatever the state carries beyond them, and the state carries more than its
    cells here -- so this case fails if the cut is placed anywhere else.
    """
    configure_dtypes()
    profile = _cell_mesh_profile()
    operator = profile.operator
    state = _cell_mesh_state()
    node_count = profile.lattice.node_count

    assert int(operator.grid.node_number) == int(node_count)
    assert int(operator.physical_node_number) > int(node_count)

    pool = _sampled_cell_pool(state, operator, pool_length=node_count)
    np.testing.assert_allclose(
        np.asarray(pool), np.asarray(state)[:node_count], rtol=0.0, atol=0.0
    )
    assert int(pool.shape[0]) == int(operator.grid.node_number)


def test_pre_fix_prefix_indexing_is_refused() -> None:
    """The pre-fix indexing is refused: this is the case the guard reddens.

    Applying the pre-fix slice -- sizing the carried block from the physical
    prefix instead of the cell count the stencil convention states -- must
    raise, and the refusal names the convention it is stated on.  With the
    guard reverted to accept the prefix, this assertion reddens.
    """
    configure_dtypes()
    profile = _cell_mesh_profile()
    operator = profile.operator
    state = _cell_mesh_state()

    with np.testing.assert_raises_regex(
        ValueError, "carries the operator's cells first"
    ):
        _sampled_cell_pool(state, operator, pool_length=operator.physical_node_number)


def test_pre_fix_prefix_indexing_shifts_the_read_off_its_own_cell() -> None:
    """The defect the guard stops: a prefix-sized slice reads a neighbour.

    Built by hand over the prefix slice, the read answers with a different
    cell's neighbourhood than the one that owns the query, which is why the
    refusal above is load-bearing rather than cosmetic.  The guard is bypassed
    here to show what the refused call would have returned.
    """
    configure_dtypes()
    profile = _cell_mesh_profile()
    operator = profile.operator
    state = _cell_mesh_state()
    node_count = int(profile.lattice.node_count)
    owner = 1
    point = jnp.asarray([2.0, 0.0])

    read = float(np.asarray(_mesh_carried_point_flux(profile, state, point)))
    np.testing.assert_allclose(
        read, _owning_cell_mix(owner) + 2.0, rtol=0.0, atol=1.0e-12
    )

    prefix = jnp.asarray(state)[: int(operator.physical_node_number)]
    points = jnp.zeros((node_count, 1, 2), dtype=jnp.float64)
    points = points.at[owner, 0].set(point)
    shifted = operator.sample_flux_field(
        prefix, operator.sample_node_flux(state), points
    )[0]
    shifted_read = float(np.asarray(shifted)[owner, 0])
    assert shifted_read != read
    # the prefix slice drags the extra physical node into the sampled block, so
    # the owning cell is read off a slot that belongs to another cell
    shifted_mix = (
        0.5 * _CELL_MESH_VALUES[owner]
        + 0.5 * _CELL_MESH_VALUES[_PHYSICAL_NODE_NUMBER - 1 + owner]
    )
    np.testing.assert_allclose(shifted_read, shifted_mix + 2.0, rtol=0.0, atol=1.0e-12)
    assert node_count == int(operator.grid.node_number)
