"""The profile support clips each cell against the signed iterate flux.

A cell participates when any of its vertices lies on the confined side of
the boundary level, whatever its centroid flux or partition label says, and
its moments are the clipped polygon's.  These tests pin that geometric
candidacy on a two-cell synthetic case under the committed chord default.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.forward_operator import ForwardFluxOperator, support_clip_mode
from nova.equilibrium.separatrix_clip import AtomicCellMesh
from nova.jax.config import configure_dtypes


def _two_cell_machine() -> AtomicCellMesh:
    """Return the atomic mesh of two unit cells on ``R in [0.5, 2.5]``."""
    cells = (
        np.asarray([[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]]),
        np.asarray([[1.5, -0.5], [2.5, -0.5], [2.5, 0.5], [1.5, 0.5]]),
    )
    centres = np.asarray([[1.0, 0.0], [2.0, 0.0]])
    return AtomicCellMesh.from_cells(cells, centroids=centres)


def _clipped_support(atomic_mesh: AtomicCellMesh):
    """Trace the chord support on the boundary level ``R = 1.75``.

    ``inside_boundary = 1.75 - R`` is positive on the confined side, so cell
    zero is wholly inside and cell one straddles the level with its left
    vertex column inside and its centroid outside.
    """
    radii = np.asarray(atomic_mesh.node_coordinates)[:, 0]
    inside_boundary = jnp.asarray(1.75 - radii, dtype=jnp.float64)
    operator = object.__new__(ForwardFluxOperator)
    operator.polarity = 1
    operator.moment_geometry = SimpleNamespace(atomic_mesh=atomic_mesh)
    operator.shared_node_flux = lambda flux: flux
    masks = DomainMasks(
        label=jnp.asarray([PlasmaDomain.CORE, PlasmaDomain.CORE], dtype=jnp.int8),
        psi_norm=jnp.asarray([0.5, 1.5]),
    )
    topology = SimpleNamespace(boundary_flux=jnp.asarray(0.0))
    support = ForwardFluxOperator._profile_support(
        operator, masks, topology, inside_boundary, jnp.asarray([0.0, 0.0])
    )
    return support, masks


def test_default_mode_is_the_signed_flux_clip():
    """The committed chord clip clips, which whole-cell booking never did."""
    configure_dtypes()
    assert support_clip_mode() == "chord"
    atomic_mesh = _two_cell_machine()
    support, _masks = _clipped_support(atomic_mesh)
    # Both cells participate under the geometric candidacy.
    assert bool(support.included[0])
    assert bool(support.included[1])


def test_centroid_outside_vertex_inside_participates_with_clipped_area():
    """A cell outside by centroid but inside by a vertex keeps its sliver."""
    configure_dtypes()
    atomic_mesh = _two_cell_machine()
    support, masks = _clipped_support(atomic_mesh)

    # The straddling cell's centroid is outside (normalised flux > 1), yet
    # its left column of vertices lies on the confined side of the level, so
    # it participates with the clipped polygon: a quarter of the full cell.
    assert float(masks.psi_norm[1]) > 1.0
    assert bool(support.included[1])
    assert int(support.vertex_count[1]) >= 4
    assert float(support.area[1]) == pytest.approx(0.25, rel=0.0, abs=1.0e-12)

    # The wholly-inside cell keeps its full atom under the same clip level.
    assert float(support.area[0]) == pytest.approx(1.0, rel=0.0, abs=1.0e-12)


def test_moment_support_promotes_the_clipped_cell_to_the_confined_label():
    """The mask promotion follows the clipped polygon, not the centroid."""
    configure_dtypes()
    atomic_mesh = _two_cell_machine()
    support, masks = _clipped_support(atomic_mesh)
    promoted = ForwardFluxOperator._moment_support_masks(masks, support)
    assert bool(promoted.core[1])
    assert bool(promoted.profile_participation[1])
