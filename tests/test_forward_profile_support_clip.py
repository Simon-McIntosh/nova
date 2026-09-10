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


def _support_matches(first, second) -> bool:
    """Return whether two supports carry byte-identical booked geometry."""
    for name in ("included", "area", "vertex_count"):
        if not np.array_equal(
            np.asarray(getattr(first, name)), np.asarray(getattr(second, name))
        ):
            return False
    return np.array_equal(
        np.asarray(first.support_vertices), np.asarray(second.support_vertices)
    )


def test_support_is_frozen_within_a_trip_and_refreshed_at_its_boundary():
    """One trip's clip geometry is the frozen record, not the live iterate.

    Two live flux states whose signed-flux clips genuinely differ (so the
    geometry would move if it were re-read per evaluation) must yield the
    identical support through ``_partition_for_state`` on one frozen
    partition -- that is what holds the Jacobian-vector products and the
    amplitude normalisation on the trip's geometry -- while a fresh partition
    read at the second state refreshes the clip to that state's live result.
    """
    configure_dtypes()
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture
    from tests.rotating_equilibrium_references import reference_cases

    case = reference_cases()["weak-rotation-reactor"].static_limit()
    machine = oracle_fixture.cached_machine(
        case, -110, wall_nodes=oracle_fixture.WALL_POINT_COUNT
    )
    operator = oracle_fixture.forward_operator(case, machine)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    terminal = np.asarray(
        oracle_fixture.exact_state(case, coordinates), dtype=np.float64
    )
    physical_count = int(operator.grid.node_number)
    physical = terminal[:physical_count]
    span = float(np.ptp(physical))
    boundary_band = np.abs(physical - float(np.median(physical))) < 0.5 * span
    perturbed = terminal.copy()
    perturbed[:physical_count] = physical + np.where(boundary_band, 1.0e-3 * span, 0.0)

    live_first = operator._profile_support(
        *_support_inputs(operator, jnp.asarray(terminal))
    )
    live_second = operator._profile_support(
        *_support_inputs(operator, jnp.asarray(perturbed))
    )
    # The two states genuinely move the boundary, so a live re-read would
    # change the geometry inside a trip.
    assert not _support_matches(live_first, live_second)

    frozen = operator._frozen_topology_partition(jnp.asarray(terminal), None)
    inside_first = operator._partition_for_state(jnp.asarray(terminal), frozen)
    inside_second = operator._partition_for_state(jnp.asarray(perturbed), frozen)
    # Within one trip two different live flux states yield the identical
    # support, and it is the frozen record rather than either live re-read.
    assert _support_matches(inside_first[3], inside_second[3])
    assert _support_matches(inside_first[3], frozen.profile_support)
    assert _support_matches(inside_second[3], frozen.profile_support)

    # A trip boundary re-reads the partition at the new state and refreshes
    # the clip to that state's own geometry.
    refreshed = operator._frozen_topology_partition(
        jnp.asarray(perturbed), None, frozen.residual_shadow
    )
    assert _support_matches(refreshed.profile_support, live_second)


def _support_inputs(operator, psi):
    """Return the profile-support arguments one state's live read needs."""
    physical = jnp.asarray(psi)[: operator.physical_node_number]
    masks, topology, _connected, _admitted = operator._fixed_design_read(physical, None)
    sample_flux = operator.sample_node_flux(psi)
    sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
    return masks, topology, physical, sample_psi_norm
