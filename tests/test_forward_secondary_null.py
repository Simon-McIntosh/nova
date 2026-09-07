"""The secondary X-point slot is confined to the wall polygon.

The second X-point slot on the forward-solve labelled read qualifies
candidates by coordinates being finite and distinct from the primary, then
ranks them by poloidal-flux distance from the axis. A candidate that the flux
locator finds outside the first wall — a coil-adjacent or centre-column saddle
that scores higher on raw flux than any in-vessel second null — used to win
that slot. This module pins the containment remedy: candidates are qualified
against the wall polygon with the same ``_points_inside_polygon`` test the
primary X-point selector applies, and the slot reports NaN when no distinct
in-vessel saddle exists rather than the best out-of-vessel one.

The fixtures are manufactured poloidal-flux maps over the same fixed-shape
``Topology`` locator the solve reads, arranged so the wall polygon either
encloses both saddle barriers or clips the outer one. That drives
:meth:`nova.equilibrium.forward.ForwardProfile._secondary_x_point` directly,
with the containment behaviour pinned to a geometry the locator resolves.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    from nova.biot.null import Null1D, Null2D
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.forward_operator import ForwardTopologyState
    from nova.equilibrium.topology import Topology
    from nova.geometry.hexstencil import hex_stencil

    from nova.equilibrium.connectivity_boundary import _points_inside_polygon


#: Axial set-off of the manufactured magnetic axis [m].
PLASMA_RADIUS = 1.7
#: Candidate separation enforced by the qualification distance [m].
QUALIFICATION_DISTANCE = 0.09
#: Grid resolution of the manufactured map (nodes per side).
GRID_SIZE = 33


def _vertical_potential(vertical):
    """Return a one-dimensional potential with two saddle barriers.

    The barriers at ``0.35`` and ``0.70`` reproduce the validated
    single-null arrangement used by the hex-flood secondary test: the first
    barrier is the primary separatrix saddle, the second the next-qualified
    saddle. The extra stationary points at ``0.52`` and ``0.98`` keep the
    potential flat enough that only the two barriers are detected.
    """
    roots = (0.0, 0.35, 0.52, 0.70, 0.98)
    derivative = np.poly1d(roots, r=True)
    return 100.0 * np.polyint(derivative)(vertical)


def _flux(radial, vertical):
    """Return the manufactured total poloidal flux at local coordinates."""
    return radial**2 + _vertical_potential(vertical)


def _manufactured_machine(z_half):
    """Return the topology, physical flux and grid node count for one wall."""
    local_radius = np.linspace(-1.25, 1.25, GRID_SIZE)
    vertical = np.linspace(-1.25, 1.25, GRID_SIZE)
    radius_grid, height_grid = np.meshgrid(
        local_radius + PLASMA_RADIUS, vertical, indexing="ij"
    )
    coordinate = np.c_[radius_grid.ravel(), height_grid.ravel()]

    angle = 2.0 * np.pi * np.arange(64) / 64
    wall = np.c_[PLASMA_RADIUS + 1.15 * np.cos(angle), z_half * np.sin(angle)]
    topology = Topology(
        Null2D.from_coordinates(
            coordinate, hex_stencil((GRID_SIZE, GRID_SIZE)), maxsize=5
        ),
        Null1D(jnp.asarray(wall)),
    )
    grid_flux = _flux(radius_grid - PLASMA_RADIUS, height_grid)
    wall_flux = _flux(wall[:, 0] - PLASMA_RADIUS, wall[:, 1])
    psi = jnp.asarray(np.r_[grid_flux.ravel(), wall_flux])
    return topology, psi, coordinate.shape[0]


def _topology_state(topology):
    """Return a diverted-resolved landmark state over the manufactured map."""
    primary = _flux(0.0, 0.35)
    return ForwardTopologyState(
        axis=jnp.asarray([PLASMA_RADIUS, 0.0]),
        axis_flux=jnp.asarray(float(_flux(0.0, 0.0))),
        boundary=jnp.asarray([PLASMA_RADIUS, 0.35]),
        boundary_flux=jnp.asarray(float(primary)),
        x_point=jnp.asarray([PLASMA_RADIUS, 0.35]),
        x_point_flux=jnp.asarray(float(primary)),
        wall_point=jnp.asarray([PLASMA_RADIUS, 0.98]),
        wall_point_flux=jnp.asarray(float(_flux(0.0, 0.98))),
        _class_margin_read=lambda: jnp.asarray(jnp.inf),
    )


def _operator_shell(topology, grid_nodes):
    """Return a solve shell carrying the operator attributes the selector uses."""
    return SimpleNamespace(
        operator=SimpleNamespace(
            physical_node_number=grid_nodes,
            polarity=1,
            _x_qualification_distance=jnp.asarray(QUALIFICATION_DISTANCE),
            topology=topology,
            _fixed_design_topology=topology,
        )
    )


@pytest.fixture(scope="module")
def full_wall_machine():
    """Return a machine whose wall encloses both saddle barriers."""
    topology, psi, grid_nodes = _manufactured_machine(z_half=1.22)
    return (
        topology,
        psi,
        grid_nodes,
        _operator_shell(topology, grid_nodes),
        _topology_state(topology),
    )


@pytest.fixture(scope="module")
def clipped_wall_machine():
    """Return a machine whose limiter clips the outer saddle barrier."""
    topology, psi, grid_nodes = _manufactured_machine(z_half=0.60)
    return (
        topology,
        psi,
        grid_nodes,
        _operator_shell(topology, grid_nodes),
        _topology_state(topology),
    )


def _candidates(shell, psi):
    """Return the finite X-point candidates the locator resolves, with r/z."""
    operator = shell.operator
    physical = jnp.asarray(psi)[: operator.physical_node_number]
    grid_flux, _wall_flux = operator.topology.split_flux_map(physical)
    _o_candidates, x_candidates = operator._fixed_design_topology.grid(grid_flux)
    candidates = np.asarray(x_candidates)
    finite = np.all(np.isfinite(candidates), axis=1)
    return candidates[finite]


def test_secondary_slot_is_nan_when_best_distinct_saddle_lies_outside_the_limiter(
    clipped_wall_machine,
):
    """An out-of-vessel second saddle no longer wins the secondary slot."""
    topology, psi, _grid_nodes, operator, state = clipped_wall_machine
    wall = np.asarray(topology.wall.coordinate)

    candidates = _candidates(operator, psi)
    assert len(candidates) >= 2
    primary = np.asarray(state.x_point)
    primary_distance = np.linalg.norm(candidates[:, :2] - primary, axis=1)
    inside_wall = np.asarray(
        _points_inside_polygon(
            candidates[:, 0], candidates[:, 1], wall[:, 0], wall[:, 1]
        )
    )
    distinct = primary_distance > QUALIFICATION_DISTANCE

    # The fixture premise: the only distinct saddle lies outside the limiter,
    # so without containment the slot would report it.
    distinct_outside = distinct & ~inside_wall
    assert not np.any(distinct & inside_wall)
    assert int(np.sum(distinct_outside)) == 1
    outside = candidates[distinct_outside][0]
    assert abs(outside[1]) > 0.60

    secondary = np.asarray(ForwardProfile._secondary_x_point(operator, psi, state))
    assert bool(np.isnan(secondary).all())


def test_diverted_fixture_keeps_its_in_vessel_second_null_unchanged(
    full_wall_machine,
):
    """Containment perturbs nothing when the second null is in-vessel."""
    topology, psi, _grid_nodes, operator, state = full_wall_machine
    wall = np.asarray(topology.wall.coordinate)

    candidates = _candidates(operator, psi)
    primary = np.asarray(state.x_point)
    primary_distance = np.linalg.norm(candidates[:, :2] - primary, axis=1)
    inside_wall = np.asarray(
        _points_inside_polygon(
            candidates[:, 0], candidates[:, 1], wall[:, 0], wall[:, 1]
        )
    )
    distinct = primary_distance > QUALIFICATION_DISTANCE
    score = candidates[:, 2] - float(np.asarray(state.axis_flux))

    # The fixture premise: every distinct saddle lies in-vessel, so the wall
    # test plays no role in the selection and the slot must be untouched.
    assert bool(np.all(inside_wall[distinct]))

    # The pre-fix selection: the highest-scoring distinct candidate anywhere.
    old_index = int(np.argmax(np.where(distinct, score, -np.inf)))
    old_selection = candidates[old_index, :2]

    secondary = np.asarray(ForwardProfile._secondary_x_point(operator, psi, state))
    assert bool(np.isfinite(secondary).all())
    np.testing.assert_allclose(secondary, old_selection, rtol=0.0, atol=1.0e-12)

    # The same selection is reached when the read is traced.
    compiled = jax.jit(
        lambda flux, landmarks: ForwardProfile._secondary_x_point(
            operator, flux, landmarks
        )
    )
    traced = np.asarray(compiled(psi, state))
    np.testing.assert_allclose(traced, secondary, rtol=0.0, atol=1.0e-12)
