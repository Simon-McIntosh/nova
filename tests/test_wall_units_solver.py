"""Solver contracts for packed vessel and limiter wall units."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.connectivity_boundary import (
    _points_inside_wall_units,
    _wall_segment_geometry,
    wall_height_shadow_mask,
)
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.steering_frames import _wall_segment_units
from nova.equilibrium.topology import TopologyState, topology_solve_receipt
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


configure_dtypes()
assert jax.config.jax_enable_x64 is True


VESSEL = np.asarray(
    [[0.0, -1.0], [2.0, -1.0], [2.0, 1.0], [0.0, 1.0]],
    dtype=np.float64,
)
BLADE = np.asarray([[1.35, -0.65], [1.15, 0.0], [1.35, 0.65]], dtype=np.float64)
BLADE_REGION = np.asarray(
    [[1.1, -0.3], [1.4, -0.3], [1.4, 0.3], [1.1, 0.3]], dtype=np.float64
)
WALL = np.concatenate((VESSEL, BLADE))
OFFSETS = np.asarray([0, len(VESSEL), len(WALL)], dtype=np.int32)
CLOSED = np.asarray([True, False])
VESSEL_KIND = np.asarray([True, False])


def _operator(*, closed_blade: bool = False) -> ForwardFluxOperator:
    """Return a matrix-free carrier retaining the synthetic packed wall."""
    wall = np.concatenate((VESSEL, BLADE_REGION)) if closed_blade else WALL
    offsets = np.asarray([0, len(VESSEL), len(wall)], dtype=np.int32)
    lattice = FluxLattice(np.linspace(0.1, 1.9, 7), np.linspace(-0.9, 0.9, 7))
    node_count = lattice.node_count

    def zero(psi_norm):
        return jnp.zeros_like(psi_norm)

    return ForwardFluxOperator(
        grid=FluxTarget(
            source_target=jnp.zeros((node_count, 1)),
            plasma_target=jnp.zeros((node_count, 1)),
            null=Null2D.from_coordinates(
                lattice.coordinate, hex_stencil(lattice.shape)
            ),
        ),
        wall=FluxTarget(
            source_target=jnp.zeros((len(wall), 1)),
            plasma_target=jnp.zeros((len(wall), 1)),
            null=Null1D(jnp.asarray(wall)),
        ),
        source=ForwardSource(core=DomainProfile(p_prime=zero, ff_prime=zero)),
        external_current=jnp.zeros(1),
        area=jnp.asarray(lattice.cell_area),
        inside_material=jnp.ones(node_count, dtype=bool),
        use_linear_moments=False,
        wall_unit_offsets=offsets,
        wall_unit_closed=np.asarray([True, closed_blade]),
        wall_unit_kinds=("vessel", "material"),
    )


def test_occupiable_containment_subtracts_the_blade() -> None:
    """Material inside the vessel is excluded without excluding its neighbour."""
    points = np.asarray([[1.25, 0.0], [0.8, 0.0]])
    wall = np.concatenate((VESSEL, BLADE_REGION))
    offsets = np.asarray([0, len(VESSEL), len(wall)], dtype=np.int32)
    contained = _points_inside_wall_units(
        points[:, 0],
        points[:, 1],
        wall[:, 0],
        wall[:, 1],
        offsets,
        np.asarray([True, True]),
        VESSEL_KIND,
    )
    np.testing.assert_array_equal(contained, [False, True])


def test_x_candidate_inside_the_blade_is_screened() -> None:
    """The topology candidate screen uses the occupiable region."""
    topology = _operator(closed_blade=True).topology
    candidates = jnp.asarray([[1.25, 0.0, 2.0], [0.8, 0.0, 1.0]])
    np.testing.assert_array_equal(topology.contained_x_candidates(candidates), [0, 1])


def test_first_contact_anchor_uses_a_bracket_inside_the_open_blade() -> None:
    """A limiter maximum can bind without wrapping through a unit endpoint."""
    topology = _operator().topology
    wall_flux = jnp.asarray([0.0, 0.1, 0.0, 0.1, 1.0, 2.0, 1.0])
    contact = topology.wall_anchor_data(wall_flux, 1)
    bracket = topology.wall_anchor_bracket(wall_flux, 1)
    np.testing.assert_array_equal(bracket, [4, 5, 6])
    np.testing.assert_allclose(contact[:2], BLADE[1], rtol=0.0, atol=1.0e-12)


def test_open_unit_has_no_terminal_or_inter_unit_segment() -> None:
    """Packed traversal closes the vessel and stops at the blade's end."""
    _start, end, valid, unit, _following = _wall_segment_geometry(
        WALL[:, 0], WALL[:, 1], OFFSETS, CLOSED
    )
    np.testing.assert_array_equal(valid, [1, 1, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(unit, [0, 0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(end[3], VESSEL[0])
    np.testing.assert_array_equal(end[6], BLADE[-1])


def test_wall_height_shadow_retains_unit_ownership() -> None:
    """The flat shadow mask marks nodes on both eligible physical units."""
    qualified = jnp.asarray([[1.0, -0.5], [jnp.nan, jnp.nan]])
    mask = wall_height_shadow_mask(
        WALL[:, 1],
        0.0,
        qualified[0],
        qualified,
        jnp.ones(len(WALL), dtype=bool),
        jnp.zeros(len(WALL), dtype=bool),
        0.01,
        0.1,
    )
    node_unit = np.searchsorted(OFFSETS[1:], np.arange(len(WALL)), side="right")
    assert set(node_unit[np.asarray(mask)]) == {0, 1}


def test_receipts_name_limiting_and_strike_units() -> None:
    """Wall ownership is explicit for a limit and the flat strike segments."""
    state = TopologyState(
        axis=jnp.asarray([0.8, 0.0]),
        axis_flux=jnp.asarray(0.0),
        boundary=jnp.asarray(BLADE[1]),
        boundary_flux=jnp.asarray(2.0),
        x_point=jnp.asarray([jnp.nan, jnp.nan]),
        x_point_flux=jnp.asarray(jnp.nan),
        wall_point=jnp.asarray(BLADE[1]),
        wall_point_flux=jnp.asarray(2.0),
        diverted=jnp.asarray(False),
        wall_unit_index=jnp.asarray(1, dtype=jnp.int32),
    )
    receipt = topology_solve_receipt((state,), solver_succeeded=True)
    assert receipt.as_dict()["limiting_unit_index"] == 1

    from nova.equilibrium.wall_mask import WallUnit

    units = (
        WallUnit(VESSEL[:, 0], VESSEL[:, 1], kind="vessel", closed=True),
        WallUnit(BLADE[:, 0], BLADE[:, 1], kind="material", closed=False),
    )
    np.testing.assert_array_equal(_wall_segment_units(units), [0, 0, 0, 0, 1, 1])
