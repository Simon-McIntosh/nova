"""Contracts of the fine-grid display flux map."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nova.biot.greens import hybrid_greens
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.display_flux import (
    DEFAULT_SPACING,
    DisplayCoupling,
    build_display_flux_map,
)
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.linalg.interpolant import Linear
from nova.equilibrium.stencil_mesh import CellCurrentMoments

#: Radial and vertical extent [m] of the manufactured lattice.
RADIUS = (0.6, 1.4)
HEIGHT = (-0.45, 0.45)
#: Conductor sections [m] and the current vector the operator carries.
CONDUCTOR_RADIUS = np.asarray([0.75, 1.25])
CONDUCTOR_HEIGHT = np.asarray([0.0, 0.0])
CONDUCTOR_WIDTH = 0.05
CONDUCTOR_CURRENT = np.asarray([1.0e5, -1.0e5])
#: The current the display build carries.  A one-percent edit to this scale is
#: the declared negative control: the node reproduction cannot hold under it.
DISPLAY_CURRENT_SCALE = 1.0


def _lattice(nodes: int = 9) -> FluxLattice:
    """Return the manufactured uniform lattice."""
    return FluxLattice(
        np.linspace(RADIUS[0], RADIUS[1], nodes),
        np.linspace(HEIGHT[0], HEIGHT[1], nodes),
    )


def _source() -> ForwardSource:
    """Return the absolute source the manufactured operator carries."""
    nodes = jnp.linspace(0.0, 1.0, 9)
    return ForwardSource(
        core=DomainProfile(
            p_prime=Linear(nodes, jnp.full(9, -1.0)),
            ff_prime=Linear(nodes, jnp.full(9, 2.0)),
        ),
        boundary_pressure=0.0,
        boundary_field_function=1.0,
    )


def _plasma_columns(target: np.ndarray, source: np.ndarray, dr: float, dz: float):
    """Return the uniform, radial and vertical plasma moment columns."""
    columns = [
        hybrid_greens(
            target[:, 0],
            target[:, 1],
            float(radius),
            float(height),
            float(dr),
            float(dz),
        )
        for radius, height in source
    ]
    uniform, radial, vertical = zip(*columns, strict=True)
    return (
        np.stack(uniform, axis=1),
        np.stack(radial, axis=1),
        np.stack(vertical, axis=1),
    )


def _profile(nodes: int = 9) -> tuple[ForwardProfile, FluxLattice]:
    """Return the manufactured forward operator and its lattice."""
    lattice = _lattice(nodes)
    coordinate = lattice.coordinate
    source_columns = [
        hybrid_greens(
            coordinate[:, 0],
            coordinate[:, 1],
            float(radius),
            float(height),
            CONDUCTOR_WIDTH,
            CONDUCTOR_WIDTH,
        )[0]
        for radius, height in zip(CONDUCTOR_RADIUS, CONDUCTOR_HEIGHT, strict=True)
    ]
    plasma_columns = _plasma_columns(
        coordinate, coordinate, lattice.radial_step, lattice.vertical_step
    )
    angle = 2.0 * np.pi * np.arange(12) / 12
    wall = np.c_[1.0 + 0.15 * np.cos(angle), 0.15 * np.sin(angle)]
    node = lattice.node_count
    profile = ForwardProfile.from_lattice(
        lattice,
        _source(),
        external_current=jnp.asarray(CONDUCTOR_CURRENT.copy()),
        source_to_grid=np.stack(source_columns, axis=1),
        plasma_to_grid=plasma_columns[0],
        source_to_wall=np.zeros((wall.shape[0], len(CONDUCTOR_CURRENT))),
        plasma_to_wall=np.zeros((wall.shape[0], node)),
        wall_coordinate=wall,
        cubic_cell_average=False,
    )
    return profile, lattice


def _coupling(profile: ForwardProfile) -> DisplayCoupling:
    """Return the target-point coupling the manufactured operator is built from."""
    lattice = profile.lattice
    cells = lattice.coordinate
    dr = lattice.radial_step
    dz = lattice.vertical_step
    drive = CONDUCTOR_CURRENT.copy()
    drive[0] *= DISPLAY_CURRENT_SCALE

    def plasma(points: np.ndarray):
        """Return the three plasma moment columns at ``points``."""
        return _plasma_columns(points, cells, dr, dz)

    def external(points: np.ndarray) -> np.ndarray:
        """Return the conductor-only flux at ``points`` [Wb]."""
        flux = np.zeros(points.shape[0], dtype=np.float64)
        for radius, height, current in zip(
            CONDUCTOR_RADIUS, CONDUCTOR_HEIGHT, drive, strict=True
        ):
            flux += (
                current
                * hybrid_greens(
                    points[:, 0],
                    points[:, 1],
                    float(radius),
                    float(height),
                    CONDUCTOR_WIDTH,
                    CONDUCTOR_WIDTH,
                )[0]
            )
        return flux

    return DisplayCoupling(plasma=plasma, external=external)


def _moments(lattice: FluxLattice) -> CellCurrentMoments:
    """Return a manufactured cell-current moment triple on ``lattice``."""
    node = lattice.node_count
    cell_current = np.zeros(node)
    core = np.hypot(lattice.coordinate[:, 0] - 1.0, lattice.coordinate[:, 1]) < 0.25
    cell_current[core] = 1.0e5
    zeros = jnp.zeros(node)
    return CellCurrentMoments(jnp.asarray(cell_current), zeros, zeros)


def _state(profile: ForwardProfile, moments: CellCurrentMoments) -> np.ndarray:
    """Return the state whose currents are ``moments``."""
    raster = np.asarray(profile.operator.raster_image(moments), dtype=np.float64)
    assert raster.size == profile.lattice.node_count
    return raster


def test_fine_map_reproduces_the_operator_raster_image():
    """Evaluating the coupling at the lattice nodes reproduces raster_image."""
    profile, lattice = _profile()
    moments = _moments(lattice)
    display = build_display_flux_map(
        profile,
        _state(profile, moments),
        coupling=_coupling(profile),
        moments=moments,
        chunk=128,
    )
    assert display.reproduction_relative_error < 1.0e-10
    assert display.max_node_mismatch_wb < 1.0e-6


def test_display_map_spacing_and_saddle_are_reported():
    """The map is drawn at the plan's spacing floor and carries its two saddles."""
    profile, lattice = _profile()
    moments = _moments(lattice)
    display = build_display_flux_map(
        profile,
        _state(profile, moments),
        coupling=_coupling(profile),
        moments=moments,
        xpoint_rz=np.asarray([1.08, 0.0]),
        chunk=128,
    )
    assert display.spacing_m <= DEFAULT_SPACING
    np.testing.assert_allclose(np.diff(display.radius), display.spacing_m)
    assert display.radius.size > lattice.radius.size
    assert np.all(np.isfinite(display.saddle_rz))
    assert np.linalg.norm(display.saddle_rz - np.asarray([1.08, 0.0])) < 0.05
    assert np.all(np.isfinite(display.lattice_saddle_rz))
    assert np.isfinite(display.saddle_offset_mm)
    assert display.max_node_mismatch_wb < 1.0e-6
    assert np.all(np.isfinite(display.flux))


def test_a_coarser_than_declared_spacing_is_refused():
    """A map coarser than the plan's floor is refused rather than drawn."""
    profile, lattice = _profile()
    moments = _moments(lattice)
    with pytest.raises(ValueError, match="coarser"):
        build_display_flux_map(
            profile,
            _state(profile, moments),
            coupling=_coupling(profile),
            moments=moments,
            spacing=2.0 * DEFAULT_SPACING,
        )
