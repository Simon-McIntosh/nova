"""The coil-edit panel renderer's display map against the operator it draws.

A panel is drawn from a stored state alone: the panel's conductor currents and
plasma moments are contracted against the machine coupling on the display
grid.  The map therefore has to agree with the operator's own ``raster_image``
at the lattice nodes, which is the one check that catches a coupling whose
frames differ from the operator's -- a circuit column dropped, a moment column
contracted in the wrong order, or a prescribed response built against points
the operator never saw.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace

from benchmarks.coil_edit_latency import DEFAULT_DISPLAY_SPACING, _panel_display_map
from nova.biot.greens import hybrid_greens
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.display_flux import DisplayCoupling
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.forward_operator import PrescribedCurrentField
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.jax.config import configure_dtypes
from nova.linalg.interpolant import Linear

configure_dtypes()
assert jax.config.jax_enable_x64 is True, (
    "the map is compared against the operator's own raster, so the test process "
    "must resolve the same working precision the operator was built in"
)

#: The re-solved state's conductor sections [m] and the current it carries.
CONDUCTOR_RADIUS = np.asarray([0.75, 1.25])
CONDUCTOR_HEIGHT = np.asarray([0.0, 0.0])
CONDUCTOR_WIDTH = 0.05
CONDUCTOR_CURRENT = np.asarray([8.0e5, -5.0e5])


def _lattice(nodes: int = 7) -> FluxLattice:
    """Return the lattice the state was solved on."""
    return FluxLattice(
        np.linspace(0.6, 1.4, nodes),
        np.linspace(-0.45, 0.45, nodes),
    )


def _source() -> ForwardSource:
    """Return the source functions the re-solved state was carried under."""
    nodes = jnp.linspace(0.0, 1.0, 9)
    return ForwardSource(
        core=DomainProfile(
            p_prime=Linear(nodes, jnp.full(9, -1.0)),
            ff_prime=Linear(nodes, jnp.full(9, 2.0)),
        ),
        boundary_pressure=0.0,
        boundary_field_function=1.0,
    )


def _conductor_columns(points: np.ndarray) -> np.ndarray:
    """Return the conductor response columns [Wb/A] at ``points``."""
    return np.column_stack(
        [
            hybrid_greens(
                points[:, 0],
                points[:, 1],
                float(radius),
                float(height),
                CONDUCTOR_WIDTH,
                CONDUCTOR_WIDTH,
            )[0]
            for radius, height in zip(CONDUCTOR_RADIUS, CONDUCTOR_HEIGHT, strict=True)
        ]
    )


def _plasma_columns(
    points: np.ndarray, cells: np.ndarray, dr: float, dz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three plasma moment columns at ``points``."""
    columns = [
        hybrid_greens(
            points[:, 0],
            points[:, 1],
            float(radius),
            float(height),
            float(dr),
            float(dz),
        )
        for radius, height in cells
    ]
    uniform, radial, vertical = zip(*columns, strict=True)
    return (
        np.stack(uniform, axis=1),
        np.stack(radial, axis=1),
        np.stack(vertical, axis=1),
    )


def _profile() -> tuple[ForwardProfile, FluxLattice]:
    """Return the re-solved state's operator, its drive carried as prescribed."""
    lattice = _lattice()
    cells = np.asarray(lattice.coordinate, dtype=float)
    dr = float(np.diff(lattice.radius).mean())
    dz = float(np.diff(lattice.height).mean())
    plasma = _plasma_columns(cells, cells, dr, dz)
    angle = 2.0 * np.pi * np.arange(12) / 12
    wall = np.c_[1.0 + 0.15 * np.cos(angle), 0.15 * np.sin(angle)]
    node = lattice.node_count
    profile = ForwardProfile.from_lattice(
        lattice,
        _source(),
        external_current=np.zeros_like(CONDUCTOR_CURRENT),
        source_to_grid=_conductor_columns(cells),
        plasma_to_grid=plasma[0],
        source_to_wall=np.zeros((wall.shape[0], CONDUCTOR_CURRENT.size)),
        plasma_to_wall=np.zeros((wall.shape[0], node)),
        wall_coordinate=wall,
        cubic_cell_average=False,
    )
    policy = PrescribedCurrentField(
        response=jnp.asarray(
            np.vstack([_conductor_columns(cells), _conductor_columns(wall)])
        ),
        current=jnp.asarray(CONDUCTOR_CURRENT),
    )
    profile = replace(
        profile,
        operator=replace(
            profile.operator,
            external_current=jnp.zeros_like(profile.operator.external_current),
            prescribed_current_field=policy,
        ),
    )
    return profile, lattice


def _moments(lattice: FluxLattice) -> CellCurrentMoments:
    """Return the re-solved state's cell-current moment triple."""
    radius = np.asarray(lattice.coordinate[:, 0], dtype=float)
    height = np.asarray(lattice.coordinate[:, 1], dtype=float)
    core = np.hypot(radius - 1.0, height) < 0.25
    return CellCurrentMoments(
        jnp.asarray(np.where(core, 1.0e5, 0.0)),
        jnp.zeros(radius.size, dtype=jnp.float64),
        jnp.zeros(radius.size, dtype=jnp.float64),
    )


def _coupling(lattice: FluxLattice) -> DisplayCoupling:
    """Return the coupling the panel renderer contracts the state against."""
    cells = np.asarray(lattice.coordinate, dtype=float)
    dr = float(np.diff(lattice.radius).mean())
    dz = float(np.diff(lattice.height).mean())
    node = cells.shape[0]

    def plasma(points: np.ndarray):
        """Return the plasma moment columns at ``points``."""
        return _plasma_columns(points, cells, dr, dz)

    def external(points: np.ndarray) -> np.ndarray:
        """Return the prescribed conductor flux at ``points`` [Wb]."""
        return _conductor_columns(points) @ CONDUCTOR_CURRENT

    return DisplayCoupling(plasma=plasma, external=external)


def test_the_panel_map_reproduces_the_operator_raster_image():
    """A state's panel map equals the operator's own raster at the nodes."""
    profile, lattice = _profile()
    moments = _moments(lattice)
    raster = np.asarray(
        profile.operator.raster_image(
            moments,
            current=jnp.zeros_like(profile.operator.external_current),
            prescribed_current=jnp.asarray(CONDUCTOR_CURRENT),
        )
    )[: lattice.node_count]
    panel = {
        "psi": raster.reshape(tuple(int(value) for value in lattice.shape)),
        "current": CONDUCTOR_CURRENT,
        "moments": tuple(
            np.asarray(value)
            for value in (
                moments.cell_current,
                moments.radial_moment,
                moments.vertical_moment,
            )
        ),
        "saddle_index": -1,
        "xpoints": np.zeros((0, 2)),
        "axis": np.asarray([1.0, 0.0]),
    }
    display = _panel_display_map(
        profile,
        {},
        panel,
        spacing=DEFAULT_DISPLAY_SPACING,
        coupling=_coupling(lattice),
    )
    assert display.reproduction_relative_error < 1.0e-10
