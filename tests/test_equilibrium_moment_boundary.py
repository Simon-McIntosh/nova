"""Push-out boundary consistency for the current-moment reconstruction.

The self-sized seed is defined by a fixed point: a uniform current disc of
radius ``a`` pushes out to a boundary whose mean minor radius is ``b(a)``, and
the seed is sized so ``a = b(a)``. That closure is what the read leans on
instead of a shape freedom, so it is pinned here end to end, through the same
gauged flux map and connectivity boundary read a machine slice would take.

The setup is synthetic but physical: a Gaussian truth current inside a
rectangular vessel, a shaping coil pair above and below the midplane whose
field puts a saddle in the total flux, and a ring of flux loops and field probes
just inside the wall. No machine data and no external files.
"""

import numpy as np

from nova.equilibrium.measurement import Magnetics, SliceMeasurement
from nova.equilibrium.moment import (
    CurrentCells,
    MomentConfig,
    MomentOrder,
    ReconstructMoment,
)
from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    from nova.equilibrium.wall_mask import inside_polygon

LIMITER_R = np.array([0.25, 1.55, 1.55, 0.25, 0.25])
LIMITER_Z = np.array([-1.05, -1.05, 1.05, 1.05, -1.05])
CENTRE = (0.9, 0.0)
PLASMA_CURRENT = 5.0e5


class Grid:
    """Reconstruction grid: node coordinates, limiter polygon and its mask."""

    def __init__(self, n_radial=49, n_vertical=65):
        self.rg = np.linspace(0.2, 1.6, n_radial)
        self.zg = np.linspace(-1.1, 1.1, n_vertical)
        radius, height = np.meshgrid(self.rg, self.zg)
        self.limiter_r, self.limiter_z = LIMITER_R, LIMITER_Z
        self.inside_limiter = inside_polygon(
            radius.ravel(), height.ravel(), LIMITER_R, LIMITER_Z
        ).reshape(height.shape)


def shaping_flux(grid, current=-1.2e6):
    """Return the coil-only flux map of a shaping pair above and below the axis."""
    from nova.biot.greens import greens_psi

    radius, height = np.meshgrid(grid.rg, grid.zg)
    flux = np.zeros(radius.shape)
    for coil_z in (-1.35, 1.35):
        loop = greens_psi(radius.ravel(), height.ravel(), 1.0, coil_z)
        flux += current * loop.reshape(radius.shape)
    return flux


def truth_cells(grid, n_side=13):
    """Return the plasma current cells, filling the vessel as a machine set does."""
    radius, height = np.meshgrid(
        np.linspace(0.32, 1.48, n_side), np.linspace(-0.96, 0.96, n_side)
    )
    inside = inside_polygon(radius.ravel(), height.ravel(), LIMITER_R, LIMITER_Z)
    return CurrentCells(
        radius.ravel(), height.ravel(), 0.10, 0.16, inside.astype(float)
    )


def probe_ring(n_probe=12):
    """Return flux loops and field probes on a ring just inside the wall."""
    angle = np.linspace(0.0, 2.0 * np.pi, n_probe, endpoint=False)
    radius = CENTRE[0] + 0.58 * np.cos(angle)
    height = CENTRE[1] + 0.92 * np.sin(angle)
    return Magnetics(
        r=np.r_[radius, radius],
        z=np.r_[height, height],
        angle=np.r_[np.rad2deg(angle), np.zeros(n_probe)],
        flux_loop=np.r_[np.zeros(n_probe, bool), np.ones(n_probe, bool)],
    )


def synthetic_slice():
    """Return the reconstruction, its measurement, and the truth centroid."""
    grid = Grid()
    cells = truth_cells(grid)
    read = ReconstructMoment(
        cells, probe_ring(), grid, major_radius=CENTRE[0], config=MomentConfig()
    )
    blob = np.exp(
        -(((cells.r - CENTRE[0]) / 0.28) ** 2 + ((cells.z - CENTRE[1]) / 0.34) ** 2)
    )
    cell_current = blob / blob.sum() * PLASMA_CURRENT
    vacuum_flux = shaping_flux(grid)
    n_sensor = read.magnetics.number
    measurement = SliceMeasurement(
        measured=read.sensor_coupling @ cell_current,
        vacuum=np.zeros(n_sensor),
        mask=np.ones(n_sensor, bool),
        scale=np.full(n_sensor, 1e-3),
        plasma_current=PLASMA_CURRENT,
        vacuum_flux=vacuum_flux,
    )
    centroid = (
        float((cell_current * cells.r).sum() / cell_current.sum()),
        float((cell_current * cells.z).sum() / cell_current.sum()),
    )
    return read, measurement, centroid


def test_the_seed_radius_is_a_fixed_point_of_its_push_out_boundary():
    """The converged seed radius equals the boundary it pushes out to."""
    read, measurement, _centroid = synthetic_slice()
    centre = read.fit_centroid(measurement)
    radius, boundary = read.self_sized_seed(measurement, centre)
    minor_radius = float(np.mean(np.asarray(boundary.radii)))
    assert 0.0 < radius < 0.6
    assert abs(minor_radius - radius) < 4 * read.config.radius_tolerance


def test_the_centroid_fit_recovers_the_truth_centroid():
    """The two-freedom filament fit lands on the truth current centroid."""
    read, measurement, centroid = synthetic_slice()
    fitted = read.fit_centroid(measurement)
    assert abs(fitted[0] - centroid[0]) < 0.05
    assert abs(fitted[1] - centroid[1]) < 0.05


def test_the_ladder_reports_a_gauged_boundary_and_gate_decision():
    """A full ladder solve returns a ring, a gauged flux pair and a gate verdict."""
    read, measurement, _centroid = synthetic_slice()
    reconstruction = read.solve(measurement, MomentOrder.QUADRUPOLE)
    assert reconstruction.ring is not None
    assert reconstruction.ring.shape[1] == 2
    assert reconstruction.order is MomentOrder.QUADRUPOLE
    assert np.isclose(reconstruction.cell_current.sum(), PLASMA_CURRENT, rtol=1e-6)
    assert reconstruction.quadrupole_shift_fraction >= 0.0
    assert reconstruction.quadrupole_applied == (
        reconstruction.quadrupole_shift_fraction < read.config.gate_shift_fraction
    )
    # the gauged map carries the coil flux; the plasma-only map does not
    assert not np.allclose(reconstruction.flux, reconstruction.plasma_flux)
    assert abs(reconstruction.axis_flux - reconstruction.boundary_flux) > 0.0


def test_the_centroid_rung_leaves_the_seed_untouched():
    """Solving at the CENTROID rung reports the self-sized disc itself."""
    read, measurement, _centroid = synthetic_slice()
    reconstruction = read.solve(measurement, MomentOrder.CENTROID)
    assert not reconstruction.quadrupole_applied
    inside = reconstruction.cell_current != 0.0
    assert np.allclose(
        reconstruction.cell_current[inside], reconstruction.cell_current[inside][0]
    )
