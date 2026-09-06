"""Flux-surface geometry assembly from accelerator-native connectivity bins."""

from dataclasses import dataclass
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.contour import Contour
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.flux_surface_geometry import (
    FluxSurfaceGeometry as HostFluxSurfaceGeometry,
)
from nova.transport import (
    CurrentDiffusion,
    EtaProfile,
    traced_assemble_flux_surface_geometry,
    flux_surface_geometry,
    traced_flux_surface_geometry,
)


@dataclass(frozen=True)
class CircularGrid:
    """Uniform rectangular grid containing a circular plasma."""

    rg: np.ndarray
    zg: np.ndarray
    inside_limiter: np.ndarray
    r0: float


def _circular_equilibrium(
    *,
    major_radius: float = 3.0,
    minor_radius: float = 0.5,
    poloidal_flux_span: float = -0.2,
    grid_points: int = 81,
):
    extent = 1.2 * minor_radius
    radius = np.linspace(major_radius - extent, major_radius + extent, grid_points)
    height = np.linspace(-extent, extent, grid_points)
    mesh_radius, mesh_height = np.meshgrid(radius, height)
    psi_n = ((mesh_radius - major_radius) / minor_radius) ** 2 + (
        mesh_height / minor_radius
    ) ** 2
    psi = poloidal_flux_span * psi_n
    inside = (mesh_radius - major_radius) ** 2 + mesh_height**2 <= (
        1.1 * minor_radius
    ) ** 2
    return psi, CircularGrid(radius, height, inside, major_radius)


def _assemble_circular_geometry(*, n_radial_cells: int = 24):
    psi, grid = _circular_equilibrium()
    geometry = flux_surface_geometry(
        psi,
        grid,
        axis_psi=0.0,
        boundary_psi=-0.2,
        profile_coefficients=np.array([1.0, 0.0]),
        coefficient_scale=np.array([8.0e5, 8.0e5]),
        ip_amperes=5.0e5,
        n_pressure=1,
        n_diamagnetic=1,
        boundary_toroidal_field=2.0,
        n_radial_cells=n_radial_cells,
    )
    assert geometry is not None
    return geometry


@cache
def _internal_circular_geometry():
    """Return host and device geometry from the same analytic circular map."""
    psi, grid = _circular_equilibrium()

    def field_function(psi_norm):
        """Return the constant toroidal-field function of the circular map."""
        return np.full_like(psi_norm, 2.0 * grid.r0)

    internal = HostFluxSurfaceGeometry.internal_geometry(
        FluxLattice(grid.rg, grid.zg),
        psi.T,
        field_function,
        axis=(grid.r0, 0.0),
        boundary_flux=-0.2,
        reference_radius=grid.r0,
        n_surface=11,
        n_theta=64,
        n_rho=25,
    )
    device = traced_flux_surface_geometry(
        jnp.asarray(psi),
        jnp.asarray(grid.rg),
        jnp.asarray(grid.zg),
        jnp.asarray(grid.inside_limiter),
        axis_psi=jnp.asarray(0.0),
        boundary_psi=jnp.asarray(-0.2),
        profile_coefficients=jnp.asarray([1.0, 0.0]),
        coefficient_scale=jnp.asarray([8.0e5, 8.0e5]),
        ip_amperes=jnp.asarray(5.0e5),
        major_radius=jnp.asarray(grid.r0),
        boundary_toroidal_field=jnp.asarray(2.0),
        n_pressure=1,
        n_diamagnetic=1,
        n_radial_cells=25,
        n_surface_bins=32,
    )
    return psi, grid, internal, device


def test_internal_geometry_loops_are_nested_and_lcfs_matches_contour():
    """Stored loops span axis to LCFS and agree with Nova's contour cut."""
    psi, grid, internal, _device = _internal_circular_geometry()
    expected_levels = np.linspace(0.0, 1.0, 11)
    np.testing.assert_array_equal(internal.surface_psi_norm, expected_levels)
    np.testing.assert_array_equal(
        internal.surface_r[0], np.full(64, internal.record.magnetic_axis[0])
    )
    np.testing.assert_array_equal(
        internal.surface_z[0], np.full(64, internal.record.magnetic_axis[1])
    )

    distance = np.sqrt(
        (internal.surface_r - internal.record.magnetic_axis[0]) ** 2
        + (internal.surface_z - internal.record.magnetic_axis[1]) ** 2
    )
    assert np.all(np.diff(np.mean(distance, axis=1)) > 0.0)

    surfaces = Contour(
        grid.rg,
        grid.zg,
        psi,
        levels=np.asarray([-0.2]),
    ).levelset(-0.2)
    contour = max(
        (surface.points for surface in surfaces if surface.closed),
        key=lambda points: points.shape[0],
    )
    lcfs = np.column_stack((internal.surface_r[-1], internal.surface_z[-1]))
    nearest = np.min(
        np.linalg.norm(lcfs[:, None, :] - contour[None, :, :], axis=2), axis=1
    )
    cell_diagonal = np.hypot(np.diff(grid.rg).min(), np.diff(grid.zg).min())
    assert float(np.max(nearest)) <= cell_diagonal


def test_internal_geometry_faces_are_finite_monotone_and_match_device_volume():
    """TORAX faces are finite and host volume agrees with the device path."""
    _psi, _grid, internal, device = _internal_circular_geometry()
    record = internal.record
    np.testing.assert_array_equal(record.rho_tor_norm, np.linspace(0.0, 1.0, 26))
    for name in record.profile_names():
        values = np.asarray(getattr(record, name))
        assert values.shape == (26,), name
        assert np.all(np.isfinite(values)), name
    for values in (record.volume, record.area, record.toroidal_flux):
        assert np.all(np.diff(values) >= 0.0)

    assert bool(device["valid"])
    device_volume = np.asarray(device["volume_face"])
    relative = np.abs(record.volume[1:] - device_volume[1:]) / np.maximum(
        np.abs(device_volume[1:]), 1.0e-12
    )
    assert int(np.argmax(relative)) == 0
    assert float(relative[0]) < 0.85
    assert float(relative[4]) < 0.15
    assert float(np.max(relative[5:])) < 0.01
    assert float(relative[-1]) < 0.005


def test_circular_geometry_has_analytic_volume_flux_and_profile_map():
    geometry = _assemble_circular_geometry(n_radial_cells=32)
    expected_volume = 2.0 * np.pi**2 * geometry.r0 * 0.5**2
    expected_toroidal_flux = np.pi * 0.5**2 * 2.0

    assert abs(geometry.volume - expected_volume) / expected_volume < 0.05
    assert abs(geometry.phi_b - expected_toroidal_flux) / expected_toroidal_flux < 0.08
    assert np.allclose(geometry.psi_n_face, geometry.rho_face**2, atol=0.04)
    assert np.all(np.diff(geometry.psi_n_face) >= 0.0)
    assert np.all(geometry.q_face > 0.0)
    assert np.all(np.isfinite(geometry.g2_face))

    current_edge = geometry.enclosed_current(geometry.psi_face)[-1]
    assert abs(current_edge - geometry.ip_amperes) / geometry.ip_amperes < 0.04


def test_nonuniform_surface_bins_preserve_fixed_output_contract():
    psi, grid = _circular_equilibrium()
    psi_n_surface = jnp.asarray([0.03, 0.09, 0.22, 0.48, 0.79, 0.97])
    volume = 2.0 * np.pi**2 * grid.r0 * 0.5**2
    span = 0.2
    gradient_metric = 4.0 * span**2 * psi_n_surface / (0.5**2 * grid.r0**2)
    bins = {
        "pn_s": psi_n_surface,
        "dv_dpn": jnp.full_like(psi_n_surface, volume),
        "inv_r2": jnp.full_like(psi_n_surface, 1.0 / grid.r0**2),
        "inv_r": jnp.full_like(psi_n_surface, 1.0 / grid.r0),
        "grad2_r2": gradient_metric,
        "v_cum": volume * psi_n_surface,
        "v_total": jnp.asarray(volume),
    }
    assembled = traced_assemble_flux_surface_geometry(
        bins,
        jnp.asarray(psi),
        jnp.asarray(grid.rg),
        jnp.asarray(grid.zg),
        jnp.asarray(grid.inside_limiter),
        axis_psi=jnp.asarray(0.0),
        boundary_psi=jnp.asarray(-span),
        profile_coefficients=jnp.asarray([1.0, 0.0]),
        coefficient_scale=jnp.asarray([8.0e5, 8.0e5]),
        ip_amperes=jnp.asarray(5.0e5),
        major_radius=jnp.asarray(grid.r0),
        boundary_toroidal_field=jnp.asarray(2.0),
        n_pressure=1,
        n_diamagnetic=1,
        n_radial_cells=12,
    )

    assert bool(assembled["valid"])
    assert assembled["rho_face"].shape == (13,)
    assert assembled["rho_cell"].shape == (12,)
    assert assembled["g2_face"].shape == (13,)
    assert np.allclose(
        np.asarray(assembled["psi_n_face"]),
        np.asarray(assembled["rho_face"]) ** 2,
        atol=0.04,
    )


def test_empty_surface_is_rejected_without_fabricated_geometry():
    psi, grid = _circular_equilibrium()
    empty = CircularGrid(
        rg=grid.rg,
        zg=grid.zg,
        inside_limiter=np.zeros_like(grid.inside_limiter),
        r0=grid.r0,
    )
    geometry = flux_surface_geometry(
        psi,
        empty,
        axis_psi=0.0,
        boundary_psi=-0.2,
        profile_coefficients=np.array([1.0, 0.0]),
        coefficient_scale=np.array([8.0e5, 8.0e5]),
        ip_amperes=5.0e5,
        n_pressure=1,
        n_diamagnetic=1,
        boundary_toroidal_field=2.0,
    )
    assert geometry is None


def test_device_assembly_is_jit_and_vmap_safe():
    psi, grid = _circular_equilibrium(grid_points=65)
    radius = jnp.asarray(grid.rg)
    height = jnp.asarray(grid.zg)
    inside = jnp.asarray(grid.inside_limiter)
    coefficients = jnp.asarray([1.0, 0.0])
    scales = jnp.asarray([8.0e5, 8.0e5])

    def assemble(one_psi, boundary_flux):
        return traced_flux_surface_geometry(
            one_psi,
            radius,
            height,
            inside,
            axis_psi=jnp.asarray(0.0),
            boundary_psi=boundary_flux,
            profile_coefficients=coefficients,
            coefficient_scale=scales,
            ip_amperes=jnp.asarray(5.0e5),
            major_radius=jnp.asarray(grid.r0),
            boundary_toroidal_field=jnp.asarray(2.0),
            n_pressure=1,
            n_diamagnetic=1,
            n_radial_cells=16,
            n_surface_bins=20,
        )

    compiled = jax.jit(assemble)(jnp.asarray(psi), jnp.asarray(-0.2))
    assert bool(compiled["valid"])
    assert compiled["psi_face"].shape == (17,)

    factors = jnp.asarray([0.98, 1.0, 1.02])
    batched = jax.vmap(assemble)(
        factors[:, jnp.newaxis, jnp.newaxis] * jnp.asarray(psi),
        factors * -0.2,
    )
    assert batched["psi_face"].shape == (3, 17)
    assert np.all(np.asarray(batched["valid"]))


def test_assembled_geometry_drives_current_diffusion():
    geometry = _assemble_circular_geometry(n_radial_cells=24)
    solver = CurrentDiffusion(
        geometry, EtaProfile(eta0=1.0e-7, contrast=0.0, shape=1.0)
    )
    times = np.array([0.0, 1.0e-4, 2.0e-4])
    step = solver.evolve(times, np.full(times.size, geometry.ip_amperes))
    prediction = solver.predict(step)

    assert step["psi_face"].shape == (3, 25)
    assert prediction["j_tor"].shape == (24,)
    assert np.all(np.isfinite(prediction["j_tor"]))
