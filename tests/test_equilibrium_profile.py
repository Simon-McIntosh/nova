"""Contracts for the JAX-native profile reconstruction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import mu_0

from nova.biot.greens import hybrid_greens
from nova.equilibrium import ProfileDegrees, ProfilePrior, ReconstructProfile
from nova.equilibrium.measurement import Magnetics


def _grid():
    grid_r = np.linspace(0.7, 1.3, 9)
    grid_z = np.linspace(-0.45, 0.45, 9)
    radius, height = np.meshgrid(grid_r, grid_z)
    inside = ((radius - 1.0) / 0.28) ** 2 + (height / 0.38) ** 2 <= 1.0
    return grid_r, grid_z, inside


def _wall():
    angle = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    return 1.0 + 0.29 * np.cos(angle), 0.39 * np.sin(angle)


def _manual_solver(*, priors=(), relaxation=0.0):
    grid_r, grid_z, inside = _grid()
    n_grid = grid_r.size * grid_z.size
    radius, height = np.meshgrid(grid_r, grid_z)
    sensor_index = np.array([20, 24, 38, 42, 56, 60])
    plasma_to_sensor = np.zeros((sensor_index.size, n_grid))
    plasma_to_sensor[np.arange(sensor_index.size), sensor_index] = 1.0
    wall_r, wall_z = _wall()
    return ReconstructProfile(
        grid_r=grid_r,
        grid_z=grid_z,
        inside_limiter=inside,
        cell_area=np.full(n_grid, (grid_r[1] - grid_r[0]) * (grid_z[1] - grid_z[0])),
        source_to_grid=np.zeros((n_grid, 1)),
        plasma_to_grid=np.eye(n_grid) * 1.0e-6,
        source_to_sensor=np.zeros((sensor_index.size, 1)),
        plasma_to_sensor=plasma_to_sensor,
        source_names=("equilibrium_field",),
        degrees=ProfileDegrees(n_pressure=2, n_diamagnetic=2),
        axis_seed=(1.0, 0.0),
        wall_r=wall_r,
        wall_z=wall_z,
        priors=priors,
        iterations=2,
        relaxation=relaxation,
        ridge=1.0e-12,
        topology_levels=12,
        topology_bisections=4,
        topology_rays=32,
    )


def _bowl(solver):
    radius, height = jnp.meshgrid(solver.grid_r, solver.grid_z)
    return ((radius - 1.0) ** 2 + 0.8 * height**2).reshape(-1)


def test_geometry_factory_uses_canonical_green_operator():
    grid_r = np.array([0.82, 1.18])
    grid_z = np.array([-0.16, 0.16])
    inside = np.ones((2, 2), dtype=bool)
    wall_r, wall_z = _wall()
    magnetics = Magnetics(
        r=np.array([1.45, 1.55]),
        z=np.array([0.0, 0.1]),
        angle=np.array([0.0, 90.0]),
        flux_loop=np.array([True, False]),
    )
    solver = ReconstructProfile.from_geometry(
        grid_r=grid_r,
        grid_z=grid_z,
        inside_limiter=inside,
        cell_width=np.array(0.08),
        cell_height=np.array(0.08),
        source_r=np.array([1.6]),
        source_z=np.array([0.0]),
        source_width=np.array([0.12]),
        source_height=np.array([0.18]),
        source_names=("vertical_field",),
        magnetics=magnetics,
        degrees=ProfileDegrees(1, 1),
        axis_seed=(1.0, 0.0),
        wall_r=wall_r,
        wall_z=wall_z,
        iterations=1,
        topology_levels=8,
        topology_bisections=3,
        topology_rays=16,
    )
    radius, height = np.meshgrid(grid_r, grid_z)
    expected = hybrid_greens(radius.ravel(), height.ravel(), 1.6, 0.0, 0.12, 0.18)[0]
    np.testing.assert_allclose(np.asarray(solver.source_to_grid[:, 0]), expected)


def test_profile_basis_matches_donor_two_term_shape():
    solver = _manual_solver()
    flux = _bowl(solver)
    basis, topology = solver._profile_basis(flux)
    beta = 0.37
    reference_radius = 1.0
    coefficients = jnp.asarray(
        [
            -beta / (2.0 * np.pi * reference_radius),
            0.0,
            -(1.0 - beta) * mu_0 * reference_radius / (2.0 * np.pi),
            0.0,
        ]
    )
    radius, _height = jnp.meshgrid(solver.grid_r, solver.grid_z)
    span = topology["psi_bnd"] - topology["psi_axis"]
    flux_norm = (flux - topology["psi_axis"]) / span
    edge = jnp.clip(1.0 - flux_norm, 0.0, 1.0)
    shape = (
        beta * radius.reshape(-1) / reference_radius
        + (1.0 - beta) * reference_radius / radius.reshape(-1)
    ) * edge
    expected = shape * topology["core_weight"].reshape(-1) * solver.cell_area
    np.testing.assert_allclose(basis @ coefficients, expected, rtol=1.0e-12)


def test_named_controls_and_prior_rows_are_strict():
    prior = ProfilePrior(
        name="magnetic moment",
        sensitivity={"pressure_0": 2.0, "diamagnetic_1": -1.0},
        target=3.0,
        sigma=0.5,
    )
    solver = _manual_solver(priors=(prior,))
    np.testing.assert_array_equal(
        solver.pack_source_currents({"equilibrium_field": 4.0}), np.array([4.0])
    )
    packed = solver.pack_coefficients(
        {
            "pressure_0": 1.0,
            "pressure_1": 2.0,
            "diamagnetic_0": 3.0,
            "diamagnetic_1": 4.0,
        }
    )
    np.testing.assert_array_equal(packed, np.arange(1.0, 5.0))
    np.testing.assert_allclose(
        np.asarray(solver._prior_matrix), np.array([[4.0, 0.0, 0.0, -2.0]])
    )
    np.testing.assert_allclose(np.asarray(solver._prior_target), np.array([6.0]))
    with pytest.raises(ValueError, match="missing equilibrium_field"):
        solver.pack_source_currents({})
    with pytest.raises(ValueError, match="unknown coefficients"):
        _manual_solver(
            priors=(
                ProfilePrior(
                    name="bad moment",
                    sensitivity={"unlabelled": 1.0},
                    target=0.0,
                    sigma=1.0,
                ),
            )
        )


def test_least_squares_recovers_profile_and_measured_current():
    solver = _manual_solver()
    initial = _bowl(solver)
    basis, _ = solver._profile_basis(initial)
    expected = jnp.asarray([0.35, -0.12, 0.42, -0.08])
    measured = solver.plasma_to_sensor @ (basis @ expected)
    plasma_current = jnp.sum(basis @ expected)
    result = solver.least_squares(
        jnp.zeros(1),
        plasma_current,
        measured,
        jnp.ones(measured.size),
        jnp.ones(measured.size, dtype=bool),
        initial,
    )
    np.testing.assert_allclose(result.coefficients, expected, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(
        jnp.sum(result.cell_current), plasma_current, rtol=1.0e-11, atol=1.0e-11
    )


def test_picard_is_jittable_and_uses_boundary_push_core():
    solver = _manual_solver(relaxation=0.25)
    initial = _bowl(solver)
    coefficients = solver.pack_coefficients(
        {
            "pressure_0": 0.4,
            "pressure_1": 0.1,
            "diamagnetic_0": 0.2,
            "diamagnetic_1": 0.05,
        }
    )
    result = jax.jit(solver.picard)(jnp.zeros(1), coefficients, initial)
    assert result.flux.shape == initial.shape
    assert result.core_weight.shape == solver.inside_limiter.shape
    assert np.isfinite(np.asarray(result.flux)).all()
    assert 0.0 < float(jnp.sum(result.core_weight)) < result.core_weight.size


def test_batched_least_squares_matches_slice_solves():
    solver = _manual_solver()
    initial = _bowl(solver)
    basis, _ = solver._profile_basis(initial)
    coefficients = jnp.asarray([[0.35, -0.12, 0.42, -0.08], [0.28, -0.08, 0.37, -0.04]])
    measured = jax.vmap(lambda c: solver.plasma_to_sensor @ (basis @ c))(coefficients)
    plasma_current = jax.vmap(lambda c: jnp.sum(basis @ c))(coefficients)
    sources = jnp.zeros((2, 1))
    scales = jnp.ones_like(measured)
    masks = jnp.ones_like(measured, dtype=bool)
    initials = jnp.stack([initial, initial])
    batched = jax.jit(solver.least_squares_batch)(
        sources, plasma_current, measured, scales, masks, initials
    )
    per_slice = [
        solver.least_squares(
            sources[index],
            plasma_current[index],
            measured[index],
            scales[index],
            masks[index],
            initials[index],
        )
        for index in range(2)
    ]
    np.testing.assert_allclose(
        batched.coefficients,
        np.stack([result.coefficients for result in per_slice]),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_shape_validation_rejects_incompatible_operator():
    grid_r, grid_z, inside = _grid()
    n_grid = grid_r.size * grid_z.size
    wall_r, wall_z = _wall()
    with pytest.raises(ValueError, match="plasma_to_grid shape"):
        ReconstructProfile(
            grid_r=grid_r,
            grid_z=grid_z,
            inside_limiter=inside,
            cell_area=np.ones(n_grid),
            source_to_grid=np.zeros((n_grid, 1)),
            plasma_to_grid=np.zeros((n_grid, n_grid - 1)),
            source_to_sensor=np.zeros((1, 1)),
            plasma_to_sensor=np.zeros((1, n_grid)),
            source_names=("field",),
            degrees=ProfileDegrees(1, 1),
            axis_seed=(1.0, 0.0),
            wall_r=wall_r,
            wall_z=wall_z,
        )
