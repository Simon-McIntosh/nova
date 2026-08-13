"""Identifiable-mode Gauss--Newton geometry recovery."""

import numpy as np
import pytest

from nova.biot.geometrysolve import (
    FluxMapFunctional,
    nuisance_basis,
    project_nuisance,
    solve_geometry,
    solve_flux_map,
    solve_linear_geometry,
    solve_linear_flux_map,
    synthetic_flux_map_recovery_ladder,
    synthetic_recovery_ladder,
)


def test_a_linear_geometry_model_is_recovered_at_the_noise_floor():
    """Every admitted physical direction is recovered without basis bias."""
    jacobian = np.array(
        [
            [8.0, 1.0],
            [2.0, 6.0],
            [-3.0, 4.0],
            [5.0, -2.0],
        ]
    )
    gram = np.array([[2.0, 0.4], [0.4, 0.7]])
    truth = np.array([0.003, -0.002])
    fit = solve_linear_geometry(
        jacobian,
        jacobian @ truth,
        np.full(4, 0.01),
        np.zeros(2),
        gram,
        resolution_limit=0.02,
    )
    assert fit.converged
    assert fit.resolved_count == 2
    assert np.allclose(fit.parameters, truth, atol=1e-10)
    assert np.linalg.norm(fit.residual) < 1e-9
    assert np.all(np.isfinite(fit.standard_error))


def test_an_unresolvable_direction_is_frozen_at_its_seed():
    """A weak mode does not turn sensor noise into an apparent deformation."""
    jacobian = np.diag([1000.0, 2.0])
    truth = np.array([0.004, 0.3])
    seed = np.array([0.001, -0.007])
    fit = solve_linear_geometry(
        jacobian,
        jacobian @ truth,
        np.ones(2),
        seed,
        np.eye(2),
        resolution_limit=0.01,
    )
    assert fit.resolved.tolist() == [True, False]
    assert fit.parameters[0] == pytest.approx(truth[0])
    assert fit.parameters[1] == pytest.approx(seed[1])
    assert np.isinf(fit.mode_standard_error[1])


def test_nuisance_columns_are_removed_before_geometry_is_fitted():
    """A current-like sensor pattern cannot masquerade as pack displacement."""
    nuisance = np.array([[1.0], [1.0], [1.0], [1.0]])
    geometry = np.array([[1.0], [-1.0], [2.0], [-2.0]])
    target = 7.0 * nuisance[:, 0] + 0.004 * geometry[:, 0]
    fit = solve_linear_geometry(
        geometry,
        target,
        np.ones(4),
        np.zeros(1),
        np.eye(1),
        nuisance_span=nuisance,
        resolution_limit=1.0,
    )
    assert fit.parameters[0] == pytest.approx(0.004)
    basis = nuisance_basis(nuisance, np.ones(4))
    assert np.linalg.norm(project_nuisance(nuisance[:, 0], basis)) < 1e-12


def test_a_nonlinear_model_converges_from_an_offset_seed():
    """Levenberg--Marquardt uses the refreshed exact Jacobian each iteration."""
    locations = np.array([0.2, 0.6, 1.1, 1.7])

    def model(parameters):
        shift, scale = parameters
        argument = scale * locations + shift
        prediction = np.sin(argument)
        jacobian = np.column_stack([np.cos(argument), locations * np.cos(argument)])
        return prediction, jacobian

    truth = np.array([0.08, 1.12])
    target, _ = model(truth)
    fit = solve_geometry(
        model,
        target,
        np.full(target.size, 1e-3),
        np.array([-0.1, 0.8]),
        np.eye(2),
        resolution_limit=1.0,
    )
    assert fit.converged
    assert fit.iterations > 1
    assert np.allclose(fit.parameters, truth, atol=1e-8)


def test_the_recovery_ladder_banks_truth_and_noise_limited_estimates():
    """Known perturbations are drawn only inside the resolvable mode space."""
    jacobian = np.diag([1000.0, 500.0, 2.0])
    ladder = synthetic_recovery_ladder(
        jacobian,
        np.ones(3),
        np.eye(3),
        np.array([0.002, 0.006]),
        resolution_limit=0.01,
        samples=32,
        seed=4,
    )
    assert ladder.truth.shape == (2, 32, 3)
    assert ladder.resolved_modes.shape == (2, 3)
    assert np.max(np.abs(ladder.truth[..., 2])) < 1e-15
    assert np.allclose(
        np.linalg.norm(ladder.truth[..., :2], axis=-1),
        ladder.amplitudes[:, None],
    )
    assert np.linalg.norm(ladder.bias, axis=1).max() < 1e-3


def test_a_flux_map_uses_only_supported_cells_and_their_uncertainty():
    """The gridded adapter preserves masking, whitening and nuisance projection."""
    target = np.array([[2.004, 2.0], [1.996, np.nan]])
    uncertainty = np.array([[0.1, 0.1], [0.1, 0.1]])
    jacobian = np.array([[[1.0], [0.0]], [[-1.0], [1000.0]]])
    nuisance = np.ones(target.shape)
    fit = solve_linear_flux_map(
        jacobian.reshape(-1, 1),
        target,
        uncertainty,
        np.zeros(1),
        np.eye(1),
        nuisance_maps=nuisance,
        resolution_limit=1.0,
    )
    assert fit.parameters[0] == pytest.approx(0.004)
    assert fit.predicted.shape == (3,)


def test_a_nonlinear_flux_model_reuses_the_geometry_core():
    """Map callbacks retain their grid shape while the common core iterates."""
    radius, height = np.meshgrid([0.4, 0.9, 1.4], [-0.5, 0.3])

    def model(parameters):
        shift, scale = parameters
        argument = scale * radius + shift * height
        predicted = np.sin(argument)
        jacobian = np.stack(
            [height * np.cos(argument), radius * np.cos(argument)], axis=-1
        )
        return predicted, jacobian

    truth = np.array([0.08, 1.12])
    target, _ = model(truth)
    fit = solve_flux_map(
        model,
        target,
        np.full(target.shape, 1e-3),
        np.array([-0.1, 0.8]),
        np.eye(2),
        resolution_limit=1.0,
    )
    assert fit.converged
    assert np.allclose(fit.parameters, truth, atol=1e-8)


def test_the_flux_map_ladder_reports_noise_limited_recovery():
    """Synthetic map recovery banks the same bias and resolution terms."""
    jacobian = np.zeros((3, 4, 3))
    jacobian[0, 0, 0] = 1000.0
    jacobian[1, 1, 1] = 500.0
    jacobian[2, 2, 2] = 2.0
    mask = np.zeros((3, 4), dtype=bool)
    mask[0, 0] = mask[1, 1] = mask[2, 2] = True
    ladder = synthetic_flux_map_recovery_ladder(
        jacobian,
        np.ones((3, 4)),
        np.eye(3),
        np.array([0.002, 0.006]),
        mask=mask,
        resolution_limit=0.01,
        samples=32,
        seed=4,
    )
    assert ladder.truth.shape == (2, 32, 3)
    assert ladder.resolved_modes.shape == (2, 3)
    assert np.max(np.abs(ladder.truth[..., 2])) < 1e-15
    assert np.linalg.norm(ladder.bias, axis=1).max() < 1e-3


def test_flux_map_shapes_are_validated_before_the_solver_runs():
    """Targets, uncertainties, masks and Jacobians retain one grid contract."""
    with pytest.raises(ValueError, match="two-dimensional"):
        FluxMapFunctional.from_maps(np.ones(4), np.ones(4))
    with pytest.raises(ValueError, match="target shape"):
        FluxMapFunctional.from_maps(np.ones((2, 2)), np.ones((2, 3)))

    def wrong_shape(parameters):
        return np.ones((2, 2)), np.ones((2, 2, parameters.size + 1))

    with pytest.raises(ValueError, match="flux jacobian"):
        solve_flux_map(
            wrong_shape,
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.zeros(1),
            np.eye(1),
        )


@pytest.mark.parametrize(
    ("noise", "message"),
    [([1.0, 0.0], "strictly positive"), ([1.0], "same shape")],
)
def test_invalid_noise_is_rejected(noise, message):
    """Noise whitening refuses undefined weights and dimension mismatches."""
    with pytest.raises(ValueError, match=message):
        solve_linear_geometry(
            np.eye(2),
            np.ones(2),
            noise,
            np.zeros(2),
            np.eye(2),
        )
