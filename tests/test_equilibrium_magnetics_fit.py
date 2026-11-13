"""Profile-coefficient fitting and its public-forward authority boundary."""

from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import MagneticsFit, MagneticsFitReceipt
from nova.equilibrium import magnetics_fit
from nova.equilibrium.measurement import Magnetics, SliceMeasurement, whitened_solve
from nova.equilibrium.solve_request import ForwardSolveReceipt, ForwardSolveRequest
from nova.equilibrium.source import DomainProfile, ForwardSource, PolynomialFluxFunction
from nova.jax.config import configure_dtypes


def test_rank_deficient_data_exposes_its_null_direction():
    response = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    target = np.array([2.0, 4.0, 6.0])
    coefficient, covariance, spectrum, null = whitened_solve(
        response, target, np.ones(3), 0.0, diagnostics=True
    )
    np.testing.assert_allclose(response @ coefficient, target, atol=2.0e-15)
    assert np.all(np.isfinite(covariance))
    assert spectrum[0] > 1.0
    assert spectrum[1] < 1.0e-14
    assert null.shape == (1, 2)
    np.testing.assert_allclose(response @ null.T, 0.0, atol=2.0e-15)


def test_regularisation_does_not_claim_data_identifiability():
    response = np.array([[1.0, 1.0]])
    _, _, spectrum, null = whitened_solve(
        response,
        np.array([1.0]),
        np.ones(1),
        1.0e-3,
        diagnostics=True,
        penalty=np.array([1.0, 4.0]),
    )
    assert spectrum.size == 1
    assert null.shape == (1, 2)
    np.testing.assert_allclose(response @ null.T, 0.0, atol=1.0e-15)


def test_current_anchor_participates_in_the_data_spectrum():
    response = np.array([[1.0, 1.0]])
    coefficient, _, spectrum, null = whitened_solve(
        response,
        np.array([3.0]),
        np.array([0.5]),
        0.0,
        anchor=(np.array([1.0, -1.0]), 1.0, 2.0),
        diagnostics=True,
    )
    np.testing.assert_allclose(coefficient, [2.0, 1.0], atol=1.0e-15)
    assert spectrum.shape == (2,)
    assert spectrum[-1] > 0.0
    assert null.shape == (0, 2)


def test_fit_receipt_and_forward_request_keep_the_authority_boundary():
    assert MagneticsFitReceipt.__module__ == "nova.equilibrium.magnetics_fit"
    assert MagneticsFit.__module__ == "nova.equilibrium.magnetics_fit"
    names = {item.name for item in fields(ForwardSolveRequest)}
    assert "measurement" not in names
    with pytest.raises(TypeError, match="measurement"):
        ForwardSolveRequest(measurement=np.zeros(4))


@pytest.fixture
def linear_fit(monkeypatch):
    """An algebraic instrument checks orchestration without a physics claim."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    source = ForwardSource(
        core=DomainProfile(
            p_prime=PolynomialFluxFunction(jnp.array([0.8, 0.1])),
            ff_prime=PolynomialFluxFunction(jnp.array([1.2])),
        )
    )
    request = ForwardSolveRequest.from_defaults(
        carrier_identity="linear-response",
        source_profile=source,
        seed_policy=magnetics_fit.ExplicitSolveSeed(jnp.zeros(3)),
    )
    response = np.array(
        [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [2.0, 1.0, 1.0], [1.0, -2.0, 1.0]]
    )
    calls = []

    def solve(active):
        assert isinstance(active, ForwardSolveRequest)
        calls.append(active)
        coefficients = np.r_[
            active.source_profile.core.p_prime.coefficients,
            active.source_profile.core.ff_prime.coefficients,
        ]
        return ForwardSolveReceipt(
            terminal_state=SimpleNamespace(
                flux=coefficients,
                cell_current=coefficients,
            ),
            qualified=True,
            termination_reason=0,
            residual_history=np.zeros(1),
            changed_mask_entries=np.zeros(1),
            support_cell_count=np.ones(1),
            globalisation_decisions=([], []),
            amplitude_history=np.ones(1),
            topology_read=None,
            polish_receipt=None,
            compilation_cache_hit=True,
            wall_seconds=0.0,
            resolved_defaults=None,
        )

    def signal(biot, coordinates, *args):
        indices = coordinates[:, 0].astype(int) - 1
        return SimpleNamespace(values=response[indices] @ biot)

    monkeypatch.setattr(magnetics_fit, "virtual_flux_loops", signal)
    monkeypatch.setattr(magnetics_fit, "virtual_poloidal_probes", signal)
    geometry = Magnetics(
        r=np.arange(1.0, 5.0),
        z=np.zeros(4),
        angle=np.zeros(4),
        flux_loop=np.array([True, False, True, False]),
    )
    fitter = MagneticsFit(
        SimpleNamespace(solve=solve),
        request,
        geometry,
        lambda receipt: receipt.terminal_state.flux,
    )
    truth = np.array([1.0, 0.3, 0.9])
    measurement = SliceMeasurement(
        measured=response @ truth,
        vacuum=np.zeros(4),
        mask=np.ones(4, dtype=bool),
        scale=np.array([0.5, 2.0, 1.0, 1.0]),
        plasma_current=truth.sum(),
    )
    return fitter, measurement, truth, calls


def test_coefficient_updates_issue_only_typed_source_requests(linear_fit):
    fitter, measurement, truth, calls = linear_fit
    receipt = fitter.fit(measurement, plasma_current_scale=1.0)
    assert receipt.converged
    assert receipt.iterations > 0
    assert receipt.forward_evaluations == len(calls)
    np.testing.assert_allclose(receipt.coefficients, truth, atol=1.0e-11)
    assert np.max(np.abs(receipt.whitened_residuals)) <= 1.0e-8
    assert receipt.undetermined_coefficients == ()
    assert receipt.singular_values.shape == (3,)
    assert all(active.target_current is None for active in calls)
    np.testing.assert_array_equal(
        fitter.request.source_profile.core.p_prime.coefficients, [0.8, 0.1]
    )


def test_masked_missing_rows_cannot_poison_the_fit(linear_fit):
    fitter, measurement, truth, _ = linear_fit
    missing = measurement.measured.copy()
    missing[1] = np.nan
    measurement = replace(
        measurement, measured=missing, mask=np.array([1, 0, 1, 1], bool)
    )
    receipt = fitter.fit(measurement, plasma_current_scale=1.0)
    assert receipt.converged
    assert receipt.whitened_residuals[1] == 0.0
    np.testing.assert_allclose(receipt.coefficients, truth, atol=1.0e-11)


def test_unqualified_forward_state_is_refused_before_sensor_read(linear_fit):
    fitter, measurement, _, _ = linear_fit
    solve = fitter.profile.solve
    fitter.profile.solve = lambda request: replace(solve(request), qualified=False)
    with pytest.raises(RuntimeError, match="did not qualify") as caught:
        fitter.fit(measurement, plasma_current_scale=1.0)
    assert isinstance(caught.value.receipt, ForwardSolveReceipt)
    assert not caught.value.receipt.qualified
