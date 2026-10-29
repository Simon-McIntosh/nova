"""Closed-form bank geometry agrees with the directional reference."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import current, support_read
from scripts.prototypes.bank_newton_step.measure import machine
from scripts.prototypes.bank_step_speed.reference import (
    _normal_geometry_booking_tangent as reference_tangent,
)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_closed_geometry_matches_directional_reference_at_trial_and_solution():
    bank, cache, roots, convention, policy, profile, quadrature, trial, _ = machine(
        "diverted"
    )
    state_path = Path(__file__).resolve().parents[1] / (
        "scripts/prototypes/bank_step_speed/converged_state.npz"
    )
    with np.load(state_path) as saved:
        assert bool(saved["converged"])
        converged = tuple(jnp.asarray(part) for part in saved["moments"])
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    index = jnp.arange(trial[0].size)
    shape = jnp.sin(1.7 * index) + jnp.cos(2.3 * index)
    direction = (shape, 0.5 * jnp.roll(shape, 17), -0.2 * jnp.roll(shape, 31))

    @jax.jit
    def read(state):
        return support_read.read(bank, cache, state, seeds, roots, convention, policy)

    @jax.jit
    def compare(state, support):
        psi = support_read.nodal_jets(bank, state)[bank.sample_index, 0]
        booked = current.book(psi, support.reading, bank.geometry, profile, quadrature)
        jets, _ = support_read._directional_field(bank, direction)
        actual = support_read._normal_geometry_booking_tangent(
            bank, state, jets, support, profile, quadrature, booked
        )
        reference = reference_tangent(
            bank, state, jets, support, profile, quadrature, booked
        )
        return jnp.concatenate(tuple(actual)), jnp.concatenate(tuple(reference))

    for label, state in (("trial", trial), ("converged", converged)):
        support = read(state)
        assert bool(support.valid)
        actual, reference = compare(state, support)
        scale = float(jnp.linalg.norm(reference))
        assert scale > 0
        relative = float(jnp.linalg.norm(actual - reference)) / scale
        print(f"GEOMETRY_REFERENCE state={label} relative={relative:.12g}")
        assert relative <= 1e-10


def test_geometry_path_contains_no_directional_autodiff():
    source = inspect.getsource(support_read._normal_geometry_booking_tangent)
    assert "jax.jvp" not in source


def test_condition_diagnostics_do_not_change_update(monkeypatch):
    matrix = jnp.asarray([[1e-6, 2e-7, 0], [0, 1, 0.2], [0, 0, 1e6]])
    moments = tuple(jnp.asarray([1.0]) for _ in range(3))
    residual = tuple(jnp.asarray([value]) for value in (1e-6, 1.0, 1e6))
    booked = current.Booking(*moments, jnp.asarray(1.0), jnp.asarray(1.0), True)
    monkeypatch.setattr(
        support_read,
        "step_residual",
        lambda *args: support_read.StepResidual(residual, booked, None, True),
    )
    monkeypatch.setattr(support_read, "step_jacobian", lambda *args, **kwargs: matrix)
    measured = support_read.newton_step(None, moments, None, None, None, None)
    fast = support_read.newton_step(
        None, moments, None, None, None, None, condition_diagnostics=False
    )
    for observed, expected in zip(fast.update, measured.update, strict=True):
        np.testing.assert_allclose(observed, expected, rtol=0, atol=0)
    assert np.isnan(float(fast.condition_before))
    assert float(measured.condition_after) < float(measured.condition_before)
