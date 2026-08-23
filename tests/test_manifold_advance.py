"""Contracts for fixed-shape topology-manifold predictor-correction."""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium import fixed_point
    from nova.equilibrium.fixed_point import (
        KrylovActionQualification,
        newton_krylov,
    )
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.manifold_advance import (
        ManifoldAdvanceQualification,
        normal_component,
        oriented_secant,
    )


def _solve(initial, previous, target, admissibility_fn, *, updates=1):
    return newton_krylov(
        lambda _state: target,
        initial,
        newton_steps=updates,
        gmres_iterations=2,
        warmup=0,
        step_cap=1.0e6,
        previous_admitted_state=previous,
        admissibility_fn=admissibility_fn,
    )


def test_secant_orientation_removes_a_tangent_sign_flip():
    forward = oriented_secant(
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([1.0, 0.0]),
    )
    reversed_basis = oriented_secant(
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([0.0, 0.0]),
        forward.tangent,
    )

    np.testing.assert_array_equal(reversed_basis.tangent, forward.tangent)
    assert (
        ManifoldAdvanceQualification(int(reversed_basis.qualification))
        is ManifoldAdvanceQualification.ACCEPTED
    )


def test_a_degenerate_secant_has_a_named_refusal_and_promotes_nothing():
    initial = jnp.asarray([0.0, 0.0])
    result = _solve(
        initial,
        initial,
        jnp.asarray([1.0, 1.0]),
        lambda _candidate: jnp.asarray(True),
    )

    assert (
        ManifoldAdvanceQualification(int(result.manifold_advance_qualification[0]))
        is ManifoldAdvanceQualification.DEGENERATE_SECANT
    )
    np.testing.assert_array_equal(result.state, initial)
    np.testing.assert_array_equal(result.advance_lengths, [0.0])


def test_corrector_projection_is_normal_to_the_secant_tangent():
    tangent = jnp.asarray([3.0, 4.0]) / 5.0
    correction = normal_component(jnp.asarray([7.0, -2.0]), tangent)

    np.testing.assert_allclose(jnp.vdot(correction, tangent), 0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        correction,
        jnp.asarray([7.0, -2.0])
        - jnp.vdot(jnp.asarray([7.0, -2.0]), tangent) * tangent,
    )

    initial = jnp.asarray([0.0, 0.0])
    previous = jnp.asarray([-1.0, 0.0])
    result = _solve(
        initial,
        previous,
        jnp.asarray([1.0, 1.0]),
        lambda _candidate: jnp.asarray(True),
    )
    secant = oriented_secant(previous, initial, initial - previous)
    predictor = initial + result.predictor_lengths[0] * secant.tangent
    applied_correction = result.state - predictor
    np.testing.assert_allclose(
        jnp.vdot(applied_correction, secant.tangent), 0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        jnp.linalg.norm(applied_correction), result.corrector_lengths[0]
    )


def test_predictor_corrector_has_fixed_receipt_shapes_and_jit_vmap_agreement():
    initial = jnp.zeros((3, 2))
    previous = jnp.asarray([[-1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])
    target = jnp.asarray([[1.0, 1.0], [1.0, 1.0], [2.0, -1.0]])

    def solve(one_initial, one_previous, one_target):
        return _solve(
            one_initial,
            one_previous,
            one_target,
            lambda candidate: jnp.all(jnp.isfinite(candidate)),
            updates=2,
        )

    compiled = jax.jit(jax.vmap(solve))(initial, previous, target)
    eager = jax.vmap(solve)(initial, previous, target)

    assert compiled.state.shape == (3, 2)
    assert compiled.trace.shape == (3, 6)
    assert compiled.manifold_advance_qualification.shape == (3, 2)
    assert compiled.manifold_admissibility.shape == (3, 2)
    assert compiled.predictor_lengths.shape == (3, 2)
    assert compiled.corrector_lengths.shape == (3, 2)
    assert compiled.advance_lengths.shape == (3, 2)
    assert compiled.newton_step_lengths.shape == (3, 2)
    assert compiled.newton_step_equivalents.shape == (3,)
    for observed, expected in zip(compiled, eager, strict=True):
        np.testing.assert_allclose(observed, expected, equal_nan=True)


def test_the_supplied_admissibility_predicate_alone_controls_promotion():
    result = _solve(
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([-1.0, 0.0]),
        jnp.asarray([1.0, 1.0]),
        lambda candidate: candidate[0] < 0.5,
    )

    assert not bool(result.manifold_admissibility[0])
    assert (
        ManifoldAdvanceQualification(int(result.manifold_advance_qualification[0]))
        is ManifoldAdvanceQualification.INADMISSIBLE_CORRECTED_STATE
    )
    np.testing.assert_array_equal(result.state, [0.0, 0.0])


@pytest.mark.parametrize("mode", ("status", "zero"))
def test_an_unqualified_or_zero_material_krylov_step_promotes_nothing(
    monkeypatch, mode
):
    def refused_gmres(_operator, right_hand_side, **_options):
        if mode == "status":
            return right_hand_side, jnp.asarray(1)
        return jnp.zeros_like(right_hand_side), jnp.asarray(0)

    monkeypatch.setattr(fixed_point.jax.scipy.sparse.linalg, "gmres", refused_gmres)
    result = _solve(
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([-1.0, 0.0]),
        jnp.asarray([1.0, 1.0]),
        lambda _candidate: jnp.asarray(True),
    )

    expected = (
        KrylovActionQualification.NONSUCCESSFUL_GMRES_STATUS
        if mode == "status"
        else KrylovActionQualification.ZERO_STEP_WITH_MATERIAL_NONLINEAR_RESIDUAL
    )
    assert (
        KrylovActionQualification(int(result.krylov_action_qualification)) is expected
    )
    assert (
        ManifoldAdvanceQualification(int(result.manifold_advance_qualification[0]))
        is ManifoldAdvanceQualification.KRYLOV_ACTION_REFUSED
    )
    np.testing.assert_array_equal(result.state, [0.0, 0.0])
    np.testing.assert_array_equal(result.advance_lengths, [0.0])


def test_the_forward_solver_seam_passes_manifold_options_to_the_shared_route():
    class ForwardShell:
        newton_steps = 1

        @staticmethod
        def flux_map(_current, _requested_class, _target_current):
            return lambda _state: jnp.asarray([1.0, 1.0])

        @staticmethod
        def _receipt(_state, history, _requested_class, _target_current):
            return history

    result = ForwardProfile._solve_accelerated(
        ForwardShell(),
        "newton_krylov",
        jnp.asarray([0.0, 0.0]),
        None,
        previous_admitted_state=jnp.asarray([-1.0, 0.0]),
        admissibility_fn=lambda candidate: jnp.all(jnp.isfinite(candidate)),
        warmup=0,
        gmres_iterations=2,
        step_cap=1.0e6,
    )

    assert (
        ManifoldAdvanceQualification(int(result.manifold_advance_qualification[0]))
        is ManifoldAdvanceQualification.ACCEPTED
    )
    assert float(result.advance_lengths[0]) > 0.0
