"""Linear-residual admission contracts for projected Krylov conditioning."""

from __future__ import annotations

import math

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.equilibrium.fixed_point import newton_krylov
    from nova.jax.config import configure_dtypes


def _scaled_quadratic_map(dimension: int):
    """Return a nonlinear fixed-point map with a separated action spectrum."""
    scales = jnp.exp(jnp.linspace(0.0, -jnp.log(200.0), dimension))

    def mapped(state):
        residual = scales * (state * state - 1.0)
        return state - residual

    return mapped


def test_resolved_krylov_direction_reaches_frozen_mask_merit_ladder_undamped():
    """A resolved raw direction contracts superlinearly on a frozen mask."""
    configure_dtypes()
    dimension = 8
    initial = jnp.full(dimension, 1.1, dtype=jnp.float64)
    frozen_mask = jnp.zeros(dimension, dtype=jnp.bool_)
    mapped = _scaled_quadratic_map(dimension)

    result = newton_krylov(
        mapped,
        initial,
        newton_steps=4,
        gmres_iterations=dimension,
        warmup=0,
        relaxation=0.5,
        step_cap=1.0e6,
        convergence_tolerance=1.0e-14,
        shadow_mask_fn=lambda _state: frozen_mask,
        promoted_shadow_mask_fn=lambda _state, _preceding: frozen_mask,
        shadowed_map_fn=lambda state, _mask: mapped(state),
        active_set_steps=1,
    )

    raw_step_sup = float((initial[0] * initial[0] - 1.0) / (2.0 * initial[0]))
    measured_step_sup = float(result.inner_iteration_proposed_step_norms[0])
    residuals_before = np.asarray(result.inner_iteration_residuals_before)
    residuals_after = np.asarray(result.inner_iteration_residuals_after)
    contraction = residuals_after[:3] / residuals_before[:3]

    assert int(result.active_set_iterations) == 1
    np.testing.assert_array_equal(result.active_set_mask_differences, [0])
    assert int(result.krylov_conditioning_count) == 0
    np.testing.assert_allclose(measured_step_sup, raw_step_sup, rtol=1.0e-12)
    assert contraction[1] < contraction[0]
    assert contraction[2] < contraction[1]
    assert bool(result.converged)
    assert float(result.residual) < 1.0e-14


def test_unresolved_linear_model_retains_conditioning_receipt():
    """A poorly resolved high-condition action still reports intervention."""
    configure_dtypes()
    dimension = 12
    diagonal = jnp.exp(jnp.linspace(0.0, -jnp.log(1.0e6), dimension))
    tangent = jnp.eye(dimension) - jnp.diag(diagonal)

    result = newton_krylov(
        lambda state: tangent @ state + jnp.ones(dimension),
        jnp.zeros(dimension),
        newton_steps=1,
        gmres_iterations=2,
        warmup=0,
        step_cap=1.0e6,
        krylov_condition_limit=math.e,
    )

    assert float(result.inner_iteration_krylov_reductions[0]) > math.sqrt(
        np.finfo(np.float64).eps
    )
    assert int(result.krylov_conditioning_count) == 1
    assert float(result.maximum_projected_krylov_condition) > math.e
