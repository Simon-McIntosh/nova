"""Typed augmented-row contracts independent of machine-sized fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import fixed_point
from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintContext,
    ConstraintMultiplier,
    ConstraintPair,
    ConstraintRecord,
    assemble_augmented_system,
    constraint_residual_jvp,
)
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.forward_operator import PrescribedCurrentField
from nova.equilibrium.solve_request import (
    ExplicitSolveSeed,
    ForwardSolveReceipt,
    ForwardSolveRequest,
    ResolvedForwardSolveDefaults,
    declared_forward_solve_policy,
)
from nova.jax.config import configure_dtypes


@dataclass(frozen=True)
class _CoordinateFunctional:
    index: int
    nonlinear: bool = False

    @property
    def row_count(self) -> int:
        return 1

    def observed(self, _profile, context, _payload):
        value = context.flux[self.index]
        return jnp.atleast_1d(value**2 if self.nonlinear else value)

    def residual(self, profile, context, _unknown, payload, target, scale):
        return (self.observed(profile, context, payload) - target) / scale

    def dual_flux_image(self, _profile, context, payload):
        image = jnp.zeros((context.flux.size, 1), dtype=context.flux.dtype)
        return image.at[self.index, 0].set(jnp.asarray(payload))


class _LinearOperator:
    def __init__(self, *, changing_mask: bool = False) -> None:
        self.prescribed_current_field = PrescribedCurrentField(
            response=jnp.asarray([[1.0], [0.0]]),
            current=jnp.asarray([0.0]),
        )
        self.source = SimpleNamespace(closure_degrees=0)
        self.changing_mask = changing_mask

    def flux_map(self, *_args):
        return lambda flux: jnp.zeros_like(flux)

    def flux_map_with_shadow(self, *_args):
        return lambda flux, shadow: jnp.where(shadow, flux, jnp.zeros_like(flux))

    def residual_shadow_mask(self, flux, _requested=None, previous_shadow=None):
        del previous_shadow
        if self.changing_mask:
            return jnp.asarray([flux[0] > 0.5, False])
        return jnp.zeros_like(flux, dtype=bool)


def _profile(*, changing_mask: bool = False) -> ForwardProfile:
    profile = object.__new__(ForwardProfile)
    profile.operator = _LinearOperator(changing_mask=changing_mask)
    profile.newton_steps = 6

    def receipt(flux, history, *_args, constraints=(), **_kwargs):
        return SimpleNamespace(
            flux=flux,
            fixed_point=history,
            constraints=constraints,
        )

    profile._receipt = receipt
    return profile


def _pair(functional, unknown, *, target, payload=None):
    return ConstraintPair(
        functional=functional,
        unknown=unknown,
        binding=ConstraintBinding(
            target=jnp.atleast_1d(target),
            tolerance=jnp.asarray([1.0e-8]),
            scale=jnp.asarray([1.0]),
            initial_unknown=jnp.asarray([0.0]),
            payload=payload,
        ),
    )


def test_linear_circuit_and_nonlinear_multiplier_rows_converge() -> None:
    configure_dtypes()
    profile = _profile()
    pairs = (
        _pair(
            _CoordinateFunctional(0),
            CircuitCurrentUnknown(jnp.asarray([1.0]), jnp.asarray([1.0])),
            target=1.0,
        ),
        _pair(
            _CoordinateFunctional(1, nonlinear=True),
            ConstraintMultiplier(jnp.asarray([1.0])),
            target=4.0,
            payload=1.0,
        ),
    )

    result = profile._solve_augmented_constraints(
        jnp.asarray([0.75, 1.75]),
        None,
        constraint_pairs=pairs,
        warmup=0,
        gmres_iterations=4,
        active_set_steps=2,
        stop_on_active_set_settlement=False,
    )

    np.testing.assert_allclose(result.flux, [1.0, 2.0], rtol=0.0, atol=1.0e-8)
    assert len(result.constraints) == 2
    np.testing.assert_allclose(result.constraints[0].physical_unknown, [1.0])
    np.testing.assert_allclose(result.constraints[1].physical_unknown, [2.0])
    assert all(bool(np.all(record.qualified)) for record in result.constraints)
    assert result.fixed_point.state.shape == (2,)
    assert result.fixed_point.row_jvp_projections.shape == (2,)


def test_residual_row_actions_match_central_differences() -> None:
    configure_dtypes()
    profile = _profile()
    flux = jnp.asarray([0.8, 1.7])
    flux_tangent = jnp.asarray([0.3, -0.2])
    unknown = jnp.asarray([0.4])
    unknown_tangent = jnp.asarray([0.1])
    for functional, target in (
        (_CoordinateFunctional(0), 1.0),
        (_CoordinateFunctional(1, nonlinear=True), 4.0),
    ):
        pair = _pair(
            functional,
            ConstraintMultiplier(jnp.asarray([1.0])),
            target=target,
            payload=1.0,
        )
        context = ConstraintContext(flux, None, None, None)
        tangent = constraint_residual_jvp(
            pair,
            profile,
            context,
            unknown,
            flux_tangent,
            unknown_tangent,
        )
        step = 1.0e-5

        def residual(trial_flux, trial_unknown):
            return pair.functional.residual(
                profile,
                context._replace(flux=trial_flux),
                trial_unknown,
                pair.binding.payload,
                pair.binding.target,
                pair.binding.scale,
            )

        difference = (
            residual(flux + step * flux_tangent, unknown + step * unknown_tangent)
            - residual(flux - step * flux_tangent, unknown - step * unknown_tangent)
        ) / (2.0 * step)
        np.testing.assert_allclose(tangent, difference, rtol=2.0e-9, atol=2.0e-9)


def test_fixed_constraint_layout_jits_and_vmaps_targets_once() -> None:
    configure_dtypes()
    profile = _profile()
    traces = {"count": 0}
    functional = _CoordinateFunctional(0)
    unknown = ConstraintMultiplier(jnp.asarray([1.0]))

    def solve_target(target):
        traces["count"] += 1
        pair = _pair(functional, unknown, target=target, payload=1.0)
        base_map = profile.operator.flux_map()
        base_shadowed = profile.operator.flux_map_with_shadow()
        system = assemble_augmented_system(
            profile,
            jnp.asarray([0.5, 0.0]),
            (pair,),
            base_map=base_map,
            base_shadow_mask=profile.operator.residual_shadow_mask,
            base_promoted_shadow_mask=profile.operator.residual_shadow_mask,
            base_shadowed_map=base_shadowed,
            requested_class=None,
            target_current=None,
        )
        history = fixed_point.newton_krylov(
            system.map_fn,
            system.initial,
            newton_steps=3,
            gmres_iterations=3,
            warmup=0,
            shadow_mask_fn=system.shadow_mask_fn,
            promoted_shadow_mask_fn=system.promoted_shadow_mask_fn,
            shadowed_map_fn=system.shadowed_map_fn,
            active_set_steps=1,
            stop_on_active_set_settlement=False,
        )
        flux, _unknown = system.split(history.state)
        return flux

    compiled = jax.jit(jax.vmap(solve_target))
    first = compiled(jnp.asarray([1.0, 1.5]))
    first.block_until_ready()
    traced = traces["count"]
    second = compiled(jnp.asarray([1.2, 1.3]))
    second.block_until_ready()

    assert traced == 1
    assert traces["count"] == traced
    np.testing.assert_allclose(first[:, 0], [1.0, 1.5], atol=1.0e-8)
    np.testing.assert_allclose(second[:, 0], [1.2, 1.3], atol=1.0e-8)


def test_typed_request_and_receipt_carry_constraint_rows() -> None:
    profile = _profile()
    pair = _pair(
        _CoordinateFunctional(0),
        ConstraintMultiplier(jnp.asarray([1.0])),
        target=1.0,
        payload=1.0,
    )
    policy = declared_forward_solve_policy()
    request = ForwardSolveRequest(
        carrier_identity="constraint-fixture",
        source_profile=profile.operator.source,
        seed_policy=ExplicitSolveSeed(jnp.asarray([0.5, 0.0])),
        policy=policy,
        route=policy.route,
        constraint_pairs=(pair,),
    )
    record = ConstraintRecord(
        observed=jnp.asarray([1.0]),
        target=jnp.asarray([1.0]),
        physical_residual=jnp.asarray([0.0]),
        scaled_residual=jnp.asarray([0.0]),
        tolerance=jnp.asarray([1.0e-8]),
        qualified=jnp.asarray([True]),
        normalized_unknown=jnp.asarray([1.0]),
        physical_unknown=jnp.asarray([1.0]),
        soft_mode_projection=jnp.asarray([0.5]),
    )
    receipt = ForwardSolveReceipt(
        terminal_state=SimpleNamespace(constraints=(record,)),
        qualified=True,
        termination_reason=1,
        residual_history=jnp.asarray([0.0]),
        mask_history=jnp.asarray([0]),
        globalisation_decisions=(jnp.asarray([0]), jnp.asarray([1.0])),
        amplitude_history=jnp.asarray([]),
        topology_read=None,
        polish_receipt=None,
        compilation_cache_hit=False,
        wall_seconds=0.0,
        resolved_defaults=ResolvedForwardSolveDefaults.from_policy(policy),
    )

    assert request.constraint_pairs == (pair,)
    assert receipt.constraints == (record,)
