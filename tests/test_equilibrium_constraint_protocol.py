"""Typed augmented-row contracts independent of machine-sized fixtures."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.greens import MU0
from nova.equilibrium import fixed_point
from nova.equilibrium.observation import CurrentMomentObservation
from nova.equilibrium.constraint import (
    BoundedExteriorFieldUnknown,
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintContext,
    ConstraintElimination,
    ConstraintMultiplier,
    ConstraintPair,
    ConstraintRecord,
    ExternalShafranovConstraint,
    FluxLevelConstraint,
    ProfileAmplitudeUnknown,
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


class _FixtureEquilibrium(NamedTuple):
    flux: jax.Array
    fixed_point: fixed_point.FixedPointResult
    constraints: tuple[ConstraintRecord, ...]
    finite: object
    topology: object = None
    normalisation: object = None
    constraint_outer: object = None


def _profile(*, changing_mask: bool = False) -> ForwardProfile:
    profile = object.__new__(ForwardProfile)
    profile.operator = _LinearOperator(changing_mask=changing_mask)
    profile.newton_steps = 6
    profile._request_compilation_cache = set()

    def receipt(flux, history, *_args, constraints=(), **_kwargs):
        return _FixtureEquilibrium(
            flux=flux,
            fixed_point=history,
            constraints=constraints,
            finite=SimpleNamespace(passed=jnp.asarray(True)),
        )

    profile._receipt = receipt
    return profile


def _pair(
    functional,
    unknown,
    *,
    target,
    payload=None,
    policy="imposed",
    elimination=None,
):
    return ConstraintPair(
        functional=functional,
        unknown=unknown,
        binding=ConstraintBinding(
            target=jnp.atleast_1d(target),
            tolerance=jnp.asarray([1.0e-8]),
            scale=jnp.asarray([1.0]),
            initial_unknown=jnp.asarray([0.0]),
            payload=payload,
            policy=policy,
            elimination=elimination,
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


def test_bounded_exterior_field_unknown_routes_and_refuses() -> None:
    configure_dtypes()
    profile = _profile()
    functional = _CoordinateFunctional(0)
    unknown = BoundedExteriorFieldUnknown(
        direction=jnp.asarray([1.0]),
        field_scale=jnp.asarray([1.0]),
        field_bound=jnp.asarray([2.0]),
    )
    context = ConstraintContext(
        flux=jnp.asarray([0.0, 0.0]),
        requested_class=None,
        target_current=None,
        shadow=None,
    )

    delta = unknown.flux_delta(
        profile,
        context,
        functional,
        None,
        jnp.asarray([1.0]),
    )
    leaves, structure = jax.tree_util.tree_flatten(unknown)
    rebuilt = jax.tree_util.tree_unflatten(structure, leaves)

    np.testing.assert_array_equal(delta, [1.0, 0.0])
    np.testing.assert_array_equal(unknown.physical_value(jnp.asarray([1.0])), [1.0])
    np.testing.assert_array_equal(unknown.physical_value(jnp.asarray([2.5])), [2.5])
    np.testing.assert_array_equal(rebuilt.direction, unknown.direction)
    np.testing.assert_array_equal(rebuilt.field_scale, unknown.field_scale)
    np.testing.assert_array_equal(rebuilt.field_bound, unknown.field_bound)
    np.testing.assert_array_equal(rebuilt.step_limit, unknown.step_limit)
    with np.testing.assert_raises_regex(
        ValueError, "exterior-field amplitude exceeds its declared finite bound"
    ):
        unknown.require_within_bound(jnp.asarray([2.5]))


def test_bounded_exterior_step_caps_backtracks_and_refuses() -> None:
    """The per-trip cap binds, the bound refusal fires, and the tangent survives."""
    configure_dtypes()
    unknown = BoundedExteriorFieldUnknown(
        direction=jnp.asarray([1.0]),
        field_scale=jnp.asarray([1.0]),
        field_bound=jnp.asarray([2.0]),
        step_limit=1.0,
    )

    step, refused = unknown.damped_step(jnp.asarray([0.0]), jnp.asarray([0.5]))
    np.testing.assert_allclose(step, [-0.5], atol=0.0)
    assert not bool(np.asarray(refused).any())

    capped, refused = unknown.damped_step(jnp.asarray([0.0]), jnp.asarray([100.0]))
    np.testing.assert_allclose(capped, [-1.0], atol=0.0)
    # the cap bounds the per-trip change without clipping onto it
    assert abs(float(capped[0])) <= 1.0
    assert not bool(np.asarray(refused).any())

    refused_step, refused = unknown.damped_step(
        jnp.asarray([-1.0e6]), jnp.asarray([1.0e12])
    )
    np.testing.assert_array_equal(refused_step, [0.0])
    assert bool(np.asarray(refused).all())

    # the value the refusal guards keeps a nonzero tangent at the bound
    tangent = jax.jvp(
        lambda value: unknown.physical_value(value),
        (jnp.asarray([2.5]),),
        (jnp.asarray([1.0]),),
    )[1]
    np.testing.assert_allclose(tangent, [1.0], atol=0.0)
    assert bool(np.asarray(unknown.bound_refusal(jnp.asarray([2.5]))).all())


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


def test_eliminated_constraint_runs_through_typed_solve_with_outer_trace() -> None:
    configure_dtypes()
    profile = _profile()
    policy = replace(declared_forward_solve_policy(), compilation_cache=False)
    controls = ConstraintElimination(
        target_step=0.5,
        target_bounds=jnp.asarray([-2.0, 2.0]),
        unknown_tolerance=1.0e-10,
        maximum_steps=5,
    )
    eliminated = _pair(
        _CoordinateFunctional(0),
        CircuitCurrentUnknown(jnp.asarray([1.0]), jnp.asarray([1.0])),
        target=1.0,
        policy="eliminated",
        elimination=controls,
    )
    request = ForwardSolveRequest(
        carrier_identity="constraint-fixture",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(jnp.asarray([0.5, 0.0])),
        policy=policy,
        route=policy.route,
        constraint_pairs=(eliminated,),
    )

    receipt = profile.solve(request)
    trace = receipt.equilibrium.constraint_outer

    assert trace is not None
    assert trace.status in {"converged", "no-root", "budget"}
    assert trace.status == "converged"
    assert 2 <= len(trace.steps) <= controls.maximum_steps
    np.testing.assert_allclose(
        [step.target for step in trace.steps], [1.0, 1.5, 0.0], atol=1.0e-12
    )
    np.testing.assert_allclose(
        [step.compensating_value for step in trace.steps],
        [1.0, 1.5, 0.0],
        atol=1.0e-10,
    )
    assert all(
        isinstance(step.inner_receipt, ForwardSolveReceipt) for step in trace.steps
    )
    assert all(len(step.inner_receipt.constraints) == 1 for step in trace.steps)
    assert bool(np.asarray(receipt.qualified))

    imposed = replace(
        eliminated,
        binding=replace(
            eliminated.binding,
            target=jnp.atleast_1d(trace.steps[-1].target),
            initial_unknown=trace.steps[-2]
            .inner_receipt.constraints[0]
            .normalized_unknown,
            policy="imposed",
            elimination=None,
        ),
    )
    imposed_receipt = profile.solve(
        replace(
            request,
            seed_policy=ExplicitSolveSeed(
                trace.steps[-2].inner_receipt.equilibrium.flux
            ),
            constraint_pairs=(imposed,),
        )
    )
    np.testing.assert_array_max_ulp(
        np.asarray(receipt.equilibrium.flux),
        np.asarray(imposed_receipt.equilibrium.flux),
        maxulp=4,
    )


def test_eliminated_constraint_reports_no_root_and_refuses_multiple_rows() -> None:
    configure_dtypes()
    profile = _profile()
    policy = replace(declared_forward_solve_policy(), compilation_cache=False)
    controls = ConstraintElimination(
        target_step=0.25,
        target_bounds=jnp.asarray([1.0, 2.0]),
        unknown_tolerance=1.0e-10,
        maximum_steps=5,
    )
    eliminated = _pair(
        _CoordinateFunctional(0),
        CircuitCurrentUnknown(jnp.asarray([1.0]), jnp.asarray([1.0])),
        target=1.5,
        policy="eliminated",
        elimination=controls,
    )
    request = ForwardSolveRequest(
        carrier_identity="constraint-fixture",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(jnp.asarray([0.5, 0.0])),
        policy=policy,
        route=policy.route,
        constraint_pairs=(eliminated,),
    )

    receipt = profile.solve(request)
    trace = receipt.equilibrium.constraint_outer

    assert trace is not None
    assert trace.status == "no-root"
    assert not bool(np.asarray(receipt.qualified))
    np.testing.assert_allclose(
        sorted(float(step.target) for step in trace.steps)[:: len(trace.steps) - 1],
        [1.0, 2.0],
    )

    budget_controls = replace(
        controls,
        target_bounds=jnp.asarray([-2.0, 2.0]),
        maximum_steps=2,
    )
    budget_pair = replace(
        eliminated,
        binding=replace(
            eliminated.binding,
            target=jnp.asarray([1.0]),
            elimination=budget_controls,
        ),
    )
    budget_receipt = profile.solve(replace(request, constraint_pairs=(budget_pair,)))
    assert budget_receipt.equilibrium.constraint_outer.status == "budget"
    assert len(budget_receipt.equilibrium.constraint_outer.steps) == 2
    assert not bool(np.asarray(budget_receipt.qualified))

    with np.testing.assert_raises_regex(
        ValueError, "exactly one eliminated constraint"
    ):
        profile.solve(replace(request, constraint_pairs=(eliminated, eliminated)))


def test_shafranov_row_requires_a_profile_amplitude_compensator() -> None:
    """The Shafranov row states a profile constraint and refuses anything else.

    A row on beta_p + l_i/2 is moved only through the source term, so a
    circuit-current or multiplier compensator would register a constraint
    nothing can drive.  The refusal is raised when the pair is built, which is
    the only moment the functional and its compensator are visible together.
    """
    configure_dtypes()
    binding = ConstraintBinding(
        target=jnp.atleast_1d(0.5),
        tolerance=jnp.asarray([1.0e-8]),
        scale=jnp.asarray([1.0]),
        initial_unknown=jnp.asarray([0.0]),
    )
    functional = ExternalShafranovConstraint(minor_radius=0.3)
    with np.testing.assert_raises_regex(
        TypeError, "must be compensated by ProfileAmplitudeUnknown"
    ):
        ConstraintPair(
            functional,
            CircuitCurrentUnknown(jnp.asarray([1.0]), jnp.asarray([1.0])),
            binding,
        )
    pair = ConstraintPair(
        functional,
        ProfileAmplitudeUnknown("pressure_gradient", jnp.asarray([1.0])),
        binding,
    )
    assert pair.row_count == 1
    assert functional.required_unknown is ProfileAmplitudeUnknown


#: Reference minor radius of the synthetic Shafranov row [m].  It is eight
#: times the imposed radius, so the row's logarithmic term is exactly zero at
#: the imposed state: the achieved amplitude is then limited by the row's own
#: floating-point rounding rather than by the magnitude of a term that the
#: comparison would have to resolve.
_SHAFRANOV_MINOR_RADIUS = 16.0
#: Radial slope of the external flux image, so the sampled vertical field is
#: ``slope / (2 pi R)`` exactly [Wb/m].
_SHAFRANOV_EXTERNAL_SLOPE = 0.5
#: Plasma current the row divides the sampled field by [A].
_SHAFRANOV_CURRENT = 1.0e6
#: Current-ring radius of the synthetic plasma at zero state [m].  The row
#: samples one radial step either side of it, so the lattice must extend a
#: step beyond every centroid the fixture visits.
_SHAFRANOV_CENTROID_BASE = 1.0
#: Current-ring radius the imposed target is stated at [m].
_SHAFRANOV_IMPOSED_RADIUS = 2.0
#: Centroid sensitivity to the state, so the compensator's own fixed-point
#: iteration contracts rather than marches away.
_SHAFRANOV_CENTROID_GAIN = -0.25
#: Base-map contraction of the synthetic flux fixed point, so the augmented
#: system has a nonsingular state block for free currents to close on.
_SHAFRANOV_FLUX_CONTRACTION = 0.5


@dataclass(frozen=True)
class _ShafranovLattice:
    """A small uniform lattice the row reads instead of a built flux grid.

    The axes are numpy arrays, as the solved lattice carries them: the shared
    cubic reader fixes its stencil origin from the axis endpoints, so an axis
    that is a traced device array cannot be read at all.
    """

    radius: np.ndarray
    height: np.ndarray
    radial_step: float
    vertical_step: float

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.radius.shape[0]), int(self.height.shape[0]))

    @property
    def node_count(self) -> int:
        rows, columns = self.shape
        return rows * columns


class _ShafranovOperator(_LinearOperator):
    """A linear operator whose profile image is one state component.

    ``drivable`` false models a solve in which the profile amplitude is not a
    free unknown: the compensator's flux image is identically zero, so no
    value of the amplitude can move the state the row observes.
    """

    image_component = 2

    def __init__(self, *, drivable: bool = True) -> None:
        super().__init__()
        self.drivable = drivable

    def flux_map(self, *_args):
        contraction = _SHAFRANOV_FLUX_CONTRACTION
        return lambda flux: contraction * jnp.asarray(flux)

    def flux_map_with_shadow(self, *_args):
        contraction = _SHAFRANOV_FLUX_CONTRACTION
        return lambda flux, shadow: jnp.where(shadow, flux, contraction * flux)

    def profile_component_image(
        self,
        flux,
        *,
        component,
        amplitude,
        requested_class=None,
        target_current=None,
    ):
        del component, requested_class, target_current
        image = jnp.zeros_like(jnp.asarray(flux))
        if not self.drivable:
            return image
        return image.at[self.image_component].set(jnp.sum(jnp.atleast_1d(amplitude)))


def _shafranov_flux_image(lattice: _ShafranovLattice) -> jax.Array:
    """Return an external flux image linear in the radial coordinate."""
    rows, columns = lattice.shape
    grid = _SHAFRANOV_EXTERNAL_SLOPE * jnp.asarray(lattice.radius)[:, None]
    return jnp.broadcast_to(grid, (rows, columns)).reshape(-1)


def _shafranov_profile(*, drivable: bool = True) -> ForwardProfile:
    """A profile whose row state and centroid response are both analytic."""
    profile = object.__new__(ForwardProfile)
    profile.operator = _ShafranovOperator(drivable=drivable)
    profile.newton_steps = 8
    profile._request_compilation_cache = set()
    radius = np.arange(0.0, 8.0)
    height = np.asarray([-1.0, 0.0, 1.0])
    profile.lattice = _ShafranovLattice(
        radius=radius,
        height=height,
        radial_step=float(radius[1] - radius[0]),
        vertical_step=float(height[1] - height[0]),
    )

    def current_moment_observation(flux, *, support, target_current=None):
        del target_current
        state = jnp.asarray(flux)
        return CurrentMomentObservation(
            plasma_current=jnp.asarray(_SHAFRANOV_CURRENT),
            centroid_r=_SHAFRANOV_CENTROID_BASE + _SHAFRANOV_CENTROID_GAIN * state[2],
            centroid_z=jnp.asarray(0.0),
            support=support,
        )

    def receipt(flux, history, *_args, constraints=(), **_kwargs):
        return _FixtureEquilibrium(
            flux=flux,
            fixed_point=history,
            constraints=constraints,
            finite=SimpleNamespace(passed=jnp.asarray(True)),
        )

    profile.current_moment_observation = current_moment_observation
    profile._receipt = receipt
    return profile


def _shafranov_observed(centroid_radius: float) -> float:
    """The Shafranov combination a linear external flux image implies."""
    force_factor = -2.0 * _SHAFRANOV_EXTERNAL_SLOPE / (MU0 * _SHAFRANOV_CURRENT)
    return float(
        force_factor - np.log(8.0 * centroid_radius / _SHAFRANOV_MINOR_RADIUS) + 1.5
    )


def _shafranov_amplitude(centroid_radius: float) -> float:
    """The amplitude whose compensated fixed point sits at one centroid radius.

    The fixed point of the augmented map is ``flux = base(flux) + delta``, so
    a profile image on one component settles at ``amplitude / (1 - c)`` and the
    centroid the row samples is that value read through the gain.
    """
    settled_flux = (
        centroid_radius - _SHAFRANOV_CENTROID_BASE
    ) / _SHAFRANOV_CENTROID_GAIN
    return settled_flux * (1.0 - _SHAFRANOV_FLUX_CONTRACTION)


def _shafranov_pair(profile, *, target):
    return _pair(
        ExternalShafranovConstraint(minor_radius=jnp.asarray(_SHAFRANOV_MINOR_RADIUS)),
        ProfileAmplitudeUnknown("pressure_gradient", jnp.asarray([1.0])),
        target=target,
        payload=(
            _shafranov_flux_image(profile.lattice),
            jnp.asarray(_SHAFRANOV_MINOR_RADIUS),
        ),
    )


def _shafranov_imposed_state(profile, centroid_radius: float) -> jax.Array:
    """The state whose compensated fixed point sits at one centroid radius."""
    settled_flux = (
        centroid_radius - _SHAFRANOV_CENTROID_BASE
    ) / _SHAFRANOV_CENTROID_GAIN
    return jnp.zeros(profile.lattice.node_count).at[2].set(settled_flux)


def _shafranov_row_observed(profile, pair, flux) -> float:
    """The row's own reading at one state, on the row's own arithmetic path.

    A target taken this way asks the solve to reproduce a value the row itself
    states at a known state, so the achieved amplitude is limited by the solve
    rather than by a second, separately rounded derivation of the same number.
    """
    context = ConstraintContext(jnp.asarray(flux), None, None, None)
    observed = pair.functional.observed(profile, context, pair.binding.payload)
    return float(np.asarray(jnp.atleast_1d(observed))[0])


def _shafranov_target_at(profile, centroid_radius: float) -> float:
    """The row's reading at the state that implies one centroid radius."""
    probe = _shafranov_pair(profile, target=0.0)
    return _shafranov_row_observed(
        profile, probe, _shafranov_imposed_state(profile, centroid_radius)
    )


def test_shafranov_row_converges_to_the_imposed_profile_amplitude() -> None:
    """The row inverts the external field onto the amplitude that implies it.

    A target the external magnetics state is imposed and the solve has to find
    the profile amplitude whose centroid implies exactly that combination.  A
    sign, factor or stencil error in the row would move the fixed point off the
    amplitude the target was built from, so the achieved amplitude is asserted
    against it to the last few bits of the representation.
    """
    configure_dtypes()
    centroid_radius = _SHAFRANOV_IMPOSED_RADIUS
    amplitude = _shafranov_amplitude(centroid_radius)
    profile = _shafranov_profile()
    target = _shafranov_target_at(profile, centroid_radius)
    pair = _shafranov_pair(profile, target=target)

    np.testing.assert_allclose(
        [target],
        [_shafranov_observed(centroid_radius)],
        rtol=0.0,
        atol=1.0e-12,
        err_msg="the row must read the imposed state as the closed form states it",
    )

    result = profile._solve_augmented_constraints(
        jnp.full(profile.lattice.node_count, 0.5),
        None,
        constraint_pairs=(pair,),
        warmup=0,
        gmres_iterations=8,
        active_set_steps=8,
        stop_on_active_set_settlement=False,
        convergence_tolerance=1.0e-15,
    )

    record = result.constraints[0]
    assert bool(np.all(np.asarray(record.qualified)))
    np.testing.assert_allclose(
        np.asarray(record.observed), [target], rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        np.asarray(result.flux)[2],
        amplitude / (1.0 - _SHAFRANOV_FLUX_CONTRACTION),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_array_max_ulp(
        np.asarray(record.physical_unknown), np.asarray([amplitude]), maxulp=4
    )
    assert result.fixed_point.row_jvp_projections.shape == (1,)


def test_shafranov_row_refuses_a_target_its_compensator_cannot_reach() -> None:
    """A row nothing can drive is refused, and the same target driven closes.

    The document the row states is that an imposed row is accepted only when
    its compensator can move the state.  Here the compensator's flux image is
    identically zero, so the target sits a fixed distance away for every
    amplitude the compensator proposes.  The terminal record must report that
    gap unqualified rather than a fit; the paired drivable arm shows the same
    target closing once the amplitude does move the state, so the refusal is
    the compensator's absence rather than an unreachable number.
    """
    configure_dtypes()
    gap = 0.25
    blocked = _shafranov_profile(drivable=False)
    target = _shafranov_target_at(blocked, _SHAFRANOV_IMPOSED_RADIUS) - gap
    seed = jnp.full(blocked.lattice.node_count, 0.5)
    blocked_pair = _shafranov_pair(blocked, target=target)
    blocked_result = blocked._solve_augmented_constraints(
        seed,
        None,
        constraint_pairs=(blocked_pair,),
        warmup=0,
        gmres_iterations=8,
        active_set_steps=8,
        stop_on_active_set_settlement=False,
    )

    blocked_record = blocked_result.constraints[0]
    assert not bool(np.asarray(blocked_record.qualified).any())
    np.testing.assert_allclose(
        np.asarray(blocked_result.flux),
        np.asarray(seed),
        rtol=0.0,
        atol=1.0e-8,
        err_msg="a row no compensator can move must not march the state to a fit",
    )
    np.testing.assert_allclose(
        np.asarray(blocked_record.physical_residual),
        [_shafranov_row_observed(blocked, blocked_pair, blocked_result.flux) - target],
        rtol=0.0,
        atol=1.0e-12,
        err_msg="the refusal is reported as the row's own unqualified reading",
    )
    assert abs(float(np.asarray(blocked_record.physical_residual)[0])) >= gap

    drivable = _shafranov_profile()
    drivable_pair = _shafranov_pair(drivable, target=target)
    drivable_result = drivable._solve_augmented_constraints(
        seed,
        None,
        constraint_pairs=(drivable_pair,),
        warmup=0,
        gmres_iterations=8,
        active_set_steps=8,
        stop_on_active_set_settlement=False,
    )

    drivable_record = drivable_result.constraints[0]
    assert bool(np.all(np.asarray(drivable_record.qualified)))
    np.testing.assert_allclose(
        np.asarray(drivable_record.observed), [target], rtol=0.0, atol=1.0e-12
    )


def _level_row_state(profile) -> tuple[jax.Array, ConstraintContext]:
    """Return a linear radial flux map and the context the level row reads."""
    lattice = profile.lattice
    radius = jnp.asarray(lattice.radius)[:, None]
    grid = jnp.broadcast_to(radius, lattice.shape)
    flux = grid.reshape(-1)
    return flux, ConstraintContext(
        flux=flux,
        requested_class=None,
        target_current=None,
        shadow=None,
    )


def test_flux_level_row_reads_the_level_and_scales_its_residual() -> None:
    configure_dtypes()
    profile = _shafranov_profile()
    row = FluxLevelConstraint(point_count=1)
    _flux, context = _level_row_state(profile)
    payload = jnp.asarray([[2.0, 0.0]])

    assert row.row_count == 1
    np.testing.assert_allclose(
        np.asarray(row.observed(profile, context, payload)),
        [2.0],
        rtol=0.0,
        atol=1.0e-12,
        err_msg="the row reads the map's own level at the declared point",
    )

    # the residual is signed against the commanded level and scaled by the
    # binding, so a level above and below the target report opposite signs
    np.testing.assert_allclose(
        np.asarray(
            row.residual(
                profile, context, None, payload, jnp.asarray([1.0]), jnp.asarray([0.5])
            )
        ),
        [2.0],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(
            row.residual(
                profile, context, None, payload, jnp.asarray([3.0]), jnp.asarray([0.5])
            )
        ),
        [-2.0],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_flux_level_row_dual_image_sums_to_one() -> None:
    configure_dtypes()
    profile = _shafranov_profile()
    row = FluxLevelConstraint(point_count=1)
    flux, context = _level_row_state(profile)

    image = row.dual_flux_image(profile, context, jnp.asarray([[2.0, 0.0]]))
    assert image.shape == (flux.size, 1)
    # an interpolated read of the level has unit total leverage on the map:
    # the reported level enters through one partition of interpolation weights
    np.testing.assert_allclose(
        float(np.sum(np.asarray(image))),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
        err_msg="the level row's flux image is a partition of unity",
    )


def test_flux_level_row_states_its_carrier_requirement() -> None:
    configure_dtypes()
    profile = SimpleNamespace(lattice=SimpleNamespace(node_count=4))
    context = ConstraintContext(
        flux=jnp.arange(4.0),
        requested_class=None,
        target_current=None,
        shadow=None,
    )
    with np.testing.assert_raises_regex(TypeError, "structured FluxLattice"):
        FluxLevelConstraint(point_count=1).observed(
            profile, context, jnp.asarray([[2.0, 0.0]])
        )


def test_unbounded_exterior_amplitude_is_reported_outside_the_field_bound() -> None:
    configure_dtypes()
    field = BoundedExteriorFieldUnknown(
        direction=jnp.eye(3),
        field_scale=jnp.asarray((1.0e-3, 1.0e-3, 1.0)),
        field_bound=jnp.asarray((2.5e-1, 2.5e-1, jnp.inf)),
        step_limit=jnp.asarray((1.0, 1.0, 1.0)),
    )

    np.testing.assert_array_equal(
        np.asarray(field.field_bound_applies), np.asarray([True, True, False])
    )
    np.testing.assert_allclose(
        np.asarray(field.physical_value(jnp.asarray((0.0, 0.0, 1.0e4)))),
        [0.0, 0.0, 1.0e4],
        rtol=0.0,
    )
    # a level past the tesla bound is never refused by the field bound
    step, refused = field.damped_step(jnp.zeros(3), jnp.asarray((0.0, 0.0, 1.0e6)))
    assert not bool(np.asarray(refused).any())
    assert float(np.asarray(step)[0]) == 0.0
    assert float(np.asarray(step)[2]) < 0.0

    # the bounded components keep refusing exactly as before, while the level
    # component is held only by its own step cap -- a level amplitude is a flux
    # offset, not a field, so the tesla bound never refuses a level step
    over_bound = jnp.asarray((2.0 * 2.5e-1 / 1.0e-3, 0.0, 1.0e6))
    step, refused = field.damped_step(over_bound, jnp.zeros(3))
    np.testing.assert_array_equal(np.asarray(step), np.zeros(3))
    np.testing.assert_array_equal(np.asarray(refused), np.asarray([True, True, False]))
