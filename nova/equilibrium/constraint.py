"""Typed data interface for constraint-augmented forward equilibria.

Constraint kinds, compensating unknowns, and per-solve bindings are separate
objects.  Their Python types and tuple positions define a static solver layout;
targets, tolerances, scales, initial values, and payloads remain ordinary JAX
leaves that can be traced and mapped over.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.greens import MU0
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.observation import MomentIntegralSupport

if TYPE_CHECKING:
    from nova.equilibrium.forward import ForwardProfile


Payload = TypeVar("Payload")
ConstraintPolicy = Literal["imposed", "eliminated"]


@dataclass(frozen=True)
class ConstraintElimination:
    """Bounded scalar search parameters for one eliminated constraint row.

    The inner augmented solve continues to impose ``ConstraintBinding.target``
    and solve its compensating unknown freely.  The outer solve varies that
    target within ``target_bounds`` until the physical compensating value
    reaches ``prescribed_unknown``.  ``target_step`` supplies the second
    scalar sample and ``maximum_steps`` bounds the number of inner receipts.
    """

    target_step: object
    target_bounds: object
    unknown_tolerance: object
    prescribed_unknown: object = 0.0
    maximum_steps: int = 6

    def __post_init__(self) -> None:
        """Require finite scalar controls and an ordered target interval."""
        step = np.asarray(self.target_step)
        bounds = np.asarray(self.target_bounds)
        tolerance = np.asarray(self.unknown_tolerance)
        prescribed = np.asarray(self.prescribed_unknown)
        if step.size != 1 or not np.all(np.isfinite(step)) or np.all(step == 0.0):
            raise ValueError("elimination target_step must be finite and nonzero")
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
            raise ValueError("elimination target_bounds must be two finite values")
        if not bounds[0] < bounds[1]:
            raise ValueError("elimination target_bounds must be strictly ordered")
        if (
            tolerance.size != 1
            or not np.all(np.isfinite(tolerance))
            or np.any(tolerance <= 0.0)
        ):
            raise ValueError(
                "elimination unknown_tolerance must be positive and finite"
            )
        if prescribed.size != 1 or not np.all(np.isfinite(prescribed)):
            raise ValueError("elimination prescribed_unknown must be finite and scalar")
        if self.maximum_steps < 2:
            raise ValueError("elimination maximum_steps must be at least two")
        object.__setattr__(self, "target_step", jnp.asarray(self.target_step))
        object.__setattr__(self, "target_bounds", jnp.asarray(self.target_bounds))
        object.__setattr__(
            self, "unknown_tolerance", jnp.asarray(self.unknown_tolerance)
        )
        object.__setattr__(
            self, "prescribed_unknown", jnp.asarray(self.prescribed_unknown)
        )


class ConstraintOuterStep(NamedTuple):
    """One target evaluation and the complete imposed inner receipt."""

    target: jax.Array
    compensating_value: jax.Array
    inner_receipt: object


class ConstraintOuterTrace(NamedTuple):
    """Bounded scalar elimination history carried by the terminal state."""

    steps: tuple[ConstraintOuterStep, ...]
    status: Literal["converged", "no-root", "budget"]
    target_bounds: jax.Array
    prescribed_unknown: jax.Array
    unknown_tolerance: jax.Array


class CompensatorRule(IntEnum):
    """How one row's compensating circuit direction was decided.

    The value is what a receipt carries, because a record row is an array
    leaf under :func:`jax.vmap`; :func:`compensator_rule_name` turns it back
    into the readable name.
    """

    EXPLICIT = 0
    DOMINANT_AUTHORITY = 1
    SINGULAR_DISTRIBUTION = 2


def compensator_rule_name(value) -> str:
    """Return the readable rule name behind one recorded integer code."""
    return CompensatorRule(int(np.asarray(value).reshape(-1)[0])).name.lower()


class ConstraintContext(NamedTuple):
    """Traced equilibrium state visible to one constraint implementation."""

    flux: jax.Array
    requested_class: jax.Array | None
    target_current: jax.Array | None
    shadow: jax.Array | None


class ConstraintFunctional(Protocol[Payload]):
    """One fixed-size physical observation and its residual row action."""

    @property
    def row_count(self) -> int: ...

    def observed(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: Payload,
    ) -> jax.Array: ...

    def residual(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        unknown: jax.Array,
        payload: Payload,
        target: jax.Array,
        scale: jax.Array,
    ) -> jax.Array: ...

    def dual_flux_image(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: Payload,
    ) -> jax.Array: ...


class CompensatingUnknown(Protocol):
    """Map normalized solver unknowns to physical values and flux images."""

    @property
    def row_count(self) -> int: ...

    def physical_value(self, normalized: jax.Array) -> jax.Array: ...

    def flux_delta(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        functional: ConstraintFunctional[object],
        payload: object,
        normalized: jax.Array,
    ) -> jax.Array: ...


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ConstraintBinding:
    """Dynamic values that bind one functional and unknown to one solve."""

    target: object
    tolerance: object
    scale: object
    initial_unknown: object
    payload: object = None
    policy: ConstraintPolicy = "imposed"
    elimination: ConstraintElimination | None = None

    def __post_init__(self) -> None:
        """Require a known solve policy; row-shape validation belongs to the pair."""
        if self.policy not in ("imposed", "eliminated"):
            raise ValueError("constraint policy must be 'imposed' or 'eliminated'")
        if self.policy == "imposed" and self.elimination is not None:
            raise ValueError("an imposed constraint cannot carry elimination controls")
        if self.policy == "eliminated" and not isinstance(
            self.elimination, ConstraintElimination
        ):
            raise ValueError(
                "an eliminated constraint needs ConstraintElimination controls"
            )
        for name in ("target", "tolerance", "scale", "initial_unknown"):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name)))

    def tree_flatten(self):
        """Keep policy static while all numerical binding values remain leaves."""
        return (
            (
                self.target,
                self.tolerance,
                self.scale,
                self.initial_unknown,
                self.payload,
                self.elimination,
            ),
            (self.policy,),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild a binding from its static policy and traced leaves."""
        (policy,) = aux_data
        target, tolerance, scale, initial_unknown, payload, elimination = children
        return cls(
            target,
            tolerance,
            scale,
            initial_unknown,
            payload,
            policy,
            elimination,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ConstraintPair:
    """One functional, one equally sized compensator, and their solve data."""

    functional: ConstraintFunctional[object]
    unknown: CompensatingUnknown
    binding: ConstraintBinding

    def __post_init__(self) -> None:
        """Validate the static square layout and trailing row dimensions."""
        rows = int(self.functional.row_count)
        if rows < 1:
            raise ValueError("a constraint must contribute at least one row")
        if int(self.unknown.row_count) != rows:
            raise ValueError("one compensating unknown is required per residual row")
        required = getattr(self.functional, "required_unknown", None)
        if required is not None and not isinstance(self.unknown, required):
            raise TypeError(
                f"{type(self.functional).__name__} must be compensated by "
                f"{required.__name__}, not {type(self.unknown).__name__}"
            )
        for name in ("target", "tolerance", "scale", "initial_unknown"):
            shape = jnp.shape(getattr(self.binding, name))
            if not shape or shape[-1] != rows:
                raise ValueError(
                    f"constraint {name} must have a trailing row dimension of {rows}"
                )
        _require_positive_if_concrete(self.binding.tolerance, "tolerance")
        _require_positive_if_concrete(self.binding.scale, "scale")

    @property
    def row_count(self) -> int:
        """Return the fixed number of residual rows in this tuple position."""
        return int(self.functional.row_count)

    def tree_flatten(self):
        """Keep the functional type static and map over unknown/binding leaves."""
        return ((self.unknown, self.binding), (self.functional,))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild a pair without changing its statically selected functional."""
        (functional,) = aux_data
        unknown, binding = children
        return cls(functional, unknown, binding)


def _require_positive_if_concrete(value: object, name: str) -> None:
    """Validate host values without forcing a traced value back to the host."""
    try:
        concrete = np.asarray(value)
    except TypeError, jax.errors.TracerArrayConversionError:
        return
    if not np.all(np.isfinite(concrete)) or np.any(concrete <= 0.0):
        raise ValueError(f"constraint {name} must be positive and finite")


def constraint_row_slices(pairs: tuple[ConstraintPair, ...]) -> tuple[slice, ...]:
    """Return static flattened row slices for one ordered constraint tuple."""
    offset = 0
    result = []
    for pair in pairs:
        result.append(slice(offset, offset + pair.row_count))
        offset += pair.row_count
    return tuple(result)


def constraint_residual_jvp(
    pair: ConstraintPair,
    profile: ForwardProfile,
    context: ConstraintContext,
    unknown: jax.Array,
    flux_tangent: jax.Array,
    unknown_tangent: jax.Array,
) -> jax.Array:
    """Differentiate the authored residual instead of accepting a second formula."""
    binding = pair.binding
    _, tangent = jax.jvp(
        lambda flux, value: pair.functional.residual(
            profile,
            context._replace(flux=flux),
            value,
            binding.payload,
            jnp.asarray(binding.target),
            jnp.asarray(binding.scale),
        ),
        (context.flux, unknown),
        (flux_tangent, unknown_tangent),
    )
    return jnp.atleast_1d(tangent)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CircuitCurrentUnknown:
    """Physical circuit-current columns selected by dimensionless directions.

    ``direction`` is the caller's when a circuit is named outright and the
    output of :func:`derive_circuit_compensators` when the direction is read
    off the constraint-response matrix instead.  ``rule`` states which of the
    two produced it and ``singular_values`` and ``authority`` carry the
    spectrum and the per-ampere authority the derivation measured, so the
    receipt can say why this direction was taken.
    """

    direction: object
    ampere_scale: object
    singular_values: object = None
    authority: object = None
    rule: CompensatorRule = CompensatorRule.EXPLICIT

    def __post_init__(self) -> None:
        direction = jnp.asarray(self.direction)
        scale = jnp.atleast_1d(jnp.asarray(self.ampere_scale))
        if direction.ndim == 1:
            direction = direction[:, None]
        if direction.ndim != 2 or direction.shape[1] != scale.shape[-1]:
            raise ValueError(
                "circuit directions must have shape (circuit_count, row_count)"
            )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "ampere_scale", scale)
        object.__setattr__(self, "rule", CompensatorRule(self.rule))
        _require_positive_if_concrete(scale, "ampere scale")

    @property
    def row_count(self) -> int:
        return int(jnp.shape(self.ampere_scale)[-1])

    def physical_value(self, normalized: jax.Array) -> jax.Array:
        return jnp.asarray(self.ampere_scale) * normalized

    def flux_delta(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        functional: ConstraintFunctional[object],
        payload: object,
        normalized: jax.Array,
    ) -> jax.Array:
        del context, functional, payload
        field = profile.operator.prescribed_current_field
        if field is None:
            raise ValueError("a circuit-current constraint needs a prescribed field")
        current_delta = jnp.asarray(self.direction) @ self.physical_value(normalized)
        return field.flux_delta(current_delta)

    def tree_flatten(self):
        """Keep the selection rule static and map over the numerical leaves."""
        return (
            (self.direction, self.ampere_scale, self.singular_values, self.authority),
            (self.rule,),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild the compensator without re-running the selection."""
        (rule,) = aux_data
        direction, ampere_scale, singular_values, authority = children
        return cls(direction, ampere_scale, singular_values, authority, rule)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ConstraintMultiplier:
    """Non-actuator multiplier applied to a functional's dual flux image."""

    multiplier_scale: object

    def __post_init__(self) -> None:
        scale = jnp.atleast_1d(jnp.asarray(self.multiplier_scale))
        object.__setattr__(self, "multiplier_scale", scale)
        _require_positive_if_concrete(scale, "multiplier scale")

    @property
    def row_count(self) -> int:
        return int(jnp.shape(self.multiplier_scale)[-1])

    def physical_value(self, normalized: jax.Array) -> jax.Array:
        return jnp.asarray(self.multiplier_scale) * normalized

    def flux_delta(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        functional: ConstraintFunctional[object],
        payload: object,
        normalized: jax.Array,
    ) -> jax.Array:
        image = functional.dual_flux_image(profile, context, payload)
        return jnp.asarray(image) @ self.physical_value(normalized)

    def tree_flatten(self):
        return ((self.multiplier_scale,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def _require_within_bound_if_concrete(value: object, bound: object, name: str) -> None:
    """Reject a concrete physical command outside its declared finite interval."""
    try:
        concrete = np.asarray(value)
        concrete_bound = np.asarray(bound)
    except TypeError, jax.errors.TracerArrayConversionError:
        return
    if not np.all(np.isfinite(concrete)):
        raise ValueError(f"{name} must be finite")
    if np.any(np.abs(concrete) > concrete_bound):
        raise ValueError(f"{name} exceeds its declared finite bound")


def _require_positive_or_unbounded_if_concrete(value: object, name: str) -> None:
    """Require positive finite limits, or an unbounded one, without forcing a trace."""
    try:
        concrete = np.asarray(value)
    except TypeError, jax.errors.TracerArrayConversionError:
        return
    if np.any(concrete <= 0.0) or np.any(np.isnan(concrete)):
        raise ValueError(f"{name} must be positive and not NaN")


def compensator_step(
    unknown: CompensatingUnknown, normalized: jax.Array, row_residual: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return the per-trip normalized unknown step and its refusal flag.

    Every augmenting unknown steps along ``-row_residual``.  A bounded
    unknown additionally caps and backtracks that step through its own
    :meth:`BoundedExteriorFieldUnknown.damped_step`, so the bounded route is a
    property of the unknown rather than a branch in the solver.
    """
    step = -jnp.asarray(row_residual)
    damped = getattr(unknown, "damped_step", None)
    if damped is None:
        return step, jnp.zeros_like(step, dtype=bool)
    return damped(normalized, row_residual)


def compensator_bound_refusal(
    unknown: CompensatingUnknown, normalized: jax.Array
) -> jax.Array | None:
    """Return an unknown's terminal bound refusal, or None when unbounded."""
    refusal = getattr(unknown, "bound_refusal", None)
    if refusal is None:
        return None
    return jnp.atleast_1d(refusal(normalized))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BoundedExteriorFieldUnknown:
    """Bounded physical field amplitudes applied through an exterior response.

    ``direction`` selects one column-vector direction per constraint row from
    the operator's prescribed exterior-field response.  ``field_scale`` maps
    the dimensionless Newton unknown to the physical unit of that response
    column and ``field_bound`` gives the largest admitted magnitude there.

    A component declared with an unbounded limit carries a unit the field bound
    has nothing to say about.  The level column of a coil-less fixture is that
    case: its amplitude is a uniform flux offset in weber, added identically
    everywhere, so it carries no poloidal field and cannot move the plasma.  It
    is reported beside the field amplitudes and never refused by the field
    bound, which :meth:`field_bound_applies` states per component.

    The bound is imposed by damped step control, never by saturating
    :meth:`physical_value`: a value clipped at its bound reaches zero tangent
    there, so a Newton step that lands on the bound can no longer be pulled
    back and the row saturates instead of converging.  Instead the per-trip
    change of the normalized unknown is capped at ``step_limit``, the step is
    backtracked until the trial field lies inside the bound, and a step that
    still exceeds it after the ladder is refused with the refusal recorded.
    Callers that need to qualify a proposed command use
    :meth:`require_within_bound` for an explicit refusal before the response
    is evaluated.
    """

    direction: object
    field_scale: object
    field_bound: object
    step_limit: object = 1.0
    backtrack_steps: int = 8

    def __post_init__(self) -> None:
        direction = jnp.asarray(self.direction)
        scale = jnp.atleast_1d(jnp.asarray(self.field_scale))
        bound = jnp.atleast_1d(jnp.asarray(self.field_bound))
        limit = jnp.atleast_1d(jnp.asarray(self.step_limit))
        if direction.ndim == 1:
            direction = direction[:, None]
        if direction.ndim != 2 or direction.shape[1] != scale.shape[-1]:
            raise ValueError(
                "exterior-field directions must have shape (field_count, row_count)"
            )
        if bound.shape != scale.shape:
            raise ValueError("exterior-field bounds must match the field scales")
        if limit.shape != scale.shape:
            raise ValueError("exterior-field step limits must match the field scales")
        if int(self.backtrack_steps) < 1:
            raise ValueError("an exterior-field backtracking ladder needs steps")
        object.__setattr__(self, "step_limit", limit)
        object.__setattr__(self, "backtrack_steps", int(self.backtrack_steps))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "field_scale", scale)
        object.__setattr__(self, "field_bound", bound)
        _require_positive_if_concrete(scale, "exterior-field scale")
        _require_positive_or_unbounded_if_concrete(bound, "exterior-field bound")
        _require_positive_if_concrete(limit, "exterior-field step limit")

    @property
    def field_bound_applies(self) -> jax.Array:
        """Return whether each amplitude is subject to the declared field bound.

        An amplitude whose limit is unbounded is reported beside the field
        amplitudes and never refused: a uniform flux offset in weber adds no
        poloidal field, so the tesla bound says nothing about it.
        """
        return jnp.isfinite(jnp.asarray(self.field_bound))

    @property
    def row_count(self) -> int:
        """Return one physical field amplitude per selected response direction."""
        return int(jnp.shape(self.field_scale)[-1])

    def physical_value(self, normalized: jax.Array) -> jax.Array:
        """Return the unclipped field amplitude in tesla.

        Bounds are imposed by :meth:`damped_step` as a recorded refusal, so the
        value keeps a nonzero tangent at the declared bound.
        """
        return jnp.asarray(self.field_scale) * normalized

    def require_within_bound(self, normalized: jax.Array) -> jax.Array:
        """Return a direct trial in tesla or refuse it before field evaluation."""
        value = jnp.asarray(self.field_scale) * normalized
        _require_within_bound_if_concrete(
            value, self.field_bound, "exterior-field amplitude"
        )
        return value

    def bound_refusal(self, normalized: jax.Array) -> jax.Array:
        """Return whether a physical field would exceed its declared bound."""
        return jnp.abs(self.physical_value(normalized)) > jnp.asarray(self.field_bound)

    def damped_step(
        self, normalized: jax.Array, row_residual: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Return the capped, backtracked normalized step and its refusal flag.

        ``row_residual`` is the scaled constraint residual at the current
        iterate, so ``-row_residual`` is the undamped Newton step on the
        normalized unknown.  The step is capped at ``step_limit`` per trip so
        one trip can never jump the declared interval, then backtracked by a
        fixed halving ladder until the trial field lies inside the bound.  A
        step that still exceeds the bound after the ladder is refused: the
        unknown holds and the refusal is recorded rather than the field being
        clipped onto the bound.
        """
        raw = -jnp.asarray(row_residual)
        magnitude = jnp.abs(raw)
        tiny = jnp.finfo(jnp.asarray(self.field_scale).dtype).tiny
        factor = jnp.minimum(
            1.0, jnp.asarray(self.step_limit) / jnp.maximum(magnitude, tiny)
        )
        normalized_value = jnp.asarray(normalized)
        refused = self.bound_refusal(normalized_value + factor * raw)
        for _ in range(int(self.backtrack_steps)):
            factor = jnp.where(refused, 0.5 * factor, factor)
            refused = self.bound_refusal(normalized_value + factor * raw)
        step = jnp.where(refused, 0.0, factor * raw)
        return step, refused

    def flux_delta(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        functional: ConstraintFunctional[object],
        payload: object,
        normalized: jax.Array,
    ) -> jax.Array:
        """Apply the field through the operator's prescribed exterior slot."""
        del context, functional, payload
        field = profile.operator.prescribed_current_field
        if field is None:
            raise ValueError(
                "an exterior-field constraint needs a prescribed exterior response"
            )
        field_delta = jnp.asarray(self.direction) @ self.physical_value(normalized)
        return field.flux_delta(field_delta)

    def tree_flatten(self):
        return (
            (self.direction, self.field_scale, self.field_bound, self.step_limit),
            self.backtrack_steps,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, backtrack_steps=aux_data)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ProfileAmplitudeUnknown:
    """Explicit pressure-gradient or poloidal-current profile amplitude."""

    component: Literal["pressure_gradient", "ff_prime"]
    amplitude_scale: object

    def __post_init__(self) -> None:
        if self.component not in ("pressure_gradient", "ff_prime"):
            raise ValueError("unknown profile-amplitude component")
        scale = jnp.atleast_1d(jnp.asarray(self.amplitude_scale))
        object.__setattr__(self, "amplitude_scale", scale)
        _require_positive_if_concrete(scale, "profile amplitude scale")

    @property
    def row_count(self) -> int:
        return int(jnp.shape(self.amplitude_scale)[-1])

    def physical_value(self, normalized: jax.Array) -> jax.Array:
        return jnp.asarray(self.amplitude_scale) * normalized

    def flux_delta(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        functional: ConstraintFunctional[object],
        payload: object,
        normalized: jax.Array,
    ) -> jax.Array:
        del functional, payload
        image = profile.operator.profile_component_image(
            context.flux,
            component=self.component,
            amplitude=self.physical_value(normalized),
            requested_class=context.requested_class,
            target_current=context.target_current,
        )
        return jnp.asarray(image)

    def tree_flatten(self):
        return ((self.amplitude_scale,), (self.component,))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (component,) = aux_data
        (amplitude_scale,) = children
        return cls(component, amplitude_scale)


class CompensatorSelection(NamedTuple):
    """What the constraint-response matrix said and which directions it gave.

    ``response`` holds the derivative of every registered observation with
    respect to every prescribed circuit current, in the observation's own
    physical unit per ampere.  ``authority`` is the same matrix divided by
    each row's declared scale, so its entries are that row's own scales moved
    per ampere and rows are comparable with one another.  ``directions`` are
    the compensating directions handed to the circuit unknowns, one column per
    row, normalised so the largest participating circuit carries unity.

    ``drivable`` lists the circuit indices the selection was allowed to reach.
    The response carrier holds a column for every conductor the operator
    prescribes, passive structure included, and a compensator that asks a
    passive ring to carry a current is not a compensator; ``authority`` and
    ``response`` stay full width so the ranking still shows what was excluded,
    while ``singular_values`` and ``directions`` describe the drivable block.
    """

    rule: CompensatorRule
    response: np.ndarray
    authority: np.ndarray
    singular_values: np.ndarray
    directions: np.ndarray
    direction_authority: np.ndarray
    row_coupling: np.ndarray
    competing: bool
    drivable: np.ndarray

    def leading_circuits(
        self, row: int, *, count: int = 4, floor: float = 1.0e-9
    ) -> tuple[int, ...]:
        """Return the circuit indices carrying most of one row's direction.

        ``floor`` is relative to the direction's largest component and exists
        to keep rounding noise in an inactive circuit out of the receipt.
        """
        column = np.abs(np.asarray(self.directions)[:, row])
        order = np.argsort(column)[::-1]
        threshold = float(floor) * column[order[0]]
        return tuple(int(index) for index in order[:count] if column[index] > threshold)


def constraint_response_matrix(
    profile: ForwardProfile,
    pairs: Sequence[ConstraintPair],
    flux: jax.Array,
    *,
    requested_class: jax.Array | None = None,
    target_current: jax.Array | None = None,
) -> jax.Array:
    """Differentiate every registered observation against every circuit current.

    The prescribed circuits enter the flux state linearly through the
    operator's response carrier, so one reverse-mode pass per residual row
    contracted with that carrier is the whole matrix, exact and at the cost of
    the rows rather than of the circuits.
    """
    pairs = tuple(pairs)
    if not pairs:
        raise ValueError("a response matrix needs at least one constraint pair")
    field = profile.operator.prescribed_current_field
    if field is None:
        raise ValueError("a constraint response matrix needs a prescribed field")
    response = jnp.asarray(field.response)
    state = jnp.ravel(jnp.asarray(flux))
    context = ConstraintContext(state, requested_class, target_current, None)
    blocks = []
    for pair in pairs:

        def observe(value, pair=pair):
            return jnp.atleast_1d(
                pair.functional.observed(
                    profile, context._replace(flux=value), pair.binding.payload
                )
            )

        jacobian = jnp.reshape(jax.jacrev(observe)(state), (pair.row_count, state.size))
        blocks.append(jacobian @ response)
    return jnp.concatenate(blocks, axis=0)


def _infinity_normalised(columns: np.ndarray) -> np.ndarray:
    """Scale each column so its largest participating circuit carries unity."""
    peak = np.max(np.abs(columns), axis=0)
    peak = np.where(peak > 0.0, peak, 1.0)
    return columns / peak


def select_compensating_directions(
    authority: np.ndarray,
    *,
    circuits: Sequence[int] | np.ndarray | None = None,
    rule: CompensatorRule | None = None,
    competition_threshold: float = 0.5,
    participation_floor: float = 0.0,
) -> CompensatorSelection:
    """Read compensating circuit directions off one normalised authority matrix.

    A row's steepest direction is its own authority row: among directions of
    equal current norm, that one buys the most motion, and its largest entry
    names the circuit with the most authority per ampere.  Two rows whose
    steepest directions are nearly parallel are asking the same circuits for
    different things, and taking each row's steepest direction would leave the
    pair fighting.  The pseudo-inverse columns of the authority matrix are the
    directions that move one row and leave the others where they are, so rows
    that compete are distributed across the circuits by the matrix's own
    singular structure instead.
    """
    authority = np.asarray(authority, dtype=float)
    if authority.ndim != 2:
        raise ValueError("the authority matrix must have shape (row, circuit)")
    rows, circuit_count = authority.shape
    if circuits is None:
        drivable = np.arange(circuit_count)
    else:
        drivable = np.unique(np.asarray(circuits, dtype=int))
        if drivable.size == 0:
            raise ValueError("a derived direction needs at least one drivable circuit")
        if drivable[0] < 0 or drivable[-1] >= circuit_count:
            raise ValueError("a drivable circuit index falls outside the response")
    if not np.all(np.isfinite(authority)):
        raise ValueError("every constraint row needs finite non-zero circuit authority")
    block = authority[:, drivable]
    norms = np.linalg.norm(block, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("every constraint row needs finite non-zero circuit authority")
    unit = block / norms[:, None]
    coupling = np.abs(unit @ unit.T)
    competing = bool(
        rows > 1 and np.max(coupling - np.eye(rows)) > float(competition_threshold)
    )
    if rule is None:
        rule = (
            CompensatorRule.SINGULAR_DISTRIBUTION
            if competing
            else CompensatorRule.DOMINANT_AUTHORITY
        )
    rule = CompensatorRule(rule)
    singular_values = np.linalg.svd(block, compute_uv=False)
    if rule is CompensatorRule.DOMINANT_AUTHORITY:
        columns = block.T
    elif rule is CompensatorRule.SINGULAR_DISTRIBUTION:
        columns = np.linalg.pinv(block)
    else:
        raise ValueError("a derived direction needs an authority or singular rule")
    columns = np.asarray(columns, dtype=float)
    if participation_floor > 0.0:
        peak = np.max(np.abs(columns), axis=0, keepdims=True)
        columns = np.where(
            np.abs(columns) >= float(participation_floor) * peak, columns, 0.0
        )
    directions = np.zeros((circuit_count, rows))
    directions[drivable] = _infinity_normalised(columns)
    direction_authority = np.einsum("rc,cr->r", authority, directions)
    if np.any(direction_authority <= 0.0):
        sign = np.where(direction_authority < 0.0, -1.0, 1.0)
        directions = directions * sign[None, :]
        direction_authority = direction_authority * sign
    return CompensatorSelection(
        rule=rule,
        response=np.zeros_like(authority),
        authority=authority,
        singular_values=singular_values,
        directions=directions,
        direction_authority=direction_authority,
        row_coupling=coupling,
        competing=competing,
        drivable=drivable,
    )


def derive_circuit_compensators(
    profile: ForwardProfile,
    pairs: Sequence[ConstraintPair],
    flux: jax.Array,
    *,
    requested_class: jax.Array | None = None,
    target_current: jax.Array | None = None,
    circuits: Sequence[int] | np.ndarray | None = None,
    rule: CompensatorRule | None = None,
    competition_threshold: float = 0.5,
    participation_floor: float = 0.0,
) -> tuple[tuple[ConstraintPair, ...], CompensatorSelection]:
    """Replace each pair's compensator with the direction the matrix implies.

    Only the direction is derived.  An ampere scale the caller already stated
    is kept, because it sets conditioning and not the converged current; a
    pair that arrives without one is given the current amplitude that moves
    its row by one declared scale, which is the same conditioning the flux
    block already carries.

    ``circuits`` names the columns the caller can actually drive.  Leaving it
    unset lets the derivation reach every column the operator prescribes,
    which on a machine whose response carrier also holds passive structure
    means asking a passive ring to carry a compensating current.
    """
    pairs = tuple(pairs)
    response = np.asarray(
        constraint_response_matrix(
            profile,
            pairs,
            flux,
            requested_class=requested_class,
            target_current=target_current,
        ),
        dtype=float,
    )
    row_slices = constraint_row_slices(pairs)
    scales = np.concatenate(
        tuple(np.ravel(np.asarray(pair.binding.scale, dtype=float)) for pair in pairs)
    )
    selection = select_compensating_directions(
        response / scales[:, None],
        circuits=circuits,
        rule=rule,
        competition_threshold=competition_threshold,
        participation_floor=participation_floor,
    )
    selection = selection._replace(response=response)
    derived = []
    for pair, row_slice in zip(pairs, row_slices, strict=True):
        columns = selection.directions[:, row_slice]
        authority = selection.direction_authority[row_slice]
        if isinstance(pair.unknown, CircuitCurrentUnknown):
            ampere_scale = jnp.asarray(pair.unknown.ampere_scale)
        else:
            ampere_scale = jnp.asarray(1.0 / authority)
        unknown = CircuitCurrentUnknown(
            direction=jnp.asarray(columns),
            ampere_scale=ampere_scale,
            singular_values=jnp.asarray(selection.singular_values),
            authority=jnp.asarray(authority),
            rule=selection.rule,
        )
        derived.append(
            ConstraintPair(
                functional=pair.functional,
                unknown=unknown,
                binding=pair.binding,
            )
        )
    return tuple(derived), selection


@dataclass(frozen=True)
class CurrentCentroidConstraint:
    """Current-centroid rows on one explicitly declared integration support."""

    components: tuple[Literal["centroid_r", "centroid_z"], ...] = ("centroid_z",)
    support: MomentIntegralSupport = MomentIntegralSupport.ALL_DOMAIN

    def __post_init__(self) -> None:
        if not self.components or any(
            item not in ("centroid_r", "centroid_z") for item in self.components
        ):
            raise ValueError("centroid components must select centroid_r or centroid_z")
        if len(set(self.components)) != len(self.components):
            raise ValueError("centroid components cannot be repeated")
        if not isinstance(self.support, MomentIntegralSupport):
            raise TypeError("centroid support must be a MomentIntegralSupport")

    @property
    def row_count(self) -> int:
        return len(self.components)

    def observed(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        del payload
        observation = profile.current_moment_observation(
            context.flux,
            support=self.support,
            target_current=context.target_current,
        )
        return jnp.stack(tuple(observation.value(name) for name in self.components))

    def residual(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        unknown: jax.Array,
        payload: object,
        target: jax.Array,
        scale: jax.Array,
    ) -> jax.Array:
        del unknown
        return (self.observed(profile, context, payload) - target) / scale

    def dual_flux_image(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        jacobian = jax.jacrev(
            lambda flux: self.observed(profile, context._replace(flux=flux), payload)
        )(context.flux)
        return jnp.moveaxis(jacobian, 0, -1)


@dataclass(frozen=True)
class FluxLevelConstraint:
    """One absolute flux-level row at each declared point.

    The row reads the total flux the map interpolates at a fixed ``(R, Z)``
    point, so it vanishes exactly where the map carries the commanded level
    there.  A uniform flux offset -- a constant added identically everywhere,
    which carries no poloidal field and therefore no force -- has unit
    leverage on this row.  On a coil-less fixture whose exterior carries no
    source, that offset is the only term that can move the level with the
    position pinned, while the solenoidal field columns contribute nothing at
    a point on their own anchor; together they are what makes the authored
    fixed point reachable.

    The points arrive as the binding payload with shape ``(point_count, 2)``
    in ``(R, Z)``, so moving a point is a new payload rather than a new
    compiled program.

    Two carriers state a flux the row can read.  A structured
    :class:`~nova.equilibrium.conservation.FluxLattice` is read by the cubic
    lattice interpolation every shape row shares.  A cell-carried mesh states
    its per-cell centroids instead and is read through the owning cell's
    own-node quadratic, which the operator evaluates with weights it fit on
    the host when the mesh was built; both reads return the point's flux in
    the state's own unit and neither adds a host callback to the traced read.
    """

    point_count: int

    def __post_init__(self) -> None:
        if int(self.point_count) < 1:
            raise ValueError("a flux-level row set needs at least one point")
        object.__setattr__(self, "point_count", int(self.point_count))

    @property
    def row_count(self) -> int:
        return self.point_count

    def observed(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        supplied = jnp.reshape(jnp.asarray(payload, dtype=jnp.float64), (-1, 2))
        if supplied.shape[0] < self.point_count:
            raise ValueError("a flux-level row needs one (R, Z) point per row")
        points = supplied[: self.point_count]
        if _carries_cell_flux_mesh(profile.lattice):
            return jax.vmap(
                lambda point: _mesh_carried_point_flux(profile, context.flux, point)
            )(points)
        grid = _lattice_grid(profile, context.flux)
        return jax.vmap(
            lambda point: sample_lattice_flux(profile.lattice, grid, point)
        )(points)

    def residual(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        unknown: jax.Array,
        payload: object,
        target: jax.Array,
        scale: jax.Array,
    ) -> jax.Array:
        del unknown
        return (self.observed(profile, context, payload) - target) / scale

    def dual_flux_image(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        return _flux_jacobian_image(self, profile, context, payload)


class WallGapTarget(NamedTuple):
    """Wall point, inward direction, the gap to close, and the flux reference."""

    origin: jax.Array
    direction: jax.Array
    gap: jax.Array
    reference_point: jax.Array | None = None

    def closing_point(self) -> jax.Array:
        """Return the point the boundary must reach to leave this gap."""
        direction = jnp.asarray(self.direction, dtype=jnp.float64)
        norm = jnp.linalg.norm(direction)
        unit = direction / jnp.where(norm > 0.0, norm, 1.0)
        return (
            jnp.asarray(self.origin, dtype=jnp.float64) + jnp.asarray(self.gap) * unit
        )


def _cubic_convolution_weights(fraction: jax.Array) -> jax.Array:
    """Return the four Keys cubic-convolution weights at one cell fraction.

    The kernel reproduces every linear function exactly and is continuously
    differentiable across cell walls, so a row that reads a flux gradient
    sees a gradient that does not jump when a control point crosses a node.
    """
    step = jnp.asarray(fraction)
    return jnp.stack(
        (
            step * (-0.5 + step * (1.0 - 0.5 * step)),
            1.0 + step * step * (-2.5 + 1.5 * step),
            step * (0.5 + step * (2.0 - 1.5 * step)),
            step * step * (-0.5 + 0.5 * step),
        )
    )


def _cell_position(support, step, count: int, value: jax.Array):
    """Return the base node index and in-cell fraction of one coordinate."""
    location = (jnp.asarray(value) - support) / step
    index = jnp.clip(jnp.floor(location), 0.0, float(count - 2))
    return index.astype(jnp.int32), location - index


def sample_lattice_flux(lattice, grid: jax.Array, point: jax.Array) -> jax.Array:
    """Interpolate the flux map at one point of the uniform lattice.

    This is the reader every shape row shares, so a control point, a
    commanded null and a gap-closing point all see one interpolation, and a
    receipt that ray-casts the achieved boundary reads the same surface the
    rows were driven onto.

    The stencil indices are clamped to the lattice, so a point outside it is
    the cubic extension of the edge cell rather than a shape error; a control
    point that has left the grid shows up in the receipt as a row that does
    not close, which is the honest report.
    """
    radial_count, vertical_count = lattice.shape
    offset = jnp.arange(-1, 3)
    radial_index, radial_fraction = _cell_position(
        float(lattice.radius[0]), float(lattice.radial_step), radial_count, point[0]
    )
    vertical_index, vertical_fraction = _cell_position(
        float(lattice.height[0]), float(lattice.vertical_step), vertical_count, point[1]
    )
    radial_nodes = jnp.clip(radial_index + offset, 0, radial_count - 1)
    vertical_nodes = jnp.clip(vertical_index + offset, 0, vertical_count - 1)
    block = grid[radial_nodes[:, None], vertical_nodes[None, :]]
    radial_weight = _cubic_convolution_weights(radial_fraction)
    vertical_weight = _cubic_convolution_weights(vertical_fraction)
    return radial_weight @ block @ vertical_weight


_LATTICE_CARRIER_ATTRIBUTES = (
    "shape",
    "radius",
    "radial_step",
    "height",
    "vertical_step",
)


def _lattice_grid(profile: ForwardProfile, flux: jax.Array) -> jax.Array:
    """Return the plasma-grid block of one flux state in lattice shape."""
    lattice = profile.lattice
    if not all(hasattr(lattice, name) for name in _LATTICE_CARRIER_ATTRIBUTES):
        raise TypeError(
            "a point-sampling row needs a structured FluxLattice carrier -- a "
            "shape, an origin and a step per axis -- or a cell-carried mesh "
            "whose per-cell flux the operator reads through sample_flux_field"
        )
    return jnp.reshape(jnp.asarray(flux)[: lattice.node_count], lattice.shape)


def _carries_cell_flux_mesh(lattice: object) -> bool:
    """Return whether a carrier states a per-cell centroid coordinate."""
    return hasattr(lattice, "coordinate")


def _mesh_carried_point_flux(
    profile: ForwardProfile, flux: jax.Array, point: jax.Array
) -> jax.Array:
    """Read the cell-carried flux at one point through the cell that owns it.

    The operator's point read evaluates each carried cell's own-node quadratic
    at the query the caller supplies *for that cell* and scatters the result
    back to that cell, so one point is placed in the slot of the cell whose
    centroid lies nearest and read from that same slot.  Nearest centroid is
    the mesh's own ownership rule, and a point on a shared edge is read by one
    of the two cells whose polynomials agree there to the fit's accuracy.

    The flux values enter the read linearly, so the read returns the point's
    flux in the state's own unit, and an offset carried identically by every
    cell moves it by exactly that offset.
    """
    operator = profile.operator
    state = jnp.asarray(flux, dtype=jnp.float64)
    centres = jnp.asarray(profile.lattice.coordinate, dtype=state.dtype)
    owner = jnp.argmin(jnp.sum((centres - point[None, :]) ** 2, axis=-1))
    points = jnp.zeros((centres.shape[0], 1, 2), dtype=state.dtype)
    points = points.at[owner, 0].set(jnp.asarray(point, dtype=state.dtype))
    values, _radial, _vertical = operator.sample_flux_field(
        state[: operator.physical_node_number],
        operator.sample_node_flux(state),
        points,
    )
    return values[owner, 0]


IsofluxReference = Literal["boundary", "reference_point"]


def _isoflux_reference(
    profile: ForwardProfile,
    context: ConstraintContext,
    reference: IsofluxReference,
    grid: jax.Array,
    point: jax.Array | None,
) -> jax.Array:
    """Return the flux level the isoflux rows are measured against.

    ``boundary`` reads the topology's own last-closed-surface level, which is
    the physical statement.  ``reference_point`` reads the map at a declared
    point instead, which states the same surface through a quantity the
    interpolation differentiates exactly: where a machine's boundary level
    comes from a polished null whose tangent the read does not carry, the
    declared point keeps the augmented Jacobian honest.
    """
    if reference == "reference_point":
        if point is None:
            raise ValueError("a reference-point isoflux row needs a reference point")
        return sample_lattice_flux(profile.lattice, grid, point)
    if reference != "boundary":
        raise ValueError("an isoflux reference is 'boundary' or 'reference_point'")
    _masks, topology = profile.operator.read(context.flux, context.requested_class)
    return topology.boundary_flux


def _flux_jacobian_image(
    functional: ConstraintFunctional[object],
    profile: ForwardProfile,
    context: ConstraintContext,
    payload: object,
) -> jax.Array:
    """Return the transposed observation Jacobian one multiplier acts through."""
    jacobian = jax.jacrev(
        lambda flux: functional.observed(profile, context._replace(flux=flux), payload)
    )(context.flux)
    return jnp.moveaxis(jacobian, 0, -1)


@dataclass(frozen=True)
class IsofluxConstraint:
    """Boundary control points asked to lie on the last closed flux surface.

    One row per control point holds the interpolated flux there against the
    boundary flux the topology read returns, so the rows vanish exactly when
    the boundary passes through every point.  The points arrive as the
    binding payload with shape ``(point_count, 2)`` in ``(R, Z)``, so steering
    a shape knob is a new payload from
    :class:`~nova.equilibrium.constraint.BoundingBoxTarget` rather than a new
    compiled program.
    """

    point_count: int
    reference: IsofluxReference = "boundary"

    def __post_init__(self) -> None:
        if int(self.point_count) < 1:
            raise ValueError("an isoflux row set needs at least one control point")
        if self.reference not in ("boundary", "reference_point"):
            raise ValueError("an isoflux reference is 'boundary' or 'reference_point'")
        object.__setattr__(self, "point_count", int(self.point_count))

    @property
    def row_count(self) -> int:
        return self.point_count

    def observed(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        supplied = jnp.reshape(jnp.asarray(payload, dtype=jnp.float64), (-1, 2))
        points = supplied[: self.point_count]
        grid = _lattice_grid(profile, context.flux)
        sampled = jax.vmap(
            lambda point: sample_lattice_flux(profile.lattice, grid, point)
        )(points)
        reference = (
            None
            if supplied.shape[0] <= self.point_count
            else supplied[self.point_count]
        )
        return sampled - _isoflux_reference(
            profile, context, self.reference, grid, reference
        )

    def residual(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        unknown: jax.Array,
        payload: object,
        target: jax.Array,
        scale: jax.Array,
    ) -> jax.Array:
        del unknown
        return (self.observed(profile, context, payload) - target) / scale

    def dual_flux_image(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        return _flux_jacobian_image(self, profile, context, payload)


@dataclass(frozen=True)
class XPointConstraint:
    """Both poloidal flux gradients at one commanded null position.

    The rows are ``dpsi/dR`` and ``dpsi/dZ`` at the payload point, taken
    through the same interpolation the isoflux rows read, so they vanish
    exactly when a stationary point of the flux map sits there.  Commanding
    the null is moving the payload point.
    """

    @property
    def row_count(self) -> int:
        return 2

    def observed(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        point = jnp.reshape(jnp.asarray(payload, dtype=jnp.float64), (2,))
        grid = _lattice_grid(profile, context.flux)
        gradient = jax.grad(
            lambda position: sample_lattice_flux(profile.lattice, grid, position)
        )(point)
        return jnp.asarray(gradient)

    def residual(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        unknown: jax.Array,
        payload: object,
        target: jax.Array,
        scale: jax.Array,
    ) -> jax.Array:
        del unknown
        return (self.observed(profile, context, payload) - target) / scale

    def dual_flux_image(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        return _flux_jacobian_image(self, profile, context, payload)


@dataclass(frozen=True)
class FieldComponentConstraint:
    """One poloidal-field component at each payload point.

    Each row reads one field component off the flux map at a fixed point,
    through the same interpolation the isoflux rows read.  Under the
    solver's total-flux convention the poloidal field is
    :math:`B_R = -(1/2\\pi R)\\,\\partial\\Phi/\\partial Z` and
    :math:`B_Z = +(1/2\\pi R)\\,\\partial\\Phi/\\partial R`, so a ``radial``
    row is the vertical flux gradient vanishing at the point — the condition
    that it is a radial turning point of the separatrix, a vertical field
    line through an outer or inner extremum — and a ``vertical`` row is the
    radial flux gradient vanishing, the upper or lower turning-point
    condition.

    ``components`` names the row's component in order; the payload is the
    matching ``(row_count, 2)`` ``(R, Z)`` table, so commanding a point is
    moving a payload row and recompiling nothing.
    """

    components: tuple[Literal["radial", "vertical"], ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("a field row set needs at least one control point")
        if any(item not in ("radial", "vertical") for item in self.components):
            raise ValueError("a field component row is 'radial' or 'vertical'")
        object.__setattr__(self, "components", tuple(self.components))

    @property
    def row_count(self) -> int:
        return len(self.components)

    def observed(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        points = jnp.reshape(jnp.asarray(payload, dtype=jnp.float64), (-1, 2))[
            : self.row_count
        ]
        grid = _lattice_grid(profile, context.flux)
        rows = []
        for point, component in zip(points, self.components, strict=True):
            gradient = jax.grad(
                lambda position: sample_lattice_flux(profile.lattice, grid, position)
            )(point)
            radius = jnp.maximum(point[0], 1.0e-6)
            if component == "radial":
                rows.append(-gradient[1] / (TOTAL_FLUX_FACTOR * radius))
            else:
                rows.append(gradient[0] / (TOTAL_FLUX_FACTOR * radius))
        return jnp.stack(rows)

    def residual(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        unknown: jax.Array,
        payload: object,
        target: jax.Array,
        scale: jax.Array,
    ) -> jax.Array:
        del unknown
        return (self.observed(profile, context, payload) - target) / scale

    def dual_flux_image(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        return _flux_jacobian_image(self, profile, context, payload)


@dataclass(frozen=True)
class WallGapConstraint:
    """One wall clearance expressed as an isoflux row at the closing point.

    The payload is a :class:`WallGapTarget`: a point on the wall, the inward
    direction the clearance is measured along, the gap itself, and optionally
    the point whose flux stands in for the boundary level.  The row is the
    flux at the closing point minus that level, so it vanishes when the
    boundary stands exactly the commanded distance off the wall.
    """

    reference: IsofluxReference = "boundary"

    def __post_init__(self) -> None:
        if self.reference not in ("boundary", "reference_point"):
            raise ValueError("a wall-gap reference is 'boundary' or 'reference_point'")

    @property
    def row_count(self) -> int:
        return 1

    def observed(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        if not isinstance(payload, WallGapTarget):
            raise TypeError("a wall-gap row needs a WallGapTarget payload")
        grid = _lattice_grid(profile, context.flux)
        sampled = sample_lattice_flux(profile.lattice, grid, payload.closing_point())
        reference = (
            None
            if payload.reference_point is None
            else jnp.asarray(payload.reference_point, dtype=jnp.float64)
        )
        return jnp.atleast_1d(
            sampled
            - _isoflux_reference(profile, context, self.reference, grid, reference)
        )

    def residual(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        unknown: jax.Array,
        payload: object,
        target: jax.Array,
        scale: jax.Array,
    ) -> jax.Array:
        del unknown
        return (self.observed(profile, context, payload) - target) / scale

    def dual_flux_image(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        return _flux_jacobian_image(self, profile, context, payload)


@dataclass(frozen=True)
class ExternalShafranovConstraint:
    r"""The Shafranov integral of the external magnetics, as one beta row.

    A large-aspect-ratio current ring is balanced by a vertical field

    .. math::

       B_v = -\frac{\mu_0 I_p}{4\pi R}\left[\ln\left(\frac{8R}{a}\right)
             + \beta_p + \frac{l_i}{2} - \frac{3}{2}\right],

    so the external magnetics alone state the combination
    :math:`\beta_p + l_i/2` the plasma must carry for force balance.  This row
    inverts that identity: its observed value is the combination implied by
    the prescribed external field at the plasma, and a solve that imposes it
    against the values the extracted profiles imply closes the loop between
    the flux-function extraction and the magnetics without fitting either.

    The external field is not a further unknown.  It is the payload: the
    prescribed conductor flux image on the lattice, which the circuits fix
    before the solve begins.  The row differentiates the state through the
    current centroid it is sampled at and the plasma current it is divided
    by, both read from the moment observation on the declared support, so a
    Newton step that moves the plasma changes what the same field implies.

    The payload is ``(external_flux, minor_radius)``: the external flux image
    in the lattice's own node ordering [Wb], and the minor radius [m] of the
    reference boundary the row is stated against.  The minor radius is a
    reference geometry rather than an unknown because this row constrains the
    scale and shape of the profiles, not the position of the plasma.

    The compensating unknown is a profile amplitude: the pressure-gradient or
    ff-prime normalisation whose flux image the source term already carries.
    A pair whose unknown is not a profile amplitude is refused when it is
    built, because then the row states a constraint nothing can move.
    """

    minor_radius: object
    support: MomentIntegralSupport = MomentIntegralSupport.ALL_DOMAIN
    required_unknown: object = ProfileAmplitudeUnknown

    def __post_init__(self) -> None:
        radius = jnp.atleast_1d(jnp.asarray(self.minor_radius))
        object.__setattr__(self, "minor_radius", radius)
        _require_positive_if_concrete(radius, "shafranov reference minor radius")
        if not isinstance(self.support, MomentIntegralSupport):
            raise TypeError("shafranov support must be a MomentIntegralSupport")

    @property
    def row_count(self) -> int:
        return 1

    def _payload(self, payload: object) -> tuple[jax.Array, jax.Array]:
        """Return the external flux image and the reference minor radius."""
        if not isinstance(payload, Sequence) or len(payload) != 2:
            raise ValueError("a shafranov row payload is (external_flux, minor_radius)")
        external_flux, minor_radius = payload
        return jnp.asarray(external_flux), jnp.asarray(minor_radius)

    def observed(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        r"""Return :math:`\beta_p + l_i/2` the external field implies here.

        The field is sampled on the lattice at the plasma's own current
        centroid, from the external flux image the payload carries, through
        the shared cubic reader so the row differentiates exactly the surface
        the shape rows read.  The identity's :math:`\mu_0` is written
        explicitly, as it is everywhere in this package.
        """
        external_flux, minor_radius = self._payload(payload)
        observation = profile.current_moment_observation(
            context.flux,
            support=self.support,
            target_current=context.target_current,
        )
        radius = observation.centroid_r
        height = observation.centroid_z
        lattice = profile.lattice
        grid = jnp.reshape(external_flux[: lattice.node_count], lattice.shape)
        step = lattice.radial_step
        point = jnp.stack((radius, height))
        upper = sample_lattice_flux(lattice, grid, point + jnp.stack((step, 0.0)))
        lower = sample_lattice_flux(lattice, grid, point - jnp.stack((step, 0.0)))
        vertical_field = (upper - lower) / (2.0 * step * TOTAL_FLUX_FACTOR * radius)
        current = observation.plasma_current
        numerator = -2.0 * TOTAL_FLUX_FACTOR * radius * vertical_field
        force_factor = numerator / (MU0 * current)
        combination = force_factor - jnp.log(8.0 * radius / minor_radius) + 1.5
        return jnp.atleast_1d(combination)

    def residual(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        unknown: jax.Array,
        payload: object,
        target: jax.Array,
        scale: jax.Array,
    ) -> jax.Array:
        """Return the implied combination against the extracted target."""
        del unknown
        return (self.observed(profile, context, payload) - target) / scale

    def dual_flux_image(
        self,
        profile: ForwardProfile,
        context: ConstraintContext,
        payload: object,
    ) -> jax.Array:
        """Return the row's direction in flux space, by autodiff of the row."""
        jacobian = jax.jacrev(
            lambda flux: self.observed(profile, context._replace(flux=flux), payload)
        )(context.flux)
        return jnp.moveaxis(jacobian, 0, -1)


class BoundingBoxRowKind(IntEnum):
    """Which physical condition one bounding-box control row applies.

    A row is a point together with one such kind: the boundary-flux rows put
    the boundary through the point, the radial-field rows put :math:`B_R=0`
    there (a radial turning point), the vertical-field rows put
    :math:`B_Z=0` there (a vertical turning point), and the null row at the
    X-point asks both gradients to vanish.
    """

    BOUNDARY_FLUX = 0
    RADIAL_FIELD = 1
    VERTICAL_FIELD = 2


@dataclass(frozen=True)
class BoundingBoxTarget:
    """Payload bundle for the low-DOF bounding-box shape row set.

    The rows are grouped by the functional that owns them: ``flux_points``
    become the boundary-flux (isoflux) rows, ``radial_field_points`` and
    ``vertical_field_points`` become the single-component field rows, and
    ``x_point`` the two-gradient null row.  Every shape figure is an ordinary
    array leaf, so steering a knob is building a new payload and recompiling
    nothing.

    Two constructions exist.  :meth:`from_control_points` reads the
    pulse-design parameter vocabulary (geometric axis, minor radius,
    elongation, the two major triangularities, the outer and inner midplane
    offsets and the quadrant squarenesses) through
    :class:`~nova.geometry.plasmapoints.ControlPoints`, the same generator
    the inverse design exposes as sliders.  :meth:`from_boundary` reads the
    achieved separatrix's own turning points — the outer and inner radial
    extrema and the upper and lower vertical extrema of the traced contour —
    so an unmoved command reproduces the achieved boundary exactly, with no
    curve fit standing between the command and the plasma.
    """

    flux_points: object
    radial_field_points: object
    vertical_field_points: object
    x_point: object | None = None
    reference_point: object | None = None

    @classmethod
    def from_boundary(
        cls,
        boundary: object,
        *,
        x_point: object | None = None,
        reference_point: object | None = None,
    ) -> "BoundingBoxTarget":
        """Read the control rows from one traced separatrix's own extrema.

        ``boundary`` is the ``(point_count, 2)`` ``(R, Z)`` contour whose own
        turning points become the rows, so the unmoved command puts the
        boundary through exactly the points it already passes.
        """
        contour = jnp.asarray(boundary, dtype=jnp.float64)
        radius, height = contour[:, 0], contour[:, 1]
        outer = contour[int(jnp.argmax(radius))]
        inner = contour[int(jnp.argmin(radius))]
        upper = contour[int(jnp.argmax(height))]
        lower = contour[int(jnp.argmin(height))]
        return cls(
            flux_points=jnp.stack((outer, upper, inner, lower)),
            radial_field_points=jnp.stack((outer, inner)),
            vertical_field_points=jnp.stack((upper, lower)),
            x_point=x_point,
            reference_point=reference_point,
        )

    @classmethod
    def from_control_points(
        cls,
        geometric_axis: object,
        minor_radius: float,
        elongation: float,
        triangularity_upper: float = 0.0,
        triangularity_lower: float = 0.0,
        triangularity_outer: float = 0.0,
        triangularity_inner: float = 0.0,
        squareness: object = 0.0,
        *,
        square: bool = False,
        x_point: object | None = None,
        reference_point: object | None = None,
    ) -> "BoundingBoxTarget":
        """Build the control rows from the pulse-design parameter vocabulary.

        The four principal points come from the geometric axis, minor radius,
        elongation and the two major triangularities; ``triangularity_outer``
        and ``triangularity_inner`` are the midplane offsets of the outer and
        inner points as fractions of the minor radius.  ``squareness`` is one
        value applied to all four quadrants or four values in
        ``(upper_outer, upper_inner, lower_inner, lower_outer)`` order; with
        ``square`` true the four quadrant points are added to the flux rows.
        ControlPoints is imported lazily because it carries the graphics and
        wall stack, which the constraint interface should not pull in.
        """
        import xarray

        from nova.geometry.plasmapoints import ControlPoints

        values = np.asarray(squareness, dtype=float)
        if values.size == 1:
            values = np.repeat(values, 4)
        if values.shape != (4,):
            raise ValueError(
                "squareness is one value or four in "
                "(upper_outer, upper_inner, lower_inner, lower_outer)"
            )
        axes = np.atleast_1d(np.asarray(geometric_axis, dtype=float))
        if axes.shape != (2,):
            raise ValueError("the geometric axis is one (R, Z) pair")
        data = xarray.Dataset(
            {
                "geometric_axis": ("point", axes.tolist()),
                "minor_radius": float(minor_radius),
                "elongation": float(elongation),
                "triangularity_upper": float(triangularity_upper),
                "triangularity_lower": float(triangularity_lower),
                "elongation_upper": float(triangularity_outer),
                "elongation_lower": float(triangularity_inner),
                "squareness_upper_outer": float(values[0]),
                "squareness_upper_inner": float(values[1]),
                "squareness_lower_inner": float(values[2]),
                "squareness_lower_outer": float(values[3]),
            }
        )
        control = ControlPoints(data, square=square)
        principal = np.stack(
            (
                np.asarray(control.outer),
                np.asarray(control.upper),
                np.asarray(control.inner),
                np.asarray(control.lower),
            )
        )
        radial = np.stack((np.asarray(control.outer), np.asarray(control.inner)))
        vertical = np.stack((np.asarray(control.upper), np.asarray(control.lower)))
        flux_points = principal
        if square:
            quadrant = np.stack(
                (
                    np.asarray(control.upper_outer),
                    np.asarray(control.upper_inner),
                    np.asarray(control.lower_inner),
                    np.asarray(control.lower_outer),
                )
            )
            flux_points = np.concatenate((principal, quadrant))
        return cls(
            flux_points=jnp.asarray(flux_points, dtype=jnp.float64),
            radial_field_points=jnp.asarray(radial, dtype=jnp.float64),
            vertical_field_points=jnp.asarray(vertical, dtype=jnp.float64),
            x_point=x_point,
            reference_point=reference_point,
        )

    @property
    def rows(self) -> list[tuple[BoundingBoxRowKind, object]]:
        """Return every control row as its ``(kind, point)`` pair."""

        def pairs(kind, points):
            return [(kind, point) for point in np.asarray(points)]

        return [
            *pairs(BoundingBoxRowKind.BOUNDARY_FLUX, self.flux_points),
            *pairs(BoundingBoxRowKind.RADIAL_FIELD, self.radial_field_points),
            *pairs(BoundingBoxRowKind.VERTICAL_FIELD, self.vertical_field_points),
        ]


class ConstraintRecord(NamedTuple):
    """Terminal physical and numerical values of one registered pair."""

    observed: jax.Array
    target: jax.Array
    physical_residual: jax.Array
    scaled_residual: jax.Array
    tolerance: jax.Array
    qualified: jax.Array
    normalized_unknown: jax.Array
    physical_unknown: jax.Array
    soft_mode_projection: jax.Array
    compensator_rule: jax.Array | None = None
    bound_refusal: jax.Array | None = None
    compensator_direction: jax.Array | None = None
    compensator_singular_values: jax.Array | None = None
    compensator_authority: jax.Array | None = None


@dataclass(frozen=True)
class AugmentedConstraintSystem:
    """Private fixed-shape map and callbacks assembled for one constraint tuple."""

    initial: jax.Array
    map_fn: Callable[[jax.Array], jax.Array]
    shadow_mask_fn: Callable[[jax.Array], jax.Array]
    promoted_shadow_mask_fn: Callable[[jax.Array, jax.Array], jax.Array]
    shadowed_map_fn: Callable[[jax.Array, jax.Array], jax.Array]
    row_jvp_observers: tuple[Callable[[jax.Array, jax.Array], jax.Array], ...]
    row_slices: tuple[slice, ...]
    flux_scale: jax.Array
    flux_size: int

    def split(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return physical flux and normalized compensator vectors."""
        return state[: self.flux_size] * self.flux_scale, state[self.flux_size :]


def assemble_augmented_system(
    profile: ForwardProfile,
    initial_flux: jax.Array,
    pairs: tuple[ConstraintPair, ...],
    *,
    base_map: Callable[[jax.Array], jax.Array],
    base_shadow_mask: Callable[[jax.Array], jax.Array],
    base_promoted_shadow_mask: Callable[[jax.Array, jax.Array], jax.Array],
    base_shadowed_map: Callable[[jax.Array, jax.Array], jax.Array],
    requested_class: jax.Array | None,
    target_current: jax.Array | None,
) -> AugmentedConstraintSystem:
    """Assemble normalized flux and compensator blocks once per tuple layout."""
    if not pairs:
        raise ValueError("an augmented system needs at least one constraint pair")
    seed = jnp.asarray(initial_flux)
    flux_size = seed.size
    row_slices = constraint_row_slices(pairs)
    row_count = row_slices[-1].stop
    flux_scale = jnp.maximum(jnp.max(jnp.abs(seed)), jnp.finfo(seed.dtype).tiny)
    initial_unknown = jnp.concatenate(
        tuple(jnp.ravel(jnp.asarray(pair.binding.initial_unknown)) for pair in pairs)
    )
    initial = jnp.concatenate((jnp.ravel(seed) / flux_scale, initial_unknown))

    def split(state):
        return state[:flux_size] * flux_scale, state[flux_size:]

    def evaluate(state, shadow=None):
        flux, unknowns = split(state)
        flux_shadow = (
            jnp.ravel(base_shadow_mask(flux))
            if shadow is None
            else jnp.ravel(shadow[:flux_size])
        )
        mapped_flux = (
            base_map(flux) if shadow is None else base_shadowed_map(flux, flux_shadow)
        )
        context = ConstraintContext(flux, requested_class, target_current, flux_shadow)
        steps = []
        for pair, row_slice in zip(pairs, row_slices, strict=True):
            value = unknowns[row_slice]
            delta = pair.unknown.flux_delta(
                profile, context, pair.functional, pair.binding.payload, value
            )
            mapped_flux = mapped_flux + jnp.where(flux_shadow, 0.0, delta)
            row = jnp.ravel(
                pair.functional.residual(
                    profile,
                    context,
                    value,
                    pair.binding.payload,
                    jnp.asarray(pair.binding.target),
                    jnp.asarray(pair.binding.scale),
                )
            )
            step, _refused = compensator_step(pair.unknown, value, row)
            steps.append(step)
        unknowns_next = unknowns + jnp.concatenate(tuple(steps))
        return jnp.concatenate((mapped_flux / flux_scale, unknowns_next))

    def shadow_mask(state):
        flux, _unknowns = split(state)
        return jnp.concatenate(
            (jnp.ravel(base_shadow_mask(flux)), jnp.zeros(row_count, dtype=bool))
        )

    def promoted_shadow_mask(state, previous):
        flux, _unknowns = split(state)
        flux_shadow = base_promoted_shadow_mask(flux, previous[:flux_size])
        return jnp.concatenate(
            (jnp.ravel(flux_shadow), jnp.zeros(row_count, dtype=bool))
        )

    def shadowed_map(state, shadow):
        return evaluate(state, shadow)

    observers = []
    for pair, row_slice in zip(pairs, row_slices, strict=True):

        def observe_row(state, direction, pair=pair, row_slice=row_slice):
            flux, unknowns = split(state)
            flux_tangent = direction[:flux_size] * flux_scale
            context = ConstraintContext(flux, requested_class, target_current, None)
            return constraint_residual_jvp(
                pair,
                profile,
                context,
                unknowns[row_slice],
                flux_tangent,
                direction[flux_size:][row_slice],
            )

        observers.append(observe_row)

    return AugmentedConstraintSystem(
        initial=initial,
        map_fn=lambda state: evaluate(state),
        shadow_mask_fn=shadow_mask,
        promoted_shadow_mask_fn=promoted_shadow_mask,
        shadowed_map_fn=shadowed_map,
        row_jvp_observers=tuple(observers),
        row_slices=row_slices,
        flux_scale=flux_scale,
        flux_size=flux_size,
    )


def constraint_records(
    profile: ForwardProfile,
    system: AugmentedConstraintSystem,
    state: jax.Array,
    pairs: tuple[ConstraintPair, ...],
    projections: jax.Array,
    *,
    requested_class: jax.Array | None,
    target_current: jax.Array | None,
) -> tuple[ConstraintRecord, ...]:
    """Evaluate terminal row qualification and physical compensator values."""
    flux, unknowns = system.split(state)
    context = ConstraintContext(flux, requested_class, target_current, None)
    records = []
    for pair, row_slice in zip(pairs, system.row_slices, strict=True):
        binding = pair.binding
        value = unknowns[row_slice]
        observed = jnp.atleast_1d(
            pair.functional.observed(profile, context, binding.payload)
        )
        target = jnp.atleast_1d(jnp.asarray(binding.target))
        physical_residual = observed - target
        scale = jnp.atleast_1d(jnp.asarray(binding.scale))
        tolerance = jnp.atleast_1d(jnp.asarray(binding.tolerance))
        circuit = (
            pair.unknown if isinstance(pair.unknown, CircuitCurrentUnknown) else None
        )
        records.append(
            ConstraintRecord(
                observed=observed,
                target=target,
                physical_residual=physical_residual,
                scaled_residual=physical_residual / scale,
                tolerance=tolerance,
                qualified=jnp.abs(physical_residual) <= tolerance,
                normalized_unknown=value,
                physical_unknown=pair.unknown.physical_value(value),
                soft_mode_projection=jnp.asarray(projections)[row_slice],
                compensator_rule=jnp.full(
                    observed.shape,
                    int(CompensatorRule.EXPLICIT if circuit is None else circuit.rule),
                    dtype=jnp.int8,
                ),
                bound_refusal=compensator_bound_refusal(pair.unknown, value),
                compensator_direction=(
                    None if circuit is None else jnp.asarray(circuit.direction)
                ),
                compensator_singular_values=(
                    None
                    if circuit is None or circuit.singular_values is None
                    else jnp.asarray(circuit.singular_values)
                ),
                compensator_authority=(
                    None
                    if circuit is None or circuit.authority is None
                    else jnp.asarray(circuit.authority)
                ),
            )
        )
    return tuple(records)


__all__ = [
    "AugmentedConstraintSystem",
    "BoundedExteriorFieldUnknown",
    "BoundingBoxRowKind",
    "BoundingBoxTarget",
    "CircuitCurrentUnknown",
    "CompensatingUnknown",
    "CompensatorRule",
    "CompensatorSelection",
    "ConstraintBinding",
    "ConstraintContext",
    "ConstraintElimination",
    "ConstraintFunctional",
    "ConstraintMultiplier",
    "ConstraintOuterStep",
    "ConstraintOuterTrace",
    "ConstraintPair",
    "ConstraintPolicy",
    "ConstraintRecord",
    "CurrentCentroidConstraint",
    "ExternalShafranovConstraint",
    "FieldComponentConstraint",
    "IsofluxConstraint",
    "IsofluxReference",
    "ProfileAmplitudeUnknown",
    "WallGapConstraint",
    "WallGapTarget",
    "XPointConstraint",
    "assemble_augmented_system",
    "compensator_bound_refusal",
    "compensator_rule_name",
    "compensator_step",
    "constraint_records",
    "constraint_residual_jvp",
    "constraint_response_matrix",
    "constraint_row_slices",
    "derive_circuit_compensators",
    "sample_lattice_flux",
    "select_compensating_directions",
]
