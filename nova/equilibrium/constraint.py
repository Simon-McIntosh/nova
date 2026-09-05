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

from nova.equilibrium.observation import MomentIntegralSupport

if TYPE_CHECKING:
    from nova.equilibrium.forward import ForwardProfile


Payload = TypeVar("Payload")
ConstraintPolicy = Literal["imposed", "eliminated"]


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

    def __post_init__(self) -> None:
        """Require a known solve policy; row-shape validation belongs to the pair."""
        if self.policy not in ("imposed", "eliminated"):
            raise ValueError("constraint policy must be 'imposed' or 'eliminated'")

    def tree_flatten(self):
        """Keep policy static while all numerical binding values remain leaves."""
        return (
            (
                self.target,
                self.tolerance,
                self.scale,
                self.initial_unknown,
                self.payload,
            ),
            (self.policy,),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild a binding from its static policy and traced leaves."""
        (policy,) = aux_data
        target, tolerance, scale, initial_unknown, payload = children
        return cls(target, tolerance, scale, initial_unknown, payload, policy)


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
    """

    rule: CompensatorRule
    response: np.ndarray
    authority: np.ndarray
    singular_values: np.ndarray
    directions: np.ndarray
    direction_authority: np.ndarray
    row_coupling: np.ndarray
    competing: bool

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
    rows = authority.shape[0]
    norms = np.linalg.norm(authority, axis=1)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(authority)):
        raise ValueError("every constraint row needs finite non-zero circuit authority")
    unit = authority / norms[:, None]
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
    singular_values = np.linalg.svd(authority, compute_uv=False)
    if rule is CompensatorRule.DOMINANT_AUTHORITY:
        columns = authority.T
    elif rule is CompensatorRule.SINGULAR_DISTRIBUTION:
        columns = np.linalg.pinv(authority)
    else:
        raise ValueError("a derived direction needs an authority or singular rule")
    columns = np.asarray(columns, dtype=float)
    if participation_floor > 0.0:
        peak = np.max(np.abs(columns), axis=0, keepdims=True)
        columns = np.where(
            np.abs(columns) >= float(participation_floor) * peak, columns, 0.0
        )
    directions = _infinity_normalised(columns)
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
    )


def derive_circuit_compensators(
    profile: ForwardProfile,
    pairs: Sequence[ConstraintPair],
    flux: jax.Array,
    *,
    requested_class: jax.Array | None = None,
    target_current: jax.Array | None = None,
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
        residuals = []
        for pair, row_slice in zip(pairs, row_slices, strict=True):
            value = unknowns[row_slice]
            delta = pair.unknown.flux_delta(
                profile, context, pair.functional, pair.binding.payload, value
            )
            mapped_flux = mapped_flux + jnp.where(flux_shadow, 0.0, delta)
            residuals.append(
                jnp.ravel(
                    pair.functional.residual(
                        profile,
                        context,
                        value,
                        pair.binding.payload,
                        jnp.asarray(pair.binding.target),
                        jnp.asarray(pair.binding.scale),
                    )
                )
            )
        rows = jnp.concatenate(tuple(residuals))
        return jnp.concatenate((mapped_flux / flux_scale, unknowns - rows))

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
    "CircuitCurrentUnknown",
    "CompensatingUnknown",
    "CompensatorRule",
    "CompensatorSelection",
    "ConstraintBinding",
    "ConstraintContext",
    "ConstraintFunctional",
    "ConstraintMultiplier",
    "ConstraintPair",
    "ConstraintPolicy",
    "ConstraintRecord",
    "CurrentCentroidConstraint",
    "ProfileAmplitudeUnknown",
    "assemble_augmented_system",
    "compensator_rule_name",
    "constraint_records",
    "constraint_response_matrix",
    "constraint_residual_jvp",
    "constraint_row_slices",
    "derive_circuit_compensators",
    "select_compensating_directions",
]
