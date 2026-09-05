"""Typed data interface for constraint-augmented forward equilibria.

Constraint kinds, compensating unknowns, and per-solve bindings are separate
objects.  Their Python types and tuple positions define a static solver layout;
targets, tolerances, scales, initial values, and payloads remain ordinary JAX
leaves that can be traced and mapped over.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.observation import MomentIntegralSupport

if TYPE_CHECKING:
    from nova.equilibrium.forward import ForwardProfile


Payload = TypeVar("Payload")
ConstraintPolicy = Literal["imposed", "eliminated"]


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
    """Physical circuit-current columns selected by dimensionless directions."""

    direction: object
    ampere_scale: object

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
        return ((self.direction, self.ampere_scale), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


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
            )
        )
    return tuple(records)


__all__ = [
    "AugmentedConstraintSystem",
    "CircuitCurrentUnknown",
    "CompensatingUnknown",
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
    "constraint_records",
    "constraint_residual_jvp",
    "constraint_row_slices",
]
