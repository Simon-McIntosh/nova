"""Device-batched forward labels for a fixed reduced equilibrium program.

The operator and its reduced coordinates are static for one labelling run.
Slice-specific currents, targets, classes and warm states stay traced leaves
of one vmapped fixed-shape program.  The result is deliberately a fixed-shape
record so a caller can pad a final batch without changing the compiled graph.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import reduced_newton
from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    CompensatorRule,
    ConstraintBinding,
    ConstraintMultiplier,
    ConstraintPair,
    CurrentCentroidConstraint,
    constraint_response_matrix,
)
from nova.equilibrium.forward import ForwardLabelledFlux
from nova.equilibrium.observation import MomentIntegralSupport

__all__ = [
    "BatchedLabelledResult",
    "BatchedLabeller",
    "CENTROID_REPORTING_QUANTUM",
    "solve_batched_labeller",
]


CENTROID_REPORTING_QUANTUM = 1.0e-12


def _reported_centroid(value: jax.Array) -> jax.Array:
    """Snap reported coordinates below physically meaningful precision."""
    quantum = jnp.asarray(CENTROID_REPORTING_QUANTUM, dtype=value.dtype)
    return jnp.rint(value / quantum) * quantum


class BatchedLabelledResult(NamedTuple):
    """One fixed-shape device result for every requested slice."""

    state: jax.Array
    converged: jax.Array
    termination: jax.Array
    trips: jax.Array
    terminal_residual: jax.Array
    achieved_centroid: jax.Array
    conditioned: jax.Array
    guard: jax.Array
    labelled_flux: ForwardLabelledFlux

    @property
    def termination_reason(self) -> jax.Array:
        """Alias used by the scalar compiled-route receipt."""
        return self.termination

    @property
    def labelled(self) -> ForwardLabelledFlux:
        """Return the consumer-facing topology fields."""
        return self.labelled_flux


def _call_kernel(
    kernel,
    reduced,
    shadow,
    base_state,
    external,
    target,
    requested,
    rows,
    *,
    has_target: bool,
    has_requested: bool,
    has_rows: bool,
):
    """Call one raw reduced kernel with only the dynamic leaves it owns."""
    arguments = (reduced, shadow, base_state)
    if kernel.__name__ == "newton_direction":
        return kernel(*arguments[:2])
    if kernel.__name__ == "initial_gather":
        if has_target and has_requested:
            return kernel(
                base_state,
                external_value=external,
                target_value=target,
                requested_value=requested,
            )
        if has_target:
            return kernel(base_state, external_value=external, target_value=target)
        if has_requested:
            return kernel(
                base_state, external_value=external, requested_value=requested
            )
        return kernel(base_state, external_value=external)
    keywords = {"external_value": external}
    if has_target:
        keywords["target_value"] = target
    if has_requested:
        keywords["requested_value"] = requested
    if has_rows:
        keywords["rows"] = rows
    return kernel(*arguments, **keywords)


def _make_solver(
    kernels: dict[str, Any],
    *,
    tolerance: float,
    newton_steps: int,
    active_set_steps: int,
    initial_unknown: jax.Array | None,
    has_target: bool,
    has_requested: bool,
    has_rows: bool,
):
    """Build a dynamic-argument copy of the production fixed trip loop."""

    factors = jnp.asarray(reduced_newton._BACKTRACKING_FACTORS, dtype=jnp.float64)
    converged_code = int(reduced_newton.FixedPointTerminationReason.CONVERGED)
    settled_code = int(reduced_newton.FixedPointTerminationReason.ACTIVE_SET_SETTLED)
    exhausted_code = int(
        reduced_newton.FixedPointTerminationReason.ACTIVE_SET_ITERATION_BUDGET_EXHAUSTED
    )

    def choose(
        reduced,
        jacobian,
        shadow,
        base_state,
        merit,
        external,
        target,
        requested,
        rows,
    ):
        scores = _call_kernel(
            kernels["step_scores"],
            reduced,
            shadow,
            base_state,
            external,
            target,
            requested,
            rows,
            has_target=has_target,
            has_requested=has_requested,
            has_rows=has_rows,
        )
        direction = kernels["direction"](jacobian, scores.residual)
        candidates = reduced[None, :] + factors[:, None] * direction[None, :]

        def score(candidate):
            return _call_kernel(
                kernels["step_scores"],
                candidate,
                shadow,
                base_state,
                external,
                target,
                requested,
                rows,
                has_target=has_target,
                has_requested=has_requested,
                has_rows=has_rows,
            )

        scored = jax.lax.map(score, candidates)
        valid = jnp.isfinite(scored.merit) & (scored.merit < merit)
        accepted = jnp.argmax(valid.astype(jnp.int32))
        return jnp.any(valid), accepted, candidates[accepted]

    def trip_body(reduced, shadow, base_state, external, target, requested, rows):
        jacobian = _call_kernel(
            kernels["jacobian"],
            reduced,
            shadow,
            base_state,
            external,
            target,
            requested,
            rows,
            has_target=has_target,
            has_requested=has_requested,
            has_rows=has_rows,
        )

        def step_body(_index, carry):
            reduced, jacobian, active, step_count, builds, rejected = carry

            def run_step(carry):
                reduced, jacobian, _active, step_count, builds, rejected = carry
                scores = _call_kernel(
                    kernels["step_scores"],
                    reduced,
                    shadow,
                    base_state,
                    external,
                    target,
                    requested,
                    rows,
                    has_target=has_target,
                    has_requested=has_requested,
                    has_rows=has_rows,
                )
                finished = jnp.isfinite(scores.flux_residual) & (
                    scores.flux_residual <= tolerance
                )

                def no_step(_):
                    return reduced, jacobian, False, step_count, builds, rejected

                def try_step(_):
                    first = choose(
                        reduced,
                        jacobian,
                        shadow,
                        base_state,
                        scores.merit,
                        external,
                        target,
                        requested,
                        rows,
                    )

                    def refresh(_):
                        refreshed = _call_kernel(
                            kernels["jacobian"],
                            reduced,
                            shadow,
                            base_state,
                            external,
                            target,
                            requested,
                            rows,
                            has_target=has_target,
                            has_requested=has_requested,
                            has_rows=has_rows,
                        )
                        selected = choose(
                            reduced,
                            refreshed,
                            shadow,
                            base_state,
                            scores.merit,
                            external,
                            target,
                            requested,
                            rows,
                        )
                        return selected, refreshed, 1

                    selected, selected_jacobian, refreshes = jax.lax.cond(
                        ~first[0], refresh, lambda _: (first, jacobian, 0), None
                    )
                    found, _accepted, candidate = selected
                    return (
                        jnp.where(found, candidate, reduced),
                        selected_jacobian,
                        found,
                        step_count + found.astype(jnp.int32),
                        builds + refreshes,
                        rejected + (~found).astype(jnp.int32),
                    )

                return jax.lax.cond(finished, no_step, try_step, None)

            return jax.lax.cond(active, run_step, lambda value: value, carry)

        return jax.lax.fori_loop(
            0,
            newton_steps,
            step_body,
            (
                reduced,
                jacobian,
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )

    def solve(initial, shadow, external, target, requested, rows):
        reduced = _call_kernel(
            kernels["initial_gather"],
            None,
            None,
            initial,
            external,
            target,
            requested,
            rows,
            has_target=has_target,
            has_requested=has_requested,
            has_rows=has_rows,
        )
        if initial_unknown is not None:
            reduced = jnp.concatenate((reduced, initial_unknown))
        state = initial
        active = jnp.asarray(True)
        converged = jnp.asarray(False)
        reason = jnp.asarray(exhausted_code, dtype=jnp.int32)
        terminal_residual = jnp.asarray(jnp.inf, dtype=initial.dtype)
        iterations = jnp.asarray(0, dtype=jnp.int32)
        converged_trip = jnp.asarray(-1, dtype=jnp.int32)
        leakage = jnp.asarray(0.0, dtype=initial.dtype)
        residuals = jnp.full((active_set_steps,), jnp.nan, dtype=initial.dtype)
        differences = jnp.full((active_set_steps,), -1, dtype=jnp.int32)
        trip_steps = jnp.zeros((active_set_steps,), dtype=jnp.int32)
        trip_builds = jnp.zeros((active_set_steps,), dtype=jnp.int32)
        trip_rejected = jnp.zeros((active_set_steps,), dtype=jnp.int32)
        trip_maps = jnp.zeros((active_set_steps,), dtype=jnp.int32)

        def outer_body(index, carry):
            (
                state,
                reduced,
                shadow,
                active,
                converged,
                reason,
                terminal_residual,
                iterations,
                converged_trip,
                leakage,
                residuals,
                differences,
                trip_steps,
                trip_builds,
                trip_rejected,
                trip_maps,
            ) = carry

            def run_trip(_):
                (
                    solved_reduced,
                    _jacobian,
                    step_count,
                    _trip_active,
                    builds,
                    rejected,
                ) = trip_body(reduced, shadow, state, external, target, requested, rows)
                closed = _call_kernel(
                    kernels["boundary"],
                    solved_reduced,
                    shadow,
                    state,
                    external,
                    target,
                    requested,
                    rows,
                    has_target=has_target,
                    has_requested=has_requested,
                    has_rows=has_rows,
                )
                next_state, promoted, difference, observed, next_reduced, excluded = (
                    closed
                )
                converged_now = (
                    jnp.isfinite(observed) & (observed <= tolerance) & (difference == 0)
                )
                settled = difference == 0
                still_active = ~converged_now & ~settled
                next_reason = jnp.where(
                    converged_now,
                    converged_code,
                    jnp.where(settled, settled_code, reason),
                )
                return (
                    next_state,
                    next_reduced,
                    promoted,
                    still_active,
                    converged | converged_now,
                    next_reason,
                    observed,
                    iterations + 1,
                    jnp.where(converged_now, index, converged_trip),
                    jnp.maximum(leakage, excluded),
                    residuals.at[index].set(observed),
                    differences.at[index].set(difference),
                    trip_steps.at[index].set(step_count),
                    trip_builds.at[index].set(builds),
                    trip_rejected.at[index].set(rejected),
                    trip_maps.at[index].set(step_count),
                )

            return jax.lax.cond(active, run_trip, lambda value: value, carry)

        return jax.lax.fori_loop(
            0,
            active_set_steps,
            outer_body,
            (
                state,
                reduced,
                shadow,
                active,
                converged,
                reason,
                terminal_residual,
                iterations,
                converged_trip,
                leakage,
                residuals,
                differences,
                trip_steps,
                trip_builds,
                trip_rejected,
                trip_maps,
            ),
        )

    return solve


def _centroid_pair(
    profile,
    state,
    target,
    *,
    requested_class=None,
    target_current=None,
):
    """Build one slice's traced response direction for centroid conditioning."""
    scale = float(np.ptp(np.asarray(profile.lattice.height)))
    seed = ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_z",),
            support=MomentIntegralSupport.ALL_DOMAIN,
        ),
        unknown=ConstraintMultiplier(multiplier_scale=jnp.asarray([1.0])),
        binding=ConstraintBinding(
            target=jnp.asarray([target]),
            tolerance=jnp.asarray([1.0e-6]),
            scale=jnp.asarray([scale]),
            initial_unknown=jnp.asarray([0.0]),
        ),
    )
    response = constraint_response_matrix(
        profile,
        (seed,),
        jnp.asarray(state),
        requested_class=requested_class,
        target_current=target_current,
    )[0]
    authority = response / jnp.asarray(scale, dtype=response.dtype)
    peak = jnp.max(jnp.abs(authority))
    direction = authority / jnp.where(peak > 0.0, peak, 1.0)
    direction_authority = jnp.dot(authority, direction)
    unknown = CircuitCurrentUnknown(
        direction=direction[:, None],
        ampere_scale=jnp.asarray([1.0 / direction_authority]),
        singular_values=jnp.asarray([jnp.linalg.norm(authority)]),
        authority=jnp.asarray([direction_authority]),
        rule=CompensatorRule.DOMINANT_AUTHORITY,
    )
    return dataclasses.replace(seed, unknown=unknown)


class BatchedLabeller:
    """Compile and execute a fixed-shape labelling program."""

    def __init__(
        self,
        profile,
        *,
        tolerance: float = reduced_newton.FIXED_POINT_RESIDUAL_TOLERANCE,
        newton_steps: int = reduced_newton.NEWTON_STEPS,
        active_set_steps: int = reduced_newton.ACTIVE_SET_STEPS,
        guard_tolerance: float = 5.0e-2,
        constraint_pairs: tuple[ConstraintPair, ...] = (),
    ):
        self.profile = profile
        self.operator = profile.operator
        self.tolerance = tolerance
        self.newton_steps = newton_steps
        self.active_set_steps = active_set_steps
        self.guard_tolerance = guard_tolerance
        self.constraint_pairs = tuple(constraint_pairs)
        self._derive_centroid_pairs = not self.constraint_pairs
        self._compiled = None

    def _build(
        self, initial, *, target_current, requested_class, current, prescribed_current
    ):
        state = jnp.asarray(initial[0])
        target = None if target_current is None else jnp.asarray(target_current[0])
        requested = None if requested_class is None else jnp.asarray(requested_class[0])
        current_value = None if current is None else jnp.asarray(current[0])
        prescribed_value = (
            None if prescribed_current is None else jnp.asarray(prescribed_current[0])
        )
        external = self.operator.external(current_value, prescribed_value)
        augmentation = None
        if self.constraint_pairs:
            augmentation = reduced_newton._row_augmentation(
                self.profile,
                self.constraint_pairs,
                state,
                requested_class=requested,
                target_current=target,
            )
        program, raw_kernels = reduced_newton._compiled_program(
            self.operator,
            state,
            requested_class=requested,
            target_current=target,
            external=external,
            program=None,
            augmentation=augmentation,
        )
        del program
        free_solver = _make_solver(
            raw_kernels
            if augmentation is None
            else reduced_newton._reduced_kernels(
                self.operator,
                reduced_newton.reduced_coordinates(
                    self.operator,
                    state,
                    requested_class=requested,
                    target_current=target,
                ),
                external,
                requested,
                target,
                augmentation=None,
            ),
            tolerance=self.tolerance,
            newton_steps=self.newton_steps,
            active_set_steps=self.active_set_steps,
            initial_unknown=None,
            has_target=target is not None,
            has_requested=requested is not None,
            has_rows=False,
        )
        conditioned_solver = None
        if augmentation is not None:
            initial_unknown = jnp.concatenate(
                tuple(
                    jnp.ravel(jnp.asarray(pair.binding.initial_unknown))
                    for pair in self.constraint_pairs
                )
            )
            conditioned_solver = _make_solver(
                raw_kernels,
                tolerance=self.tolerance,
                newton_steps=self.newton_steps,
                active_set_steps=self.active_set_steps,
                initial_unknown=initial_unknown,
                has_target=target is not None,
                has_requested=requested is not None,
                has_rows=True,
            )

        def one(
            initial_value,
            external_value,
            target_value,
            requested_value,
            reference,
            active,
            condition_target,
        ):
            shadow = jnp.ravel(
                jnp.asarray(
                    self.operator.residual_shadow_mask(initial_value, requested_value),
                    dtype=bool,
                )
            )
            free = free_solver(
                initial_value,
                shadow,
                external_value,
                target_value,
                requested_value,
                None,
            )
            if initial_unknown is not None:
                free = (
                    free[0],
                    jnp.concatenate((free[1], jnp.zeros_like(initial_unknown))),
                    *free[2:],
                )
            free_state = free[0]
            free_centroid = self.profile.current_moment_observation(
                free_state,
                support=MomentIntegralSupport.ALL_DOMAIN,
                target_current=target_value,
            ).stack()[1:]
            has_reference = jnp.all(jnp.isfinite(reference))
            guard = jnp.logical_not(has_reference) | (
                jnp.linalg.norm(free_centroid - reference) <= self.guard_tolerance
            )
            needs = active & (jnp.logical_not(free[4]) | jnp.logical_not(guard))

            def conditioned(_):
                if conditioned_solver is None:
                    return free, jnp.asarray(True)
                if self._derive_centroid_pairs:
                    rows = (
                        _centroid_pair(
                            self.profile,
                            initial_value,
                            condition_target[0],
                            requested_class=requested_value,
                            target_current=target_value,
                        ),
                    )
                else:
                    rows = tuple(
                        dataclasses.replace(
                            pair,
                            binding=dataclasses.replace(
                                pair.binding,
                                target=jnp.asarray(condition_target),
                            ),
                        )
                        for pair in self.constraint_pairs
                    )
                row_arguments = reduced_newton._RowArguments(
                    rows,
                    jnp.maximum(
                        jnp.max(jnp.abs(free_state)), jnp.finfo(free_state.dtype).tiny
                    ),
                    target_value,
                )
                result = conditioned_solver(
                    free_state,
                    jnp.ravel(
                        jnp.asarray(
                            self.operator.residual_shadow_mask(
                                free_state, requested_value
                            ),
                            dtype=bool,
                        )
                    ),
                    external_value,
                    target_value,
                    requested_value,
                    row_arguments,
                )
                return result, jnp.asarray(True)

            solved, conditioned = jax.lax.cond(
                needs, conditioned, lambda _: (free, False), None
            )
            state_value = solved[0]
            masks, topology = self.operator.read(state_value, requested_value)
            labelled = self.profile._labelled_flux(state_value, masks, topology)
            centroid = self.profile.current_moment_observation(
                state_value,
                support=MomentIntegralSupport.ALL_DOMAIN,
                target_current=target_value,
            ).stack()[1:]
            centroid = _reported_centroid(centroid)
            return (
                jnp.where(active, state_value, initial_value),
                jnp.where(active, solved[4], False),
                jnp.where(active, solved[5], -1),
                jnp.where(active, solved[7], 0),
                jnp.where(active, solved[6], jnp.nan),
                jnp.where(active, centroid, jnp.nan),
                jnp.where(active, conditioned, False),
                jnp.where(active, guard, False),
                labelled,
            )

        mapped = jax.vmap(one)
        device_mesh = jax.sharding.Mesh(
            np.asarray(jax.devices(), dtype=object), ("batch",)
        )
        batch_sharding = jax.sharding.NamedSharding(
            device_mesh, jax.sharding.PartitionSpec("batch")
        )
        self._compiled = jax.jit(
            mapped,
            in_shardings=(
                batch_sharding,
                batch_sharding,
                batch_sharding if target is not None else None,
                batch_sharding if requested is not None else None,
                batch_sharding,
                batch_sharding,
                batch_sharding,
            ),
        )
        self._batch_sharding = batch_sharding
        return self._compiled

    def solve(
        self,
        initial,
        *,
        current=None,
        prescribed_current=None,
        target_current=None,
        requested_class=None,
        reference_centroid=None,
        centroid_target=None,
        active=None,
    ) -> BatchedLabelledResult:
        """Run one padded batch and perform one host read of all result leaves."""
        initial = jnp.asarray(initial)
        if initial.ndim != 2:
            raise ValueError("initial must have shape (batch, state)")
        batch = initial.shape[0]
        devices = jax.devices()
        if batch % len(devices):
            raise ValueError("batch size must be divisible by the visible device count")
        zeros_current = None if current is None else jnp.asarray(current)
        zeros_prescribed = (
            None if prescribed_current is None else jnp.asarray(prescribed_current)
        )
        target = None if target_current is None else jnp.asarray(target_current)
        requested = None if requested_class is None else jnp.asarray(requested_class)
        if centroid_target is not None and self._derive_centroid_pairs:
            if (
                prescribed_current is None
                and self.operator.prescribed_current_field is None
            ):
                raise ValueError(
                    "centroid conditioning needs a prescribed current response"
                )
            if not self.constraint_pairs:
                self.constraint_pairs = (
                    _centroid_pair(
                        self.profile,
                        initial[0],
                        jnp.asarray(centroid_target).reshape((batch, -1))[0, 0],
                        requested_class=(None if requested is None else requested[0]),
                        target_current=None if target is None else target[0],
                    ),
                )
        if reference_centroid is None:
            reference = jnp.full((batch, 2), jnp.nan, dtype=initial.dtype)
        else:
            reference = jnp.asarray(reference_centroid)
        if centroid_target is None:
            if self.constraint_pairs:
                static_target = np.concatenate(
                    [
                        np.asarray(pair.binding.target).reshape(-1)
                        for pair in self.constraint_pairs
                    ]
                )
                condition_target = jnp.broadcast_to(
                    jnp.asarray(static_target, dtype=initial.dtype),
                    (batch, static_target.size),
                )
            else:
                condition_target = jnp.zeros((batch, 1), dtype=initial.dtype)
        else:
            condition_target = jnp.asarray(centroid_target).reshape((batch, -1))
        if active is None:
            active_value = jnp.ones((batch,), dtype=bool)
        else:
            active_value = jnp.asarray(active, dtype=bool)
        compiled = self._compiled
        if compiled is None:
            compiled = self._build(
                initial,
                target_current=target,
                requested_class=requested,
                current=zeros_current,
                prescribed_current=zeros_prescribed,
            )
        if zeros_current is None and zeros_prescribed is None:
            external = self.operator.external()
            external = jnp.broadcast_to(external, (batch,) + external.shape)
        else:
            values = zeros_current if zeros_current is not None else zeros_prescribed
            external = jax.vmap(
                lambda value: self.operator.external(
                    value if zeros_current is not None else None,
                    value if zeros_prescribed is not None else None,
                )
            )(values)
        batch_sharding = getattr(self, "_batch_sharding", None)
        if batch_sharding is not None:
            initial = jax.device_put(initial, batch_sharding)
            external = jax.device_put(external, batch_sharding)
            if target is not None:
                target = jax.device_put(target, batch_sharding)
            if requested is not None:
                requested = jax.device_put(requested, batch_sharding)
            reference = jax.device_put(reference, batch_sharding)
            active_value = jax.device_put(active_value, batch_sharding)
            condition_target = jax.device_put(condition_target, batch_sharding)
        output = compiled(
            initial,
            external,
            target,
            requested,
            reference,
            active_value,
            condition_target,
        )
        return BatchedLabelledResult(*jax.device_get(output))


def solve_batched_labeller(profile, initial, **kwargs) -> BatchedLabelledResult:
    """Construct a labeller and solve one batch."""
    options = {
        key: kwargs.pop(key)
        for key in tuple(kwargs)
        if key
        in {
            "tolerance",
            "newton_steps",
            "active_set_steps",
            "guard_tolerance",
            "constraint_pairs",
        }
    }
    return BatchedLabeller(profile, **options).solve(initial, **kwargs)
