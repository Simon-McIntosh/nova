"""Focused contracts for the compiled-slice public-entry cache."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import pytest

from nova.equilibrium import reduced_newton


class _Operator:
    def __init__(self) -> None:
        self.external_current = jnp.asarray([3.0, 4.0])
        self.prescribed_current_field = None
        self.external_calls = 0

    def external(self, current=None, prescribed_current=None):
        assert prescribed_current is None
        self.external_calls += 1
        return jnp.asarray(self.external_current if current is None else current)


def _fields(state):
    return {
        "state": state,
        "reduced": state[:1],
        "terminal_residual": 0.0,
        "active_set_iterations": 0,
        "converged": True,
        "termination_reason": 0,
        "active_set_residuals": [],
        "active_set_mask_differences": [],
        "newton_steps_per_trip": [],
        "jacobian_builds_per_trip": [],
        "rejected_steps_per_trip": [],
        "map_evaluations_per_trip": [],
        "off_support_leakage": 0.0,
    }


@pytest.fixture
def public_entry(monkeypatch) -> tuple[Callable[..., object], list[int]]:
    coordinate_builds = [0]
    coordinates = reduced_newton.ReducedCoordinates(
        cells=jnp.asarray([0], dtype=jnp.int32),
        leaves=("cell_current",),
        cell_number=2,
    )

    def derive(*args, **kwargs):
        coordinate_builds[0] += 1
        return coordinates

    def compiled_result(operator, initial, **kwargs):
        program, _ = reduced_newton._compiled_program(
            operator,
            initial,
            requested_class=kwargs["requested_class"],
            target_current=kwargs["target_current"],
            external=kwargs["external"],
            default_external=kwargs["default_external"],
            program=kwargs["program"],
            augmentation=kwargs["augmentation"],
        )
        return _fields(initial), program

    monkeypatch.setattr(reduced_newton, "reduced_coordinates", derive)
    monkeypatch.setattr(reduced_newton, "_reduced_kernels", lambda *args, **kwargs: {})
    monkeypatch.setattr(reduced_newton, "_compiled_result", compiled_result)
    reduced_newton._compiled_program_cache.clear()
    yield reduced_newton.solve_reduced_newton_compiled, coordinate_builds
    reduced_newton._compiled_program_cache.clear()


def test_cached_public_entry_skips_seed_and_default_exterior_derivation(public_entry):
    """An unchanged immutable seed reaches the executable without preparation."""
    solve, coordinate_builds = public_entry
    operator = _Operator()
    seed = jnp.asarray([1.0, 2.0])

    first = solve(operator, seed)
    second = solve(operator, seed)

    assert second.program is first.program
    assert coordinate_builds == [1]
    assert operator.external_calls == 1


def test_conductor_edit_reuses_the_program_but_recomputes_its_traced_exterior(
    public_entry,
):
    """A conductor edit is data, not a cache key or a stale exterior hit."""
    solve, coordinate_builds = public_entry
    operator = _Operator()
    seed = jnp.asarray([1.0, 2.0])

    first = solve(operator, seed)
    operator.external_current = jnp.asarray([5.0, 6.0])
    edited = solve(operator, seed)

    assert edited.program.cache_key == first.program.cache_key
    assert coordinate_builds == [1]
    assert operator.external_calls == 2
    assert edited.program.external is not first.program.external


def test_equal_but_distinct_seed_rederives_coordinates(public_entry):
    """Object identity never aliases coordinates for a different seed array."""
    solve, coordinate_builds = public_entry
    operator = _Operator()
    seed = jnp.asarray([1.0, 2.0])

    first = solve(operator, seed)
    second = solve(operator, jnp.array(seed))

    assert second.program.cache_key == first.program.cache_key
    assert coordinate_builds == [2]
    assert operator.external_calls == 2
