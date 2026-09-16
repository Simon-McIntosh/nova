"""Compilation identity for forward solves whose exterior field changes."""

from __future__ import annotations

import hashlib

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import fixed_point, reduced_newton
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import configure_dtypes


CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
)
REQUESTED_CELLS = -300
RELATIVE_IDENTITY_TOLERANCE = 1.0e-14


def _certificate_row(case_name: str):
    """Return the production operator, seed and request for one analytic row."""
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, REQUESTED_CELLS)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(case_name, exact, coordinates)
    empty_operator = certificate.oracle_fixture.forward_operator(source_case, machine)
    exact_moments = certificate.oracle_fixture.exact_current_moments(
        source_case, empty_operator, oracle_state
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_moments)
    exact_internal = certificate.oracle_fixture._internal_flux_image(
        empty_operator, exact_coefficients
    )
    operator = certificate.oracle_fixture.forward_operator(
        source_case, machine, oracle_state - exact_internal
    )
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=certificate.recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        case_name, source_case, operator, exact_moments
    )
    seed, requested_class, _seed_receipt = certificate._production_seed(
        profile,
        case_name,
        target_current,
        centroid,
        current_receipt,
    )
    request = certificate._certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"solovev:{case_name}:{REQUESTED_CELLS}",
    )
    return profile, jnp.asarray(seed), requested_class, target_current, request


def _maps(profile, requested_class, target_current):
    """Return the traced map callbacks and their shadow readers."""
    operator = profile.operator
    mapped = operator.traced_flux_map(requested_class, target_current)
    shadowed = operator.traced_flux_map_with_shadow(requested_class, target_current)

    def shadow_mask(state, active_operator=operator):
        return active_operator.residual_shadow_mask(state, requested_class)

    def promoted_shadow_mask(state, previous, active_operator=operator):
        return active_operator.residual_shadow_mask(
            state, requested_class, previous_shadow=previous
        )

    return mapped, shadowed, shadow_mask, promoted_shadow_mask


def _lower_certificate_solve(row, external=None):
    """Lower the production fixed-point program with exterior flux as input."""
    profile, seed, requested_class, target_current, request = row
    program = profile._accelerated_history_program(
        "newton_krylov",
        requested_class=requested_class,
        target_current=target_current,
        **request.policy.kernel_options(),
    )
    if external is None:
        external = profile.operator.external()
    return program.lower(seed, external, profile.operator), external


def _digest(lowered) -> str:
    """Return the SHA-256 identity of one StableHLO module."""
    text = lowered.as_text(dialect="stablehlo")
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.slow
def test_certificate_rows_share_one_solve_program_per_mesh():
    """Two exterior values on one mesh lower to one StableHLO program."""
    configure_dtypes()
    row = _certificate_row(CASES[0])
    fixture_external = row[0].operator.external()
    scaled_external = fixture_external * jnp.asarray(0.9, dtype=fixture_external.dtype)
    lowered = tuple(
        _lower_certificate_solve(row, external)[0]
        for external in (fixture_external, scaled_external)
    )
    assert _digest(lowered[0]) == _digest(lowered[1])


@pytest.mark.slow
def test_traced_exterior_preserves_the_certificate_terminal_state():
    """Binding or tracing one exterior agrees to floating-point precision."""
    configure_dtypes()
    profile, seed, requested_class, target_current, request = _certificate_row(CASES[0])
    operator = profile.operator
    external = operator.external()
    mapped, shadowed, shadow_mask, promoted_shadow_mask = _maps(
        profile, requested_class, target_current
    )
    options = request.policy.kernel_options()
    traced = fixed_point.newton_krylov(
        mapped,
        seed,
        shadow_mask_fn=shadow_mask,
        promoted_shadow_mask_fn=promoted_shadow_mask,
        shadowed_map_fn=shadowed,
        map_arguments=(external, operator),
        callback_arguments=(operator,),
        **options,
    )
    bound = fixed_point.newton_krylov(
        operator.flux_map(
            requested_class=requested_class,
            target_current=target_current,
        ),
        seed,
        shadow_mask_fn=shadow_mask,
        promoted_shadow_mask_fn=promoted_shadow_mask,
        shadowed_map_fn=operator.flux_map_with_shadow(
            requested_class=requested_class,
            target_current=target_current,
        ),
        **options,
    )
    traced_state = np.asarray(traced.state)
    bound_state = np.asarray(bound.state)
    max_absolute_flux_difference = float(np.max(np.abs(traced_state - bound_state)))
    flux_scale = float(np.max(np.abs(bound_state)))
    max_relative_flux_difference = max_absolute_flux_difference / flux_scale
    residual_difference = abs(float(traced.residual) - float(bound.residual))
    residual_scale = abs(float(bound.residual))
    relative_residual_difference = residual_difference / residual_scale
    assert max_relative_flux_difference <= RELATIVE_IDENTITY_TOLERANCE
    assert relative_residual_difference <= RELATIVE_IDENTITY_TOLERANCE


@pytest.mark.slow
def test_compiled_slice_program_traces_conductor_exterior():
    """Two prescribed conductor states reuse one compiled slice StableHLO."""
    configure_dtypes()
    profile, seed, requested_class, target_current, _request = _certificate_row(
        CASES[0]
    )
    operator = profile.operator
    first_current = jnp.asarray(operator.external_current)
    second_current = first_current * jnp.asarray(0.8, dtype=first_current.dtype)
    common = {
        "requested_class": requested_class,
        "target_current": target_current,
        "newton_steps": 1,
        "active_set_steps": 1,
    }
    first = reduced_newton.solve_reduced_newton_compiled(
        operator, seed, current=first_current, **common
    )
    second = reduced_newton.solve_reduced_newton_compiled(
        operator, seed, current=second_current, **common
    )
    assert second.program.slice_solver is first.program.slice_solver

    shadow = jnp.ravel(
        jnp.asarray(operator.residual_shadow_mask(seed, requested_class), dtype=bool)
    )
    solver = first.program.slice_solver
    first_hlo = solver.lower(seed, shadow, operator.external(first_current), operator)
    second_hlo = solver.lower(seed, shadow, operator.external(second_current), operator)
    assert _digest(first_hlo) == _digest(second_hlo)
