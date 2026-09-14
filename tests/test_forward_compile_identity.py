"""Compilation identity for forward solves whose exterior field changes."""

from __future__ import annotations

import hashlib

import jax
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

    def shadow_mask(state):
        return operator.residual_shadow_mask(state, requested_class)

    def promoted_shadow_mask(state, previous):
        return operator.residual_shadow_mask(
            state, requested_class, previous_shadow=previous
        )

    return mapped, shadowed, shadow_mask, promoted_shadow_mask


def _lower_certificate_solve(row):
    """Lower the production fixed-point program with exterior flux as input."""
    profile, seed, requested_class, target_current, request = row
    mapped, shadowed, shadow_mask, promoted_shadow_mask = _maps(
        profile, requested_class, target_current
    )

    def solve(external):
        return fixed_point.newton_krylov(
            mapped,
            seed,
            shadow_mask_fn=shadow_mask,
            promoted_shadow_mask_fn=promoted_shadow_mask,
            shadowed_map_fn=shadowed,
            map_arguments=(external,),
            **request.policy.kernel_options(),
        )

    external = profile.operator.external()
    return jax.jit(solve).lower(external), external


def _digest(lowered) -> str:
    """Return the SHA-256 identity of one StableHLO module."""
    text = lowered.as_text(dialect="stablehlo")
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.slow
def test_certificate_rows_share_one_solve_program_per_mesh():
    """Two analytic exteriors on the same mesh lower to one StableHLO program."""
    configure_dtypes()
    rows = tuple(_certificate_row(case_name) for case_name in CASES)
    lowered = tuple(_lower_certificate_solve(row)[0] for row in rows)
    assert _digest(lowered[0]) == _digest(lowered[1])


@pytest.mark.slow
def test_traced_exterior_preserves_the_certificate_terminal_state():
    """Binding or tracing one exterior gives the same terminal solve bits."""
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
        map_arguments=(external,),
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
    assert np.array_equal(np.asarray(traced.state), np.asarray(bound.state))
    assert np.array_equal(np.asarray(traced.residual), np.asarray(bound.residual))


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
    first_hlo = solver.lower(seed, shadow, operator.external(first_current))
    second_hlo = solver.lower(seed, shadow, operator.external(second_current))
    assert _digest(first_hlo) == _digest(second_hlo)
