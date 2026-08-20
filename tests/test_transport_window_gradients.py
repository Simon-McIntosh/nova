"""Implicit reverse-mode checks for the converged transport window."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nova.jax.config import configure_dtypes
from nova.transport.coupled_window import implicit_window_state
from nova.transport.torax_geometry import torax_geometry_from_fsa


def _geometry_record(radial_cells: int = 4) -> dict[str, object]:
    major_radius = 3.0
    minor_radius = 0.5
    vacuum_field = 2.0
    plasma_current = 5.0e5
    rho_face = np.linspace(0.0, 1.0, radial_cells + 1)
    radius_face = minor_radius * rho_face
    phi_boundary = np.pi * minor_radius**2 * vacuum_field
    vpr_face = 4.0 * np.pi**2 * major_radius * minor_radius**2 * rho_face
    poloidal_field = np.maximum(
        4.0e-7 * np.pi * plasma_current * rho_face / (2.0 * np.pi * minor_radius),
        1.0e-6,
    )
    gradient = np.maximum(poloidal_field * major_radius, 1.0e-6)
    b2_face = vacuum_field**2 + poloidal_field**2
    return {
        "rho_face": rho_face,
        "vpr_face": vpr_face,
        "inv_r_face": np.full_like(rho_face, 1.0 / major_radius),
        "g3_face": np.full_like(rho_face, 1.0 / major_radius**2),
        "int_dl_over_bp_face": 2.0
        * np.pi
        * np.maximum(radius_face, 1.0e-6)
        / poloidal_field,
        "grad_psi_face": gradient,
        "grad_psi2_face": gradient**2,
        "grad_psi2_over_r2_face": gradient**2 / major_radius**2,
        "f_face": np.full_like(rho_face, major_radius * vacuum_field),
        "flux_sign": 1.0,
        "psi_face": 0.2 * rho_face**2,
        "ip_profile_face": plasma_current * rho_face**2,
        "phi_b": phi_boundary,
        "r_in_face": major_radius - radius_face,
        "r_out_face": major_radius + radius_face,
        "delta_upper_face": np.zeros_like(rho_face),
        "delta_lower_face": np.zeros_like(rho_face),
        "elongation_face": np.ones_like(rho_face),
        "b2_face": b2_face,
        "inv_b2_face": 1.0 / b2_face,
    }


def test_implicit_gradient_matches_finite_difference_across_caps_and_geometry():
    """The adjoint crosses the direct JAX geometry adapter, not the iteration."""
    configure_dtypes()
    record = _geometry_record()
    phi_boundary = jnp.asarray(record["phi_b"])
    seed = jnp.asarray([0.35, -0.2])

    def exchange(state, source_strength):
        traced_record = dict(record)
        traced_record["phi_b"] = phi_boundary * (1.0 + 0.08 * source_strength)
        geometry = torax_geometry_from_fsa(traced_record)
        geometry_drive = geometry.Phi_face[-1] / phi_boundary
        return jnp.tanh(
            jnp.asarray([0.18, 0.11]) * state
            + jnp.asarray([0.07, -0.04]) * source_strength
            + jnp.asarray([0.03, 0.02]) * geometry_drive
        )

    def functional(source_strength, iteration_cap):
        state = implicit_window_state(
            exchange,
            seed,
            source_strength,
            iteration_cap=iteration_cap,
            tolerance=1.0e-13,
        )
        return jnp.dot(jnp.asarray([1.2, -0.7]), state) + 0.4 * jnp.dot(state, state)

    source_strength = jnp.asarray(0.6)
    short_cap_gradient = jax.grad(lambda value: functional(value, 24))(source_strength)
    long_cap_gradient = jax.grad(lambda value: functional(value, 48))(source_strength)
    difference_step = 1.0e-5
    finite_difference = (
        functional(source_strength + difference_step, 48)
        - functional(source_strength - difference_step, 48)
    ) / (2.0 * difference_step)

    np.testing.assert_allclose(
        short_cap_gradient,
        long_cap_gradient,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        long_cap_gradient,
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-10,
    )

    state = implicit_window_state(
        exchange,
        seed,
        source_strength,
        iteration_cap=24,
        tolerance=1.0e-13,
    )
    assert float(jnp.max(jnp.abs(state - exchange(state, source_strength)))) <= 1.0e-13
