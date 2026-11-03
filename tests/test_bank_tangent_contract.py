"""Bound the device storage used by local booking derivatives."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import pytest

from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine
from scripts.prototypes.bank_step_batch.measure import lanes_of, seeds_for

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tangent_programs():
    if jax.default_backend() != "gpu":
        pytest.skip("device-memory contract requires a GPU")
    assert jax.config.jax_enable_x64
    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine(
        "diverted"
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    support = jax.jit(
        lambda state: support_read.read(
            bank, cache, state, seeds, roots, convention, policy
        )
    )(moments)
    booked = jax.jit(
        lambda state, reading: (
            support_read.step_residual(
                bank, state, state, reading, profile, quadrature
            ).booked
        )
    )(moments, support)

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    def tangent(state, reading, result):
        return support_read._local_booking_sensitivities(
            bank, state, reading, profile, gradient, quadrature, result
        )

    single = jax.jit(tangent).lower(moments, support, booked).compile()
    stacked = lanes_of(moments, 16)
    lane_seeds = seeds_for(roots, 16)
    lane_support = jax.jit(
        lambda state, lane: support_read.read_batch(
            bank, cache, state, lane, lane, convention, policy
        )
    )(stacked, lane_seeds)
    lane_booked = jax.jit(
        lambda state, reading: jax.vmap(
            lambda one, one_read: (
                support_read.step_residual(
                    bank, one, one, one_read, profile, quadrature
                ).booked
            )
        )(state, reading)
    )(stacked, lane_support)
    batched = (
        jax.jit(jax.vmap(tangent, in_axes=(0, 0, 0)))
        .lower(stacked, lane_support, lane_booked)
        .compile()
    )
    return single, batched, (stacked, lane_support, lane_booked)


def test_single_tangent_temporary_bytes(tangent_programs):
    single, _, _ = tangent_programs
    assert single.memory_analysis().temp_size_in_bytes < 3_000_000_000


def test_sixteen_lane_tangent_temporary_bytes(tangent_programs):
    _, batched, arguments = tangent_programs
    assert batched.memory_analysis().temp_size_in_bytes < 55_000_000_000
    jax.block_until_ready(batched(*arguments))
