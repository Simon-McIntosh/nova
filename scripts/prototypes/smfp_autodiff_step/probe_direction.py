"""Check the step derivative against central differences on a smooth direction.

The gate's own direction is a ramp over the cell list, so a perturbed state
either side of the seed can leave the read's validity window; the derivative
identity is then untestable rather than failed.  This probe perturbs along a
smooth unit-scale field whose norm matches the state and reports the held-read
and fresh-read comparisons separately, so a booking derivative and a saddle
geometry derivative are told apart.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import support_read
from scripts.prototypes.smfp_autodiff_step.fixture import Certificate, emit

LENGTHS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


def smooth_direction(moments):
    cells = moments[0].size
    phase = 2.0 * jnp.pi * jnp.arange(cells) / cells
    base = jnp.cos(phase)
    scale = jnp.linalg.norm(jnp.concatenate(moments)) / jnp.sqrt(3.0 * cells)
    return (
        scale * base,
        0.5 * scale * jnp.roll(base, 17),
        -0.2 * scale * jnp.roll(base, 31),
    )


def relative(actual, reference):
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    denominator = np.max(np.abs(reference))
    if not np.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return float(np.max(np.abs(actual - reference)) / denominator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    certificate = Certificate(args.store)
    bank = certificate.bank
    moments = certificate.seed_moments
    direction = smooth_direction(moments)
    emit(
        "DIRECTION",
        norm=float(jnp.linalg.norm(jnp.concatenate(direction))),
        state_norm=float(jnp.linalg.norm(jnp.concatenate(moments))),
    )
    read = jax.jit(
        lambda m: support_read.read(
            bank, certificate.cache, m, certificate.seed, certificate.roots,
            certificate.convention, certificate.policy,
        )
    )
    residual = jax.jit(
        lambda base, candidate, support: support_read.step_residual(
            bank, base, candidate, support, certificate.profile, certificate.quadrature
        )
    )
    applied = jax.jit(
        lambda m, direction, support, booked: support_read.step_jacobian_vector(
            bank, m, direction, support, certificate.profile, certificate.gradient,
            certificate.quadrature, booked, fresh_geometry=True,
        )
    )
    cached = jax.jit(
        lambda m, direction, support, booked: support_read.step_jacobian_vector(
            bank, m, direction, support, certificate.profile, certificate.gradient,
            certificate.quadrature, booked, fresh_geometry=False,
        )
    )
    support = read(moments)
    result = residual(moments, moments, support)
    emit(
        "SEED",
        valid=bool(support.valid),
        residual=float(jnp.max(jnp.abs(jnp.concatenate(result.residual)))),
        booked_valid=bool(result.booked.valid),
    )
    booked = result.booked
    fresh_action = np.asarray(jnp.concatenate(applied(moments, direction, support, booked)))
    cached_action = np.asarray(
        jnp.concatenate(cached(moments, direction, support, booked))
    )
    for length in LENGTHS:
        high = tuple(
            value + length * change
            for value, change in zip(moments, direction, strict=True)
        )
        low = tuple(
            value - length * change
            for value, change in zip(moments, direction, strict=True)
        )
        high_support, low_support = read(high), read(low)
        held_high = residual(moments, high, support)
        held_low = residual(moments, low, support)
        fresh_high = residual(high, high, high_support)
        fresh_low = residual(low, low, low_support)
        held_central = (
            jnp.concatenate(held_high.residual)
            - jnp.concatenate(held_low.residual)
        ) / (2.0 * length)
        fresh_central = (
            jnp.concatenate(fresh_high.residual)
            - jnp.concatenate(fresh_low.residual)
        ) / (2.0 * length)
        emit(
            "LENGTH",
            length=length,
            held_valid=(bool(held_high.valid), bool(held_low.valid)),
            fresh_valid=(bool(high_support.valid), bool(low_support.valid)),
            held_cached=relative(cached_action, held_central),
            held_fresh=relative(fresh_action, held_central),
            fresh_fresh=relative(fresh_action, fresh_central),
            fresh_cached=relative(cached_action, fresh_central),
        )


if __name__ == "__main__":
    main()