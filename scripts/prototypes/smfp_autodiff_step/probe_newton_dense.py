"""Solve the bank equation with the step's own dense Jacobian and a line search.

The shipped step assembles the same matrix and hands it to a block GMRES arm
that does not converge it at this certificate.  This probe takes the assembled
matrix, solves it exactly, and iterates the fresh read, so the section's residual
clause is measured against the linearisation the step actually produces.  The
per-step wall is recorded so the arm is never mistaken for the warm step.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import support_read
from scripts.prototypes.smfp_autodiff_step.fixture import Certificate, emit


def magnitude(result):
    return float(jnp.max(jnp.abs(jnp.concatenate(result.residual))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--target", type=float, default=1e-10)
    args = parser.parse_args()
    certificate = Certificate(args.store)
    bank = certificate.bank
    read = jax.jit(
        lambda m: support_read.read(
            bank, certificate.cache, m, certificate.seed, certificate.roots,
            certificate.convention, certificate.policy,
        )
    )
    residual = jax.jit(
        lambda base, candidate, support: support_read.step_residual(
            bank,
            base,
            candidate,
            support,
            certificate.profile,
            certificate.quadrature,
        )
    )
    dense = jax.jit(
        lambda m, support, booked: support_read.step_jacobian(
            bank, m, support, certificate.profile, certificate.gradient,
            certificate.quadrature, booked, fresh_geometry=True,
        )
    )
    moments = certificate.seed_moments
    support = read(moments)
    current = residual(moments, moments, support)
    before = magnitude(current)
    emit("SEED", residual=before, valid=bool(support.valid))
    steps = 0
    while before > args.target and steps < args.max_steps:
        started = perf_counter()
        matrix = np.asarray(dense(moments, support, current.booked))
        matrix_wall = 1e3 * (perf_counter() - started)
        rhs = -np.concatenate(tuple(np.asarray(value) for value in current.residual))
        update = np.linalg.solve(matrix, rhs)
        emit(
            "DENSE",
            index=steps,
            assembly_ms=matrix_wall,
            condition=float(np.linalg.cond(matrix)),
            update_norm=float(np.linalg.norm(update)),
        )
        accepted = False
        for half in range(8):
            length = 2.0**-half
            candidate = tuple(
                value
                + length
                * jnp.asarray(
                    update[index * value.size : (index + 1) * value.size]
                ).reshape(value.shape)
                for index, value in enumerate(moments)
            )
            next_support = read(candidate)
            refreshed = residual(candidate, candidate, next_support)
            after = magnitude(refreshed)
            if bool(next_support.valid) and np.isfinite(after) and after < before:
                emit(
                    "ACCEPT",
                    index=steps,
                    length=length,
                    residual=after,
                )
                moments, support, current, before = (
                    candidate,
                    next_support,
                    refreshed,
                    after,
                )
                accepted = True
                break
        if not accepted:
            emit("NO_DESCENT", index=steps, residual=before)
            break
        steps += 1
    emit(
        "FINAL",
        steps=steps,
        residual=before,
        converged=bool(before <= args.target),
        valid=bool(support.valid),
    )


if __name__ == "__main__":
    main()