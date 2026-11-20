"""Solve the step's own dense Jacobian directly, bypassing the Krylov arm.

The gate's step reports 400 unconverged Krylov iterations with a linear
residual larger than the residual it started from.  This probe takes the same
dense Jacobian and solves it exactly, so a failed update is attributed either
to the factorization (the Krylov arm) or to the Jacobian itself (the dense arm):
if the exact solve also fails to reduce the fresh-read residual, the matrix is
the defect, not the solver.
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


def magnitude(result):
    return float(jnp.max(jnp.abs(jnp.concatenate(result.residual))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    certificate = Certificate(args.store)
    bank = certificate.bank
    moments = certificate.seed_moments
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
    support = read(moments)
    result = residual(moments, moments, support)
    before = magnitude(result)
    emit("SEED", residual=before, valid=bool(support.valid))
    for fresh in (True, False):
        dense = jax.jit(
            lambda m, support, booked, fresh=fresh: support_read.step_jacobian(
                bank, m, support, certificate.profile, certificate.gradient,
                certificate.quadrature, booked, fresh_geometry=fresh,
            )
        )
        matrix = np.asarray(dense(moments, support, result.booked))
        emit(
            "DENSE",
            fresh_geometry=fresh,
            shape=list(matrix.shape),
            condition=float(np.linalg.cond(matrix)),
            finite=bool(np.all(np.isfinite(matrix))),
        )
        rhs = -np.concatenate(tuple(np.asarray(value) for value in result.residual))
        update = np.linalg.solve(matrix, rhs)
        emit(
            "SOLVE",
            fresh_geometry=fresh,
            update_sup=float(np.max(np.abs(update))),
            update_norm=float(np.linalg.norm(update)),
            linear_residual=float(
                np.linalg.norm(matrix @ update - rhs) / np.linalg.norm(rhs)
            ),
        )
        for half in range(6):
            length = 2.0**-half
            candidate = tuple(
                value + length * jnp.asarray(
                    update[index * value.size : (index + 1) * value.size]
                ).reshape(value.shape)
                for index, value in enumerate(moments)
            )
            support_next = read(candidate)
            after = magnitude(residual(candidate, candidate, support_next))
            emit(
                "TRIAL",
                fresh_geometry=fresh,
                length=length,
                residual=after,
                valid=bool(support_next.valid),
            )


if __name__ == "__main__":
    main()