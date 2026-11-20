"""Time the step's stages separately in one warm process, read excluded.

Every stage is its own jitted program over the same seed state, so the walls
are comparable and no stage inherits another's compile.  The two Jacobian arms
split the wall into the fixed-geometry booking sensitivities and the moving
saddle normal form whose tangent a fresh read supplies; the whole step is timed
last as the boundary the section's clause uses.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import numpy as np

from nova.equilibrium import support_read
from scripts.prototypes.smfp_autodiff_step.fixture import Certificate, emit


def timed(name, call, *arguments, repeats=5):
    program = jax.jit(call)
    result = program(*arguments)
    jax.block_until_ready(result)
    walls = []
    for _ in range(repeats):
        started = perf_counter()
        result = program(*arguments)
        jax.block_until_ready(result)
        walls.append(1e3 * (perf_counter() - started))
    emit("STAGE", stage=name, median_ms=float(np.median(walls)), walls_ms=walls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    print("REVISION=" + os.environ.get("SMFP_REVISION", "unknown"), flush=True)
    certificate = Certificate(args.store)
    emit(
        "PROCESS",
        backend=jax.default_backend(),
        devices=[str(device) for device in jax.local_devices()],
        x64=bool(jax.config.jax_enable_x64),
    )
    moments = certificate.seed_moments
    support = certificate.read(moments)
    jax.block_until_ready(support)
    emit(
        "SUPPORT",
        valid=bool(support.valid),
        reading_valid=bool(support.reading.valid),
        qualified=bool(support.reading.qualified),
    )
    gradient = certificate.gradient
    quadrature = certificate.quadrature
    profile = certificate.profile
    bank = certificate.bank

    def residual(m):
        return support_read.step_residual(bank, m, m, support, profile, quadrature)

    booked = jax.jit(residual)(moments)
    jax.block_until_ready(booked)
    timed("step_residual", residual, moments)
    for label, fresh in (("frozen", False), ("fresh", True)):
        timed(
            f"jacobian_{label}",
            lambda m, fresh=fresh: support_read.step_jacobian(
                bank,
                m,
                support,
                profile,
                gradient,
                quadrature,
                booked.booked,
                fresh_geometry=fresh,
            ),
            moments,
        )
    for label, fresh in (("frozen", False), ("fresh", True)):
        timed(
            f"whole_step_{label}",
            lambda m, fresh=fresh: support_read.newton_step(
                bank,
                m,
                support,
                profile,
                gradient,
                quadrature,
                fresh_geometry=fresh,
                condition_diagnostics=False,
            ),
            moments,
        )


if __name__ == "__main__":
    main()
