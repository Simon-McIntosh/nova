"""Test each same-iterate update against central differences of a fresh read."""

# ruff: noqa: E402

from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step import measure


def emit(name, **values):
    print(name + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def run():
    (
        operands,
        cache,
        roots,
        convention,
        policy,
        profile,
        quadrature,
        moments,
        analytic,
    ) = measure.machine("diverted")
    seed = roots + 0.03 * jnp.median(operands.geometry.pitch)
    support = support_read.read(
        operands, cache, moments, seed, roots, convention, policy
    )
    held = {}

    def capture(bank, route, tolerance=1e-6):
        held["bank"] = jnp.asarray(bank)
        held["route"] = jnp.asarray(route)
        return float(
            np.linalg.norm(np.asarray(bank - route)) / np.linalg.norm(np.asarray(route))
        )

    measure._assert_update_agreement = capture
    measure.compare_route(
        operands,
        cache,
        roots,
        convention,
        policy,
        profile,
        quadrature,
        moments,
        support,
        analytic,
    )
    active_quadrature = replace(quadrature, target_current=jnp.sum(moments[0]))

    @jax.jit
    def full(bank, window, candidate):
        read = support_read.read(
            bank, window, candidate, seed, roots, convention, policy
        )
        result = support_read.step_residual(
            bank, candidate, candidate, read, profile, active_quadrature
        )
        return jnp.stack(result.residual, axis=-1), result.valid

    base, base_valid = full(operands, cache, moments)
    scale = float(jnp.linalg.norm(base))
    emit("BASE", residual_norm=scale, valid=bool(base_valid))
    for kind, direction in held.items():
        direction_parts = tuple(direction[:, i] for i in range(3))
        frozen = jnp.stack(
            support_read.step_jacobian_vector(
                operands,
                moments,
                direction_parts,
                support,
                profile,
                lambda radius, flux: jnp.zeros_like(radius),
                active_quadrature,
            ),
            axis=-1,
        )
        emit(
            "FROZEN",
            kind=kind,
            relative_newton=float(jnp.linalg.norm(base + frozen) / scale),
        )
        for length in (1e-5, 1e-6, 1e-7):
            plus = tuple(
                a + length * d for a, d in zip(moments, direction_parts, strict=True)
            )
            minus = tuple(
                a - length * d for a, d in zip(moments, direction_parts, strict=True)
            )
            high, high_valid = full(operands, cache, plus)
            low, low_valid = full(operands, cache, minus)
            difference = (high - low) / (2 * length)
            emit(
                "FRESH_DIFFERENCE",
                kind=kind,
                length=length,
                high_valid=bool(high_valid),
                low_valid=bool(low_valid),
                relative_newton=float(jnp.linalg.norm(base + difference) / scale),
                frozen_gap=float(
                    jnp.linalg.norm(frozen - difference) / jnp.linalg.norm(difference)
                ),
            )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    emit(
        "HEADER",
        revision=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        tree=str(root),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
    )
    run()
