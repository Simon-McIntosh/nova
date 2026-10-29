"""Check the local saddle geometry tangent against fresh-read differences."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine


def emit(name, **values):
    print(name + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def run():
    operands, cache, roots, convention, policy, profile, quadrature, moments, _ = (
        machine("diverted")
    )
    seeds = roots + 0.03 * jnp.median(operands.geometry.pitch)

    @jax.jit
    def fresh(bank, window, state):
        support = support_read.read(
            bank, window, state, seeds, roots, convention, policy
        )
        result = support_read.step_residual(
            bank, state, state, support, profile, quadrature
        )
        return jnp.stack(result.residual, axis=-1), result.valid, support

    @jax.jit
    def tangent(bank, state, direction, support):
        return jnp.stack(
            support_read.step_jacobian_vector(
                bank,
                state,
                direction,
                support,
                profile,
                lambda radius, flux: jnp.zeros_like(radius),
                quadrature,
                fresh_geometry=True,
            ),
            axis=-1,
        )

    @jax.jit
    def frozen(bank, state, direction, support):
        return jnp.stack(
            support_read.step_jacobian_vector(
                bank,
                state,
                direction,
                support,
                profile,
                lambda radius, flux: jnp.zeros_like(radius),
                quadrature,
            ),
            axis=-1,
        )

    base, valid, support = fresh(operands, cache, moments)
    assert bool(valid)
    index = jnp.arange(moments[0].size)
    shape = jnp.sin(1.7 * index) + jnp.cos(2.3 * index)
    shape = 0.4 * (shape - jnp.mean(shape)) / jnp.max(jnp.abs(shape))
    direction = (shape, 0.5 * jnp.roll(shape, 17), -0.2 * jnp.roll(shape, 31))
    predicted = tangent(operands, moments, direction, support)
    frozen_prediction = frozen(operands, moments, direction, support)
    jax.block_until_ready((predicted, frozen_prediction))
    emit(
        "BASE",
        residual_norm=float(jnp.linalg.norm(base)),
        tangent_norm=float(jnp.linalg.norm(predicted)),
        valid=bool(valid),
    )
    best = float("inf")
    for length in (1e-5, 1e-6, 1e-7):
        plus = tuple(a + length * d for a, d in zip(moments, direction, strict=True))
        minus = tuple(a - length * d for a, d in zip(moments, direction, strict=True))
        high, high_valid, _ = fresh(operands, cache, plus)
        low, low_valid, _ = fresh(operands, cache, minus)
        difference = (high - low) / (2 * length)
        relative = float(
            jnp.linalg.norm(predicted - difference) / jnp.linalg.norm(difference)
        )
        old_relative = float(
            jnp.linalg.norm(frozen_prediction - difference)
            / jnp.linalg.norm(difference)
        )
        per_cell = np.asarray(jnp.linalg.norm(predicted - difference, axis=-1))
        worst = np.argsort(per_cell)[-8:][::-1]
        emit(
            "FRESH_JVP",
            length=length,
            relative=relative,
            frozen_relative=old_relative,
            high_valid=bool(high_valid),
            low_valid=bool(low_valid),
            observed_norm=float(jnp.linalg.norm(difference)),
            worst_cells=worst.tolist(),
            worst_errors=per_cell[worst].tolist(),
            worst_normal=np.asarray(support.reading.normal_form_cells)[worst].tolist(),
        )
        best = min(best, relative)
    if best > 1e-3:
        raise AssertionError(f"fresh-read tangent differs by {best}")


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
    assert jax.config.jax_enable_x64
    run()
