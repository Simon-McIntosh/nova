"""Measure contracted booking tangents and their device-memory estimates."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import nova
from nova.equilibrium import current, support_read
from scripts.prototypes.bank_newton_step.measure import machine
from scripts.prototypes.bank_step_batch.measure import lanes_of, seeds_for


def emit(name, **values):
    print(name, json.dumps(values, sort_keys=True, default=float), flush=True)


def timed(executable, arguments):
    result = executable(*arguments)
    jax.block_until_ready(result)
    walls = []
    for _ in range(3):
        started = perf_counter()
        result = executable(*arguments)
        jax.block_until_ready(result)
        walls.append(1e3 * (perf_counter() - started))
    return sorted(walls)[1]


def memory(executable):
    analysis = executable.memory_analysis()
    return {
        name: int(getattr(analysis, name))
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--hlo", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    emit(
        "HEADER",
        arm=args.arm,
        revision=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        tree=str(root),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        current=current.__file__,
        backend=jax.default_backend(),
    )
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

    compiled = jax.jit(tangent).lower(moments, support, booked).compile()
    Path(args.hlo).write_text(compiled.as_text())
    emit(
        "TANGENT",
        memory=memory(compiled),
        wall_ms=timed(compiled, (moments, support, booked)),
        device=jax.devices()[0].memory_stats(),
    )
    step = (
        jax.jit(
            lambda state, reading: support_read.newton_step(
                bank,
                state,
                reading,
                profile,
                gradient,
                quadrature,
                fresh_geometry=True,
                condition_diagnostics=False,
            )
        )
        .lower(moments, support)
        .compile()
    )
    emit("STEP", wall_ms=timed(step, (moments, support)))

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
    try:
        batched = (
            jax.jit(jax.vmap(tangent, in_axes=(0, 0, 0)))
            .lower(stacked, lane_support, lane_booked)
            .compile()
        )
        estimate = memory(batched)
        emit("BATCH_COMPILED", lanes=16, memory=estimate)
        result = batched(stacked, lane_support, lane_booked)
        jax.block_until_ready(result)
        emit(
            "BATCH_EXECUTED",
            lanes=16,
            memory=estimate,
            device=jax.devices()[0].memory_stats(),
        )
    except Exception as error:
        emit("BATCH_FAILED", lanes=16, error=repr(error))


if __name__ == "__main__":
    main()
