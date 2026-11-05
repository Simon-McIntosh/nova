"""Time the solve's own fused units, one jitted call each, at the solve shapes.

The per-stage split times each stage as a separate program, so the compiler
cannot share work between them and the sum can exceed the whole solve. This
receipt measures the units the while_loop actually contains — one read, one
Newton step, one residual — so the split can be read against a fused total.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine

ITERATIONS = 12


def emit(label, **values):
    print(label + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def timed(name, lanes, function, *args, repeats):
    value = function(*args)
    jax.block_until_ready(value)
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        value = function(*args)
        jax.block_until_ready(value)
        samples.append(1e3 * (perf_counter() - started))
    emit("UNIT", lanes=lanes, name=name, median_ms=median(samples), min_ms=min(samples))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    emit(
        "HEADER",
        backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64),
        nova=nova.__file__,
        repeats=args.repeats,
    )
    operands, cache, roots, convention, policy, profile, quadrature, moments, _ = (
        machine("diverted")
    )

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    seeds = roots + 0.03 * jnp.median(operands.geometry.pitch)
    for lanes in (1, 16):
        scales = jnp.linspace(2e-5, 6e-5, 16)[:lanes]
        states = tuple(value[None] * (1 + scales[:, None]) for value in moments)
        seeds_batch = jnp.broadcast_to(seeds, (lanes, *seeds.shape))
        previous_batch = jnp.broadcast_to(roots, (lanes, *roots.shape))

        @jax.jit
        def read(state):
            return support_read.read_batch(
                operands,
                cache,
                state,
                seeds_batch,
                previous_batch,
                convention,
                policy,
                -1.0,
                iterations=ITERATIONS,
            )

        @jax.jit
        def step(state, support):
            return jax.vmap(
                lambda one, read_one: support_read.newton_step(
                    operands,
                    one,
                    read_one,
                    profile,
                    gradient,
                    quadrature,
                    fresh_geometry=True,
                    condition_diagnostics=False,
                )
            )(state, support)

        @jax.jit
        def magnitude(state, support):
            return jax.vmap(
                lambda one, read_one: support_read.step_residual(
                    operands, one, one, read_one, profile, quadrature
                )
            )(state, support)

        @jax.jit
        def body(state):
            support = read(state)
            stepped = jax.vmap(
                lambda one, read_one: support_read.newton_step(
                    operands,
                    one,
                    read_one,
                    profile,
                    gradient,
                    quadrature,
                    fresh_geometry=True,
                    condition_diagnostics=False,
                )
            )(state, support)
            candidate = tuple(
                value + change
                for value, change in zip(state, stepped.update, strict=True)
            )
            next_read = read(candidate)
            return next_read, magnitude(candidate, next_read)

        support = timed("read", lanes, read, states, repeats=args.repeats)
        timed("step", lanes, step, states, support, repeats=args.repeats)
        timed("magnitude", lanes, magnitude, states, support, repeats=args.repeats)
        timed("body", lanes, body, states, repeats=args.repeats)


if __name__ == "__main__":
    main()
