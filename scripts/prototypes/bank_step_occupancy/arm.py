"""Profile one unsynchronised warm bank Newton step and certify its solve."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine


def emit(name, **values):
    print(name, json.dumps(values, sort_keys=True, default=float), flush=True)


def norm(residual):
    return float(
        np.linalg.norm(
            np.concatenate([np.asarray(v).ravel() for v in residual.residual])
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    root = Path(support_read.__file__).resolve().parents[2]
    emit(
        "HEADER",
        revision=args.revision,
        tree=str(root),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        devices=[str(device) for device in jax.devices()],
        command="arm.py --revision " + args.revision + " --trace " + str(args.trace),
    )
    assert jax.config.jax_enable_x64
    assert jax.default_backend() == "gpu"

    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine(
        "diverted"
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    def gradient(radius, flux):
        return jnp.zeros_like(radius)
    read = jax.jit(
        lambda state: support_read.read(
            bank, cache, state, seeds, roots, convention, policy
        )
    )
    residual = jax.jit(
        lambda state, support: support_read.step_residual(
            bank, state, state, support, profile, quadrature
        )
    )
    step = jax.jit(
        lambda state, support: support_read.newton_step(
            bank,
            state,
            support,
            profile,
            gradient,
            quadrature,
            fresh_geometry=True,
            condition_diagnostics=False,
        )
    )

    support = read(moments)
    jax.block_until_ready(support)
    jax.block_until_ready(step(moments, support))
    walls = []
    for _ in range(3):
        started = perf_counter()
        result = step(moments, support)
        jax.block_until_ready(result)
        walls.append(1000 * (perf_counter() - started))
    emit(
        "WALL",
        samples_ms=walls,
        median_ms=sorted(walls)[1],
        jitted_dispatches=1,
        host_syncs_inside_step=0,
        terminal_syncs=1,
    )

    args.trace.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(args.trace), create_perfetto_link=False)
    with jax.profiler.TraceAnnotation("whole_warm_step"):
        result = step(moments, support)
        jax.block_until_ready(result)
    jax.profiler.stop_trace()
    emit("TRACE", path=str(args.trace))

    state = moments
    for iteration in range(20):
        support = read(state)
        before = norm(residual(state, support))
        emit("ITERATE", index=iteration, fresh_residual=before)
        if before <= 1e-10:
            break
        correction = step(state, support)
        state = tuple(
            value + delta for value, delta in zip(state, correction.update, strict=True)
        )
    terminal_support = read(state)
    terminal = norm(residual(state, terminal_support))
    emit(
        "SOLVE",
        steps=iteration,
        fresh_residual=terminal,
        valid=bool(np.asarray(terminal_support.valid)),
    )
    assert terminal <= 1e-10


if __name__ == "__main__":
    main()
