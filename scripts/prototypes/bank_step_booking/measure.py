"""Compare fresh-read bank solves with and without a carried Newton factor."""

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
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_converge.fresh_solve import (
    analytic_oracle,
    magnitude,
)
from scripts.prototypes.bank_newton_step.measure import machine


def emit(name, **values):
    print(name, json.dumps(values, sort_keys=True, default=float), flush=True)


def run(arm, trace, max_steps):
    bank, cache, roots, convention, policy, profile, quadrature, moments, analytic = (
        machine("diverted")
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)

    @jax.jit
    def read(state):
        return support_read.read(bank, cache, state, seeds, roots, convention, policy)

    @jax.jit
    def residual(state, support):
        return support_read.step_residual(
            bank, state, state, support, profile, quadrature
        )

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    @jax.jit
    def fresh_step(state, support):
        return support_read.newton_step(
            bank,
            state,
            support,
            profile,
            gradient,
            quadrature,
            fresh_geometry=True,
            condition_diagnostics=False,
        )

    @jax.jit
    def carried_step(state, support, factor, ratio):
        return support_read.newton_step(
            bank,
            state,
            support,
            profile,
            gradient,
            quadrature,
            fresh_geometry=True,
            condition_diagnostics=False,
            factor_cache=factor,
            contraction_ratio=ratio,
        )

    support = read(moments)
    first = fresh_step(moments, support)
    jax.block_until_ready(first)
    if arm == "reuse":
        jax.block_until_ready(
            carried_step(moments, support, first.factor, jnp.asarray(0.1))
        )
    factor = None
    previous = None
    walls = []
    refactors = 0
    for index in range(max_steps):
        support = read(moments)
        current = residual(moments, support)
        before = magnitude(current)
        emit("ITERATE", index=index, fresh_residual=before, valid=bool(support.valid))
        if bool(support.valid) and before <= 1e-10:
            break
        ratio = before / previous if previous is not None else float("inf")
        use_cache = arm == "reuse" and factor is not None
        started = perf_counter()
        if use_cache:
            correction = carried_step(moments, support, factor, jnp.asarray(ratio))
        else:
            correction = fresh_step(moments, support)
        jax.block_until_ready(correction)
        wall_ms = 1e3 * (perf_counter() - started)
        walls.append(wall_ms)
        refactors += bool(correction.refactored)
        emit(
            "LINEAR",
            index=index,
            wall_ms=wall_ms,
            contraction_ratio=ratio,
            refactored=bool(correction.refactored),
            linear_residual=float(correction.linear_residual),
        )
        if trace is not None and index == 1:
            options = jax.profiler.ProfileOptions()
            options.enable_hlo_proto = False
            options.host_tracer_level = 0
            options.python_tracer_level = 0
            jax.profiler.start_trace(
                str(trace), create_perfetto_link=False, profiler_options=options
            )
            with jax.profiler.TraceAnnotation("stage:whole_step"):
                if use_cache:
                    traced = carried_step(moments, support, factor, jnp.asarray(ratio))
                else:
                    traced = fresh_step(moments, support)
                jax.block_until_ready(traced)
            jax.profiler.stop_trace()
            emit("TRACE", path=str(trace))
        accepted = False
        for half in range(13):
            length = 2.0**-half
            candidate = tuple(
                value + length * delta
                for value, delta in zip(moments, correction.update, strict=True)
            )
            candidate_support = read(candidate)
            candidate_state = residual(candidate, candidate_support)
            after = magnitude(candidate_state)
            if bool(candidate_support.valid) and np.isfinite(after) and after < before:
                emit("ACCEPT", index=index, length=length, fresh_residual=after)
                moments = candidate
                factor = correction.factor
                previous = before
                accepted = True
                break
        if not accepted:
            emit("NO_DESCENT", index=index, fresh_residual=before)
            break
    support = read(moments)
    terminal = magnitude(residual(moments, support))
    converged = bool(support.valid) and terminal <= 1e-10
    from tests.equilibrium import test_topology_read as fixture

    _, _, wall, _, _ = fixture._analytic_inputs("diverted")
    oracle, _ = analytic_oracle(analytic, roots, support.reading, wall)
    emit("ORACLE", **oracle)
    emit(
        "FINAL",
        arm=arm,
        steps=index,
        fresh_residual=terminal,
        converged=converged,
        refactors=refactors,
        step_walls_ms=walls,
        warm_step_median_ms=float(np.median(walls[1:])) if len(walls) > 1 else None,
    )
    assert converged
    assert oracle["axis_ratio"] <= 5e-4
    assert oracle["saddle_ratio"] <= 5e-4
    assert oracle["boundary_ratio"] <= 1e-4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("fresh", "reuse"), required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()
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
    run(args.arm, args.trace, args.max_steps)
