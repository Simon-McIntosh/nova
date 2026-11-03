"""Time the bank step's stages under one command-buffer arm at one batch width."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine

MEASURED_STAGES = []


def emit(kind, **data):
    print(kind, json.dumps(data, sort_keys=True, default=float), flush=True)


def measure(name, function, *args):
    compiled = jax.jit(function)
    result = compiled(*args)
    jax.block_until_ready(result)
    walls = []
    for _ in range(3):
        started = perf_counter()
        result = compiled(*args)
        jax.block_until_ready(result)
        walls.append(1e3 * (perf_counter() - started))
    emit("STAGE", name=name, walls_ms=walls, median_ms=sorted(walls)[1])
    MEASURED_STAGES.append((name, compiled, args))
    return result


def gradient(radius, flux):
    return jnp.zeros_like(radius)


def stack(width, tree):
    return jax.tree.map(lambda value: jnp.stack([value] * width), tree)


def residual_norm(source, width):
    if width == 1:
        values = np.concatenate([np.asarray(v).ravel() for v in source.residual])
        return float(np.linalg.norm(values))
    lanes = np.concatenate(
        [np.asarray(v).reshape(width, -1) for v in source.residual], axis=1
    )
    return float(np.max(np.linalg.norm(lanes, axis=1)))


def one_lane_newton(bank, moments, support, profile, quadrature):
    return support_read.newton_step(
        bank,
        moments,
        support,
        profile,
        gradient,
        quadrature,
        fresh_geometry=True,
        condition_diagnostics=False,
    )


def solve(width, bank, cache, roots, convention, policy, profile, quadrature, moments):
    """Newton-iterate to a fresh-read solved-equation residual of at most 1e-10."""
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    if width == 1:
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
            lambda state, support: one_lane_newton(
                bank, state, support, profile, quadrature
            )
        )
    else:
        seeds_b = stack(width, seeds)
        roots_b = stack(width, roots)
        read = jax.jit(
            lambda state: support_read.read_batch(
                bank, cache, state, seeds_b, roots_b, convention, policy
            )
        )
        residual = jax.jit(
            lambda state, support: jax.vmap(
                lambda lane, support_lane: support_read.step_residual(
                    bank, lane, lane, support_lane, profile, quadrature
                )
            )(state, support)
        )
        step = jax.jit(
            lambda state, support: jax.vmap(
                lambda lane, support_lane: one_lane_newton(
                    bank, lane, support_lane, profile, quadrature
                )
            )(state, support)
        )

    state = moments
    index = -1
    for index in range(20):
        support = read(state)
        before = residual_norm(residual(state, support), width)
        emit(
            "ITERATE",
            index=index,
            fresh_residual=before,
            valid=bool(np.all(np.asarray(support.valid))),
        )
        if before <= 1e-10:
            break
        correction = step(state, support)
        state = tuple(
            value + delta
            for value, delta in zip(state, correction.update, strict=True)
        )
    support = read(state)
    terminal = residual_norm(residual(state, support), width)
    emit(
        "SOLVE",
        steps=index,
        fresh_residual=terminal,
        valid=bool(np.all(np.asarray(support.valid))),
    )
    return state, terminal


def measure_stages(width, bank, cache, roots, convention, policy, profile, quadrature, moments):
    if width == 1:
        seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
        support = measure(
            "read",
            lambda c, m: support_read.read(
                bank, c, m, seeds, roots, convention, policy
            ),
            cache,
            moments,
        )
        residual = measure(
            "residual_and_booking",
            lambda m, s: support_read.step_residual(
                bank, m, m, s, profile, quadrature
            ),
            moments,
            support,
        )
        booked = residual.booked
        measure(
            "local_booking_tangent",
            lambda m, s, p: support_read._local_booking_sensitivities(
                bank, m, s, profile, gradient, quadrature, p
            ),
            moments,
            support,
            booked,
        )
        matrix = measure(
            "jacobian_with_geometry",
            lambda m, s, p: support_read.step_jacobian(
                bank, m, s, profile, gradient, quadrature, p, fresh_geometry=True
            ),
            moments,
            support,
            booked,
        )
        balanced, row, column = measure(
            "equilibration",
            lambda a, m, p: support_read._equilibrate(a, m, p),
            matrix,
            moments,
            booked,
        )
        factor = measure("lu_factor", linalg.lu_factor, balanced)
        right = -jnp.concatenate(residual.residual)
        measure(
            "lu_solve",
            lambda f, v, r, c: c * linalg.lu_solve(f, v / r),
            factor,
            right,
            row,
            column,
        )
        measure(
            "whole_step",
            lambda m, s: one_lane_newton(bank, m, s, profile, quadrature),
            moments,
            support,
        )
        return

    moments_b = stack(width, moments)
    seeds_b = stack(width, roots + 0.03 * jnp.median(bank.geometry.pitch))
    roots_b = stack(width, roots)

    support_b = measure(
        "read",
        lambda state: support_read.read_batch(
            bank, cache, state, seeds_b, roots_b, convention, policy
        ),
        moments_b,
    )

    def batched_residual(state, support):
        return jax.vmap(
            lambda lane, support_lane: support_read.step_residual(
                bank, lane, lane, support_lane, profile, quadrature
            )
        )(state, support)

    residual_b = measure("residual_and_booking", batched_residual, moments_b, support_b)
    booked_b = residual_b.booked

    def batched_tangent(state, support, booked):
        return jax.vmap(
            lambda m, s, p: support_read._local_booking_sensitivities(
                bank, m, s, profile, gradient, quadrature, p
            )
        )(state, support, booked)

    measure("local_booking_tangent", batched_tangent, moments_b, support_b, booked_b)

    def batched_jacobian(state, support, booked):
        return jax.vmap(
            lambda m, s, p: support_read.step_jacobian(
                bank, m, s, profile, gradient, quadrature, p, fresh_geometry=True
            )
        )(state, support, booked)

    matrix_b = measure(
        "jacobian_with_geometry", batched_jacobian, moments_b, support_b, booked_b
    )

    def batched_equilibrate(source, state, booked):
        return jax.vmap(lambda a, m, p: support_read._equilibrate(a, m, p))(
            source, state, booked
        )

    balanced_b, row_b, column_b = measure(
        "equilibration", batched_equilibrate, matrix_b, moments_b, booked_b
    )
    factor_b = measure("lu_factor", jax.vmap(linalg.lu_factor), balanced_b)
    right_b = -jnp.concatenate(
        [jnp.ravel(value) for value in residual_b.residual], axis=-1
    )
    measure(
        "lu_solve",
        lambda f, v, r, c: c * jax.vmap(linalg.lu_solve)(f, v / r),
        factor_b,
        right_b,
        row_b,
        column_b,
    )
    measure(
        "whole_step",
        lambda m, s: jax.vmap(
            lambda lane, support_lane: one_lane_newton(
                bank, lane, support_lane, profile, quadrature
            )
        )(m, s),
        moments_b,
        support_b,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("off", "on"), required=True)
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    width = arguments.width

    root = Path(__file__).resolve().parents[3]
    emit(
        "HEADER",
        arm=arguments.arm,
        width=width,
        revision=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        tree=str(root),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
    )
    assert jax.config.jax_enable_x64

    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine(
        "diverted"
    )
    measure_stages(
        width, bank, cache, roots, convention, policy, profile, quadrature, moments
    )
    moments_final, terminal = solve(
        width, bank, cache, roots, convention, policy, profile, quadrature, moments
    )
    output = arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, *[np.asarray(value) for value in moments_final])
    emit("STATE", path=str(output), fresh_residual=terminal)

    trace = Path(os.environ["BANK_STEP_TRACE_DIR"])
    options = jax.profiler.ProfileOptions()
    options.enable_hlo_proto = False
    options.host_tracer_level = 0
    options.python_tracer_level = 0
    jax.profiler.start_trace(
        str(trace), create_perfetto_link=False, profiler_options=options
    )
    for name, compiled, args in MEASURED_STAGES:
        with jax.profiler.TraceAnnotation(f"stage:{name}"):
            result = compiled(*args)
            jax.block_until_ready(result)
    jax.profiler.stop_trace()
    emit("TRACE_PATH", path=str(trace))


if __name__ == "__main__":
    main()