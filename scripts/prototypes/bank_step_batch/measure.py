"""Batch the warm bank Newton step over independent slices and time it.

Every arm runs the same operands at the diverted 550-cell certificate state.
A batch holds distinct per-slice states, each a small perturbation of the
converged moments, so a batch that shared one lane's state across the axis
would not reproduce the single-step result. The step is
``support_read.step_batch``: one read decision for the batch and one ``vmap``
over the lanes.
"""

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
from jax.scipy import linalg
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine

BATCHES = (1, 4, 16, 64)
REPEATS = 5
STEPS = 40


def emit(kind, **data):
    print(kind, json.dumps(data, sort_keys=True, default=float), flush=True)


def gradient(radius, flux):
    return jnp.zeros_like(radius)


def lanes_of(moments, count):
    """One distinct state per slice, each a small multiple away from the base."""
    offsets = [(index - (count - 1) / 2) * 2e-5 for index in range(count)]
    return tuple(
        jnp.stack([value * (1.0 + scale) for scale in offsets]) for value in moments
    )


def seeds_for(roots, count):
    return jnp.broadcast_to(jnp.asarray(roots), (count, *jnp.shape(roots)))


def timed(compiled, args, repeats=REPEATS):
    result = compiled(*args)
    jax.block_until_ready(result)
    walls = []
    for _ in range(repeats):
        started = perf_counter()
        result = compiled(*args)
        jax.block_until_ready(result)
        walls.append(1e3 * (perf_counter() - started))
    return min(walls), walls


def memory_mb():
    stats = jax.devices()[0].memory_stats()
    if not stats:
        return None
    return {
        "bytes_in_use": stats.get("bytes_in_use"),
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        "bytes_limit": stats.get("bytes_limit"),
    }


def journal(path, record):
    with open(path, "a") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=float) + "\n")


def stage_programs(bank, profile, quadrature):
    """One batched program per independently timed stage of the step."""

    def residual(state, support, _booked):
        return support_read.step_residual(
            bank, state, state, support, profile, quadrature
        ).booked

    def tangent(state, support, booked):
        return support_read._local_booking_sensitivities(
            bank, state, support, profile, gradient, quadrature, booked
        )

    def geometry(state, support, booked):
        direction = tuple(jnp.ones_like(value) for value in state)
        return support_read._normal_geometry_booking_tangent(
            bank,
            state,
            support_read._directional_field(bank, direction)[0],
            support,
            profile,
            quadrature,
            booked,
        )

    return dict(
        residual_and_booking=residual,
        local_booking_tangent=tangent,
        geometry_tangent=geometry,
        assembly=lambda state, support, booked: support_read.step_jacobian(
            bank,
            state,
            support,
            profile,
            gradient,
            quadrature,
            booked,
            fresh_geometry=True,
        ),
    )


def _record_arm(run_dir, kind, **fields):
    journal(run_dir / "stage.jsonl", {"kind": kind, **fields})
    emit(kind, **fields)


def run_stage_arm(run_dir):
    bank, cache, roots, convention, policy, profile, quadrature, moments, _analytic = (
        machine("diverted")
    )
    stages = stage_programs(bank, profile, quadrature)

    def read_program(state, lane_seeds):
        return support_read.read_batch(
            bank, cache, state, lane_seeds, lane_seeds, convention, policy
        )

    def step_program(state, lane_seeds):
        return support_read.step_batch(
            bank,
            cache,
            state,
            lane_seeds,
            lane_seeds,
            convention,
            policy,
            profile,
            gradient,
            quadrature,
            fresh_geometry=True,
        )

    for count in BATCHES:
        stacked = lanes_of(moments, count)
        seeds = seeds_for(roots, count)
        support = jax.jit(read_program)(stacked, seeds)
        booked = jax.jit(
            lambda state, lane_support: jax.vmap(
                lambda one, one_support: support_read.step_residual(
                    bank, one, one, one_support, profile, quadrature
                ).booked
            )(state, lane_support)
        )(stacked, support)
        jax.block_until_ready(support.reading.boundary_flux)

        memory = memory_mb()
        _record_arm(run_dir, "BATCH", slices=count, memory=memory)

        for name, function in stages.items():
            batched = jax.jit(jax.vmap(function, in_axes=(0, 0, 0)))
            wall, walls = timed(batched, (stacked, support, booked))
            _record_arm(
                run_dir,
                "STAGE",
                slices=count,
                name=name,
                min_ms=wall,
                per_slice_ms=wall / count,
                walls_ms=walls,
            )

        matrix = jax.jit(jax.vmap(stages["assembly"], in_axes=(0, 0, 0)))(
            stacked, support, booked
        )
        jax.block_until_ready(matrix)
        equilibration = jax.jit(
            jax.vmap(
                lambda a, m, p: support_read._equilibrate(a, m, p), in_axes=(0, 0, 0)
            )
        )
        balanced, row, column = equilibration(matrix, stacked, booked)
        lu_factor = jax.jit(jax.vmap(lambda a: linalg.lu_factor(a)))
        factor = lu_factor(balanced)
        jax.block_until_ready(factor[0])
        for name, compiled, args in (
            ("equilibration", equilibration, (matrix, stacked, booked)),
            ("lu_factor", lu_factor, (balanced,)),
        ):
            wall, walls = timed(compiled, args)
            _record_arm(
                run_dir,
                "STAGE",
                slices=count,
                name=name,
                min_ms=wall,
                per_slice_ms=wall / count,
                walls_ms=walls,
            )
        jax.block_until_ready(row)

        for name, function in (("read", read_program), ("whole_step", step_program)):
            compiled = jax.jit(function)
            wall, walls = timed(compiled, (stacked, seeds))
            trace_dir = run_dir / f"trace-s{count}-{name}"
            trace_dir.mkdir(parents=True, exist_ok=True)
            jax.profiler.start_trace(str(trace_dir), create_perfetto_link=False)
            compiled(stacked, seeds)
            jax.block_until_ready(compiled(stacked, seeds))
            jax.profiler.stop_trace()
            _record_arm(
                run_dir,
                "WALL",
                slices=count,
                name=name,
                min_ms=wall,
                per_slice_ms=wall / count,
                walls_ms=walls,
                memory=memory_mb(),
            )


def run_converge_arm(run_dir):
    import tests.equilibrium.test_topology_read as fixture
    from scripts.prototypes.bank_newton_converge.fresh_solve import analytic_oracle

    bank, cache, roots, convention, policy, profile, quadrature, moments, analytic = (
        machine("diverted")
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    count = 16
    stacked = lanes_of(moments, count)
    lane_seeds = seeds_for(roots, count)
    _, _, wall, _, _ = fixture._analytic_inputs("diverted")

    def read_program(state):
        return support_read.read_batch(
            bank, cache, state, lane_seeds, lane_seeds, convention, policy
        )

    def residual_program(state, lane_support):
        return jax.vmap(
            lambda one, one_support: support_read.step_residual(
                bank, one, one, one_support, profile, quadrature
            )
        )(state, lane_support)

    read = jax.jit(read_program)
    residual = jax.jit(residual_program)
    step = jax.jit(
        lambda state: support_read.step_batch(
            bank,
            cache,
            state,
            lane_seeds,
            lane_seeds,
            convention,
            policy,
            profile,
            gradient,
            quadrature,
            fresh_geometry=True,
        )
    )

    state = stacked
    per_lane_steps = [None] * count
    iteration_walls = []
    for index in range(STEPS):
        support = read(state)
        current = residual(state, support)
        sizes = jnp.max(jnp.abs(jnp.concatenate(current.residual, axis=1)), axis=1)
        jax.block_until_ready(sizes)
        for lane in range(count):
            if (
                per_lane_steps[lane] is None
                and float(sizes[lane]) <= 1e-10
                and bool(support.valid[lane])
            ):
                per_lane_steps[lane] = index
        emit(
            "ITERATE",
            index=index,
            residual=[float(value) for value in sizes],
            valid=[bool(value) for value in support.valid],
            pending=sum(value is None for value in per_lane_steps),
        )
        if all(value is not None for value in per_lane_steps):
            break
        started = perf_counter()
        correction = step(state)
        jax.block_until_ready(correction.step.update)
        iteration_walls.append(1e3 * (perf_counter() - started))
        state = tuple(
            value + change for value, change in zip(state, correction.step.update, strict=True)
        )

    final_support = read(state)
    final = residual(state, final_support)
    finals = jnp.max(jnp.abs(jnp.concatenate(final.residual, axis=1)), axis=1)
    jax.block_until_ready(finals)
    for lane in range(count):
        reading = jax.tree_util.tree_map(lambda value: value[lane], final_support.reading)
        oracle, _roots = analytic_oracle(analytic, np.asarray(seeds), reading, wall)
        record = {
            "kind": "LANE",
            "lane": lane,
            "steps": per_lane_steps[lane],
            "fresh_residual": float(finals[lane]),
            "valid": bool(final_support.valid[lane]),
            **oracle,
        }
        journal(run_dir / "converge.jsonl", record)
        emit("LANE", **record)
    emit(
        "CONVERGE",
        lanes=count,
        max_steps=max(per_lane_steps),
        mean_steps=float(np.mean(per_lane_steps)),
        iteration_ms=iteration_walls,
        per_slice_ms=float(np.mean(iteration_walls)) / count if iteration_walls else None,
        max_fresh_residual=float(jnp.max(finals)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("stage", "converge", "both"), default="both")
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    emit(
        "HEADER",
        revision=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        tree=str(root),
        cwd=str(Path.cwd()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        devices=[str(device) for device in jax.devices()],
    )
    assert jax.config.jax_enable_x64
    if arguments.arm in ("stage", "both"):
        run_stage_arm(arguments.run_dir)
    if arguments.arm in ("converge", "both"):
        run_converge_arm(arguments.run_dir)


if __name__ == "__main__":
    main()