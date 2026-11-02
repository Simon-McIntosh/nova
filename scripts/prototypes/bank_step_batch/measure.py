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


def chunked(function, count, chunk=8):
    """Apply ``function`` to lanes in fixed chunks, so peak memory is bounded.

    A batch larger than ``chunk`` lanes shares the launch cost within each
    chunk and not across chunks, which is the price of keeping every
    intermediate inside the card's memory.
    """
    inner = jax.vmap(function, in_axes=(0, 0, 0))

    def run(stacked, support, booked):
        parts = []
        for start in range(0, count, chunk):
            width = min(chunk, count - start)
            window = jax.tree_util.tree_map(
                lambda value: jax.lax.slice_in_dim(value, start, start + width),
                (stacked, support, booked),
            )
            parts.append(inner(*window))
        return jax.tree_util.tree_map(lambda *pieces: jnp.concatenate(pieces), *parts)

    return run


def chunked_args(function, count, chunk=8):
    """The same chunking for a program that already takes a batched leading axis."""

    def run(first, second):
        parts = []
        for start in range(0, count, chunk):
            width = min(chunk, count - start)
            window = jax.tree_util.tree_map(
                lambda value: jax.lax.slice_in_dim(value, start, start + width),
                first,
            )
            parts.append(
                function(window, jax.lax.slice_in_dim(second, start, start + width))
            )
        return jax.tree_util.tree_map(lambda *pieces: jnp.concatenate(pieces), *parts)

    return run


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
                lambda one, one_support: (
                    support_read.step_residual(
                        bank, one, one, one_support, profile, quadrature
                    ).booked
                )
            )(state, lane_support)
        )(stacked, support)
        jax.block_until_ready(support.reading.boundary_flux)

        memory = memory_mb()
        _record_arm(run_dir, "BATCH", slices=count, memory=memory)

        for name, function in stages.items():
            batched = jax.jit(chunked(function, count))
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

        matrix = jax.jit(chunked(stages["assembly"], count))(stacked, support, booked)
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
            if count == 64 and name == "whole_step":
                # The 64-slice whole step ends the process with a segmentation
                # fault inside the profiler trace, after the stage rows land.
                emit("SKIPPED", slices=count, name=name, reason="segfault-in-trace")
                continue
            compiled = jax.jit(chunked_args(function, count))
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


def publish_case():
    """Build the certificate case with the shared operand cache published."""
    from scripts.prototypes.bank_step_batch.fixture import build_case

    case = build_case("diverted", 64)
    emit(
        "PUBLISH",
        nodes=len(case["nodes"]),
        matrix_bytes=int(case["matrix"].nbytes),
    )


def _kernel_totals(trace_dir):
    import gzip

    totals = {}
    for path in Path(trace_dir).rglob("*.trace.json.gz"):
        with gzip.open(path, "rt") as stream:
            try:
                document = json.load(stream)
            except json.JSONDecodeError:
                continue
        events = document.get("traceEvents", ())
        device = {
            event.get("pid")
            for event in events
            if event.get("name") == "process_name"
            and "/device:GPU" in str(event.get("args", {}).get("name", ""))
        }
        for event in events:
            if event.get("ph") != "X" or "dur" not in event:
                continue
            if event.get("pid") not in device:
                continue
            name = str(event.get("name", ""))
            if name.startswith(("cuGraphLaunch", "command_buffer", "memcpy")):
                continue
            entry = totals.setdefault(name, [0.0, 0])
            entry[0] += float(event["dur"])
            entry[1] += 1
    ranked = sorted(totals.items(), key=lambda item: item[1][0], reverse=True)
    return [
        {"name": name, "ms": total / 1e3, "calls": calls}
        for name, (total, calls) in ranked[:5]
    ]


def run_plain_kernel_arm(run_dir):
    """Trace the booking tangent at one lane with no batch map at all."""
    bank, cache, roots, convention, policy, profile, quadrature, moments, _analytic = (
        machine("diverted")
    )
    stages = stage_programs(bank, profile, quadrature)
    seeds = seeds_for(roots, 1)
    stacked = lanes_of(moments, 1)
    support = jax.jit(
        lambda state, lane_seeds: support_read.read_batch(
            bank, cache, state, lane_seeds, lane_seeds, convention, policy
        )
    )(stacked, seeds)
    one = tuple(value[0] for value in stacked)
    one_support = jax.tree_util.tree_map(lambda value: value[0], support)
    booked = jax.jit(
        lambda state, lane_support: (
            support_read.step_residual(
                bank, state, state, lane_support, profile, quadrature
            ).booked
        )
    )(one, one_support)
    compiled = jax.jit(stages["local_booking_tangent"])
    args = (one, one_support, booked)
    compiled(*args)
    jax.block_until_ready(compiled(*args))
    trace_dir = run_dir / "trace-tangent-plain"
    trace_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(trace_dir), create_perfetto_link=False)
    compiled(*args)
    jax.block_until_ready(compiled(*args))
    jax.profiler.stop_trace()
    emit("KERNELS", slices=1, batch=False, top=_kernel_totals(trace_dir))


def run_kernel_arm(run_dir):
    """Profile the booking tangent with and without the batch's vmap."""
    bank, cache, roots, convention, policy, profile, quadrature, moments, _analytic = (
        machine("diverted")
    )
    stages = stage_programs(bank, profile, quadrature)
    for count in (1, 4):
        stacked = lanes_of(moments, count)
        seeds = seeds_for(roots, count)
        support = jax.jit(
            lambda state, lane_seeds: support_read.read_batch(
                bank, cache, state, lane_seeds, lane_seeds, convention, policy
            )
        )(stacked, seeds)
        booked = jax.jit(
            lambda state, lane_support: jax.vmap(
                lambda one, one_support: (
                    support_read.step_residual(
                        bank, one, one, one_support, profile, quadrature
                    ).booked
                )
            )(state, lane_support)
        )(stacked, support)
        compiled = jax.jit(chunked(stages["local_booking_tangent"], count))
        args = (stacked, support, booked)
        compiled(*args)
        jax.block_until_ready(compiled(*args))
        trace_dir = run_dir / f"trace-tangent-{count}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(trace_dir), create_perfetto_link=False)
        compiled(*args)
        jax.block_until_ready(compiled(*args))
        jax.profiler.stop_trace()
        emit("KERNELS", slices=count, top=_kernel_totals(trace_dir))


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
    # The sixteen-lane tangent asks for 119 GiB in one map, so the step runs
    # eight lanes at a time, as the stage arm does.
    step = jax.jit(
        chunked_args(
            lambda state, lanes: support_read.step_batch(
                bank,
                cache,
                state,
                lanes,
                lanes,
                convention,
                policy,
                profile,
                gradient,
                quadrature,
                fresh_geometry=True,
            ),
            count,
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
        correction = step(state, lane_seeds)
        jax.block_until_ready(correction.step.update)
        iteration_walls.append(1e3 * (perf_counter() - started))
        state = tuple(
            value + change
            for value, change in zip(state, correction.step.update, strict=True)
        )

    final_support = read(state)
    final = residual(state, final_support)
    finals = jnp.max(jnp.abs(jnp.concatenate(final.residual, axis=1)), axis=1)
    jax.block_until_ready(finals)
    for lane in range(count):
        reading = jax.tree_util.tree_map(
            lambda value: value[lane], final_support.reading
        )
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
        emit("LANE", **{key: value for key, value in record.items() if key != "kind"})
    emit(
        "CONVERGE",
        lanes=count,
        max_steps=max(per_lane_steps),
        mean_steps=float(np.mean(per_lane_steps)),
        iteration_ms=iteration_walls,
        per_slice_ms=float(np.mean(iteration_walls)) / count
        if iteration_walls
        else None,
        max_fresh_residual=float(jnp.max(finals)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=("stage", "converge", "kernel", "plain", "publish", "both"),
        default="both",
    )
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
    if arguments.arm in ("kernel", "both"):
        run_kernel_arm(arguments.run_dir)
    if arguments.arm == "plain":
        run_plain_kernel_arm(arguments.run_dir)
    if arguments.arm == "publish":
        publish_case()


if __name__ == "__main__":
    main()
