"""Measure the tracked read over a steady batch, with the decision threaded per lane.

The batched read is one program over a stack of slices. The question this
prototype answers is whether the batch carries the *steady* tracked branch: the
read's rebuild decision must be false in every lane, and the batch must execute
the carry branch once for all lanes rather than paying the whole-mesh rebuild in
each lane, which is what a ``vmap`` of the read's own conditional compiles to.

Three arms are timed at the diverted 550-cell certificate state on one device:

``vmap_conditional``
    ``jax.vmap`` of ``read`` with the tracking state stacked and the decision
    left to the read. The conditional becomes a select, so both branches run.
``vmap_carried``
    ``jax.vmap`` of ``read`` with ``rebuild=False``: the carried decision, no
    branch at all. This is the arm the previous record closed the bound with.
``batch``
    ``support_read.read_batch``: the per-lane decision is formed from batched
    stencils and one ``lax.cond`` on ``any`` lane chooses the branch for the
    whole batch.

Per-lane rebuild flags are printed for every arm, so a batch said to be steady
shows it per lane rather than asserting it. Kernel launches and device time come
from ``jax.profiler``; copy bytes are a property of the timed region, which
returns device arrays and blocks only outside it.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp

from nova.equilibrium import support_read

assert jax.config.jax_enable_x64 is True

ROOT = Path(__file__).resolve().parents[3]
BATCHES = (1, 4, 16, 64)
KEY = {1: "single", 4: "four", 16: "sixteen", 64: "sixtyfour"}
POLICY_RADIUS = 0.054806712567833996


def emit(token, **row):
    print(token + " " + json.dumps(row, sort_keys=True, default=float), flush=True)


def tiled(array, count):
    return jnp.broadcast_to(array, (count, *array.shape))


def stacked(moments, count):
    return tuple(tiled(moment, count) for moment in moments)


def vmap_conditional(operands, cache, convention, policy):
    def one(moments, seeds, previous, track):
        return support_read.read(
            operands,
            cache,
            moments,
            seeds,
            previous,
            convention,
            policy,
            tracking=track,
        )

    return jax.vmap(one)


def vmap_carried(operands, cache, convention, policy):
    def one(moments, seeds, previous, track):
        return support_read.read(
            operands,
            cache,
            moments,
            seeds,
            previous,
            convention,
            policy,
            tracking=track,
            rebuild=False,
        )

    return jax.vmap(one)


def batch_read(operands, cache, convention, policy):
    def one(moments, seeds, previous, track):
        return support_read.read_batch(
            operands,
            cache,
            moments,
            seeds,
            previous,
            convention,
            policy,
            tracking=track,
        )

    return one


def wall(program, arguments, repeats):
    jax.block_until_ready(program(*arguments))
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        result = program(*arguments)
        jax.block_until_ready(result)
        samples.append((started, perf_counter()))
    return samples, result


def flags(operands, cache, case, moments, seeds, tracking):
    """The per-lane rebuild decision, read off the batched stencils."""
    convention = case["convention"]
    stack = jax.vmap(
        lambda state, seed, warm: support_read.stationary_stencil(
            operands, cache, state, seed, warm, convention
        )
    )(moments, seeds, seeds)
    return jax.vmap(support_read.rebuild_needed)(stack, tracking)


def counts(program, arguments):
    from scripts.prototypes.tracked_read_batch.profile_kernels import trace

    path, rows = trace(program, arguments, 1)
    launches = sum(count for _, (count, _) in rows)
    device_ms = sum(nanoseconds for _, (_, nanoseconds) in rows) / 1e6
    return {
        "launches": launches,
        "device_ms": device_ms,
        "trace": str(path),
        "kernels": rows[:5],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=550)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        f"# revision={revision} tree={ROOT} module={support_read.__file__} "
        f"cwd={Path.cwd().resolve()} backend={jax.default_backend()} "
        f"devices={[str(device) for device in jax.local_devices()]}",
        flush=True,
    )

    from scripts.prototypes.tracked_read.measure import bank_for, build_kind, cache_for

    case = build_kind("diverted", args.cells, POLICY_RADIUS)
    operands, digest = bank_for(case, args.store)
    cache = cache_for(operands)
    convention, policy = case["convention"], case["policy"]
    roots = jnp.asarray(case["roots"])
    converged = case["states"]["converged"]
    trial = case["states"]["trial"]
    emit(
        "BANK",
        digest=digest,
        cells=args.cells,
        bucket=operands.bucket,
        nodes=int(operands.active_nodes),
    )

    # The unbatched steady read, and what it costs carried.
    single = support_read.read(
        operands, cache, converged, roots, roots, convention, policy
    )
    jax.block_until_ready(single)
    track = support_read.tracking_of(single)
    emit("SINGLE_STEADY", rebuilt=bool(single.rebuilt), valid=bool(single.valid))
    carried = jax.jit(
        lambda m, t: support_read.read(
            operands,
            cache,
            m,
            roots,
            roots,
            convention,
            policy,
            tracking=t,
            rebuild=False,
        )
    )
    samples, _ = wall(carried, (converged, track), args.repeats)
    emit(
        "SINGLE_WALL",
        variant="carried",
        per_call_ms=min((b - a) * 1e3 for a, b in samples),
    )

    arms = (
        ("vmap_conditional", vmap_conditional(operands, cache, convention, policy)),
        ("vmap_carried", vmap_carried(operands, cache, convention, policy)),
        ("batch", batch_read(operands, cache, convention, policy)),
    )
    rows = {}
    for count in BATCHES:
        state = stacked(converged, count)
        seeds = tiled(roots, count)
        first = jax.jit(arms[0][1])(state, seeds, seeds, None)
        jax.block_until_ready(first)
        track = jax.vmap(support_read.tracking_of)(first)
        jax.block_until_ready(track)
        per_lane = flags(operands, cache, case, state, seeds, track)
        emit(
            "REBUILD_FLAGS",
            batch=count,
            per_lane=[bool(value) for value in jnp.ravel(per_lane)],
            any_lane=bool(jnp.any(per_lane)),
        )
        for name, function in arms:
            program = jax.jit(function)
            samples, result = wall(program, (state, seeds, seeds, track), args.repeats)
            per_slice = [((b - a) * 1e3) / count for a, b in samples]
            row = {
                "batch": count,
                "arm": name,
                "per_call_ms": min(per_slice) * count,
                "per_slice_ms": min(per_slice),
                "mean_per_slice_ms": sum(per_slice) / len(per_slice),
                "rebuilt_per_lane": [
                    bool(value) for value in jnp.ravel(result.rebuilt)
                ],
                "repeats": args.repeats,
            }
            emit("BATCH_ROW", **row)
            rows[f"{KEY[count]}/{name}"] = row

        # A batch whose last lane is another state: how many lanes ask for a
        # rebuild, and whether the batch takes the rebuild branch because of it.
        mixed = tuple(
            jnp.stack([converged[index]] * (count - 1) + [trial[index]])
            for index in range(len(converged))
        )
        mixed_flags = flags(operands, cache, case, mixed, seeds, track)
        function = batch_read(operands, cache, convention, policy)
        program = jax.jit(function)
        result = program(mixed, seeds, seeds, track)
        jax.block_until_ready(result)
        emit(
            "MIXED_FLAGS",
            batch=count,
            per_lane_asking=[bool(value) for value in jnp.ravel(mixed_flags)],
            rebuild_fraction=float(jnp.mean(mixed_flags.astype(jnp.float32))),
            batch_took_rebuild=bool(jnp.any(result.rebuilt)),
        )

    # Kernel launches and device time at one and sixteen slices for the shipped
    # arm, and at one slice for the two vmap arms.
    for count in (1, 16):
        state = stacked(converged, count)
        seeds = tiled(roots, count)
        first = jax.jit(arms[0][1])(state, seeds, seeds, None)
        track = jax.vmap(support_read.tracking_of)(first)
        for name, function in arms:
            program = jax.jit(function)
            jax.block_until_ready(program(state, seeds, seeds, track))
            row = counts(program, (state, seeds, seeds, track))
            emit("KERNELS", batch=count, arm=name, **row)
            rows[f"{KEY[count]}/{name}"].update(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print("EXIT=0")


if __name__ == "__main__":
    main()
