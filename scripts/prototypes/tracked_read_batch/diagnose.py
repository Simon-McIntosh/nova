"""Decompose the batched read's per-call wall against the unbatched read.

The batched read at one slice was measured at 54.6 ms against 1.27 ms for the
same read called once on the same device. This separates the candidates: the
rebuild conditional under ``vmap``, the carried decision, the batch axis, and
the program's own size.

Every number is a wall measured on the device between two ``block_until_ready``
calls over repeated runs, and the program size is reported as counts of HLO op
classes read from the compiled module -- not as a launch count, which needs a
profiler this node does not carry.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp

from nova.equilibrium import support_read
from scripts.prototypes.tracked_read.measure import build_kind, cache_for
from scripts.prototypes.tracked_read_batch.measure import batched, tiled

assert jax.config.jax_enable_x64 is True


def emit(token, **row):
    print(token + " " + json.dumps(row, sort_keys=True, default=float), flush=True)


def op_classes(program, arguments):
    """HLO op classes of the lowered module, counted from its text."""
    lines = program.lower(*arguments).as_text(dialect="hlo").splitlines()
    return dict(
        hlo_lines=len(lines),
        fusion=sum(1 for line in lines if "fusion(" in line),
        custom_call=sum(1 for line in lines if "custom-call(" in line),
        while_loop=sum(1 for line in lines if line.strip().startswith("while")),
        conditional=sum(1 for line in lines if line.strip().startswith("conditional")),
    )


def wall(program, arguments, repeats):
    jax.block_until_ready(program(*arguments))
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        result = program(*arguments)
        jax.block_until_ready(result)
        samples.append((perf_counter() - started) * 1e3)
    return min(samples), sum(samples) / len(samples)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--policy", type=float, default=0.054806712567833996)
    args = parser.parse_args(argv[1:])

    case = build_kind("diverted", args.cells, args.policy)
    operands = support_read.SupportOperands.from_rows(
        case["geometry"],
        case["matrix"],
        case["exterior"],
        case["nodes"],
        case["sample_index"],
        bucket=len(case["nodes"]),
    )
    cache = cache_for(operands)
    convention, policy = case["convention"], case["policy"]
    moments = case["states"]["converged"]
    roots = jnp.asarray(case["roots"])
    emit("DEVICE", backend=jax.default_backend(), devices=[str(d) for d in jax.devices()])

    # The unbatched read, both ways: rebuilding the decision every call, and
    # carrying it with the conditional still in the program.
    single = support_read.read(operands, cache, moments, roots, roots, convention, policy)
    jax.block_until_ready(single)
    program = jax.jit(
        lambda m: support_read.read(
            operands, cache, m, roots, roots, convention, policy
        )
    )
    low, high = wall(program, (moments,), args.repeats)
    emit("SINGLE", variant="rebuild_every_call", per_call_ms=low, mean_ms=high, **op_classes(program, (moments,)))

    track = support_read.tracking_of(single)
    carried = jax.jit(
        lambda m, t: support_read.read(
            operands, cache, m, roots, roots, convention, policy, tracking=t, rebuild=False
        )
    )
    low, high = wall(carried, (moments, track), args.repeats)
    emit("SINGLE", variant="carried", per_call_ms=low, mean_ms=high, **op_classes(carried, (moments, track)))

    # The batched read: the same state stacked over the slice axis.
    for count in (1, 4, 16, 64):
        stacked = tuple(tiled(moment, count) for moment in moments)
        seeded = tiled(roots, count)
        for variant, rebuild in (
            ("carried", False),
            ("conditional", None),
            ("forces_rebuild", True),
        ):
            one = jax.jit(batched(operands, cache, convention, policy, rebuild=rebuild))
            opening = jax.jit(batched(operands, cache, convention, policy))
            first = opening(stacked, seeded, seeded, None)
            jax.block_until_ready(first)
            lanes = jax.vmap(support_read.tracking_of)(first)
            jax.block_until_ready(lanes)
            low, high = wall(one, (stacked, seeded, seeded, lanes), args.repeats)
            emit(
                "BATCH",
                batch=count,
                variant=variant,
                per_call_ms=low,
                per_slice_ms=low / count,
                mean_ms=high,
                **op_classes(one, (stacked, seeded, seeded, lanes)),
            )


if __name__ == "__main__":
    main(sys.argv)