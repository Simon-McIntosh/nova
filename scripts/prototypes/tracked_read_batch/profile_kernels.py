"""Kernel launches and device time, from the profiler rather than from HLO text.

Each program is traced with ``jax.profiler.trace`` and the xplane the tracer
writes is read back with ``jax.profiler.ProfileData``: one row per kernel event
carrying its name and its device duration, which is what a launch count means.
The batched read is traced at one and at four slices as well as once
unbatched, so a kernel that appears only in the batched path can be said to
cost per call or per slice by whether its event count grows with the batch.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax

from nova.equilibrium import support_read
from scripts.prototypes.tracked_read.measure import build_kind, cache_for
from scripts.prototypes.tracked_read_batch.measure import batched, tiled

assert jax.config.jax_enable_x64 is True


def emit(token, **row):
    print(token + " " + json.dumps(row, sort_keys=True, default=float), flush=True)


def xplane_of(directory):
    found = sorted(Path(directory).rglob("*.xplane.pb"))
    return found[-1] if found else None


# The device track is named for the CUDA stream that carries it, e.g.
# "Stream #14(MemcpyH2D,Compute,MemcpyD2D)"; host tracks are named
# "Host Threads/<pid>" and "python".
DEVICE = re.compile(r"stream|device|gpu", re.IGNORECASE)


def kernels(directory):
    """Kernel events of one trace, and the per-line totals they came from.

    A trace holds host tracks as well as the device one, so the events are
    kept per line: a Python frame's duration on a host line is not a kernel,
    and counting both together reports a launch count that no device issued.
    """
    path = xplane_of(directory)
    if path is None:
        return None, [], []
    data = jax.profiler.ProfileData.from_file(str(path))
    lines = []
    totals = defaultdict(lambda: [0, 0])
    for plane in data.planes:
        for line in plane.lines:
            count = sum(1 for event in line.events if event.duration_ns)
            lines.append((line.name, count))
            if not DEVICE.search(line.name):
                continue
            for event in line.events:
                if not event.duration_ns:
                    continue
                entry = totals[event.name]
                entry[0] += 1
                entry[1] += event.duration_ns
    return path, sorted(totals.items(), key=lambda item: -item[1][1]), lines


def trace(program, arguments, repeats):
    directory = tempfile.mkdtemp(prefix="trb-trace-")
    with jax.profiler.trace(directory, create_perfetto_link=False):
        for _ in range(repeats):
            jax.block_until_ready(program(*arguments))
    path, rows, lines = kernels(directory)
    emit("TRACE_LINES", path=str(path), lines=lines)
    shutil.rmtree(directory, ignore_errors=True)
    return path, rows


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=550)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--policy", type=float, default=0.054806712567833996)
    parser.add_argument("--top", type=int, default=12)
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
    roots = jax.numpy.asarray(case["roots"])
    emit(
        "DEVICE", backend=jax.default_backend(), cells=args.cells, repeats=args.repeats
    )

    # Both unbatched arms: the rebuilding read, which is what a caller that
    # passes no tracking gets, and the carried read, which is the one the
    # 1.2 ms wall belongs to. Tracing only the first would compare the batch
    # against a read nobody runs in the steady state.
    single = jax.jit(
        lambda m: support_read.read(
            operands, cache, m, roots, roots, convention, policy
        )
    )
    opening = single(moments)
    jax.block_until_ready(opening)
    path, rows = trace(single, (moments,), args.repeats)
    single_rows = dict(rows)
    emit("TRACE", arm="single", path=str(path), kernels=len(rows), repeats=args.repeats)
    for name, (count, total) in rows[: args.top]:
        emit(
            "SINGLE_KERNEL",
            name=name,
            count=count,
            total_ns=total,
            per_call_ns=total / args.repeats,
        )

    track = support_read.tracking_of(opening)
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
    jax.block_until_ready(carried(moments, track))
    path, rows = trace(carried, (moments, track), args.repeats)
    carried_rows = dict(rows)
    emit(
        "TRACE", arm="carried", path=str(path), kernels=len(rows), repeats=args.repeats
    )
    for name, (count, total) in rows[: args.top]:
        emit(
            "CARRIED_KERNEL",
            name=name,
            count=count,
            total_ns=total,
            per_call_ns=total / args.repeats,
        )

    arms = {}
    for count in (1, 4):
        stacked = tuple(tiled(moment, count) for moment in moments)
        seeded = tiled(roots, count)
        program = jax.jit(batched(operands, cache, convention, policy, rebuild=False))
        opening = jax.jit(batched(operands, cache, convention, policy))
        first = opening(stacked, seeded, seeded, None)
        jax.block_until_ready(first)
        lanes = jax.vmap(support_read.tracking_of)(first)
        jax.block_until_ready(lanes)
        jax.block_until_ready(program(stacked, seeded, seeded, lanes))
        path, rows = trace(program, (stacked, seeded, seeded, lanes), args.repeats)
        arms[count] = dict(rows)
        emit(
            "TRACE",
            arm=f"batch{count}",
            path=str(path),
            kernels=len(rows),
            repeats=args.repeats,
        )
        for name, (events, total) in rows[: args.top]:
            emit(
                "BATCH_KERNEL",
                batch=count,
                name=name,
                count=events,
                total_ns=total,
                per_call_ns=total / args.repeats,
                per_slice_ns=total / args.repeats / count,
            )

    one, four = arms[1], arms[4]
    emit(
        "STEADY",
        carried_events=sum(value[0] for value in carried_rows.values()),
        carried_ns=sum(value[1] for value in carried_rows.values()) / args.repeats,
        rebuilding_events=sum(value[0] for value in single_rows.values()),
        rebuilding_ns=sum(value[1] for value in single_rows.values()) / args.repeats,
        batch1_ns=sum(value[1] for value in one.values()) / args.repeats,
        batch4_ns=sum(value[1] for value in four.values()) / args.repeats,
    )
    for name in sorted(set(one) - set(carried_rows)):
        events_one = one[name][0]
        events_four = four.get(name, (0, 0))[0]
        emit(
            "ADDED_KERNEL",
            name=name,
            events_at_one=events_one,
            events_at_four=events_four,
            scales=(
                "per slice"
                if events_one and events_four >= 3 * events_one
                else "per call"
            ),
            ns_per_call_at_one=one[name][1] / args.repeats,
            ns_per_call_at_four=four.get(name, (0, 0))[1] / args.repeats,
        )


if __name__ == "__main__":
    main(sys.argv)
