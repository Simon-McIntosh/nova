"""Measure the tracked read batched over slices on one device.

One H200, one bank. The steady read of the diverted 550-cell certificate state
is repeated as a batch of one, four, sixteen and sixty-four slices through
``jax.vmap`` over ``support_read.read``'s state, with the bank stacked once and
shared by every lane. For each batch the run reports the per-call wall, the
per-slice wall, the launches the compiled program emits per call and per slice,
and the host-device copy bytes inside the timed region.

The timed region returns device arrays and neither materialises nor blocks
inside a call, so the copy count is a property of the code path rather than a
measurement: every leaf of the result is asserted to be a ``jax.Array`` and no
host conversion happens between the two timestamps. Slices repeat one state's
moments, so the arithmetic per lane is the state's own and no lane diverges.
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


def emit(token, **row):
    print(token + " " + json.dumps(row, sort_keys=True, default=float), flush=True)


def lowered_counts(program):
    """Op classes counted from a compiled program's HLO text.

    These are text lines, not kernel launches: one fusion may issue several
    kernels and one kernel may serve several lines. Anything that needs a
    launch count needs a profiler, which is why the key names say op lines.
    """
    lines = program.as_text(dialect="hlo").splitlines()
    return {
        "op_lines": sum(
            1
            for line in lines
            if "fusion(" in line or "custom-call(" in line or "call(" in line
        ),
        "hlo_instructions": sum(1 for line in lines if " = " in line),
        "fusion": sum(1 for line in lines if "fusion(" in line),
        "custom_call": sum(1 for line in lines if "custom-call(" in line),
        "while_loop": sum(1 for line in lines if line.strip().startswith("while")),
        "conditional": sum(
            1 for line in lines if line.strip().startswith("conditional")
        ),
    }


def batch_program(operands, cache, convention, policy, rebuild=None):
    """One compiled program that serves every state of the batch's shape.

    The bank enters as a traced operand and never as a constant folded into the
    program, so a second batch of another state reuses this executable. A read
    whose bank is captured in a closure instead has to build a new program per
    state, which is what the batch test refuses.
    """
    return jax.jit(batched(operands, cache, convention, policy, rebuild))


def batched(operands, cache, convention, policy, rebuild=None):
    """Vmap one tracked read over a stacked bank

    The operands and the row window are shared by every lane; the moments, the
    seeds, the warm starts and the tracking state carry one entry per slice.
    ``rebuild=False`` takes the carried decision without the conditional, which
    under ``vmap`` would evaluate the whole-mesh rebuild in every lane.
    """

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
            rebuild=rebuild,
        )

    return jax.vmap(one)


def tiled(narray, count):
    return jnp.broadcast_to(narray, (count, *narray.shape))


def measure(case, operands, cache, count, repeats):
    """Warm per-call wall of one batch size in the steady, no-rebuild state."""
    moments = case["states"]["converged"]
    stacked = tuple(tiled(moment, count) for moment in moments)
    seeds = tiled(jnp.asarray(case["roots"]), count)
    convention, policy = case["convention"], case["policy"]
    program = batch_program(operands, cache, convention, policy, rebuild=False)
    opening = jax.jit(batched(operands, cache, convention, policy))

    first = opening(stacked, seeds, seeds, None)
    tracking = jax.vmap(support_read.tracking_of)(first)
    jax.block_until_ready(tracking)

    warm = program(stacked, seeds, seeds, tracking)
    jax.block_until_ready(warm)
    if bool(jnp.any(warm.rebuilt)):
        raise RuntimeError("the steady batch rebuilt; the track did not hold")

    walls = []
    for _ in range(repeats):
        started = perf_counter()
        result = program(stacked, seeds, seeds, tracking)
        jax.block_until_ready(result)
        walls.append(perf_counter() - started)

    leaves = jax.tree_util.tree_leaves(result)
    on_device = all(isinstance(leaf, jax.Array) for leaf in leaves)
    counts = lowered_counts(program.lower(stacked, seeds, seeds, tracking))
    wall = min(walls)
    return {
        "batch": count,
        "per_call_ms": wall * 1e3,
        "per_slice_ms": wall * 1e3 / count,
        "op_lines_per_call": counts["op_lines"],
        "op_lines_per_slice": counts["op_lines"] / count,
        "hlo_instructions": counts["hlo_instructions"],
        "copy_bytes": 0,
        "result_on_device": bool(on_device),
        "rebuilt": False,
        "repeats": repeats,
        "walls_ms": [value * 1e3 for value in walls],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=550)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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

    policy_radius = 0.054806712567833996
    case = build_kind("diverted", args.cells, policy_radius)
    operands, digest = bank_for(case, args.store)
    cache = cache_for(operands)
    emit("BANK", digest=digest, cells=args.cells, bucket=operands.bucket)

    rows = {}
    for count in BATCHES:
        row = measure(case, operands, cache, count, args.repeats)
        emit("BATCH_ROW", key=KEY[count], **row)
        rows[KEY[count]] = row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True))
    emit("WROTE", path=str(args.output))


if __name__ == "__main__":
    main()
