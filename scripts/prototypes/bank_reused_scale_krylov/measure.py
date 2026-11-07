"""Time one batched bank Jacobian assembly against one batched product.

The decision this node opens with: a later Newton update could avoid
assembling the dense Jacobian by running GMRES over
``support_read.step_jacobian_vector`` under the scales and block inverses of
the previous update. That is only worth doing if a product is much cheaper
than an assembly, because a later update needs about forty products.

``seeds`` builds the fixture once and saves the lane states as arrays, so
every arm and every revision starts from identical bytes: rebuilding the
fixture runs the booking, and two revisions of the booking start their solve
about 1e-14 apart, which moves a read count.

``jacobian`` times one batched ``step_jacobian`` against one batched
``step_jacobian_vector`` at a width, from those saved arrays. The read is
hoisted out of both timings, so the measurement is the assembly against the
product they both sit behind.

Every mode prints one ``RESULT`` JSON line on stdout after the header lines,
so a caller reads numbers rather than prose.
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

from nova.equilibrium import support_read  # noqa: E402
from scripts.prototypes.bank_solve_profile import fixture  # noqa: E402


def _revision():
    root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    for directory in (".git", os.path.join("..", "..", "..", ".git")):
        head = os.path.join(root, directory, "HEAD")
        try:
            with open(head, encoding="utf-8") as handle:
                value = handle.read().strip()
            if value.startswith("ref: "):
                target = os.path.join(root, directory, value[5:])
                with open(target, encoding="utf-8") as handle:
                    return handle.read().strip()
            return value
        except OSError:
            continue
    return "unknown"


def _header(args):
    import nova

    print(f"# nova={nova.__file__}", flush=True)
    print(f"# cwd={os.getcwd()}", flush=True)
    print(f"# revision={_revision()}", flush=True)
    print(f"# platforms={[str(device) for device in jax.devices()]}", flush=True)
    print(f"# argv={' '.join(['measure.py', *args])}", flush=True)


def _batch(width):
    return fixture.build(width)


def _wide_scales():
    """The nested scale bank the width sweep uses, out to 256 lanes.

    The profile fixture builds sixteen lanes, so a wider batch needs its own
    scale bank: the first sixteen scales are the fixture's own, then each
    doubling adds the midpoints of the gaps so the set stays nested and every
    scale is distinct.
    """
    scales = list(np.linspace(fixture.SCALE_LOW, fixture.SCALE_HIGH, 16))
    divisions = 15
    while len(scales) < 256:
        scales.extend(
            fixture.SCALE_LOW
            + (fixture.SCALE_HIGH - fixture.SCALE_LOW) * (index + 0.5) / divisions
            for index in range(divisions)
        )
        divisions *= 2
    scales = np.asarray(scales[:256])
    assert len(np.unique(scales)) == 256
    return scales


def build_seeds(args):
    batch = _batch(16)
    os.makedirs(args.seeds, exist_ok=True)
    path = os.path.join(args.seeds, "seeds-wide.npz")
    scales = _wide_scales()
    np.savez_compressed(
        path,
        scales=scales,
        seeds=np.asarray(batch.seeds),
        previous=np.asarray(batch.roots),
        **{
            f"state_{index}": np.asarray(value)[None] * (1 + scales[:, None])
            for index, value in enumerate(batch.moments)
        },
    )
    print(f"# wrote {path} with {scales.size} distinct lane scales", flush=True)
    return {"mode": "seeds", "width": int(scales.size), "seeds": path}


def _load_seeds(path, width):
    saved = np.load(path)
    batch = _batch(16)
    states = tuple(jnp.asarray(saved[f"state_{index}"][:width]) for index in range(3))
    seeds = jnp.broadcast_to(jnp.asarray(saved["seeds"]), (width, 2, 2))
    previous = jnp.broadcast_to(jnp.asarray(saved["previous"]), (width, 2, 2))
    return batch, states, seeds, previous


def _time(call, reps):
    jax.block_until_ready(call())
    start = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(call())
    return (time.perf_counter() - start) / reps


def _memory_peak():
    stats = jax.devices()[0].memory_stats()
    return None if stats is None else stats.get("peak_bytes_in_use")


def _read(batch, moments):
    def one(values):
        return support_read.read(
            batch.operands,
            batch.cache,
            values,
            batch.seeds,
            batch.roots,
            batch.convention,
            batch.policy,
        )

    return jax.jit(jax.vmap(one))(moments)


def run_jacobian(args):
    batch, states, _, _ = _load_seeds(args.seeds, args.width)
    cells = states[0].shape[-1]
    direction = tuple(value * 0.5 for value in states)
    support = _read(batch, states)

    assemble = jax.jit(
        lambda values, reading: support_read.step_jacobian(
            batch.operands,
            values,
            reading,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            fresh_geometry=True,
        )
    )
    product = jax.jit(
        lambda values, change, reading: support_read.step_jacobian_vector(
            batch.operands,
            values,
            change,
            reading,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            fresh_geometry=True,
        )
    )
    assembly = jax.block_until_ready(assemble(states, support))
    applied = jax.block_until_ready(product(states, direction, support))
    shape_ok = assembly.shape[-1] == applied[0].size
    assemble_ms = _time(lambda: assemble(states, support), args.reps) * 1e3
    product_ms = _time(lambda: product(states, direction, support), args.reps) * 1e3
    break_even = assemble_ms / product_ms
    return {
        "mode": "jacobian",
        "width": args.width,
        "cells": int(cells),
        "matrix_dimension": int(assembly.shape[-1]),
        "shape_consistent": bool(shape_ok),
        "assemble_ms": assemble_ms,
        "product_ms": product_ms,
        "assemble_per_lane_ms": assemble_ms / args.width,
        "product_per_lane_ms": product_ms / args.width,
        "break_even_iterations": break_even,
        "forty_products_ms": 40.0 * product_ms,
        "forty_products_cheaper": bool(40.0 * product_ms < assemble_ms),
        "peak_device_bytes": _memory_peak(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["seeds", "jacobian"])
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    _header([f"--width={args.width}", f"--reps={args.reps}"])
    payload = {"seeds": build_seeds, "jacobian": run_jacobian}[args.mode](args)
    print("RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()