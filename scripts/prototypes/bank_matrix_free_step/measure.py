"""Time the bank step's assembled Jacobian against its matrix-free product.

The fixture, the seeds and the lane perturbations are built once by
``seeds`` and saved as arrays, so a base arm and a head arm of two revisions
start from identical bytes: rebuilding the seed would follow each revision's
own booking and move the solve's read count at about 1e-14 relative.

Modes
-----
seeds    build the shared state arrays for a width and write them to a file
jacobian time one batched assembly against one batched Jacobian-vector product
update   time one batched Newton update, assembled and matrix-free, with the
         GMRES iterations each lane needed
solve    time a whole batched solve, assembled against matrix-free

Every mode prints one ``RESULT`` JSON line and nothing else on stdout, so a
caller reads the numbers rather than parsing prose.
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


def _header():
    import nova

    print(f"# nova={nova.__file__}", flush=True)
    print(f"# cwd={os.getcwd()}", flush=True)
    print(f"# revision={_revision()}", flush=True)
    print(f"# platforms={[str(device) for device in jax.devices()]}", flush=True)


def _revision():
    head = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".git")
    try:
        with open(os.path.join(head, "HEAD"), encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return "unknown"


def _batch(width):
    return fixture.build(width)


def build_seeds(args):
    batch = _batch(args.width)
    directory = args.seeds
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"seeds-{args.width}.npz")
    np.savez(
        path,
        states=np.stack([np.asarray(value[0]) for value in batch.states]),
        seeds=np.asarray(batch.seeds_batch),
        previous=np.asarray(batch.previous_batch),
    )
    print(f"# wrote {path}", flush=True)
    return {"seeds": path, "width": args.width}


def _load_seeds(path):
    """Return the saved lane states as device arrays on the batch's machine."""
    saved = np.load(path)
    # The saved array is (components, lanes, cells).
    components, lanes = saved["states"].shape[:2]
    batch = _batch(lanes)
    states = tuple(jnp.asarray(saved["states"][index]) for index in range(components))
    return (
        batch,
        states,
        jnp.asarray(saved["seeds"]),
        jnp.asarray(saved["previous"]),
    )


def _time(call, reps):
    warmed = jax.block_until_ready(call())
    del warmed
    start = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(call())
    return (time.perf_counter() - start) / reps


def _memory_peak():
    stats = jax.devices()[0].memory_stats()
    return None if stats is None else stats.get("peak_bytes_in_use")


def _read(batch, moments):
    return jax.jit(
        lambda bank, cache, values: support_read.read_batch(
            bank,
            cache,
            values,
            batch.seeds_batch,
            batch.previous_batch,
            batch.convention,
            batch.policy,
        )
    )(batch.operands, batch.cache, moments)


def run_jacobian(args):
    batch, states, _, _ = _load_seeds(args.seeds)
    cells = states[0].shape[-1]
    direction = tuple(value * 0.5 for value in states)

    def assemble():
        return support_read.step_jacobian(
            batch.operands,
            states,
            _read(batch, states),
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            fresh_geometry=True,
        )

    support = _read(batch, states)

    def product():
        return support_read.step_jacobian_vector(
            batch.operands,
            states,
            direction,
            support,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            fresh_geometry=True,
        )

    assemble_ms = _time(lambda: jax.jit(assemble)(), args.reps) * 1e3
    product_ms = _time(lambda: jax.jit(product)(), args.reps) * 1e3
    return {
        "mode": "jacobian",
        "width": args.width,
        "cells": cells,
        "assemble_ms": assemble_ms,
        "product_ms": product_ms,
        "ratio": assemble_ms / product_ms,
    }


def _update(batch, states, matrix_free):
    extra = {"matrix_free": True} if matrix_free else {}

    def one(values, reading):
        return support_read.newton_step(
            batch.operands,
            values,
            reading,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            fresh_geometry=True,
            **extra,
        )

    support = _read(batch, states)
    return jax.jit(lambda values, readings: jax.vmap(one)(values, readings))(
        states, support
    )


def run_update(args):
    batch, states, _, _ = _load_seeds(args.seeds)
    step = _update(batch, states, bool(args.matrix_free))
    iterations = np.asarray(jax.block_until_ready(step).krylov_iterations)
    call = lambda: _update(batch, states, bool(args.matrix_free))  # noqa: E731
    wall_ms = _time(call, args.reps) * 1e3
    return {
        "mode": "update",
        "width": args.width,
        "matrix_free": bool(args.matrix_free),
        "update_ms": wall_ms,
        "per_lane_ms": wall_ms / args.width,
        "iterations": [int(value) for value in iterations],
        "iterations_median": float(np.median(iterations)),
        "iterations_max": int(iterations.max()),
    }


def _solve(batch, states, seeds, previous, matrix_free):
    # A revision without the product path must be called without the keyword.
    extra = {"matrix_free": True} if matrix_free else {}

    def call():
        return support_read.solve_batch(
            batch.operands,
            batch.cache,
            states,
            seeds,
            previous,
            batch.convention,
            batch.policy,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            **extra,
        )

    return jax.jit(call)


def run_solve(args):
    batch, states, seeds, previous = _load_seeds(args.seeds)
    solve = _solve(batch, states, seeds, previous, bool(args.matrix_free))
    result = jax.block_until_ready(solve())
    wall_ms = _time(solve, args.reps) * 1e3
    return {
        "mode": "solve",
        "width": args.width,
        "matrix_free": bool(args.matrix_free),
        "wall_ms": wall_ms,
        "per_slice_ms": wall_ms / args.width,
        "reads": [int(value) for value in np.asarray(result.reads)],
        "updates": [int(value) for value in np.asarray(result.steps)],
        "residual": [float(value) for value in np.asarray(result.residual)],
        "converged": bool(np.all(np.asarray(result.converged))),
        "moments_digest": float(
            sum(np.sum(np.asarray(value)) for value in result.moments[0:1])
        ),
        "peak_device_bytes": _memory_peak(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["seeds", "jacobian", "update", "solve"])
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--matrix-free", dest="matrix_free", type=int, default=0)
    parser.add_argument("--reps", type=int, default=5)
    arguments = parser.parse_args()
    _header()
    handler = {
        "seeds": build_seeds,
        "jacobian": run_jacobian,
        "update": run_update,
        "solve": run_solve,
    }[arguments.mode]
    payload = handler(arguments)
    print("RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
