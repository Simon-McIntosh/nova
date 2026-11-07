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
from scripts.prototypes.bank_solve_trace.analyse import events as trace_events  # noqa: E402


def _revision():
    import subprocess

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    try:
        return subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
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
    """Host wall per call, reported as both the median and the mean.

    This is host wall around a synchronization, not device time: it includes
    dispatch and any host-side work between kernels. The device-arm cost comes
    from :func:`_trace`.
    """
    jax.block_until_ready(call())
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(call())
        samples.append(1e3 * (time.perf_counter() - start))
    return float(np.median(samples)), float(np.mean(samples))


def _trace(call, directory):
    """Device time and kernel-launch count from one profiled call.

    The warm call is outside the trace, so the recorded device events belong
    to exactly one invocation of the arm. ``trace_events`` is the bank solve
    trace's own reader of an XLA profile, so the device-ms and launch figures
    here are the same quantities that record reports.
    """
    jax.block_until_ready(call())
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 1
    options.enable_hlo_proto = False
    options.raise_error_on_start_failure = True
    with jax.profiler.trace(directory, profiler_options=options):
        jax.block_until_ready(call())
    device, _, _ = trace_events(directory)
    return {
        "device_ms": sum(event["duration"] for event in device) / 1e6,
        "launches": len(device),
    }


def _trace_dir(args):
    import pathlib

    if not args.trace:
        raise SystemExit("jacobian mode needs --trace <directory>")
    directory = pathlib.Path(args.trace)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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

    # `vmap` fuses the lanes into one kernel graph and is the batched cost;
    # `map` walks the lanes inside one compiled program, which bounds the
    # intermediate memory when the fused graph exceeds the device.
    def assemble_one(values, reading):
        return support_read.step_jacobian(
            batch.operands,
            values,
            reading,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            fresh_geometry=True,
        )

    def product_one(values, change, reading):
        return support_read.step_jacobian_vector(
            batch.operands,
            values,
            change,
            reading,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            fresh_geometry=True,
        )

    if args.execution == "map":
        assemble = jax.jit(
            lambda values, reading: jax.lax.map(
                lambda pair: assemble_one(pair[0], pair[1]), (values, reading)
            )
        )
        product = jax.jit(
            lambda values, change, reading: jax.lax.map(
                lambda triple: product_one(triple[0], triple[1], triple[2]),
                (values, change, reading),
            )
        )
    else:
        assemble = jax.jit(jax.vmap(assemble_one))
        product = jax.jit(jax.vmap(product_one))
    assembly = jax.block_until_ready(assemble(states, support))
    applied = jax.block_until_ready(product(states, direction, support))
    # The assembly is one (lanes, 3c, 3c) block per lane; the product returns
    # a three-component update of (lanes, c) arrays.
    shape_ok = (
        assembly.ndim == 3
        and tuple(assembly.shape[-2:]) == (3 * cells, 3 * cells)
        and len(applied) == 3
        and all(tuple(value.shape[-1:]) == (cells,) for value in applied)
    )
    # A false shape check means the arms did not measure the same system, so
    # the run must not report a ratio at all.
    assert shape_ok, (
        f"shape mismatch: assembly {assembly.shape}, product components "
        f"{tuple(value.shape for value in applied)}, cells {cells}"
    )
    wall_assemble_median, wall_assemble_mean = _time(
        lambda: assemble(states, support), args.reps
    )
    wall_product_median, wall_product_mean = _time(
        lambda: product(states, direction, support), args.reps
    )
    traced = _trace_dir(args)
    assemble_device = _trace(
        lambda: assemble(states, support), traced / "assemble"
    )
    product_device = _trace(
        lambda: product(states, direction, support), traced / "product"
    )
    break_even_device = assemble_device["device_ms"] / product_device["device_ms"]
    return {
        "mode": "jacobian",
        "execution": args.execution,
        "width": args.width,
        "cells": int(cells),
        "matrix_dimension": int(assembly.shape[-1]),
        "shape_consistent": bool(shape_ok),
        "assemble_host_wall_ms_median": wall_assemble_median,
        "assemble_host_wall_ms_mean": wall_assemble_mean,
        "product_host_wall_ms_median": wall_product_median,
        "product_host_wall_ms_mean": wall_product_mean,
        "assemble_device_ms": assemble_device["device_ms"],
        "product_device_ms": product_device["device_ms"],
        "assemble_launches": assemble_device["launches"],
        "product_launches": product_device["launches"],
        "assemble_device_per_lane_ms": assemble_device["device_ms"] / args.width,
        "product_device_per_lane_ms": product_device["device_ms"] / args.width,
        "break_even_device_iterations": break_even_device,
        "forty_products_device_ms": 40.0 * product_device["device_ms"],
        "forty_products_cheaper": bool(
            40.0 * product_device["device_ms"] < assemble_device["device_ms"]
        ),
        "peak_device_bytes": _memory_peak(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["seeds", "jacobian"])
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--execution", choices=["vmap", "map"], default="vmap")
    parser.add_argument("--trace", default=None)
    args = parser.parse_args()
    _header([f"--width={args.width}", f"--reps={args.reps}"])
    payload = {"seeds": build_seeds, "jacobian": run_jacobian}[args.mode](args)
    print("RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()