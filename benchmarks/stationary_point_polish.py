"""Measure stationary-point polish latency and tolerance-exit savings."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.jax.config import configure_dtypes
from nova.linalg.tensor_spline import fit_tensor_spline


def _fixed_cap_positions(spline, seeds, valid, steps=8):
    """Run every Newton lane for the complete extraction budget."""
    lower = jnp.stack((spline.radial[0], spline.vertical[0]))
    upper = jnp.stack((spline.radial[-1], spline.vertical[-1]))

    def step(_iteration, point):
        evaluation = spline.evaluate(point[..., 0], point[..., 1])
        determinant = (
            evaluation.radial_second_derivative * evaluation.vertical_second_derivative
            - evaluation.mixed_derivative**2
        )
        safe = jnp.where(
            jnp.abs(determinant) > jnp.finfo(point.dtype).tiny, determinant, 1.0
        )
        radial_step = (
            evaluation.vertical_second_derivative * evaluation.radial_derivative
            - evaluation.mixed_derivative * evaluation.vertical_derivative
        ) / safe
        vertical_step = (
            evaluation.radial_second_derivative * evaluation.vertical_derivative
            - evaluation.mixed_derivative * evaluation.radial_derivative
        ) / safe
        candidate = jnp.stack(
            (
                jnp.clip(point[..., 0] - radial_step, lower[0], upper[0]),
                jnp.clip(point[..., 1] - vertical_step, lower[1], upper[1]),
            ),
            axis=-1,
        )
        return jnp.where(valid[..., None], candidate, point)

    return jax.lax.fori_loop(0, steps, step, seeds)


def _latencies(function, argument, repetitions):
    function(argument)
    jax.block_until_ready(function(argument))
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        jax.block_until_ready(function(argument))
        samples.append((time.perf_counter_ns() - started) / 1.0e3)
    values = np.asarray(samples)
    return {
        "median_us": float(np.median(values)),
        "p95_us": float(np.percentile(values, 95)),
        "minimum_us": float(np.min(values)),
    }


def run(repetitions):
    configure_dtypes()
    radial = jnp.linspace(0.2, 2.0, 65, dtype=jnp.float64)
    vertical = jnp.linspace(-1.4, 1.3, 81, dtype=jnp.float64)
    mesh_r, mesh_z = jnp.meshgrid(radial, vertical)
    centre = jnp.asarray((1.13, -0.27), dtype=jnp.float64)

    def field(radial_centre):
        radial_offset = mesh_r - radial_centre
        vertical_offset = mesh_z - centre[1]
        return (
            radial_offset**2
            + 0.4 * radial_offset**3
            - 1.3 * vertical_offset**2
            + 0.2 * vertical_offset**3
        )

    values = field(centre[0])
    spline = fit_tensor_spline(radial, vertical, values)
    valid = jnp.ones((32,), dtype=bool)
    offsets = jnp.linspace(-0.18, 0.18, 32)
    cold_seeds = centre + jnp.stack((offsets, -0.7 * offsets), axis=-1)
    warm_seeds = jnp.broadcast_to(centre, cold_seeds.shape)

    cold = jax.jit(lambda seeds: polish_stationary_points(spline, seeds, valid))
    warm = jax.jit(lambda seeds: polish_stationary_points(spline, seeds, valid))
    fixed = jax.jit(lambda seeds: _fixed_cap_positions(spline, seeds, valid))

    shifts = jnp.linspace(0.0, 2.0e-3, 16)
    sequence_values = jax.vmap(lambda shift: field(centre[0] + shift))(shifts)
    sequence_splines = jax.vmap(
        lambda field: fit_tensor_spline(radial, vertical, field)
    )(sequence_values)

    @jax.jit
    def tracked(splines):
        def refine(seed, current_spline):
            result = polish_stationary_points(current_spline, seed, valid)
            return result["position_rz"], result["iteration_count"]

        return jax.lax.scan(refine, warm_seeds, splines)

    cold_result = cold(cold_seeds)
    warm_result = warm(warm_seeds)
    tracked_result = tracked(sequence_splines)
    jax.block_until_ready((cold_result, warm_result, tracked_result))
    cold_counts = np.asarray(cold_result["iteration_count"])
    warm_counts = np.asarray(warm_result["iteration_count"])
    tracked_counts = np.asarray(tracked_result[1])

    return {
        "device": str(jax.devices()[0]),
        "platform": jax.default_backend(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "repetitions": repetitions,
        "cold_tolerance_exit": {
            **_latencies(cold, cold_seeds, repetitions),
            "maximum_iterations": int(cold_counts.max()),
            "mean_iterations": float(cold_counts.mean()),
        },
        "warm_tolerance_exit": {
            **_latencies(warm, warm_seeds, repetitions),
            "maximum_iterations": int(warm_counts.max()),
            "mean_iterations": float(warm_counts.mean()),
        },
        "tracked_sequence": {
            **_latencies(tracked, sequence_splines, repetitions),
            "fields": int(shifts.size),
            "maximum_iterations_per_field": tracked_counts.max(axis=1).tolist(),
            "mean_iterations": float(tracked_counts.mean()),
        },
        "batched_fixed_cap": {
            **_latencies(fixed, cold_seeds, repetitions),
            "iterations": 8,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.repetitions)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
