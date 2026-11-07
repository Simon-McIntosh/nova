"""Per-slice reads, updates, rebuilds and wall of a warm-started bank slice.

Each process runs the same eight-slice ramp for each stated drive step. Slice 0
is solved cold from the fixture's sixteen perturbed seeds; every later slice
seeds ``support_read.solve_batch`` from its own previous converged state and
the stationary points that state read. The seed states are built once and
saved, so the measured processes start from identical arrays.

The first role, ``builder``, converges slice 0 and writes those arrays. It also
solves one slice cold at the largest ramp amplitude, which is the check that
the exact solution of a scaled profile is the scaled exact solution: the
analytic oracle is then the unit-drive oracle scaled by the slice's amplitude.
"""

# ruff: noqa: E402
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_solve_profile import fixture as profile_fixture
from scripts.prototypes.bank_warm_start import streams

ROOT = Path(__file__).resolve().parents[3]
MAX_STEPS = 60
TOLERANCE = 1e-10


def emit(label, **values):
    print(label + " " + json.dumps(values, sort_keys=True, default=_default), flush=True)


def _default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, jax.Array):
        return np.asarray(value).tolist()
    if isinstance(value, tuple):
        return [_default(item) for item in value]
    raise TypeError(type(value))


_COMPILED = {}


def solve(machine, moments, seeds, previous, amplitude):
    """One batched slice at a traced drive amplitude.

    The jitted callable is built once per machine: a fresh ``jax.jit`` wrapper
    per call compiles a fresh program, which would time a compile per slice.
    """

    compiled = _COMPILED.get("solve")
    if compiled is None:
        compiled = jax.jit(_one(machine))
        _COMPILED["solve"] = compiled
    return compiled(moments, seeds, previous, amplitude)


def _one(machine):
    def one(moments_leaf, seeds_leaf, previous_leaf, amplitude_leaf):
        profile_leaf = streams.scaled_profile(machine.profile, amplitude_leaf)
        return support_read.solve_batch(
            machine.operands,
            machine.cache,
            moments_leaf,
            seeds_leaf,
            previous_leaf,
            machine.convention,
            machine.policy,
            profile_leaf,
            profile_fixture.gradient,
            machine.quadrature,
            max_steps=MAX_STEPS,
            reuse_factor=False,
            contraction_limit=0.5,
            tolerance=TOLERANCE,
        )

    return one


def receipt(result, wall_ms, amplitude, machine, step, slice_index, role):
    converged = np.asarray(result.converged)
    residual = np.asarray(result.residual)
    reads = np.asarray(result.reads)
    steps = np.asarray(result.steps)
    refactors = np.asarray(result.refactors)
    fresh_before = np.asarray(result.residual_path)[:, 0]
    reading = result.support.reading
    boundary = np.asarray(reading.boundary_flux)
    axis_flux = np.asarray(reading.axis_flux)
    span = boundary - axis_flux
    position = np.asarray(result.support.stationary.position)
    axis_error = np.linalg.norm(position[:, 0] - np.asarray(machine.analytic_axis), axis=-1)
    saddle_error = np.linalg.norm(
        position[:, 1] - np.asarray(machine.analytic_saddle), axis=-1
    )
    oracle_boundary = amplitude * np.asarray(machine.analytic_boundary_flux)
    oracle_span = amplitude * np.asarray(
        machine.analytic_boundary_flux - machine.analytic_axis_flux
    )
    emit(
        "SLICE",
        role=role,
        step=step,
        slice=slice_index,
        amplitude=float(amplitude),
        reads=reads.tolist(),
        updates=steps.tolist(),
        refactors=refactors.tolist(),
        fresh_before=fresh_before.tolist(),
        terminal_residual=residual.tolist(),
        converged=converged.tolist(),
        wall_ms=wall_ms,
        boundary_flux=boundary.tolist(),
        span=span.tolist(),
        axis_error=axis_error.tolist(),
        saddle_error=saddle_error.tolist(),
        boundary_error=(boundary - oracle_boundary).tolist(),
        oracle_span=float(np.asarray(oracle_span).ravel()[0]),
        minor_radius=machine.minor_radius,
    )


def timed_solve(machine, moments, seeds, previous, amplitude):
    result = solve(machine, moments, seeds, previous, amplitude)
    jax.block_until_ready(result.residual)
    started = perf_counter()
    result = solve(machine, moments, seeds, previous, amplitude)
    jax.block_until_ready(result.residual)
    return result, 1e3 * (perf_counter() - started)


def build_states(machine, path):
    """Converge slice 0 cold and save the arrays every process starts from."""
    result, wall = timed_solve(
        machine, machine.cold_states, machine.cold_seeds, machine.cold_seeds, 1.0
    )
    receipt(result, wall, 1.0, machine, 0.0, 0, "builder")
    states = tuple(np.asarray(value) for value in result.moments)
    stationary = np.asarray(result.support.stationary.position)
    np.savez(
        path,
        m0=states[0],
        m1=states[1],
        m2=states[2],
        stationary=stationary,
    )
    # The exact solution of a scaled profile is the scaled exact solution.
    amplitude = 1.0 + (streams.SLICES - 1) * max(streams.STEPS)
    scaled_result = jax.jit(
        lambda state, seed, drive: solve(machine, state, seed, seed, drive)
    )(machine.cold_states, machine.cold_seeds, amplitude)
    jax.block_until_ready(scaled_result.residual)
    exact = np.stack([np.asarray(value) for value in result.moments])
    probe = np.stack([np.asarray(value) for value in scaled_result.moments])
    emit(
        "SCALING",
        amplitude=amplitude,
        relative_gap=float(np.max(np.abs(probe - amplitude * exact)) / np.max(np.abs(probe))),
        probe_residual=float(np.max(np.asarray(scaled_result.residual))),
        steps=np.asarray(scaled_result.steps).tolist(),
        reads=np.asarray(scaled_result.reads).tolist(),
    )


def run_chains(machine, path, role):
    archive = np.load(path, allow_pickle=False)
    moments = tuple(jnp.asarray(archive[name]) for name in ("m0", "m1", "m2"))
    stationary = jnp.asarray(archive["stationary"])
    for step in streams.STEPS:
        amplitudes = streams.amplitudes(step)
        current, seeds = moments, stationary
        for index in range(1, streams.SLICES):
            result, wall = timed_solve(
                machine, current, seeds, seeds, float(amplitudes[index])
            )
            receipt(
                result,
                wall,
                float(amplitudes[index]),
                machine,
                step,
                index,
                role,
            )
            current = tuple(np.asarray(value) for value in result.moments)
            current = tuple(jnp.asarray(value) for value in current)
            seeds = result.support.stationary.position


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("builder", "runner"), required=True)
    parser.add_argument("--streams", type=Path, required=True)
    args = parser.parse_args()
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = "unversioned-archive"
    emit(
        "HEADER",
        role=args.role,
        revision=revision,
        tree=str(ROOT),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64),
        pid=os.getpid(),
    )
    assert jax.config.jax_enable_x64
    machine = streams.build()
    if args.role == "builder":
        build_states(machine, args.streams)
    run_chains(machine, args.streams, args.role)


if __name__ == "__main__":
    main()