"""Direct against continuation on a saved warm-start slice, on one H200.

Each arm starts from the same saved arrays: the diverted 550-cell certificate's
slice-0 converged state and the stationary points its own read returned
(``warm-start-streams.npz``). The direct arm asks the bank solve for the full
drive change in one call; the continuation arm advances the drive in adaptive
warm-started sub-steps.

Both arms call a solve compiled once for a traced drive. Compiling per sub-step
would time a compile per step, because the scaled profile closes over the drive
as a Python scalar.
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
from nova.equilibrium import drive_continuation, support_read
from scripts.prototypes.bank_warm_start import streams

ROOT = Path(__file__).resolve().parents[3]
MAX_STEPS = 60
TOLERANCE = 1e-10
STEPS = (1e-3, 1e-2)
SLICES = 8


def emit(label, **values):
    print(label + " " + json.dumps(values, sort_keys=True, default=_default), flush=True)


def _default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, jax.Array):
        return np.asarray(value).tolist()
    if isinstance(value, (tuple, list)):
        return [_default(item) for item in value]
    raise TypeError(type(value))


def _gradient():
    from scripts.prototypes.bank_solve_profile import fixture as profile_fixture

    return profile_fixture.gradient


_COMPILED = {}


def solve(drive, moments, seeds, previous, *, max_steps, tolerance, operands=None):
    """One batched solve at a traced drive, compiled once per static pair."""
    from scripts.prototypes.bank_solve_profile import fixture as profile_fixture

    machine = _MACHINE
    key = (int(max_steps), float(tolerance))

    def one(drive_leaf, state, seed, warm):
        profile = streams.scaled_profile(machine.profile, drive_leaf)
        return support_read.solve_batch(
            machine.operands,
            machine.cache,
            state,
            seed,
            warm,
            machine.convention,
            machine.policy,
            profile,
            profile_fixture.gradient,
            machine.quadrature,
            max_steps=jax.core.concrete_or_error(int, max_steps),
            tolerance=jax.core.concrete_or_error(float, tolerance),
            reuse_factor=False,
            contraction_limit=0.5,
        )

    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = jax.jit(one)
        _COMPILED[key] = compiled
    return compiled(jnp.asarray(drive), moments, seeds, previous)


def _signature(result):
    reading = result.support.reading
    position = np.asarray(result.support.stationary.position)
    return {
        "position": position.tolist(),
        "boundary_flux": np.asarray(reading.boundary_flux).tolist(),
        "axis_flux": np.asarray(reading.axis_flux).tolist(),
        "membership_positive": int(np.sum(np.asarray(reading.membership) > 0)),
        "normal_form": int(np.sum(np.asarray(reading.normal_form_cells))),
    }


def _step_result(result, wall_ms, label, machine, step, amplitude, extra):
    converged = np.asarray(result.converged)
    residual = np.asarray(result.residual)
    reads = np.asarray(result.reads)
    updates = np.asarray(result.steps)
    position = np.asarray(result.support.stationary.position)
    axis_error = np.linalg.norm(position[:, 0] - np.asarray(machine.analytic_axis), axis=-1)
    saddle_error = np.linalg.norm(
        position[:, 1] - np.asarray(machine.analytic_saddle), axis=-1
    )
    reading = result.support.reading
    span = np.asarray(reading.boundary_flux) - np.asarray(reading.axis_flux)
    emit(
        "ARM",
        arm=label,
        step=step,
        amplitude=float(amplitude),
        converged=int(np.sum(converged)),
        lanes=int(converged.size),
        residual_max=float(np.max(residual)),
        reads_batch=int(np.max(reads)),
        reads_sum=int(np.sum(reads)),
        updates_batch=int(np.max(updates)),
        updates_sum=int(np.sum(updates)),
        wall_ms=wall_ms,
        axis_error=axis_error.tolist(),
        saddle_error=saddle_error.tolist(),
        minor_radius=machine.minor_radius,
        span=span.tolist(),
        tail_residual=residual.tolist(),
        **extra,
    )
    return _signature(result)


def run_slice(machine, saved, step, index, lanes=16):
    amplitude = 1.0 + step * index
    moments = tuple(
        jnp.asarray(np.load(saved)[name])[:lanes] for name in ("m0", "m1", "m2")
    )
    seeds = _seed_array(saved)[:lanes]
    previous = seeds
    direct = solve(
        amplitude,
        moments,
        seeds,
        previous,
        max_steps=MAX_STEPS,
        tolerance=TOLERANCE,
    )
    jax.block_until_ready(direct.residual)
    started = perf_counter()
    direct = solve(
        amplitude,
        moments,
        seeds,
        previous,
        max_steps=MAX_STEPS,
        tolerance=TOLERANCE,
    )
    jax.block_until_ready(direct.residual)
    direct_wall = 1e3 * (perf_counter() - started)
    direct_signature = _step_result(
        direct, direct_wall, "direct", machine, step, amplitude, {"certified": None}
    )

    def continuation_solve(
        _operands,
        _cache,
        state,
        seed,
        warm,
        _convention,
        _policy,
        drive,
        _gradient_value,
        _quadrature,
        _axis_sign=-1.0,
        *,
        max_steps=60,
        tolerance=1e-10,
        **_options,
    ):
        return solve(
            drive, state, seed, warm, max_steps=max_steps, tolerance=tolerance
        )

    started = perf_counter()
    reached = drive_continuation.advance_drive(
        None,
        None,
        moments,
        seeds,
        previous,
        None,
        None,
        lambda drive: drive,
        _gradient(),
        None,
        start_drive=1.0,
        target_drive=amplitude,
        solve=continuation_solve,
    )
    jax.block_until_ready(reached.result.residual)
    continuation_wall = 1e3 * (perf_counter() - started)
    continuation_signature = _step_result(
        reached.result,
        continuation_wall,
        "continuation",
        machine,
        step,
        amplitude,
        {
            "certified": bool(reached.certified),
            "substeps": len(reached.substeps),
            "substep_rows": [
                {
                    "drive": row.drive,
                    "change": row.change,
                    "sub_step": row.sub_step,
                    "tolerance": row.tolerance,
                    "updates": row.updates,
                    "reads": row.reads,
                    "residual": row.residual,
                    "converged": bool(row.converged),
                    "final": bool(row.final),
                }
                for row in reached.substeps
            ],
            "reason": reached.reason,
        },
    )
    # Same-drive agreement between the states the two arms reach.
    direct_state = np.stack([np.asarray(value) for value in direct.moments])
    reached_state = np.stack([np.asarray(value) for value in reached.result.moments])
    scale = float(np.max(np.abs(direct_state)))
    emit(
        "PAIR",
        step=step,
        amplitude=float(amplitude),
        direct_converged=int(np.sum(np.asarray(direct.converged))),
        certified=bool(reached.certified),
        state_relative_gap=float(np.max(np.abs(reached_state - direct_state)) / scale),
        direct_position=direct_signature["position"],
        continuation_position=continuation_signature["position"],
        direct_boundary=direct_signature["boundary_flux"],
        continuation_boundary=continuation_signature["boundary_flux"],
        direct_membership=direct_signature["membership_positive"],
        continuation_membership=continuation_signature["membership_positive"],
    )


def _seed_array(saved):
    with np.load(saved) as archive:
        return jnp.asarray(archive["stationary"])


_MACHINE = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--steps", type=float, nargs="+", default=list(STEPS))
    parser.add_argument("--slices", type=int, nargs="+", default=[1])
    parser.add_argument("--lanes", type=int, default=16)
    args = parser.parse_args()
    global _MACHINE
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
        revision=revision,
        tree=str(ROOT),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        driver=drive_continuation.__file__,
        backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64),
        steps=args.steps,
        slices=args.slices,
        lanes=args.lanes,
        pid=os.getpid(),
    )
    assert jax.config.jax_enable_x64
    _MACHINE = streams.build()
    emit("BUILT", minor_radius=_MACHINE.minor_radius)
    for step in args.steps:
        for index in args.slices:
            run_slice(_MACHINE, args.streams, step, index, args.lanes)
    emit("DONE", steps=args.steps, slices=args.slices)


if __name__ == "__main__":
    main()