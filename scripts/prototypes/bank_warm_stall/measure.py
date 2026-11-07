"""Why a 1e-2 warm-started bank slice stalls where a cold start converges.

One H200 allocation runs two arms that solve the same slice of the diverted
certificate at the same drive: the warm arm starts from the previous slice's
converged state, the cold arm from the fixture's own seed. Both arms load the
same saved arrays and both are observed through :mod:`observed`, so every
update, trial, read and Newton factor carries its own line.
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

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_solve_profile import fixture as profile_fixture
from scripts.prototypes.bank_warm_stall import observed
from scripts.prototypes.bank_warm_start import streams

ROOT = Path(__file__).resolve().parents[0]
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent

MAX_STEPS = 60
TOLERANCE = 1e-10
SLICE = 1


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


def emit(sink, label, **values):
    line = label + " " + json.dumps(values, sort_keys=True, default=_default)
    print(line, flush=True)
    if sink is not None:
        sink.write(line + "\n")


def compiled(machine):
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

    return jax.jit(one)


def summary(result, amplitude):
    return {
        "amplitude": float(amplitude),
        "reads": np.asarray(result.reads).tolist(),
        "updates": np.asarray(result.steps).tolist(),
        "refactors": np.asarray(result.refactors).tolist(),
        "fresh_before": np.asarray(result.residual_path)[:, 0].tolist(),
        "terminal_residual": np.asarray(result.residual).tolist(),
        "converged": np.asarray(result.converged).tolist(),
        "residual_path": np.asarray(result.residual_path).tolist(),
        "update_path": np.asarray(result.update_path).tolist(),
        "boundary_class": np.asarray(result.support.reading.boundary_class).tolist(),
        "reason": np.asarray(result.support.reading.reason).tolist(),
        "valid": np.asarray(result.support.valid).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("warm", "cold"), required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--amplitude", type=float, default=1.01)
    args = parser.parse_args()
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        revision = "unversioned"

    machine = streams.build()
    archive = np.load(args.states, allow_pickle=False)
    if args.role == "warm":
        moments = tuple(jax.device_put(archive[name]) for name in ("m0", "m1", "m2"))
        seeds = jax.device_put(archive["stationary"])
        previous = seeds
    else:
        moments = machine.cold_states
        seeds = machine.cold_seeds
        previous = machine.cold_seeds

    sink = open(args.events, "w", buffering=1)
    emit(
        sink,
        "HEADER",
        role=args.role,
        revision=revision,
        tree=str(ROOT),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64),
        amplitude=args.amplitude,
        pid=os.getpid(),
    )
    assert jax.config.jax_enable_x64

    def write(line):
        sink.write(line + "\n")

    undo, _ = observed.install(write)
    try:
        solve = compiled(machine)
        result = solve(moments, seeds, previous, jnp.asarray(args.amplitude))
        jax.block_until_ready(result.residual)
        undo()
        emit(sink, "SUMMARY", **summary(result, args.amplitude))
    finally:
        undo()
        sink.close()


if __name__ == "__main__":
    main()
