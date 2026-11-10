"""Compare plain, secant and bank-Jacobian tangent seeds on common streams."""

# ruff: noqa: E402
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from nova.jax.config import configure_dtypes

configure_dtypes()
import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
from nova.equilibrium import drive_continuation, support_read as bank
from scripts.prototypes.bank_warm_start import streams
from scripts.prototypes.bank_warm_predictor.program import programs
from scripts.prototypes.bank_solve_profile.fixture import gradient

assert jax.config.jax_enable_x64


def tangent(machine, moments, stationary, drive, target):
    """Solve the assembled fresh-geometry Jacobian against the drive derivative."""
    support = bank.read_batch(
        machine.operands,
        machine.cache,
        moments,
        stationary,
        stationary,
        machine.convention,
        machine.policy,
    )
    profile = streams.scaled_profile(machine.profile, drive)

    def one(state, reading):
        residual = bank.step_residual(
            machine.operands, state, state, reading, profile, machine.quadrature
        )
        matrix = bank.step_jacobian(
            machine.operands,
            state,
            reading,
            profile,
            gradient,
            machine.quadrature,
            residual.booked,
            fresh_geometry=True,
        )
        balanced, row, column = bank._equilibrate(matrix, state, residual.booked)
        inverse, indices = bank._block_preconditioner(balanced)

        def residual_at(amplitude):
            result = bank.step_residual(
                machine.operands,
                state,
                state,
                reading,
                streams.scaled_profile(machine.profile, amplitude),
                machine.quadrature,
            )
            return jnp.concatenate(result.residual)

        _, derivative = jax.jvp(residual_at, (drive,), (jnp.ones_like(drive),))
        delta, condition, relative, _ = bank._block_gmres(
            balanced, -derivative / row, inverse, indices
        )
        delta = column * delta
        delta = jnp.where(condition * relative <= 1e-8, delta, jnp.nan)
        return tuple(
            x + (target - drive) * dx
            for x, dx in zip(
                state, jnp.split(delta, (state[0].size, 2 * state[0].size)), strict=True
            )
        )

    return jax.vmap(one)(moments, support)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["plain", "secant", "tangent"], required=True)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        "HEADER",
        json.dumps(
            dict(
                revision=revision,
                cwd=str(Path.cwd().resolve()),
                module=drive_continuation.__file__,
                backend=jax.default_backend(),
                devices=str(jax.devices()),
                x64=jax.config.jax_enable_x64,
                arm=args.arm,
                width=args.width,
                wall_label=f"titan P100, width {args.width}",
            )
        ),
        flush=True,
    )
    assert jax.default_backend() == "gpu"
    machine = streams.build()
    archive = np.load(args.streams, allow_pickle=False)
    initial = tuple(
        jnp.asarray(archive[name][: args.width]) for name in ("m0", "m1", "m2")
    )
    roots = jnp.asarray(archive["stationary"][: args.width])
    bank_inputs, solve_program, tangent_compiled = programs(machine, tangent)

    def solve(machine, moments, seeds, previous, amplitude):
        return solve_program(*bank_inputs, moments, seeds, previous, amplitude)

    def tangent_program(moments, stationary, drive, target):
        return tangent_compiled(*bank_inputs, moments, stationary, drive, target)

    secant_program = jax.jit(
        lambda m, p, d, pd, t: drive_continuation.predict_seed(
            m, d, t, previous_moments=p, previous_drive=pd
        )
    )
    arrays = {}
    rows = []
    for step in (1e-5, 1e-3):
        current, stationary = initial, roots
        older = None
        drive, old_drive = 1.0, None
        for index in range(1, 8):
            target = 1.0 + step * index

            def seed():
                if args.arm == "tangent":
                    return tangent_program(
                        current, stationary, jnp.asarray(drive), jnp.asarray(target)
                    )
                if args.arm == "secant" and older is not None:
                    return secant_program(
                        current,
                        older,
                        jnp.asarray(drive),
                        jnp.asarray(old_drive),
                        jnp.asarray(target),
                    )
                return current

            predicted = seed()
            jax.block_until_ready(predicted)
            result = solve(machine, predicted, stationary, stationary, target)
            jax.block_until_ready(result.residual)
            start = perf_counter()
            predicted = seed()
            jax.block_until_ready(predicted)
            prediction_ms = (perf_counter() - start) * 1e3
            start = perf_counter()
            result = solve(machine, predicted, stationary, stationary, target)
            jax.block_until_ready(result.residual)
            solve_ms = (perf_counter() - start) * 1e3
            jets = jax.vmap(lambda m: bank.nodal_jets(machine.operands, m))(
                result.moments
            )
            key = f"{step:g}-{index}"
            arrays[key + "-flux"] = np.asarray(jets[..., 0])
            arrays[key + "-position"] = np.asarray(result.support.stationary.position)
            arrays[key + "-boundary"] = np.asarray(result.support.reading.boundary_flux)
            arrays[key + "-span"] = np.asarray(
                result.support.reading.boundary_flux - result.support.reading.axis_flux
            )
            row = dict(
                arm=args.arm,
                width=args.width,
                wall_label=f"titan P100, width {args.width}",
                step=step,
                slice=index,
                seed_residual=np.asarray(result.residual_path)[:, 0].tolist(),
                residual=np.asarray(result.residual).tolist(),
                reads=np.asarray(result.reads).tolist(),
                updates=np.asarray(result.steps).tolist(),
                converged=np.asarray(result.converged).tolist(),
                prediction_ms=prediction_ms,
                solve_ms=solve_ms,
                batch_ms=prediction_ms + solve_ms,
                per_slice_ms=(prediction_ms + solve_ms) / args.width,
                minor_radius=machine.minor_radius,
            )
            print("SLICE", json.dumps(row), flush=True)
            rows.append(row)
            if args.warmup and index == 2:
                print("SUMMARY two-slice warmup complete", flush=True)
                return
            assert np.all(result.converged) and np.max(result.residual) <= 1e-10
            older, old_drive = current, drive
            current, drive = result.moments, target
            stationary = result.support.stationary.position
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    np.savez(args.out.with_suffix(".npz"), **arrays)
    print(
        "SUMMARY",
        args.arm,
        "14 slice measurements complete; terminal residual <= 1e-10",
        flush=True,
    )


if __name__ == "__main__":
    main()
