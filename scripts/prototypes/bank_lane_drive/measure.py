"""Count mixed-drive continuation using one numerical solve for both drivers.

The scalar driver cannot name mixed physical drives. Its scalar coordinate
therefore parameterises both ramps, and a common lane-profile adapter feeds
those amplitudes into the numerical solve used by both arms.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64
import jax.numpy as jnp  # noqa: E402

from nova.equilibrium import support_read as bank  # noqa: E402
from scripts.prototypes.bank_solve_profile.fixture import gradient  # noqa: E402
from scripts.prototypes.bank_warm_predictor.receipts import array_hash  # noqa: E402
from scripts.prototypes.bank_warm_start import streams  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("main", "head"), required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("measured_driver", args.driver)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    print(
        "HEADER",
        json.dumps(
            dict(
                revision=args.revision,
                cwd=str(Path.cwd()),
                module=driver.__file__,
                numerical_module=bank.__file__,
                backend=jax.default_backend(),
                devices=str(jax.devices()),
                x64=jax.config.jax_enable_x64,
                arm=args.arm,
            )
        ),
        flush=True,
    )
    assert jax.default_backend() == "gpu"
    machine = streams.build()
    archive = np.load(args.streams, allow_pickle=False)
    moments = tuple(jnp.asarray(archive[name][:8]) for name in ("m0", "m1", "m2"))
    stationary = jnp.asarray(archive["stationary"][:8])
    inputs = (
        machine.operands,
        machine.cache,
        machine.convention,
        machine.policy,
        machine.quadrature,
    )
    steps = jnp.asarray([1e-5] * 4 + [1e-3] * 4)
    target = 1.0 + 7 * steps
    identity = dict(
        seed=array_hash((*moments, stationary)),
        bank=array_hash(jax.tree_util.tree_leaves(inputs)),
        steps=np.asarray(steps).tolist(),
        target=np.asarray(target).tolist(),
    )
    identity["hash"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    print("INPUTS", json.dumps(identity), flush=True)

    @jax.jit
    def solve_program(
        operands,
        cache,
        convention,
        policy,
        quadrature,
        moments,
        seeds,
        previous,
        amplitude,
        tolerance,
    ):
        return bank.solve_batch(
            operands,
            cache,
            moments,
            seeds,
            previous,
            convention,
            policy,
            lambda lane: streams.scaled_profile(machine.profile, amplitude[lane]),
            gradient,
            quadrature,
            max_steps=driver.UPDATE_BUDGET,
            tolerance=tolerance,
            reuse_factor=False,
        )

    rows = []

    def solve(
        _operands,
        _cache,
        state,
        seeds,
        previous,
        _convention,
        _policy,
        drive,
        _gradient,
        _quadrature,
        _axis_sign,
        *,
        max_steps,
        tolerance,
        **_options,
    ):
        amplitudes = (
            jnp.ones(8) + (drive - 1.0) * (steps / 1e-3)
            if args.arm == "main"
            else drive
        )
        tolerances = jnp.broadcast_to(jnp.asarray(tolerance), (8,))
        started = perf_counter()
        result = solve_program(*inputs, state, seeds, previous, amplitudes, tolerances)
        jax.block_until_ready(result)
        rows.append(
            dict(
                drive=np.asarray(amplitudes).tolist(),
                updates=np.asarray(result.steps).tolist(),
                reads=np.asarray(result.reads).tolist(),
                residual=np.asarray(result.residual).tolist(),
                converged=np.asarray(result.converged).tolist(),
                tolerance=np.asarray(tolerances).tolist(),
                seconds=perf_counter() - started,
            )
        )
        args.out.with_suffix(".progress.json").write_text(json.dumps(rows, indent=2))
        print("SUBSOLVE", len(rows), json.dumps(rows[-1]), flush=True)
        return result

    started = perf_counter()
    result = driver.advance_drive(
        machine.operands,
        machine.cache,
        moments,
        stationary,
        stationary,
        machine.convention,
        machine.policy,
        lambda d: d,
        gradient,
        machine.quadrature,
        solve=solve,
        start_drive=1.0 if args.arm == "main" else jnp.ones(8),
        target_drive=1.007 if args.arm == "main" else target,
        initial_relative_step=1e-3 if args.arm == "main" else steps,
        max_substeps=64,
    )
    elapsed = perf_counter() - started
    assert result.certified, result.reason
    assert np.all(result.result.converged)
    assert np.max(result.result.residual) <= 1e-10
    jets = jax.vmap(lambda m: bank.nodal_jets(machine.operands, m))(
        result.result.moments
    )
    arrays = dict(
        flux=np.asarray(jets[..., 0]),
        position=np.asarray(result.result.support.stationary.position),
        boundary=np.asarray(result.result.support.reading.boundary_flux),
        span=np.asarray(
            result.result.support.reading.boundary_flux
            - result.result.support.reading.axis_flux
        ),
    )
    with args.out.with_suffix(".npz").open("wb") as output:
        np.savez(output, **arrays)
    record = dict(
        arm=args.arm,
        revision=args.revision,
        inputs=identity,
        numerical_module=bank.__file__,
        driver_module=driver.__file__,
        completed=True,
        rows=rows,
        reads=result.reads,
        updates=result.updates,
        subsolves=len(result.substeps),
        residual=np.asarray(result.result.residual).tolist(),
        certified=result.certified,
        minor_radius=machine.minor_radius,
        seconds=elapsed,
        wall_label="titan P100, width 8, no speed verdict",
        target=np.asarray(target).tolist(),
    )
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(
        "SUMMARY",
        json.dumps(
            {
                key: record[key]
                for key in (
                    "arm",
                    "reads",
                    "updates",
                    "subsolves",
                    "residual",
                    "seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
