"""Count certified secant-seeded slices with optional cross-slice factors."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
from time import perf_counter
import jax
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64
import jax.numpy as jnp
import numpy as np
from nova.equilibrium import drive_continuation, support_read as bank
from scripts.prototypes.bank_warm_start import streams
from scripts.prototypes.bank_solve_profile.fixture import gradient
from scripts.prototypes.bank_warm_predictor import receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("main", "head"), required=True)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    print(
        "HEADER",
        json.dumps(
            dict(
                cwd=str(root),
                module=bank.__file__,
                predictor_module=drive_continuation.__file__,
                devices=str(jax.devices()),
                x64=jax.config.jax_enable_x64,
                arm=args.arm,
                width=args.width,
            )
        ),
        flush=True,
    )
    assert Path(bank.__file__).resolve() == root / "nova/equilibrium/support_read.py"
    assert jax.default_backend() == "gpu"
    machine = streams.build()
    archive = np.load(args.streams, allow_pickle=False)
    initial = tuple(
        jnp.asarray(archive[name][: args.width]) for name in ("m0", "m1", "m2")
    )
    roots = jnp.asarray(archive["stationary"][: args.width])
    assert len(roots) == args.width
    inputs = (
        machine.operands,
        machine.cache,
        machine.convention,
        machine.policy,
        machine.quadrature,
    )
    metadata = receipts.provenance(root, initial, roots, inputs, args.width)
    metadata["arm"] = args.arm
    print("INPUTS", json.dumps(metadata), flush=True)

    @jax.jit
    def solve(
        operands,
        cache,
        convention,
        policy,
        quadrature,
        moments,
        points,
        amplitude,
        factor=None,
        factor_valid=True,
    ):
        options = dict(max_steps=60, tolerance=1e-10, reuse_factor=False)
        if args.arm == "head":
            options.update(
                reuse_factor=True, factor_cache=factor, factor_valid=factor_valid
            )
        return bank.solve_batch(
            operands,
            cache,
            moments,
            points,
            points,
            convention,
            policy,
            streams.scaled_profile(machine.profile, amplitude),
            gradient,
            quadrature,
            **options,
        )

    predict = jax.jit(
        lambda m, p, d, pd, t: drive_continuation.predict_seed(
            m, d, t, previous_moments=p, previous_drive=pd
        )
    )
    rows, arrays = [], {}
    for step in (1e-5, 1e-3):
        current, stationary, older = initial, roots, None
        drive, old_drive, factor, valid = 1.0, None, None, True
        for index in range(1, 8):
            target = 1.0 + step * index
            start = perf_counter()
            predicted = (
                current
                if older is None
                else predict(current, older, drive, old_drive, target)
            )
            result = solve(*inputs, predicted, stationary, target, factor, valid)
            jax.block_until_ready(result)
            wall = (perf_counter() - start) * 1000
            jets = jax.vmap(lambda m: bank.nodal_jets(machine.operands, m))(
                result.moments
            )
            key = f"{step:g}-{index}"
            for suffix, value in (
                ("flux", jets[..., 0]),
                ("position", result.support.stationary.position),
                ("boundary", result.support.reading.boundary_flux),
                (
                    "span",
                    result.support.reading.boundary_flux
                    - result.support.reading.axis_flux,
                ),
            ):
                arrays[key + "-" + suffix] = np.asarray(value)
            row = dict(
                arm=args.arm,
                step=step,
                slice=index,
                width=args.width,
                residual=np.asarray(result.residual).tolist(),
                converged=np.asarray(result.converged).tolist(),
                reads=np.asarray(result.reads).tolist(),
                updates=np.asarray(result.steps).tolist(),
                refactors=np.asarray(result.refactors).tolist(),
                batch_ms=wall,
                minor_radius=machine.minor_radius,
            )
            print("SLICE", json.dumps(row), flush=True)
            assert np.all(result.converged) and np.max(result.residual) <= 1e-10
            rows.append(row)
            receipts.persist(
                args.out, rows, arrays, metadata, complete=(len(rows) == 14)
            )
            older, old_drive = current, drive
            current, drive, stationary = (
                result.moments,
                target,
                result.support.stationary.position,
            )
            if args.arm == "head":
                factor, valid = result.factor, result.factor_valid
            if args.warmup and index == 2:
                print("SUMMARY warmup complete", flush=True)
                return
    print("SUMMARY 14 slices certified; residual <= 1e-10", flush=True)


if __name__ == "__main__":
    main()
