"""Paired scalar booking and complete native read timing with runtime operands."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nova.equilibrium import current, support_read as sr
    from scripts.prototypes.bank_newton_step.measure import machine

    assert jax.config.jax_enable_x64 is True
    assert Path(current.__file__).resolve().is_relative_to(root)
    payload = dict(
        revision=args.revision,
        module=current.__file__,
        cwd=str(Path.cwd().resolve()),
        instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        backend=jax.default_backend(),
        job=os.environ.get("SLURM_JOB_ID"),
        tile=current._SCALAR_CUT_BATCH_WIDTH,
        rows=[],
        complete=False,
    )
    print("HEADER", json.dumps(payload), flush=True)
    args.output.write_text(json.dumps(payload, indent=2))
    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine(
        "diverted"
    )

    def booking(bank, state, support, rule):
        def one(values, reading):
            psi = sr.nodal_jets(bank, values)[bank.sample_index, 0]
            return sr.book(bank, reading, psi, profile, rule)

        return jax.vmap(one)(state, support)

    def whole(bank, cache, state, seeds, previous, tracking, rule):
        support = sr.read_batch(
            bank, cache, state, seeds, previous, convention, policy, tracking=tracking
        )
        return support, booking(bank, state, support, rule)

    for width in (1, 16):
        scales = jnp.linspace(2e-5, 6e-5, 16)[:width]
        state = tuple(value[None] * (1 + scales[:, None]) for value in moments)
        seeds = jnp.broadcast_to(
            roots + 0.03 * jnp.median(bank.geometry.pitch), (width, *roots.shape)
        )
        prior = jnp.broadcast_to(roots, seeds.shape)
        initial = jax.block_until_ready(
            jax.jit(whole)(bank, cache, state, seeds, prior, None, quadrature)
        )[0]
        assert bool(jnp.all(initial.valid))
        cut = (initial.reading.membership > 0) & (
            initial.reading.membership < 1 - 1e-14
        )
        conic = cut & ~initial.reading.normal_form_cells
        counts = np.asarray(jnp.sum(conic, axis=-1)).tolist()
        assert min(counts) > 32
        for kind in ("booking", "carried", "full"):
            if kind == "booking":
                fn, values = booking, (bank, state, initial, quadrature)
            else:
                tracking = sr.tracking_of(initial) if kind == "carried" else None
                fn, values = (
                    whole,
                    (bank, cache, state, seeds, prior, tracking, quadrature),
                )
            started = time.perf_counter()
            program = jax.jit(fn).lower(*values).compile()
            compile_seconds = time.perf_counter() - started
            result = jax.block_until_ready(program(*values))
            samples = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = jax.block_until_ready(program(*values))
                samples.append(1000 * (time.perf_counter() - started))
            booked = result if kind == "booking" else result[1]
            assert bool(jnp.all(booked.valid)) and bool(
                jnp.all(jnp.isfinite(booked.raw_current))
            )
            if kind != "booking":
                assert bool(jnp.all(result[0].rebuilt == (kind == "full")))
            row = dict(
                width=width,
                kind=kind,
                median_ms=statistics.median(samples),
                samples_ms=samples,
                compile_seconds=compile_seconds,
                conic_population=counts,
                conic_buckets=[
                    (n + payload["tile"] - 1) // payload["tile"] for n in counts
                ],
                raw_current=np.asarray(booked.raw_current).tolist(),
            )
            payload["rows"].append(row)
            args.output.write_text(json.dumps(payload, indent=2))
            print("MEASURE", json.dumps(row), flush=True)
    payload["complete"] = True
    args.output.write_text(json.dumps(payload, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
