"""Compare scalar conic widths on identical precomputed physical flux samples."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import current, support_read as sr
from scripts.prototypes.bank_newton_step.measure import machine


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    bank, cache, roots, convention, policy, profile, rule, moments, _ = machine(
        "diverted"
    )
    scales = jnp.linspace(2e-5, 6e-5, 16)
    state = tuple(value[None] * (1 + scales[:, None]) for value in moments)
    seeds = jnp.broadcast_to(roots, (16, *roots.shape))
    support = jax.jit(
        lambda b, c, m, s: sr.read_batch(b, c, m, s, s, convention, policy)
    )(bank, cache, state, seeds)
    psi = jax.jit(
        jax.vmap(lambda state: sr.nodal_jets(bank, state)[bank.sample_index, 0])
    )(state)
    geometry = bank.geometry
    payload = dict(
        instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        module=current.__file__,
        cwd=str(Path.cwd().resolve()),
        rows=[],
    )
    print("HEADER", json.dumps(payload), flush=True)
    reference = None
    for width in (32, 64, 128, 256):
        with patch.object(current, "_SCALAR_CUT_BATCH_WIDTH", width):
            program = (
                jax.jit(
                    lambda psi, read, geometry, rule: jax.vmap(
                        lambda p, r: current.book(p, r, geometry, profile, rule)
                    )(psi, read)
                )
                .lower(psi, support.reading, geometry, rule)
                .compile()
            )
        result = jax.block_until_ready(program(psi, support.reading, geometry, rule))
        assert bool(jnp.all(result.valid))
        if reference is None:
            reference = jax.tree.map(np.asarray, result)
        for a, b in zip(result, reference, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            jax.block_until_ready(program(psi, support.reading, geometry, rule))
            samples.append(1000 * (time.perf_counter() - start))
        row = dict(
            width=width, median_ms=statistics.median(samples), samples_ms=samples
        )
        payload["rows"].append(row)
        args.output.write_text(json.dumps(payload, indent=2))
        print("MEASURE", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
