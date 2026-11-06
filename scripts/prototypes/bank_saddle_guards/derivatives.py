"""Compare saddle derivatives on shared trial and converged certificate states."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from pathlib import Path
import shlex
import sys

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import fixed_points, support_read, topology
from scripts.prototypes.bank_solve_profile.fixture import build


def compare(run):
    with np.load(run / "derivatives-base.npz") as reference:
        with np.load(run / "derivatives-head.npz") as candidate:
            assert candidate.files == reference.files
            for key in candidate.files:
                value, expected = candidate[key], reference[key]
                assert np.isfinite(value).all() and np.any(value != 0)
                error = float(np.max(np.abs(value - expected)))
                print(
                    f"PARITY {key} absolute_error={error:.17g} limit=1e-15", flush=True
                )
                np.testing.assert_allclose(value, expected, atol=1e-15, rtol=0)
                altered = value.copy()
                altered.flat[0] += 1e-12
                detected = float(np.max(np.abs(altered - expected)))
                assert detected > 1e-15
                print(
                    f"CONTROL {key} absolute_error={detected:.17g} refused=True",
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--arm", choices=("base", "head"), required=True)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    print(
        f"# revision: {args.revision} tree: {Path.cwd().resolve()} "
        f"command: {shlex.join(sys.orig_argv)}",
        flush=True,
    )
    print(
        f"revision={args.revision} nova={nova.__file__} "
        f"module={fixed_points.__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    assert jax.config.jax_enable_x64
    if args.compare_only:
        compare(args.run)
        return
    batch = build(16)
    assert len(batch.operands.geometry.centre) == 550
    common = args.run / "derivative-inputs.npz"
    if args.arm == "base":
        with np.load(args.states) as saved:
            arrays = {f"trial_{i}": np.asarray(v) for i, v in enumerate(batch.moments)}
            arrays.update(
                {f"converged_{i}": saved[f"moment_{i}"][12] for i in range(3)}
            )
            arrays["roots"] = np.asarray(batch.roots)
            np.savez(common, **arrays)
    with np.load(common) as saved:
        roots = jnp.asarray(saved["roots"])
        states = [
            tuple(jnp.asarray(saved[f"{name}_{i}"]) for i in range(3))
            for name in ("trial", "converged")
        ]
    bank, cache = batch.operands, batch.cache
    pitch = jnp.median(bank.geometry.pitch)

    @jax.jit
    def evaluate(bank, cache, moments, roots):
        solved = fixed_points.solve(
            cache, jnp.stack(moments, axis=-1), roots, roots, pitch, -1.0
        )
        field = fixed_points.field_from_jets(
            support_read.nodal_jets(bank, moments),
            bank.sample_index,
            bank.mesh,
            bank_rows=bank.rows,
            bank_moments=jnp.stack(moments, axis=-1),
            bank_exterior=bank.exterior,
        )
        return solved, topology._curvature_derivatives(field, solved.position[1], pitch)

    arrays = {}
    for name, moments in zip(("trial", "converged"), states):
        solved, values = jax.block_until_ready(evaluate(bank, cache, moments, roots))
        assert bool(solved.valid[1])
        arrays[f"{name}_point"] = np.asarray(solved.position[1])
        for rank, value in zip((3, 4), values):
            arrays[f"{name}_{rank}"] = np.asarray(value)
    np.savez(args.run / f"derivatives-{args.arm}.npz", **arrays)
    if args.arm == "head":
        compare(args.run)


if __name__ == "__main__":
    main()
