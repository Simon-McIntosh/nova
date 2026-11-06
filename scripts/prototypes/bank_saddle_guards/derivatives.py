"""Compare saddle derivatives on shared trial and converged certificate states."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import fixed_points, support_read, topology
from scripts.prototypes.bank_solve_profile.fixture import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--arm", choices=("base", "head"), required=True)
    args = parser.parse_args()
    print(
        f"revision={args.revision} nova={nova.__file__} "
        f"module={fixed_points.__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    assert jax.config.jax_enable_x64
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
        with np.load(args.run / "derivatives-base.npz") as reference:
            for key, value in arrays.items():
                error = float(np.max(np.abs(value - reference[key])))
                print(
                    f"PARITY {key} absolute_error={error:.17g} limit=1e-15", flush=True
                )
                np.testing.assert_allclose(value, reference[key], atol=1e-15, rtol=0)


if __name__ == "__main__":
    main()
