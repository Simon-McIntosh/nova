"""Measure staged bank trajectories and their fresh-state precision."""

from scripts.prototypes.bank_solve_read import measure as reference

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

support_read = reference.support_read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--lanes", type=int, nargs="+", default=[1, 16])
    args = parser.parse_args()
    reference.emit(
        "HEADER",
        revision=args.revision,
        cwd=str(Path.cwd().resolve()),
        nova=reference.nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64),
        job=os.environ.get("SLURM_JOB_ID"),
    )
    assert jax.config.jax_enable_x64
    bank, cache, roots, convention, policy, profile, quadrature, moments, analytic = (
        reference.machine("diverted")
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    _, _, wall, _, _ = reference.fixture._analytic_inputs("diverted")
    receipt = {"revision": args.revision, "module": support_read.__file__, "rows": []}
    for lanes in args.lanes:
        scales = jnp.linspace(2e-5, 6e-5, lanes)
        states = tuple(value[None] * (1 + scales[:, None]) for value in moments)
        seeds_batch = jnp.broadcast_to(seeds, (lanes, *seeds.shape))
        roots_batch = jnp.broadcast_to(roots, (lanes, *roots.shape))

        def run(state, operands):
            return support_read.solve_batch(
                operands,
                cache,
                state,
                seeds_batch,
                roots_batch,
                convention,
                policy,
                profile,
                lambda radius, flux: jnp.zeros_like(radius),
                quadrature,
                max_steps=60,
                reuse_factor=False,
                contraction_limit=0.5,
            )

        start = perf_counter()
        program = jax.jit(run).lower(states, bank).compile()
        reference.emit("COMPILE", lanes=lanes, seconds=perf_counter() - start)
        for call in range(2):
            start = perf_counter()
            result = jax.block_until_ready(program(states, bank))
            wall_ms = 1000 * (perf_counter() - start)
            reading = result.support.reading
            lane_read = jax.tree.map(lambda value: value[0], reading)
            oracle, oracle_roots = reference.analytic_oracle(
                analytic, roots, lane_read, wall
            )
            flux = jax.vmap(lambda state: support_read.nodal_jets(bank, state)[:, 0])(
                result.moments
            )
            row = dict(
                lanes=lanes,
                call=call,
                wall_ms=wall_ms,
                per_slice_ms=wall_ms / lanes,
                steps=np.asarray(result.steps).tolist(),
                residual=np.asarray(result.residual).tolist(),
                converged=np.asarray(result.converged).tolist(),
                freezes=np.asarray(result.freezes).tolist(),
                unfreezes=np.asarray(result.unfreezes).tolist(),
                residual_path=np.asarray(result.residual_path).tolist(),
                update_path=np.asarray(result.update_path).tolist(),
                axis=np.asarray(reading.axis).tolist(),
                saddle=np.asarray(reading.boundary).tolist(),
                boundary_flux=np.asarray(reading.boundary_flux).tolist(),
                span=oracle["flux_span"],
                minor_radius=oracle["minor_radius"],
                flux=np.asarray(flux).tolist(),
                oracle=oracle,
                oracle_roots=np.asarray(oracle_roots).tolist(),
            )
            receipt["rows"].append(row)
            args.output.write_text(json.dumps(receipt, indent=2))
            reference.emit(
                "SOLVE",
                **{
                    key: row[key]
                    for key in (
                        "lanes",
                        "call",
                        "wall_ms",
                        "per_slice_ms",
                        "steps",
                        "residual",
                        "converged",
                        "unfreezes",
                    )
                },
            )
            assert all(row["converged"]) and max(row["residual"]) <= 1e-10


if __name__ == "__main__":
    main()
