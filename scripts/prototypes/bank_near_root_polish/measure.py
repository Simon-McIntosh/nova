"""Measure common-seed perturbations with one compiled sixteen-lane solve."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.bank_solve_read import measure as reference

support_read = reference.support_read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--process", type=int, required=True)
    args = parser.parse_args()
    reference.emit(
        "HEADER",
        revision=args.revision,
        nova=reference.nova.__file__,
        module=support_read.__file__,
        cwd=str(Path.cwd().resolve()),
        x64=bool(jax.config.jax_enable_x64),
        devices=str(jax.devices()),
    )
    assert jax.config.jax_enable_x64
    bank, cache, roots, convention, policy, profile, quadrature, moments, analytic = (
        reference.machine("diverted")
    )
    common = args.run / "common-seeds.npz"
    if not common.exists():
        assert args.arm == "base" and args.process == 0
        scales = jnp.linspace(2e-5, 6e-5, 16)
        states = tuple(
            np.asarray(value[None] * (1 + scales[:, None])) for value in moments
        )
        seeds = np.asarray(
            jnp.broadcast_to(
                roots + 0.03 * jnp.median(bank.geometry.pitch), (16, *roots.shape)
            )
        )
        previous = np.asarray(jnp.broadcast_to(roots, seeds.shape))
        rng = np.random.default_rng(1931)
        payload = dict(seeds=seeds, previous=previous)
        for variant in range(6):
            for channel, state in enumerate(states):
                noise = rng.uniform(-1, 1, state.shape)
                noise /= np.max(np.abs(noise))
                payload[f"state_{variant}_{channel}"] = (
                    state.copy()
                    if variant == 0
                    else state + 1e-14 * np.max(np.abs(state)) * noise
                )
        np.savez(common, **payload)
    with np.load(common) as saved:
        states = [
            tuple(jnp.asarray(saved[f"state_{v}_{c}"]) for c in range(3))
            for v in range(6)
        ]
        seeds, previous = jnp.asarray(saved["seeds"]), jnp.asarray(saved["previous"])
    reference.emit(
        "COMMON_SEEDS", sha256=hashlib.sha256(common.read_bytes()).hexdigest()
    )

    def solve(state, operands, seed, prior):
        return support_read.solve_batch(
            operands,
            cache,
            state,
            seed,
            prior,
            convention,
            policy,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
            max_steps=60,
        )

    started = perf_counter()
    program = jax.jit(solve).lower(states[0], bank, seeds, previous).compile()
    reference.emit("COMPILE", seconds=perf_counter() - started)
    jax.block_until_ready(program(states[0], bank, seeds, previous))
    _, _, wall, _, _ = reference.fixture._analytic_inputs("diverted")
    receipt = dict(revision=args.revision, process=args.process, arm=args.arm, rows=[])
    output = args.run / f"{args.arm}-{args.process}.json"
    for variant, state in enumerate(states):
        walls = []
        for _ in range(3):
            started = perf_counter()
            result = jax.block_until_ready(program(state, bank, seeds, previous))
            walls.append(1000 * (perf_counter() - started))
        reading = result.support.reading
        oracle, oracle_roots = reference.analytic_oracle(
            analytic, roots, jax.tree.map(lambda v: v[0], reading), wall
        )
        flux = jax.vmap(lambda s: support_read.nodal_jets(bank, s)[:, 0])(
            result.moments
        )
        row = dict(
            variant=variant,
            lanes=16,
            call=variant,
            wall_ms=float(np.median(walls)),
            walls_ms=walls,
            reads=np.asarray(result.reads).tolist(),
            steps=np.asarray(result.steps).tolist(),
            converged=np.asarray(result.converged).tolist(),
            residual=np.asarray(result.residual).tolist(),
            residual_path=np.asarray(result.residual_path).tolist(),
            axis=np.asarray(reading.axis).tolist(),
            saddle=np.asarray(reading.boundary).tolist(),
            boundary_flux=np.asarray(reading.boundary_flux).tolist(),
            flux=np.asarray(flux).tolist(),
            span=oracle["flux_span"],
            minor_radius=oracle["minor_radius"],
            oracle=oracle,
            oracle_roots=np.asarray(oracle_roots).tolist(),
        )
        receipt["rows"].append(row)
        output.write_text(json.dumps(receipt))
        reference.emit(
            "ROW",
            **{
                k: row[k]
                for k in (
                    "variant",
                    "reads",
                    "steps",
                    "residual",
                    "converged",
                    "wall_ms",
                )
            },
        )
    reference.emit("COMPLETE", reads=[row["reads"][0] for row in receipt["rows"]])


if __name__ == "__main__":
    main()
