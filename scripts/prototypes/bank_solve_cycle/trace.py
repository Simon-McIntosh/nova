"""Observe read geometry in the device loop without retranscribing its updates."""

from scripts.prototypes.bank_solve_read import measure as reference

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    support_read = reference.support_read
    print("MODULE", support_read.__file__, flush=True)
    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = (
        reference.machine("diverted")
    )
    lanes = 16
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    states = tuple(
        value[None] * (1 + jnp.linspace(2e-5, 6e-5, lanes)[:, None])
        for value in moments
    )
    points = jnp.broadcast_to(roots, (lanes, *roots.shape))
    seed_points = jnp.broadcast_to(seeds, (lanes, *seeds.shape))
    original = support_read.read_batch
    observations = []
    fields = (
        "membership",
        "edge_interval",
        "fragment_area",
        "fragment_selected",
        "normal_form_cells",
        "field_coefficients",
        "boundary_flux",
    )

    def observe(values):
        observations.append(
            {
                name: np.asarray(value).copy()
                for name, value in zip(fields, values, strict=True)
            }
        )

    def read(*args, **kwargs):
        result = original(*args, **kwargs)
        jax.debug.callback(
            observe, tuple(getattr(result.reading, key) for key in fields), ordered=True
        )
        return result

    support_read.read_batch = read
    result = jax.jit(
        lambda state, operands: support_read.solve_batch(
            operands,
            cache,
            state,
            seed_points,
            points,
            convention,
            policy,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
        )
    )(states, bank)
    jax.block_until_ready(result)
    jax.effects_barrier()
    path = np.asarray(result.residual_path)
    rows = []
    for index, (left, right) in enumerate(zip(observations, observations[1:])):
        fields_changed = {}
        for field in fields:
            a, b = left[field], right[field]
            delta = np.abs(a.astype(float) - b.astype(float)).reshape(lanes, -1)
            fields_changed[field] = {
                "max": np.max(delta, axis=1).tolist(),
                "changed": np.count_nonzero(delta, axis=1).tolist(),
            }
        rows.append(
            {
                "iterate": index + 1,
                "residual_before": path[:, index].tolist(),
                "residual_after": path[:, index + 1].tolist(),
                "fields": fields_changed,
            }
        )
    args.output.write_text(
        json.dumps({"steps": np.asarray(result.steps).tolist(), "rows": rows}, indent=2)
    )
    print("TRACE", len(observations), "STEPS", np.asarray(result.steps), flush=True)


if __name__ == "__main__":
    main()
