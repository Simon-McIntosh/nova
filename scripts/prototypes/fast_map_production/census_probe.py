"""Inspect the production Newton root contract at a moving-field tie."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import fixed_points
from scripts.prototypes.fast_map_production import tie_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    import nova

    print(
        f"nova={nova.__file__} fixed_points={fixed_points.__file__} "
        f"cwd={Path.cwd().resolve()} backend={jax.default_backend()}",
        flush=True,
    )
    state = tie_probe.build("diverted", 250, args.cache)
    sampled = tie_probe.make_sampler(state)
    prior, warm_count = tie_probe.warm(state, sampled, -0.2, 0.17, 0.01)
    offsets = (0.17323108126295567, 0.17323108126295570)
    rows = []
    for offset in offsets:
        shifted = jnp.asarray(offset * state["scale"])
        roots, reading, _, _, _, _ = sampled(shifted, prior)
        jets = replace(
            state["jets"],
            exterior=state["jets"].exterior + shifted * state["jet_slope"],
        )
        window = fixed_points.RowCache(
            jets.mesh,
            jets.nodes,
            jnp.arange(len(jets.nodes), dtype=jnp.int32),
            jets.rows,
            jets.exterior,
        )
        seeds = roots.position if not rows else rows[0]["seeds"]
        axis_sign = jnp.sign(roots.jet.hessian[0, 0, 0])
        solved = fixed_points.solve(
            window,
            state["reference"],
            seeds,
            seeds,
            jnp.median(state["geometry"].pitch),
            axis_sign,
            iterations=12,
        )
        jax.block_until_ready(solved)
        rows.append(
            {
                "offset": offset,
                "seeds": np.asarray(seeds),
                "census_valid": np.asarray(roots.valid).tolist(),
                "census_positions": np.asarray(roots.position).tolist(),
                "census_determinants": np.linalg.det(
                    np.asarray(roots.jet.hessian)
                ).tolist(),
                "solved_positions": np.asarray(solved.position).tolist(),
                "solved_valid": np.asarray(solved.valid).tolist(),
                "solved_status": np.asarray(solved.status).tolist(),
                "solved_iterations": np.asarray(solved.iterations).tolist(),
                "solved_gradient": np.linalg.norm(
                    np.asarray(solved.jet.gradient), axis=-1
                ).tolist(),
                "solved_determinants": np.linalg.det(
                    np.asarray(solved.jet.hessian)
                ).tolist(),
                "reading_class": int(reading.boundary_class),
            }
        )
    rows[0]["seeds"] = rows[0]["seeds"].tolist()
    rows[1]["seeds"] = rows[1]["seeds"].tolist()
    result = {
        "cells": len(state["geometry"].centre),
        "warm_count": warm_count,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
