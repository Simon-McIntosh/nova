"""Scan the 282-cell tie window for census departures and the borderline cross."""

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

from nova.equilibrium import topology
from scripts.prototypes.fast_map_production import tie_probe


def cell_cross(field, geometry, policy, cell):
    """Smallest signed cross product of the quadratic seed of one cell."""
    coefficients = field.sampled_coefficient(geometry)
    hessian = jnp.stack(
        (
            jnp.stack((2 * coefficients[:, 3], coefficients[:, 4]), axis=-1),
            jnp.stack((coefficients[:, 4], 2 * coefficients[:, 5]), axis=-1),
        ),
        axis=1,
    )
    _conditioned, safe = jax.vmap(topology._conditioned_hessian, in_axes=(0, None))(
        hessian, policy.hessian_tolerance
    )
    displacement = -jnp.linalg.solve(safe, coefficients[:, 1:3, None])[..., 0]
    seeds = geometry.centre + geometry.pitch[:, None] * displacement
    slot = jnp.arange(geometry.vertices.shape[1])
    following = jnp.where(
        slot[None] + 1 < geometry.vertex_count[:, None], slot[None] + 1, 0
    )
    end = jnp.take_along_axis(geometry.vertices, following[..., None], axis=1)
    edge = end - geometry.vertices
    offset_seed = seeds[:, None] - geometry.vertices
    cross = topology._cross_plane(edge, offset_seed)
    rounding = (
        4.0
        * jnp.finfo(seeds.dtype).eps
        * (jnp.abs(edge[..., 0]) + jnp.abs(edge[..., 1]))
        * (jnp.abs(offset_seed[..., 0]) + jnp.abs(offset_seed[..., 1]))
    )
    live = slot[None] < geometry.vertex_count[:, None]
    row = jnp.where(live, cross, jnp.inf)[cell]
    bound = jnp.where(live, rounding, 0.0)[cell]
    order = jnp.argsort(jnp.where(live, cross, jnp.inf)[cell])
    return row, bound, order, seeds[cell]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scan", default="0.16,0.18,41")
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    import nova

    print(f"nova={nova.__file__} cwd={Path.cwd().resolve()} backend={jax.default_backend()}",
          flush=True)
    state = tie_probe.build("diverted", 250, args.cache)
    sampled = tie_probe.make_sampler(state)
    prior, warm_count = tie_probe.warm(state, sampled, -0.2, 0.16, 0.01)
    geometry = state["geometry"]
    convention = state["convention"]
    policy = state["policy"]

    def counts(offset, carry):
        roots, *_ = sampled(jnp.asarray(offset * state["scale"]), carry)
        eigen = np.linalg.eigvalsh(
            np.asarray(convention.sigma) * np.asarray(roots.jet.hessian)
        )
        valid = np.asarray(roots.valid)
        saddles = valid & (eigen[:, 0] < 0) & (eigen[:, 1] > 0)
        return valid, saddles, np.asarray(roots.position)

    lo, hi, count = (float(part) for part in args.scan.split(","))
    window = []
    carry = prior
    for value in np.linspace(lo, hi, int(count)):
        valid, saddles, position = counts(value, carry)
        window.append(
            {"offset": float(value), "valid": int(valid.sum()),
             "saddles": int(saddles.sum())}
        )
        updated = np.asarray(carry).copy()
        for slot, live in enumerate(valid):
            if live and slot < len(updated):
                updated[slot] = position[slot]
        carry = jnp.asarray(updated)

    fixed = []
    for offset in (0.17323108126295622, 0.17323108126295625):
        valid, saddles, _position = counts(offset, prior)
        fixed.append(
            {"offset": offset, "valid": int(valid.sum()), "saddles": int(saddles.sum())}
        )

    probes = []
    for offset in (0.17323108126295622, 0.17323108126295625):
        shifted = jnp.asarray(offset * state["scale"])
        varied = replace(state["jets"], exterior=state["jets"].exterior + shifted * state["jet_slope"])
        field = varied.field(state["reference"])
        row, bound, order, seed = cell_cross(field, geometry, policy, 263)
        row = np.asarray(row)
        bound = np.asarray(bound)
        order = np.asarray(order)
        probes.append(
            {
                "offset": offset,
                "seed": [float(x) for x in np.asarray(seed)],
                "min_cross": float(row[order[0]]),
                "min_cross_edge": int(order[0]),
                "bound": float(bound[order[0]]),
                "cross_all": [float(x) for x in row],
                "bound_all": [float(x) for x in bound],
            }
        )
    result = {
        "cells": len(geometry.centre),
        "warm_count": warm_count,
        "scan": window,
        "fixed_prior": fixed,
        "probes": probes,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    for item in fixed:
        print("FIXED_PRIOR", repr(item["offset"]), item["valid"], item["saddles"], flush=True)
    for item in window:
        print("SCAN", repr(item["offset"]), item["valid"], item["saddles"], flush=True)
    for item in probes:
        print("PROBE", repr(item["offset"]), "seed", item["seed"], flush=True)
        print("   min_cross", item["min_cross"], "bound", item["bound"], flush=True)
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()