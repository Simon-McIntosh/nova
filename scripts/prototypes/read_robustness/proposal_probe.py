"""Measure candidate coverage of the own-cell and edge-neighbour proposals."""

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

from scripts.prototypes.fast_map_production import tie_probe


def candidates(field, geometry):
    coefficients = np.asarray(field.sampled_coefficient(geometry))
    hessian = np.stack(
        (
            np.stack((2 * coefficients[:, 3], coefficients[:, 4]), axis=-1),
            np.stack((coefficients[:, 4], 2 * coefficients[:, 5]), axis=-1),
        ),
        axis=1,
    )
    displacement = -np.linalg.solve(hessian, coefficients[:, 1:3, None])[..., 0]
    seeds = (
        np.asarray(geometry.centre) + np.asarray(geometry.pitch)[:, None] * displacement
    )
    polygons = np.asarray(geometry.vertices)
    counts = np.asarray(geometry.vertex_count)
    neighbours = np.asarray(geometry.neighbour)
    ring = np.concatenate((np.arange(len(seeds))[:, None], neighbours), axis=1)
    valid = ring >= 0
    ring = np.maximum(ring, 0)
    vertices = polygons[ring]
    vertex_count = counts[ring]
    slot = np.arange(vertices.shape[2])
    following = np.where(
        slot[None, None, :] + 1 < vertex_count[:, :, None], slot + 1, 0
    )
    end = np.take_along_axis(vertices, following[..., None], axis=2)
    edge = end - vertices
    relative = seeds[:, None, None, :] - vertices
    cross = edge[..., 0] * relative[..., 1] - edge[..., 1] * relative[..., 0]
    live = slot[None, None, :] < vertex_count[:, :, None]
    inside = np.all(np.where(live, cross >= -1e-12, True), axis=2) & valid
    return seeds, inside


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    import nova

    print(
        f"nova={nova.__file__} cwd={Path.cwd().resolve()} "
        f"backend={jax.default_backend()}",
        flush=True,
    )
    rows = []
    for requested, offsets in (
        (250, (0.1732, 0.17325, 0.1735)),
        (500, (0.122, 0.1225, 0.123)),
    ):
        state = tie_probe.build("diverted", requested, args.cache)
        for offset in offsets:
            varied = replace(
                state["jets"],
                exterior=state["jets"].exterior
                + jnp.asarray(offset * state["scale"]) * state["jet_slope"],
            )
            seeds, inside = candidates(
                varied.field(state["reference"]), state["geometry"]
            )
            row = {
                "cells": len(seeds),
                "offset": offset,
                "own": int(inside[:, 0].sum()),
                "ring": int(inside.any(axis=1).sum()),
                "own_indices": np.flatnonzero(inside[:, 0]).tolist(),
                "ring_indices": np.flatnonzero(inside.any(axis=1)).tolist(),
            }
            rows.append(row)
            print(row, flush=True)
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
