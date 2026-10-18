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
from scripts.prototypes.read_robustness import census as census_module
from scripts.prototypes.read_robustness.census_base_step import containment_only_census


def candidates(field, state, census_function):
    """Read the candidate count after the census's own admission checks."""
    return census_function(
        field,
        state["geometry"],
        state["convention"],
        state["policy"],
        state["previous"],
    ).candidates


def counter(state, census_function):
    """Evaluate one census arm at offsets on the same moving field."""

    @jax.jit
    def count(offset):
        varied = replace(
            state["jets"],
            exterior=state["jets"].exterior + offset * state["jet_slope"],
        )
        return candidates(varied.field(state["reference"]), state, census_function)

    return count


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
    own_census = containment_only_census()
    rows = []
    for requested, offsets in (
        (250, (0.1732, 0.17325, 0.1735)),
        (500, (0.122, 0.1225, 0.123)),
    ):
        state = tie_probe.build("diverted", requested, args.cache)
        own_count = counter(state, own_census)
        ring_count = counter(state, census_module.census)
        geometry = state["geometry"]
        reserve = min(
            len(geometry.centre),
            max(
                state["policy"].null_capacity * (geometry.neighbour.shape[1] + 1),
                state["previous"].shape[0],
            ),
        )
        for offset in offsets:
            shifted = jnp.asarray(offset * state["scale"])
            row = {
                "cells": len(geometry.centre),
                "offset": offset,
                "own": int(np.asarray(own_count(shifted))),
                "ring": int(np.asarray(ring_count(shifted))),
                "reserve": reserve,
            }
            rows.append(row)
            print(row, flush=True)
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
