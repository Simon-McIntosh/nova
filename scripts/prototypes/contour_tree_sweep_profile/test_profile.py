"""Receipt identity and visit-order sensitivity of the sweep instrument."""

from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from scripts.prototypes.contour_tree_sweep_profile.profile import (
    assert_identity,
    assert_sweep_identity,
    inputs,
    instrumented_sweep,
    load_builder,
    replay_build,
    roots_with_depth,
)


def test_pointer_counter_sees_known_chain() -> None:
    roots, depth = roots_with_depth(
        jnp.array([1, 2, 2], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    assert int(depth) == 2
    assert int(roots[0]) == 2


@pytest.mark.parametrize("kind", ("limited", "diverted"))
@pytest.mark.parametrize("cells", (550, 2000, 5000, 10000))
def test_instrumented_receipt_identity(kind: str, cells: int) -> None:
    module = load_builder(Path(os.environ["CONTOUR_TREE_MODULE"]))
    baseline = load_builder(Path(os.environ["REFERENCE_CONTOUR_TREE_MODULE"]))
    arguments, realised_cells = inputs(kind, cells)
    cpu = jax.local_devices(backend="cpu")[0]
    expected = baseline.build_contour_tree(
        *(jax.device_put(argument, cpu) for argument in arguments)
    )
    jax.block_until_ready(expected)
    reference = module.build_contour_tree(*arguments)
    jax.block_until_ready(reference)
    assert_identity(expected, reference)
    mutation = os.environ.get("MUTATE_VISIT_ORDER") == "1"
    vertex_flux, vertex_valid, vertex_is_wall, edges, edge_valid, sigma = arguments
    signed = jnp.asarray(sigma, dtype=vertex_flux.dtype) * vertex_flux
    adjacency = module._carrier_adjacency(edges, edge_valid, vertex_flux.size)
    neighbours, neighbour_valid, _ = adjacency
    counts = {}
    passes = {}
    for name, descending in (("join", True), ("split", False)):
        sweep_args = (
            signed,
            vertex_valid,
            vertex_is_wall,
            neighbours,
            neighbour_valid,
            descending,
            vertex_flux.size,
            edges.shape[0],
        )
        native = module._sweep_tree(*sweep_args)
        counted, depth = instrumented_sweep(module, sweep_args, mutation=mutation)
        assert_sweep_identity(native, counted)
        assert depth["visits"] == vertex_flux.size
        counts[name] = depth
        passes[name] = counted
    candidate = replay_build(module, arguments, adjacency, passes)
    assert_identity(reference, candidate)
    if not mutation:
        output = Path(os.environ["PROFILE_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        (output / f"counts-{kind}-{cells}.json").write_text(
            json.dumps(
                {
                    "case": kind,
                    "requested_cells": cells,
                    "realised_cells": realised_cells,
                    "vertices": int(vertex_flux.size),
                    "counts": counts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
