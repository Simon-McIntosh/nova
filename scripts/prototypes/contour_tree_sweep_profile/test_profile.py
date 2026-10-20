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
    inputs,
    instrumented_build,
    load_builder,
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
    arguments, realised_cells = inputs(kind, cells)
    reference = module.build_contour_tree(*arguments)
    jax.block_until_ready(reference)
    mutation = os.environ.get("MUTATE_VISIT_ORDER") == "1"
    executable, depths = instrumented_build(module, arguments, mutation=mutation)
    candidate = executable(*arguments)
    jax.block_until_ready(candidate)
    counts = {}
    for name, visits in depths.items():
        assert len(visits) == arguments[0].size
        counts[name] = {
            "visits": len(visits),
            "roots_total": sum(visits),
            "roots_max": max(visits),
        }
    if not mutation:
        output = Path(os.environ["PROFILE_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        (output / f"counts-{kind}-{cells}.json").write_text(
            json.dumps(
                {
                    "case": kind,
                    "requested_cells": cells,
                    "realised_cells": realised_cells,
                    "vertices": int(arguments[0].size),
                    "counts": counts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    assert_identity(reference, candidate)
