"""Compare the map JVP with a detached booked-moment handoff."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.fine_grid_convergence import handoff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--start-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    context = build(args.cells, args.cache)
    saved = json.loads(args.start_json.read_text())
    base = jnp.asarray(saved["terminal_state"]["moments"]) / context.scale
    direction = jnp.sin(jnp.arange(base.size).reshape(base.shape) + 1)
    direction = direction / jnp.linalg.norm(direction)

    def equation(point):
        state = context.map(point * context.scale)
        return state.moment_residual / context.scale

    def measure():
        value, tangent = jax.jvp(equation, (base,), (direction,))
        rows = []
        for step in (1e-4, 3e-5, 1e-5):
            trial = equation(base + step * direction)
            remainder = np.asarray(trial - value - step * tangent)
            rows.append([float(np.linalg.norm(remainder[:, col])) for col in range(3)])
        slopes = [
            [
                float(
                    np.log(rows[first][col] / rows[first + 1][col])
                    / np.log((1e-4, 3e-5)[first] / (3e-5, 1e-5)[first])
                )
                for col in range(3)
            ]
            for first in range(2)
        ]
        return {"remainders": rows, "slopes": slopes}

    positive = measure()

    real_weight = handoff.booked_blend
    handoff.booked_blend = lambda gap: jax.lax.stop_gradient(real_weight(gap))
    negative = measure()
    handoff.booked_blend = real_weight
    positive_pass = all(
        1.7 < slope < 2.3 for row in positive["slopes"] for slope in row
    )
    negative_pass = all(
        1.7 < slope < 2.3 for row in negative["slopes"] for slope in row
    )
    result = {
        "cells": len(context.reference),
        "positive": positive,
        "detached_handoff_weight": negative,
        "positive_pass": positive_pass,
        "negative_pass": negative_pass,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"CONTROL positive_pass={positive_pass} negative_pass={negative_pass}",
        flush=True,
    )
    if not positive_pass or negative_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
