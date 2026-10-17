"""Measure derivative and continuity checks at delivered conic-owner roots."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np

import nova
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.xpoint_cell_owner.runtime_solve import (
    DIRECTION_BLOCK,
    factor,
    residual,
    without_profile,
)


def save(path, result):
    path.write_text(json.dumps(result, indent=2) + "\n")


def gauss_newton_gradient(point, operands, scale, profile, component="halfplane"):
    """Accumulate the transposed Jacobian product from forward columns."""
    value = residual(point, operands, scale, profile, component)
    count = point.size
    blocks = (count + DIRECTION_BLOCK - 1) // DIRECTION_BLOCK
    columns = jnp.zeros((blocks * DIRECTION_BLOCK, count), point.dtype)

    def body(block, packed):
        indices = block * DIRECTION_BLOCK + jnp.arange(DIRECTION_BLOCK)
        directions = (indices[:, None] == jnp.arange(count)[None]).astype(point.dtype)

        def column(direction):
            return jax.jvp(
                lambda trial: residual(trial, operands, scale, profile, component),
                (point,),
                (direction,),
            )[1]

        products = jax.vmap(column)(directions)
        return jax.lax.dynamic_update_slice(
            packed, products, (block * DIRECTION_BLOCK, 0)
        )

    columns = jax.lax.fori_loop(0, blocks, body, columns)
    return value, columns[:count] @ value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--start-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--component",
        choices=(
            "halfplane",
            "bell",
            "saddle",
            "pinned",
            "anchored",
            "shifted",
            "ordinary",
            "oriented",
        ),
        default="halfplane",
    )
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() in ("gpu", "cpu")
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    saved = json.loads(args.start_json.read_text())
    if "terminal_state" not in saved or not saved.get("history"):
        raise ValueError("root checks require a delivered cold state")
    context = build(args.cells, args.cache)
    state = jnp.asarray(saved["terminal_state"]["moments"]).reshape(-1)
    state = state / jnp.tile(context.scale, len(context.reference))
    operands = without_profile(context.operands)
    profile = context.profile
    scale = context.scale
    mapped = jax.jit(
        lambda point, carrier, unit: residual(
            point, carrier, unit, profile, args.component
        )
    )
    value, gradient = jax.jit(
        lambda point, carrier, unit: gauss_newton_gradient(
            point, carrier, unit, profile, args.component
        )
    )(state, operands, scale)
    value = np.asarray(value)
    result = {
        "cells": len(context.reference),
        "cold_converged": bool(saved.get("converged")),
        "residual": float(np.max(np.abs(value))),
        "gauss_newton_gradient_two": float(jnp.linalg.norm(gradient)),
        "taylor": [],
        "continuity": [],
    }
    save(args.output, result)
    print(
        f"ROOT cells={result['cells']} component={args.component} "
        f"residual={result['residual']:.12g} "
        f"gradient={result['gauss_newton_gradient_two']:.12g}",
        flush=True,
    )
    points = jnp.arange(state.size, dtype=state.dtype)
    random_directions = [
        jnp.sin(points * frequency + phase)
        for frequency, phase in ((0.13, 0.3), (0.29, 1.4), (0.47, 2.7))
    ]
    random_directions = [item / jnp.linalg.norm(item) for item in random_directions]
    steps = (1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7)
    tangent_program = jax.jit(
        lambda point, direction, carrier, unit: jax.jvp(
            lambda trial: residual(trial, carrier, unit, profile, args.component),
            (point,),
            (direction,),
        )
    )
    for index, direction in enumerate(random_directions):
        centre, tangent = tangent_program(state, direction, operands, scale)
        remainders = []
        for step in steps:
            trial = mapped(state + step * direction, operands, scale)
            error = np.asarray((trial - centre - step * tangent).reshape(-1, 3))
            remainders.append(
                [float(np.linalg.norm(error[:, block])) for block in range(3)]
            )
        slopes = [
            [
                float(np.log(left / right) / np.log(first / second))
                if left > 0 and right > 0
                else None
                for left, right in zip(previous, following, strict=True)
            ]
            for first, second, previous, following in zip(
                steps[:-1], steps[1:], remainders[:-1], remainders[1:], strict=True
            )
        ]
        row = {
            "direction": index,
            "steps": steps,
            "remainders": remainders,
            "slopes": slopes,
        }
        result["taylor"].append(row)
        save(args.output, result)
        print(f"TAYLOR {json.dumps(row)}", flush=True)

    lu, pivots = jax.jit(
        lambda point, carrier, unit: factor(
            point, carrier, unit, profile, args.component
        )
    )(state, operands, scale)
    newton = jsp_linalg.lu_solve((lu, pivots), -jnp.asarray(value))
    newton = newton / jnp.linalg.norm(newton)
    directions = [("newton", newton)] + [
        (f"random-{index}", direction)
        for index, direction in enumerate(random_directions[:2])
    ]
    offsets = (-1e-5, -3e-6, -1e-6, 0.0, 1e-6, 3e-6, 1e-5)
    for name, direction in directions:
        centre, tangent = tangent_program(state, direction, operands, scale)
        rows = []
        for offset in offsets:
            trial = mapped(state + offset * direction, operands, scale)
            difference = np.asarray(trial - centre)
            remainder = np.asarray(trial - centre - offset * tangent)
            rows.append(
                {
                    "step": offset,
                    "difference_inf": float(np.max(np.abs(difference))),
                    "linearized_error_inf": float(np.max(np.abs(remainder))),
                }
            )
        result["continuity"].append({"direction": name, "rows": rows})
        save(args.output, result)
        print(f"CONTINUITY direction={name} rows={json.dumps(rows)}", flush=True)
    result["taylor_pass"] = all(
        slope is not None and 1.8 <= slope <= 2.2
        for direction in result["taylor"]
        for row in direction["slopes"][1:4]
        for slope in row
    )
    result["continuity_finite"] = all(
        np.isfinite(row["difference_inf"]) and np.isfinite(row["linearized_error_inf"])
        for direction in result["continuity"]
        for row in direction["rows"]
    )
    save(args.output, result)
    print(
        f"SUMMARY taylor_pass={result['taylor_pass']} "
        f"continuity_finite={result['continuity_finite']}",
        flush=True,
    )
    if not result["taylor_pass"] or not result["continuity_finite"]:
        raise ValueError("root derivative or continuity check failed")


if __name__ == "__main__":
    main()
