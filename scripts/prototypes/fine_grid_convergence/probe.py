"""Measure map derivatives and the saddle handoff at a stalled state."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

import nova
from nova.equilibrium import topology
from scripts.prototypes.fast_map_production import kernel, program
from scripts.prototypes.fast_read.ordinary_cut import bell_moments
from scripts.prototypes.fine_grid_convergence.context import build


def _normal_fraction(context, reading, indices):
    geometry = context.geometry
    form = reading.saddle_form
    fragments = jax.vmap(
        topology._normal_form_cell_fragments, in_axes=(0, 0, None, None)
    )(
        geometry.vertices[indices],
        geometry.vertex_count[indices],
        form,
        context.policy.edge_tolerance,
    )
    sector = kernel._local_axis_sector(geometry, reading.axis, form)
    positive = jnp.nonzero(form.positive, size=2, fill_value=0)[0]
    selected = positive == sector
    area = jnp.sum(jnp.where(selected[None], fragments.area[:, positive], 0), axis=1)
    return area / geometry.full_area[indices]


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
    shape = context.reference.shape

    def equation(point):
        return (
            context.map(point.reshape(shape) * context.scale).moment_residual
            / context.scale
        ).reshape(-1)

    residual, factor = jax.linearize(equation, base.reshape(-1))
    residual = np.asarray(residual)
    length = residual.size
    product = jax.jit(factor)
    operator = LinearOperator(
        (length, length),
        matvec=lambda vector: np.asarray(product(jnp.asarray(vector))),
        dtype=np.float64,
    )
    direction, code = gmres(
        operator, -residual, restart=48, maxiter=6, rtol=1e-10, atol=0
    )
    print(
        f"linear_code={code} residual={np.max(np.abs(residual)):.12g} "
        f"direction_norm={np.linalg.norm(direction):.12g}",
        flush=True,
    )
    generators = [("newton", direction)]
    for seed in (17, 29, 43):
        random = np.random.default_rng(seed).standard_normal(length)
        generators.append((f"random-{seed}", random / np.linalg.norm(random)))

    def details(point):
        moments = point.reshape(shape) * context.scale
        reading = context.reading(moments)
        ordinary, _, four = bell_moments(
            context.geometry, reading, context.profile, context.convention.sigma
        )
        fraction, _, _ = kernel.bell_fraction_with_crossings(
            context.geometry, reading, context.convention.sigma
        )
        saddle, _ = program._saddle_moments(context.geometry, reading, context.profile)
        return reading, fraction, ordinary, saddle, four

    detailed = jax.jit(details)
    reading, fraction, ordinary, saddle, four = detailed(base.reshape(-1))
    frame = np.asarray(reading.normal_form_cells | four)
    frame_ids = np.flatnonzero(frame)
    neighbours = np.asarray(context.geometry.neighbour)[frame_ids].reshape(-1)
    indices = np.unique(np.r_[frame_ids, neighbours[neighbours >= 0]].astype(np.int32))
    indices = indices[:32]
    normal_fraction = _normal_fraction(context, reading, indices)
    result = {
        "cells": len(context.reference),
        "stalled_residual": float(np.max(np.abs(residual))),
        "linear_code": int(code),
        "linear_residual_ratio": float(
            np.linalg.norm(operator @ direction + residual) / np.linalg.norm(residual)
        ),
        "direction_norm": float(np.linalg.norm(direction)),
        "directions": {},
        "handoff": [],
    }
    for name, vector in generators:
        tangent = np.asarray(product(jnp.asarray(vector))).reshape(shape)
        base_block = residual.reshape(shape)
        rows = []
        for step in np.logspace(-2, -8, 7):
            trial = np.asarray(equation(base.reshape(-1) + step * vector)).reshape(
                shape
            )
            remainder = trial - base_block - step * tangent
            rows.append(
                {
                    "step": float(step),
                    "blocks": [
                        float(np.linalg.norm(remainder[:, col])) for col in range(3)
                    ],
                    "hot_cell": int(np.argmax(np.max(np.abs(remainder), axis=1))),
                    "max_remainder": float(np.max(np.abs(remainder))),
                }
            )
        result["directions"][name] = rows
        print(
            f"TAYLOR {name} h=1e-4 blocks={rows[2]['blocks']} "
            f"h=1e-5 blocks={rows[3]['blocks']}",
            flush=True,
        )
    step = 1e-6
    offsets = []
    for sign in (-1, 1):
        offsets.append(
            detailed(base.reshape(-1) + sign * step * jnp.asarray(direction))
        )
    left_normal = np.asarray(_normal_fraction(context, offsets[0][0], indices))
    right_normal = np.asarray(_normal_fraction(context, offsets[1][0], indices))
    for offset, index in enumerate(indices):
        normal = float(normal_fraction[offset])
        member = float(reading.geometric_membership[index])
        ordinary_area = float(fraction[index])
        ordinary_moment = np.asarray(ordinary[index])
        saddle_moment = np.asarray(saddle[index])
        left, right = offsets
        row = {
            "cell": int(index),
            "booking": bool(reading.booking_cells[index]),
            "four_crossings": bool(four[index]),
            "path_weight": float(reading.path_weight[index]),
            "normal_membership": normal,
            "normal_membership_derivative_left": float(
                (normal - left_normal[offset]) / step
            ),
            "normal_membership_derivative_right": float(
                (right_normal[offset] - normal) / step
            ),
            "ordinary_membership": ordinary_area,
            "ordinary_membership_derivative_left": float(
                (ordinary_area - float(left[1][index])) / step
            ),
            "ordinary_membership_derivative_right": float(
                (float(right[1][index]) - ordinary_area) / step
            ),
            "selected_membership": member,
            "ordinary_moments": ordinary_moment.tolist(),
            "normal_moments": saddle_moment.tolist(),
            "ordinary_derivative_left": (
                (ordinary_moment - np.asarray(left[2][index])) / step
            ).tolist(),
            "ordinary_derivative_right": (
                (np.asarray(right[2][index]) - ordinary_moment) / step
            ).tolist(),
            "normal_derivative_left": (
                (saddle_moment - np.asarray(left[3][index])) / step
            ).tolist(),
            "normal_derivative_right": (
                (np.asarray(right[3][index]) - saddle_moment) / step
            ).tolist(),
        }
        result["handoff"].append(row)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    cell_count = len(context.reference)
    print(
        f"SUMMARY cells={cell_count} residual={result['stalled_residual']:.12g} "
        f"linear_residual_ratio={result['linear_residual_ratio']:.12g} "
        f"handoff_cells={len(indices)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
