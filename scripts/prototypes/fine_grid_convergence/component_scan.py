"""Locate the edge factor that drops support along a Newton direction."""

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
from scripts.prototypes.fast_map_production import kernel

original_path = kernel._continuous_path_weight

from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.fine_grid_convergence.edge_components import edge_components


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
    kernel._continuous_path_weight = original_path
    saved = json.loads(args.start_json.read_text())
    shape = context.reference.shape
    base = jnp.asarray(saved["terminal_state"]["moments"]) / context.scale

    def equation(point):
        state = context.map(point.reshape(shape) * context.scale)
        return (state.moment_residual / context.scale).reshape(-1)

    residual, tangent = jax.linearize(equation, base.reshape(-1))
    residual = np.asarray(residual)
    product = jax.jit(tangent)
    operator = LinearOperator(
        (residual.size, residual.size),
        matvec=lambda vector: np.array(product(jnp.asarray(vector)), copy=True),
        dtype=np.float64,
    )
    direction, code = gmres(
        operator, -residual, restart=48, maxiter=6, rtol=1e-10, atol=0
    )
    result = {
        "cells": len(context.reference),
        "base_residual": float(np.max(np.abs(residual))),
        "linear_code": int(code),
        "linear_residual_ratio": float(
            np.linalg.norm(operator @ direction + residual) / np.linalg.norm(residual)
        ),
        "direction_norm": float(np.linalg.norm(direction)),
        "steps": [],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"BASE residual={result['base_residual']:.12g} "
        f"linear_code={code} direction_norm={result['direction_norm']:.12g}",
        flush=True,
    )
    mapped = jax.jit(context.map)
    geometry = context.geometry
    seed = jnp.argmin(jnp.sum((geometry.centre - context.seeds[0]) ** 2, axis=1))
    radius = jnp.maximum(
        context.policy.normal_form_radius,
        context.policy.normal_form_pitch_floor * jnp.median(geometry.pitch),
    )

    def detail(moments):
        reading = context.reading(moments)
        components = edge_components(
            geometry,
            reading.field_coefficients,
            reading.saddle_form,
            reading.axis,
            seed,
            context.convention.sigma,
            context.convention.sigma * reading.boundary_flux,
            radius,
            jnp.asarray(True),
        )
        return reading, components

    inspect = jax.jit(detail)
    tangent = jnp.asarray(direction)
    for amount in (0, 1e-4, 2e-4, 1e-3):
        moments = (base.reshape(-1) + amount * tangent).reshape(shape) * context.scale
        state = mapped(moments)
        reading, components = inspect(moments)
        jax.block_until_ready((state, reading, components))
        (
            flux,
            frame,
            edge,
            margin,
            sector,
            near,
            sector_margin,
            sector_span,
            length_width,
        ) = map(np.asarray, components)
        original_flux = np.clip(0.5 + margin / 1e-8, 0, 1)
        original_edge = np.minimum(original_flux, frame)
        neighbouring = np.asarray(geometry.neighbour)
        touched = np.argwhere((neighbouring >= 0) & (original_edge < 0.999))
        local = []
        local_cells = (8, 80, 82) if args.cells == 500 else (47, 48, 50)
        for cell in local_cells:
            for slot, peer in enumerate(neighbouring[cell]):
                if peer >= 0:
                    local.append(
                        {
                            "cell": cell,
                            "slot": slot,
                            "peer": int(peer),
                            "flux_margin": float(margin[cell, slot]),
                            "flux_weight_original": float(original_flux[cell, slot]),
                            "flux_weight_geometric": float(flux[cell, slot]),
                            "frame_weight": float(frame[cell, slot]),
                            "sector_weight": float(sector[cell, slot]),
                            "sector_margin": float(sector_margin[cell, slot]),
                            "sector_span": float(sector_span[cell, slot]),
                            "original_sector_width": float(length_width),
                            "near_weight": float(near[cell]),
                        }
                    )
        cell_ids = (8, 82) if args.cells == 500 else (47, 50)
        scaled_residual = np.asarray(state.moment_residual / context.scale)
        peak_cell, peak_block = np.unravel_index(
            np.argmax(np.abs(scaled_residual)), scaled_residual.shape
        )
        row = {
            "step": amount,
            "residual": float(jnp.max(jnp.abs(state.moment_residual / context.scale))),
            "boundary_class": int(reading.boundary_class),
            "booking_ids": np.flatnonzero(np.asarray(reading.booking_cells)).tolist(),
            "supported_cells": int(jnp.sum(reading.selected_cells)),
            "peak_residual_cell": int(peak_cell),
            "peak_residual_block": int(peak_block),
            "peak_residual_value": float(scaled_residual[peak_cell, peak_block]),
            "saddle": np.asarray(reading.saddle_form.position).tolist(),
            "cells": [
                {
                    "cell": cell,
                    "booking": bool(reading.booking_cells[cell]),
                    "path_weight": float(reading.path_weight[cell]),
                    "normal_form": bool(reading.normal_form_cells[cell]),
                    "membership": float(reading.membership[cell]),
                    "geometric_membership": float(reading.geometric_membership[cell]),
                    "current_residual": float(
                        (state.moment_residual / context.scale)[cell, 0]
                    ),
                }
                for cell in cell_ids
            ],
            "local_edges": local,
            "weak_edges": [
                [
                    int(cell),
                    int(slot),
                    int(neighbouring[cell, slot]),
                    float(original_flux[cell, slot]),
                    float(frame[cell, slot]),
                    float(flux[cell, slot]),
                    float(margin[cell, slot]),
                ]
                for cell, slot in touched
            ],
        }
        result["steps"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        neighbour = row["cells"][1]
        print(
            f"STEP h={amount:.8g} residual={row['residual']:.12g} "
            f"booking={row['booking_ids']} weight={neighbour['path_weight']:.6g} "
            f"cell={cell_ids[1]} current={neighbour['current_residual']:.12g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
