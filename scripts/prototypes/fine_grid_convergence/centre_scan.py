"""Compare sector openings at the analytic reference state."""

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
from scripts.prototypes.fast_map_production import kernel

original_path = kernel._continuous_path_weight

from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.fine_grid_convergence.edge_components import edge_components
from scripts.prototypes.fine_grid_convergence.path_weight import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    context = build(args.cells, args.cache)
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

    corrected, parts = jax.jit(detail)(context.reference)
    jax.block_until_ready((corrected, parts))
    kernel._continuous_path_weight = original_path
    original = jax.jit(context.reading)(context.reference)
    jax.block_until_ready(original)
    install()
    flux, frame, edge, margin, sector, near, sector_margin, sector_span, width = map(
        np.asarray, parts
    )
    opened = np.clip(1 + sector_margin / np.maximum(sector_span, 1e-12), 0, 1)
    old = np.clip(0.5 + sector_margin / width, 0, 1)
    neighbours = np.asarray(geometry.neighbour)
    neighbour_edges = np.asarray(geometry.neighbour_edge)
    peer = np.maximum(neighbours, 0)
    peer_edge = np.maximum(neighbour_edges, 0)
    original_flux = np.clip(0.5 + margin / 1e-8, 0, 1)

    def paths(sector_weight):
        frame_weight = 1 - near[:, None] * (1 - sector_weight)
        frame_weight = np.minimum(frame_weight, frame_weight[peer, peer_edge])
        edge_weight = np.where(
            neighbours >= 0, np.minimum(original_flux, frame_weight), 0
        )
        strength = np.zeros(len(context.reference))
        strength[int(seed)] = 1
        for _ in range(len(strength)):
            incoming = np.minimum(strength[peer], edge_weight)
            strength = np.maximum(strength, np.max(incoming, axis=1))
        return strength

    baseline_path = paths(old)
    full_span_path = paths(opened)
    scale_screen = []
    for exponent in (2, 3, 4, 5, 6):
        scaled_span = sector_span * flux**exponent
        tapered = np.clip(1 + sector_margin / np.maximum(scaled_span, 1e-12), 0, 1)
        strength = paths(tapered)
        area = np.asarray(original.geometric_membership) * strength
        scale_screen.append(
            {
                "exponent": exponent,
                "support": int(np.sum(strength > 0)),
                "cell_5_path": float(strength[5]),
                "cell_6_path": float(strength[6]),
                "maximum_membership_change": float(
                    np.max(np.abs(area - np.asarray(original.membership)))
                ),
            }
        )
    candidates = np.argwhere((neighbours >= 0) & (opened - old > 0.01))
    rows = []
    for cell, slot in candidates:
        rows.append(
            {
                "cell": int(cell),
                "slot": int(slot),
                "peer": int(neighbours[cell, slot]),
                "margin": float(sector_margin[cell, slot]),
                "span": float(sector_span[cell, slot]),
                "old": float(old[cell, slot]),
                "opened": float(opened[cell, slot]),
                "flux": float(flux[cell, slot]),
                "flux_weight_original": float(original_flux[cell, slot]),
                "near": float(near[cell]),
            }
        )
    rows.sort(key=lambda row: row["opened"] - row["old"], reverse=True)
    path_difference = np.asarray(corrected.path_weight - original.path_weight)
    membership_difference = np.asarray(corrected.membership - original.membership)
    changed = np.argsort(-np.abs(membership_difference))[:20]
    result = {
        "cells": len(context.reference),
        "old_width": float(width),
        "baseline_path_error": float(
            np.max(np.abs(baseline_path - np.asarray(original.path_weight)))
        ),
        "full_span_path_error": float(
            np.max(np.abs(full_span_path - np.asarray(corrected.path_weight)))
        ),
        "scale_screen": scale_screen,
        "opened_edge_count": len(rows),
        "largest_openings": rows[:30],
        "support_original": int(np.sum(np.asarray(original.selected_cells))),
        "support_corrected": int(np.sum(np.asarray(corrected.selected_cells))),
        "largest_membership_changes": [
            {
                "cell": int(cell),
                "original": float(original.membership[cell]),
                "corrected": float(corrected.membership[cell]),
                "path_difference": float(path_difference[cell]),
            }
            for cell in changed
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"SUMMARY cells={result['cells']} width={width:.8g} "
        f"opened_edges={len(rows)} original_support={result['support_original']} "
        f"corrected_support={result['support_corrected']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
