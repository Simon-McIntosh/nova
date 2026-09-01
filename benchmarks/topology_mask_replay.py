"""Replay a cached MAST private mask through production topology kernels."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.flux_surface_connectivity import (
    fit_tensor_spline,
    hex_edge_admissibility,
    label_saddle_aware_hex_connected_components,
    private_flux_mask,
)
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = (
    ROOT / "docs/figures/topology-visual-corroboration/mast-topology-operands.npz"
)
DEFAULT_OUTPUT = ROOT / "docs/figures/hex-cell-single-grid/mask-replay-gate.json"
GENERATOR = (
    ROOT / "docs/figures/topology-visual-corroboration/generate_topology_visuals.py"
)


def _generator_module():
    spec = importlib.util.spec_from_file_location(
        "topology_operand_generator", GENERATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load topology operand generator {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected_row(cache: Path) -> tuple[dict[str, object], str]:
    generator = _generator_module()
    identity = generator._source_authority(generator.MAST_AUTHORITY)["source_identity"]
    rows = generator._read_cache(cache, identity)
    matches = [
        row
        for row in rows
        if int(row["shot"]) == 22086
        and int(row["frame"]) == 43
        and row["arm"] == "mixed"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one MAST 22086/43 mixed row, got {len(matches)}")
    return matches[0], identity


def run(
    cache: Path = DEFAULT_CACHE, output: Path = DEFAULT_OUTPUT
) -> dict[str, object]:
    """Replay candidate selection and the private mask from cached operands only."""
    configure_dtypes()
    row, source_identity = _selected_row(cache)
    coordinate = np.asarray(row["cell_rz"], dtype=float)
    cached_labels = np.asarray(row["domain_labels"], dtype=np.int8)
    values = np.asarray(row["per_cell_flux_values"], dtype=float)
    candidate_points = np.asarray(row["x_candidates"], dtype=float)
    cached_candidate_labels = np.asarray(
        row["per_candidate_domain_labels"], dtype=np.int8
    )
    polygons = np.asarray(row["current_cell_polygons"], dtype=float)
    atomic_signed_flux = np.asarray(row["atomic_node_signed_flux"], dtype=float)
    if values.shape != (len(coordinate),):
        raise RuntimeError("cache lacks one flux value per topology cell")
    if cached_candidate_labels.shape != (len(candidate_points), len(coordinate)):
        raise RuntimeError("cache lacks one domain-label map per X candidate")
    if polygons.shape[0] != len(coordinate) or polygons.shape[-1] != 2:
        raise RuntimeError("cache lacks one current-cell polygon per topology cell")
    if not atomic_signed_flux.size:
        raise RuntimeError("cache lacks signed flux at shared atomic nodes")

    radius = np.unique(coordinate[:, 0])
    height = np.unique(coordinate[:, 1])
    shape = (height.size, radius.size)
    if radius.size * height.size != len(coordinate):
        raise RuntimeError("cached topology cells are not a tensor carrier")
    field = values.reshape((radius.size, height.size)).T
    spline = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(field)
    )
    present = np.all(np.isfinite(candidate_points), axis=1)
    safe_points = np.where(present[:, None], candidate_points, 0.0)
    candidate_flux = np.asarray(
        spline(jnp.asarray(safe_points[:, 0]), jnp.asarray(safe_points[:, 1]))
    )
    candidate_flux = np.where(present, candidate_flux, np.nan)
    axis_point = np.asarray(row["selected_o"], dtype=float)[0]
    axis_flux = float(spline(axis_point[0], axis_point[1]))
    selected_index = int(np.nanargmin(np.abs(candidate_flux - axis_flux)))
    cached_selected_point = np.asarray(row["selected_x"], dtype=float)[0]
    cached_selected_index = int(
        np.nanargmin(np.linalg.norm(candidate_points - cached_selected_point, axis=1))
    )

    generator = _generator_module()
    rings, shared_edges = generator._raster_hex_partition_geometry(
        jnp.asarray(radius), jnp.asarray(height)
    )
    np.testing.assert_array_equal(np.asarray(rings), hex_stencil(shape))
    inside_material = cached_labels != int(PlasmaDomain.EXCLUDED_MATERIAL)
    inside_material = inside_material.reshape((radius.size, height.size)).T
    seed_flat = int(np.argmin(np.sum((coordinate - axis_point) ** 2, axis=1)))
    axis_seed = np.zeros(len(coordinate), dtype=bool)
    axis_seed[seed_flat] = True
    axis_seed = axis_seed.reshape((radius.size, height.size)).T

    replayed_candidate_labels = np.full_like(cached_candidate_labels, -1)
    replayed_candidate_private = []
    for candidate_index in np.flatnonzero(present):
        level = float(candidate_flux[candidate_index])
        closed = np.where(axis_flux >= level, field >= level, field < level)
        confined = inside_material & closed
        admissible = hex_edge_admissibility(
            jnp.asarray(field),
            jnp.asarray(radius),
            jnp.asarray(height),
            jnp.asarray(level),
            jnp.asarray(axis_flux),
            shared_edges,
        )
        component_labels = label_saddle_aware_hex_connected_components(
            jnp.asarray(confined),
            jnp.asarray(rings),
            admissible,
            confined.size,
        )
        private = np.asarray(
            private_flux_mask(component_labels, jnp.asarray(axis_seed)), dtype=bool
        )
        connected = confined & ~private
        labels = np.full(shape, int(PlasmaDomain.COMMON_SOL), dtype=np.int8)
        labels[~inside_material] = int(PlasmaDomain.EXCLUDED_MATERIAL)
        labels[connected] = int(PlasmaDomain.CORE)
        labels[private] = int(PlasmaDomain.PRIVATE_FLUX)
        replayed_candidate_labels[candidate_index] = labels.T.reshape(-1)
        replayed_candidate_private.append(int(private.sum()))

    selected_labels = replayed_candidate_labels[selected_index]
    replayed_private = selected_labels == int(PlasmaDomain.PRIVATE_FLUX)
    committed_private = cached_labels == int(PlasmaDomain.PRIVATE_FLUX)
    differing = replayed_private != committed_private
    candidate_label_differences = np.count_nonzero(
        replayed_candidate_labels[present] != cached_candidate_labels[present]
    )
    selected_matches = selected_index == cached_selected_index
    symmetric_difference = int(np.count_nonzero(differing))
    receipt = {
        "schema": "nova.topology-mask-cache-replay",
        "operand": "MAST 22086/43 mixed",
        "cache": str(cache.resolve()),
        "cache_source_identity": source_identity,
        "cache_staleness_check": "passed",
        "grid_shape": list(shape),
        "cell_count": int(len(coordinate)),
        "x_candidate_count": int(np.count_nonzero(present)),
        "selected_x_candidate_index": selected_index,
        "cached_selected_x_candidate_index": cached_selected_index,
        "selected_x_candidate_matches": selected_matches,
        "selected_x_position_residual_m": float(
            np.linalg.norm(candidate_points[selected_index] - cached_selected_point)
        ),
        "candidate_private_cell_counts": replayed_candidate_private,
        "per_candidate_domain_label_differences": int(candidate_label_differences),
        "committed_private_cells": int(committed_private.sum()),
        "replayed_private_cells": int(replayed_private.sum()),
        "symmetric_difference_private_mask_cells": symmetric_difference,
        "current_cell_polygon_count": int(polygons.shape[0]),
        "current_cell_polygon_capacity": int(polygons.shape[1]),
        "atomic_node_signed_flux_count": int(atomic_signed_flux.size),
        "atomic_node_signed_flux_zero_count": int(
            np.count_nonzero(atomic_signed_flux == 0.0)
        ),
        "passes": bool(
            selected_matches
            and candidate_label_differences == 0
            and symmetric_difference == 0
        ),
        "falsifier": "fires on any nonzero private-mask symmetric difference",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.cache, arguments.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
