"""Census the off-axis polygon edge reduction on the weak-rotation geometry.

The certificate carrier stores scalar closed-form moment matrices.  This
diagnostic evaluates the packed uniform row on the same authored cells, resolves
that row into edge contributions, and compares each contribution with a direct
angle quadrature of the unreduced Urankar antiderivative.  A rectangular carrier
at the same requested cell count is measured through the identical calculation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from shapely.geometry import LineString

from nova.biot.polygon import pack_section, pad_batch
from nova.biot.polygonanalytic import _edge_flux, packed_analytic_greens
from nova.frame.coilset import CoilSet
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture
from tests.rotating_equilibrium_references import reference_cases


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/figures/uniform-cell-clip-and-coupling/off-axis-error/census.json"
FIGURE = (
    ROOT
    / "docs/figures/uniform-cell-clip-and-coupling/off-axis-error/error-contours.svg"
)
REQUESTED_CELLS = (110, 300)
OUTBOARD_WINDOW = (7.25, 7.75, 0.0, 0.5)
REFERENCE_NODES = 192
REFERENCE_CHECK_NODES = 144
MAX_CENSUS_SOURCES = 48


@dataclass(frozen=True)
class Carrier:
    """Authored cells and target geometry for one tiling."""

    tiling: str
    requested_cells: int
    node: np.ndarray
    area: np.ndarray
    polygons: tuple[np.ndarray, ...]
    cut: np.ndarray


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _clean_vertices(vertices: np.ndarray) -> np.ndarray:
    """Remove adjacent duplicate polygon vertices without moving an edge."""
    scale = max(float(np.max(np.abs(vertices))), float(np.ptp(vertices)), 1.0)
    tolerance = 128.0 * np.finfo(float).eps * scale
    kept = [vertices[0]]
    for vertex in vertices[1:]:
        if np.linalg.norm(vertex - kept[-1]) > tolerance:
            kept.append(vertex)
    if len(kept) > 1 and np.linalg.norm(kept[-1] - kept[0]) <= tolerance:
        kept.pop()
    return np.asarray(kept, dtype=np.float64)


def _material_geometry(case, requested_cells: int, tiling: str) -> Carrier:
    """Build one authored tiling inside the analytic limiter."""
    wall = fixture.limiter_contour(case, points=fixture.WALL_POINT_COUNT)
    if tiling == "hexagonal":
        machine = fixture.cached_machine(
            case, -requested_cells, wall_nodes=fixture.WALL_POINT_COUNT
        )
        polygons = tuple(
            np.asarray(item, dtype=np.float64) for item in machine.cell_polygons
        )
        boundary = LineString(wall)
        cut = np.asarray(
            [
                boundary.intersects(LineString(np.vstack((item, item[0]))))
                for item in polygons
            ]
        )
        return Carrier(
            tiling=tiling,
            requested_cells=requested_cells,
            node=np.asarray(machine.node, dtype=np.float64),
            area=np.asarray(machine.area, dtype=np.float64),
            polygons=polygons,
            cut=cut,
        )

    coilset = CoilSet(dplasma=-requested_cells, tplasma="rectangle")
    coilset.firstwall.insert(wall, turn="rectangle")
    plasma = np.asarray(coilset.subframe.loc[:, "plasma"], dtype=bool)
    material = np.asarray(coilset.subframe.loc[:, "poly"], dtype=object)[plasma]
    polygons = tuple(
        _clean_vertices(np.asarray(item.poly.exterior.coords)[:-1, :2])
        for item in material
    )
    node = np.c_[
        np.asarray(coilset.subframe.loc[plasma, "x"], dtype=np.float64),
        np.asarray(coilset.subframe.loc[plasma, "z"], dtype=np.float64),
    ]
    boundary = LineString(wall)
    cut = np.asarray([item.poly.intersects(boundary) for item in material])
    return Carrier(
        tiling=tiling,
        requested_cells=requested_cells,
        node=node,
        area=np.asarray([item.poly.area for item in material], dtype=np.float64),
        polygons=polygons,
        cut=cut,
    )


def _edge_quadrature(target_r, target_z, edge, which: int, nodes: int) -> np.ndarray:
    """Directly integrate one edge limit before the closed-form reduction."""
    ra, za, rb, zb = edge
    b1 = (rb - ra) / (zb - za)
    a02 = 1.0 + b1 * b1
    node, weight = np.polynomial.legendre.leggauss(nodes)
    alpha = 0.25 * np.pi * (node + 1.0)
    radius = np.atleast_1d(target_r)[:, None]
    height = np.atleast_1d(target_z)[:, None]
    cos_phi = -np.cos(2.0 * alpha)[None, :]
    sin_phi = np.sin(2.0 * alpha)[None, :]
    sin_two_phi = np.sin(2.0 * (np.pi - 2.0 * alpha))[None, :]
    plane_radius = ra - b1 * (za - height)
    level = (zb - height) if which else (za - height)
    source_radius = plane_radius + b1 * level
    offset = source_radius - radius * cos_phi
    plane_offset = plane_radius - radius * cos_phi
    ring_distance_squared = level * level + (radius * sin_phi) ** 2
    plane_distance_squared = plane_offset**2 + a02 * (radius * sin_phi) ** 2
    distance = np.sqrt(ring_distance_squared + offset**2)
    gamma = level + b1 * offset
    integrand = (
        gamma * distance / (2.0 * a02)
        + level * radius * cos_phi * np.arcsinh(offset / np.sqrt(ring_distance_squared))
        + (plane_distance_squared + 2.0 * a02 * radius * cos_phi * plane_offset)
        / (2.0 * a02 * np.sqrt(a02))
        * np.arcsinh(gamma / np.sqrt(plane_distance_squared))
        - 0.5
        * radius**2
        * sin_two_phi
        * np.arctan(
            (level * offset - b1 * ring_distance_squared)
            / (radius * sin_phi * distance)
        )
    )
    quadrature_weight = 0.25 * np.pi * weight * -np.cos(2.0 * alpha)
    return 4.0 * (integrand @ quadrature_weight)


def _correlation(first: np.ndarray, second: np.ndarray) -> dict[str, float | None]:
    """Return Pearson and rank correlations, retaining undefined results."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 3 or np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(stats.pearsonr(first, second).statistic),
        "spearman": float(stats.spearmanr(first, second).statistic),
    }


def _patch_enrichment(node: np.ndarray, error: np.ndarray) -> dict[str, float | int]:
    r_min, r_max, z_min, z_max = OUTBOARD_WINDOW
    selected = (
        (node[:, 0] >= r_min)
        & (node[:, 0] <= r_max)
        & (node[:, 1] >= z_min)
        & (node[:, 1] <= z_max)
    )
    squared = np.asarray(error, dtype=np.float64) ** 2
    domain_fraction = float(np.mean(selected))
    error_fraction = float(
        np.sum(squared[selected]) / max(np.sum(squared), np.finfo(float).tiny)
    )
    return {
        "node_count": int(np.count_nonzero(selected)),
        "domain_fraction": domain_fraction,
        "squared_error_fraction": error_fraction,
        "enrichment": error_fraction / max(domain_fraction, np.finfo(float).tiny),
    }


def _source_indices(carrier: Carrier, anchor: np.ndarray) -> np.ndarray:
    """Select cut, anchor-near, outboard, and domain-spanning source strata."""
    count = min(MAX_CENSUS_SOURCES, len(carrier.polygons))
    selected = set(np.flatnonzero(carrier.cut).tolist())
    anchor_distance = np.linalg.norm(carrier.node - anchor, axis=1)
    selected.update(np.argsort(anchor_distance)[:16].tolist())
    window_centre = np.asarray([0.5 * (OUTBOARD_WINDOW[0] + OUTBOARD_WINDOW[1]), 0.25])
    outboard_distance = np.linalg.norm(carrier.node - window_centre, axis=1)
    selected.update(np.argsort(outboard_distance)[:16].tolist())
    ordered = np.lexsort((carrier.node[:, 1], carrier.node[:, 0]))
    spread = ordered[np.linspace(0, len(ordered) - 1, count, dtype=np.intp)]
    for index in spread:
        if len(selected) >= count:
            break
        selected.add(int(index))
    if len(selected) > count:
        protected = set(np.flatnonzero(carrier.cut).tolist())
        optional = sorted(
            selected - protected, key=lambda index: anchor_distance[index]
        )
        selected = protected | set(optional[: max(count - len(protected), 0)])
    return np.asarray(sorted(selected), dtype=np.intp)


def _packed_uniform_matrix(carrier: Carrier, source_indices: np.ndarray) -> np.ndarray:
    """Evaluate the packed uniform row in bounded target/source tiles."""
    target_tile = 24
    source_tile = 24
    selected = tuple(carrier.polygons[index] for index in source_indices)
    result = np.empty((len(carrier.node), len(selected)), dtype=np.float64)
    for target_start in range(0, len(carrier.node), target_tile):
        target_stop = min(target_start + target_tile, len(carrier.node))
        targets = carrier.node[target_start:target_stop]
        for source_start in range(0, len(selected), source_tile):
            source_stop = min(source_start + source_tile, len(selected))
            sections = selected[source_start:source_stop]
            edge, weight, norm = pad_batch(sections)
            pair = np.arange(len(targets) * len(sections))
            rows, columns = np.divmod(pair, len(sections))
            values = packed_analytic_greens(
                np,
                targets[rows, 0],
                targets[rows, 1],
                edge[:, :, columns],
                weight[:, columns],
                norm[columns],
            )[0]
            result[target_start:target_stop, source_start:source_stop] = np.asarray(
                values
            ).reshape(len(targets), len(sections))
    return result


def _measure_carrier(case, carrier: Carrier) -> tuple[dict[str, Any], np.ndarray]:
    """Measure every live source edge against an independent angle integral."""
    started = perf_counter()
    target_r = carrier.node[:, 0]
    target_z = carrier.node[:, 1]
    if np.any(target_r <= 0.0):
        raise RuntimeError("the off-axis census encountered a machine-axis target")
    anchor = np.asarray(case.magnetic_axis, dtype=np.float64)
    source_indices = _source_indices(carrier, anchor)
    closed = np.zeros((len(target_r), len(source_indices)), dtype=np.float64)
    reference = np.zeros_like(closed)
    reference_check = np.zeros_like(closed)
    per_source_rms = np.zeros(len(source_indices), dtype=np.float64)
    per_source_aspect = np.zeros(len(source_indices), dtype=np.float64)
    edge_absolute = []
    edge_reference = []
    edge_check = []
    live_edge_count = 0

    for source, source_index in enumerate(source_indices):
        vertices = carrier.polygons[source_index]
        edges, weights, norm = pack_section(vertices)
        source_errors = []
        spans = np.ptp(vertices, axis=0)
        per_source_aspect[source] = max(spans) / max(min(spans), np.finfo(float).tiny)
        for edge, weight in zip(edges, weights, strict=True):
            if weight == 0.0:
                continue
            live_edge_count += 1
            computed = (
                0.5
                * norm
                * target_r
                * (
                    _edge_flux(target_r, target_z, edge, 0, 128)
                    - _edge_flux(target_r, target_z, edge, 1, 128)
                )
            )
            expected = (
                0.5
                * norm
                * target_r
                * (
                    _edge_quadrature(target_r, target_z, edge, 0, REFERENCE_NODES)
                    - _edge_quadrature(target_r, target_z, edge, 1, REFERENCE_NODES)
                )
            )
            checked = (
                0.5
                * norm
                * target_r
                * (
                    _edge_quadrature(target_r, target_z, edge, 0, REFERENCE_CHECK_NODES)
                    - _edge_quadrature(
                        target_r, target_z, edge, 1, REFERENCE_CHECK_NODES
                    )
                )
            )
            closed[:, source] += computed
            reference[:, source] += expected
            reference_check[:, source] += checked
            difference = computed - expected
            source_errors.append(difference)
            edge_absolute.append(np.abs(difference))
            edge_reference.append(expected)
            edge_check.append(expected - checked)
        if source_errors:
            per_source_rms[source] = float(
                np.sqrt(np.mean(np.asarray(source_errors, dtype=np.float64) ** 2))
            )

    packed = _packed_uniform_matrix(carrier, source_indices)
    packed_scalar_delta = packed - closed
    current = (
        np.asarray(
            case.toroidal_current_density(
                carrier.node[source_indices, 0], carrier.node[source_indices, 1]
            ),
            dtype=np.float64,
        )
        * carrier.area[source_indices]
    )
    error_map = (packed - reference) @ current
    reference_image = reference @ current
    absolute_edges = np.concatenate(edge_absolute)
    reference_edges = np.concatenate(edge_reference)
    check_edges = np.concatenate(edge_check)
    reference_rms = float(np.sqrt(np.mean(reference_edges**2)))
    image_rms = float(np.sqrt(np.mean(reference_image**2)))
    error_absolute = np.abs(error_map)
    distance_anchor = np.linalg.norm(carrier.node - anchor, axis=1)
    weighted_centre = np.sum(carrier.node * error_absolute[:, None] ** 2, axis=0) / max(
        np.sum(error_absolute**2), np.finfo(float).tiny
    )
    result = {
        "tiling": carrier.tiling,
        "requested_cells": carrier.requested_cells,
        "realised_cells": len(carrier.node),
        "live_edge_count": live_edge_count,
        "target_count": len(carrier.node),
        "pair_count": int(closed.size),
        "source_census": {
            "sampled": len(source_indices),
            "available": len(carrier.polygons),
            "fraction": float(len(source_indices) / len(carrier.polygons)),
            "selection": (
                "all cut cells plus anchor, outboard, and domain-spanning strata"
            ),
            "indices": source_indices.tolist(),
        },
        "all_targets_off_axis": True,
        "characteristic_pitch_m": float(np.sqrt(np.median(carrier.area))),
        "cut_cell_count": int(np.count_nonzero(carrier.cut)),
        "edge_contribution_error": {
            "absolute_max_wb_per_a": float(np.max(absolute_edges)),
            "absolute_rms_wb_per_a": float(np.sqrt(np.mean(absolute_edges**2))),
            "rms_relative_to_edge_reference": float(
                np.sqrt(np.mean(absolute_edges**2))
                / max(reference_rms, np.finfo(float).tiny)
            ),
            "direct_quadrature_check_rms_relative": float(
                np.sqrt(np.mean(check_edges**2))
                / max(reference_rms, np.finfo(float).tiny)
            ),
            "reference_nodes": REFERENCE_NODES,
            "reference_check_nodes": REFERENCE_CHECK_NODES,
        },
        "packed_to_scalar_edge_sum": {
            "absolute_max_wb_per_a": float(np.max(np.abs(packed_scalar_delta))),
            "relative_rms": float(
                np.sqrt(np.mean(packed_scalar_delta**2))
                / max(float(np.sqrt(np.mean(closed**2))), np.finfo(float).tiny)
            ),
        },
        "current_weighted_error_map": {
            "absolute_max_wb": float(np.max(error_absolute)),
            "absolute_rms_wb": float(np.sqrt(np.mean(error_map**2))),
            "rms_relative_to_reference_image": float(
                np.sqrt(np.mean(error_map**2)) / max(image_rms, np.finfo(float).tiny)
            ),
            "squared_error_centre_rz_m": weighted_centre.tolist(),
            "distance_of_error_centre_to_normalisation_anchor_m": float(
                np.linalg.norm(weighted_centre - anchor)
            ),
            "correlation_with_cut_target": _correlation(
                error_absolute, carrier.cut.astype(np.float64)
            ),
            "correlation_with_distance_to_normalisation_anchor": _correlation(
                error_absolute, distance_anchor
            ),
            "outboard_window": _patch_enrichment(carrier.node, error_map),
        },
        "source_geometry": {
            "correlation_edge_rms_with_aspect": _correlation(
                per_source_rms, per_source_aspect
            ),
            "maximum_aspect": float(np.max(per_source_aspect)),
        },
        "elapsed_seconds": perf_counter() - started,
    }
    return result, error_map


def _plot(carriers, maps, output: Path) -> None:
    """Draw shared-level line contours for both tilings and resolutions."""
    positive = np.concatenate([np.abs(values) for values in maps])
    nonzero = positive[positive > 0.0]
    lower = max(float(np.percentile(nonzero, 10.0)), np.finfo(float).tiny)
    upper = float(np.max(nonzero))
    levels = np.geomspace(lower, upper, 9) if upper > lower else np.asarray([upper])
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), constrained_layout=True)
    for axis, carrier, error in zip(axes.ravel(), carriers, maps, strict=True):
        axis.tricontour(
            carrier.node[:, 0],
            carrier.node[:, 1],
            np.maximum(np.abs(error), lower),
            levels=levels,
            linewidths=0.8,
            colors="C3",
        )
        axis.set_aspect("equal")
        axis.set_title(
            f"{carrier.tiling} · {carrier.requested_cells} requested · "
            f"{len(carrier.node)} realised"
        )
        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
    figure.suptitle("Off-axis polygon-edge error · shared absolute levels [Wb]")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def measure(output: Path = OUTPUT, figure: Path = FIGURE) -> dict[str, Any]:
    """Run the complete weak-rotation hexagonal/rectangular discriminator."""
    configure_dtypes()
    case = reference_cases()["weak-rotation-reactor"].static_limit()
    carriers = []
    results = []
    maps = []
    for requested_cells in REQUESTED_CELLS:
        for tiling in ("hexagonal", "rectangular"):
            carrier = _material_geometry(case, requested_cells, tiling)
            result, error_map = _measure_carrier(case, carrier)
            carriers.append(carrier)
            results.append(result)
            maps.append(error_map)
            edge_relative = result["edge_contribution_error"][
                "rms_relative_to_edge_reference"
            ]
            packed_relative = result["packed_to_scalar_edge_sum"]["relative_rms"]
            print(
                f"CENSUS tiling={tiling} requested={requested_cells} "
                f"realised={len(carrier.node)} edge_rel={edge_relative:.9g} "
                f"packed_rel={packed_relative:.9g}",
                flush=True,
            )
            partial = {
                "schema": "nova.solovev-off-axis-edge-census.v1",
                "case": "weak-rotation-reactor-static",
                "source_revision": _source_revision(),
                "completed": False,
                "rows": results,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(partial, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    _plot(carriers, maps, figure)
    worst_packed = max(
        row["packed_to_scalar_edge_sum"]["relative_rms"] for row in results
    )
    worst_edge = max(
        row["edge_contribution_error"]["rms_relative_to_edge_reference"]
        for row in results
    )
    receipt = {
        "schema": "nova.solovev-off-axis-edge-census.v1",
        "case": "weak-rotation-reactor-static",
        "source_revision": _source_revision(),
        "completed": True,
        "route_audit": {
            "certificate_carrier_builder": (
                "scripts/analytic_oracle_fixtures/measure.py::build_machine"
            ),
            "certificate_matrix_builder": "polygon_analytic_flux_moments_batched",
            "packed_kernel_serves_banked_certificate": False,
            "interpretation": (
                "the packed comparison is a discriminator on identical authored "
                "geometry; changing packed_analytic_moments cannot change the "
                "banked certificate axis ladder"
            ),
        },
        "axis_error_context_mm": {"110": 13.352807, "300": 30.247708, "500": 52.413422},
        "outboard_window_rz_m": list(OUTBOARD_WINDOW),
        "rows": results,
        "figure": str(figure.relative_to(ROOT)),
        "headline": {
            "worst_packed_to_scalar_relative_rms": worst_packed,
            "worst_edge_reduction_relative_rms": worst_edge,
            "classification": (
                "packed_kernel_not_supported_as_axis_error_cause"
                if worst_packed < 1.0e-9
                else "packed_kernel_discrepancy_detected"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    arguments = parser.parse_args()
    receipt = measure(arguments.output, arguments.figure)
    print(json.dumps(receipt["headline"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
