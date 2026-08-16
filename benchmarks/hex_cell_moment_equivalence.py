"""Compare one exact linear-current hexagon with refined plasma elements.

The reference route contracts the uniform, radial, and vertical analytic
polygon blocks for flux and both field components.  The comparison route
hands the identical boundary to :class:`nova.frame.polygrid.PolyGrid`, then
uses each boundary-fitted element as a low-order point loop carrying the exact
integral of the linear density over that element.  Refinement therefore tests
the representation without introducing a current-conservation discrepancy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from nova.biot.greens import greens_bz_br, greens_psi
from nova.biot.polygonanalytic import (
    polygon_analytic_field_moments,
    polygon_analytic_flux_moments,
)
from nova.frame.polygrid import PolyGrid

DEFAULT_OUTPUT = Path("docs/figures/plasma-edge-current-representation")
QUANTITIES = ("psi", "B_R", "B_Z")
SOURCE_CENTRE = np.array([3.0, 0.1])
SOURCE_RADIUS = 0.30
DENSITIES = {
    "uniform": np.array([2.3e6, 0.0, 0.0]),
    "linear": np.array([2.3e6, 1.1e6, -0.8e6]),
}


def _hexagon() -> np.ndarray:
    angle = np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    return SOURCE_CENTRE + SOURCE_RADIUS * np.column_stack(
        (np.cos(angle), np.sin(angle))
    )


def _targets() -> tuple[np.ndarray, dict[str, list[int]]]:
    near_angle = np.array([0.08, 0.82, 1.72, 2.82, 3.78, 5.18])
    far_angle = np.array([0.28, 1.18, 2.18, 3.38, 4.38, 5.58])
    near = SOURCE_CENTRE + 1.22 * SOURCE_RADIUS * np.column_stack(
        (np.cos(near_angle), np.sin(near_angle))
    )
    far = SOURCE_CENTRE + 4.0 * SOURCE_RADIUS * np.column_stack(
        (np.cos(far_angle), np.sin(far_angle))
    )
    targets = np.concatenate((near, far))
    return targets, {"near": list(range(6)), "far": list(range(6, 12))}


def _moment_response(vertices: np.ndarray, targets: np.ndarray, coefficients):
    area = Polygon(vertices).area
    flux = polygon_analytic_flux_moments(
        targets[:, 0], targets[:, 1], vertices, expansion_point=SOURCE_CENTRE
    )
    radial, vertical = polygon_analytic_field_moments(
        targets[:, 0], targets[:, 1], vertices, expansion_point=SOURCE_CENTRE
    )
    weights = area * np.asarray(coefficients, dtype=float)
    return np.asarray(
        [
            np.tensordot(weights, np.asarray(flux), axes=1),
            np.tensordot(weights, np.asarray(radial), axes=1),
            np.tensordot(weights, np.asarray(vertical), axes=1),
        ]
    )


def _plasma_elements(vertices: np.ndarray, requested_cells: int):
    """Return nova-generated cells clipped back to the authored boundary."""
    grid = PolyGrid(
        poly=vertices,
        delta=-requested_cells,
        turn="hexagon",
        tile=True,
        trim=True,
        nturn=1.0,
    )
    boundary = Polygon(vertices)
    cells = []
    for wrapped in grid.frame.poly:
        clipped = wrapped.poly.intersection(boundary)
        if not clipped.is_empty and clipped.area > 0.0:
            cells.append(clipped)
    area = np.asarray([cell.area for cell in cells])
    centre = np.asarray([[cell.centroid.x, cell.centroid.y] for cell in cells])
    return cells, area, centre


def _element_response(targets, area, centre, coefficients):
    delta = centre - SOURCE_CENTRE
    density = (
        coefficients[0] + coefficients[1] * delta[:, 0] + coefficients[2] * delta[:, 1]
    )
    current = area * density
    target_r = targets[:, 0, None]
    target_z = targets[:, 1, None]
    source_r = centre[None, :, 0]
    source_z = centre[None, :, 1]
    psi = greens_psi(target_r, target_z, source_r, source_z) @ current
    bz, br = greens_bz_br(target_r, target_z, source_r, source_z)
    return np.asarray((psi, br @ current, bz @ current)), current


def _fraction(computed: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(computed - reference) / np.linalg.norm(reference))


def _fit_order(size: list[float], error: list[float]) -> float:
    return float(np.polyfit(np.log(size), np.log(error), 1)[0])


def run_equivalence(refinements=(48, 120, 300, 720)) -> dict:
    """Run both coefficient cases over near and far target sets."""
    vertices = _hexagon()
    targets, regions = _targets()
    exact_area = Polygon(vertices).area
    meshes = []
    for requested in refinements:
        _, area, centre = _plasma_elements(vertices, requested)
        meshes.append(
            {
                "requested_cells": int(requested),
                "cell_count": int(len(area)),
                "characteristic_size": float(np.sqrt(exact_area / len(area))),
                "area": area,
                "centre": centre,
            }
        )

    result = {
        "geometry": {
            "vertices_m": vertices.tolist(),
            "area_m2": exact_area,
            "source_centre_m": SOURCE_CENTRE.tolist(),
        },
        "targets": {
            "coordinates_m": targets.tolist(),
            "regions": regions,
        },
        "moment_blocks": {
            quantity: ["uniform", "radial", "vertical"] for quantity in QUANTITIES
        },
        "refinements": [
            {
                key: mesh[key]
                for key in ("requested_cells", "cell_count", "characteristic_size")
            }
            for mesh in meshes
        ],
        "cases": {},
    }
    for case, coefficients in DENSITIES.items():
        reference = _moment_response(vertices, targets, coefficients)
        responses = []
        current_residuals = []
        exact_current = exact_area * coefficients[0]
        for mesh in meshes:
            response, current = _element_response(
                targets, mesh["area"], mesh["centre"], coefficients
            )
            responses.append(response)
            current_residuals.append(
                float(abs(np.sum(current) - exact_current) / abs(exact_current))
            )
        size = [mesh["characteristic_size"] for mesh in meshes]
        region_metrics = {}
        for region, indices in regions.items():
            region_metrics[region] = {}
            for row, quantity in enumerate(QUANTITIES):
                fractions = [
                    _fraction(response[row, indices], reference[row, indices])
                    for response in responses
                ]
                region_metrics[region][quantity] = {
                    "fractions": fractions,
                    "fitted_order": _fit_order(size, fractions),
                    "finest_fraction": fractions[-1],
                }
        result["cases"][case] = {
            "density_coefficients": {
                "uniform_A_per_m2": float(coefficients[0]),
                "radial_A_per_m3": float(coefficients[1]),
                "vertical_A_per_m3": float(coefficients[2]),
            },
            "exact_total_current_A": float(exact_current),
            "current_relative_residuals": current_residuals,
            "regions": region_metrics,
        }
    return result


def _write_figure(result: dict, path: Path) -> None:
    size = np.asarray([row["characteristic_size"] for row in result["refinements"]])
    width, height = 1050, 620
    panel_width, panel_height = 300, 220
    left, top = 72, 58
    column_gap, row_gap = 38, 70
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="title description">',
        '<title id="title">Hexagonal cell moment equivalence convergence</title>',
        '<desc id="description">Relative disagreement between exact moment blocks '
        "and refined low-order plasma elements for flux and both field "
        "components.</desc>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="system-ui, sans-serif" fill="#222">',
        '<text x="525" y="25" text-anchor="middle" font-size="17">Single-cell '
        "moments vs boundary-fitted low-order plasma mesh</text>",
    ]
    for column, quantity in enumerate(QUANTITIES):
        for row, case in enumerate(DENSITIES):
            x0 = left + column * (panel_width + column_gap)
            y0 = top + row * (panel_height + row_gap)
            metrics = result["cases"][case]["regions"]
            values = np.asarray(
                metrics["near"][quantity]["fractions"]
                + metrics["far"][quantity]["fractions"]
            )
            log_x = np.log10(size)
            log_y = np.log10(values)
            x_min, x_max = float(log_x.min()), float(log_x.max())
            y_min, y_max = float(log_y.min()), float(log_y.max())
            y_pad = max(0.12 * (y_max - y_min), 0.15)
            y_min -= y_pad
            y_max += y_pad

            def point_coordinates(fractions):
                x = x0 + (log_x - x_min) / (x_max - x_min) * panel_width
                y = (
                    y0
                    + panel_height
                    - ((np.log10(fractions) - y_min) / (y_max - y_min) * panel_height)
                )
                return list(zip(x, y, strict=True))

            lines.extend(
                [
                    f'<rect x="{x0}" y="{y0}" width="{panel_width}" '
                    f'height="{panel_height}" fill="none" stroke="#777"/>',
                    f'<text x="{x0 + panel_width / 2}" y="{y0 - 12}" '
                    f'text-anchor="middle" font-size="13">{case}: {quantity}</text>',
                    f'<text x="{x0 + panel_width / 2}" y="{y0 + panel_height + 34}" '
                    'text-anchor="middle" font-size="11">characteristic element '
                    "size [m]</text>",
                ]
            )
            if column == 0:
                lines.append(
                    f'<text x="{x0 - 48}" y="{y0 + panel_height / 2}" '
                    'text-anchor="middle" font-size="11" '
                    f'transform="rotate(-90 {x0 - 48} {y0 + panel_height / 2})">'
                    "relative L2 disagreement</text>"
                )
            for region, colour, marker in (
                ("near", "#1769aa", "circle"),
                ("far", "#c44e52", "square"),
            ):
                metric = result["cases"][case]["regions"][region][quantity]
                points = point_coordinates(np.asarray(metric["fractions"]))
                joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                lines.append(
                    f'<polyline points="{joined}" fill="none" stroke="{colour}" '
                    'stroke-width="1.8"/>'
                )
                for x, y in points:
                    if marker == "circle":
                        lines.append(
                            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{colour}"/>'
                        )
                    else:
                        lines.append(
                            f'<rect x="{x - 3:.2f}" y="{y - 3:.2f}" width="6" '
                            f'height="6" fill="{colour}"/>'
                        )
                legend_y = y0 + 17 + (0 if region == "near" else 16)
                lines.extend(
                    [
                        f'<line x1="{x0 + 10}" y1="{legend_y}" x2="{x0 + 30}" '
                        f'y2="{legend_y}" stroke="{colour}" stroke-width="1.8"/>',
                        f'<text x="{x0 + 36}" y="{legend_y + 4}" font-size="10">'
                        f"{region}, p={metric['fitted_order']:.2f}</text>",
                    ]
                )
            lines.extend(
                [
                    f'<text x="{x0}" y="{y0 + panel_height + 15}" font-size="9">'
                    f"{10**x_min:.3f}</text>",
                    f'<text x="{x0 + panel_width}" y="{y0 + panel_height + 15}" '
                    f'text-anchor="end" font-size="9">{10**x_max:.3f}</text>',
                ]
            )
    lines.extend(("</g>", "</svg>"))
    path.write_text("\n".join(lines) + "\n")


def write_artifacts(result: dict, output: Path = DEFAULT_OUTPUT) -> tuple[Path, Path]:
    """Bank the numerical findings and their convergence figure."""
    output.mkdir(parents=True, exist_ok=True)
    findings = output / "hex_cell_moment_equivalence_findings.json"
    figure = output / "hex_cell_moment_equivalence_convergence.svg"
    findings.write_text(json.dumps(result, indent=2) + "\n")
    _write_figure(result, figure)
    return findings, figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_equivalence()
    findings, figure = write_artifacts(result, args.output)
    print(findings)
    print(figure)


if __name__ == "__main__":
    main()
