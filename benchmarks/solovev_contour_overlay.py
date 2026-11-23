# ruff: noqa: E501
"""Compare banked Solov'ev flux contours without evaluating equilibrium physics.

The input archive contains accepted forward-solve states and the analytic
reference evaluated on the same nodes.  This module verifies the archive,
forms piecewise-linear normalized-flux contours, measures their radial shape
difference, and writes a compact SVG plus a machine-readable findings file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
from scipy.interpolate import LinearNDInterpolator


DEFAULT_BANK = Path(
    "/run/user/39486/claude-39486/-home-ITER-mcintos-Code-nova/"
    "4c0db14a-a820-4e6a-a348-0801eab85fba/scratchpad/flux-state-bank"
)
DEFAULT_OUTPUT = Path("docs/figures/plasma-edge-current-representation")
COARSE_FILE = "coarse_legacy_t0000.npz"
FINE_FILE = "fine_homotopy_t5875.npz"
ANGLE_COUNT = 720
RADIAL_SAMPLE_COUNT = 1025
LEVEL_COLOURS = (
    "#440154",
    "#482878",
    "#3e4a89",
    "#31688e",
    "#26828e",
    "#1f9e89",
    "#35b779",
    "#6ece58",
    "#b5de2b",
    "#fde725",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bank_entries(bank: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = bank / "state-bank-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "nova.flux-state-bank.v1":
        raise ValueError(f"unsupported state-bank schema: {manifest.get('schema')!r}")
    entries = {entry["file"]: entry for entry in manifest["entries"]}
    for name in (COARSE_FILE, FINE_FILE):
        entry = entries.get(name)
        if entry is None:
            raise ValueError(f"state bank omits required checkpoint {name}")
        if not entry["verification"]["verified"]:
            raise ValueError(f"state bank marks {name} unverified")
        actual = _sha256(bank / name)
        if actual != entry["sha256"]:
            raise ValueError(f"state-bank digest mismatch for {name}: {actual}")
    if manifest.get("normalized_contour_level_count") != 21:
        raise ValueError("state bank does not contain the required 21 contour levels")
    return manifest, entries


def _load_checkpoint(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _wall_radii(origin: np.ndarray, angles: np.ndarray, wall: np.ndarray) -> np.ndarray:
    """Return the first wall intersection along every ray from ``origin``."""
    direction = np.column_stack((np.cos(angles), np.sin(angles)))
    start = wall
    edge = np.roll(wall, -1, axis=0) - start
    offset = start - origin
    denominator = _cross(direction[:, None, :], edge[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        ray = _cross(offset[None, :, :], edge[None, :, :]) / denominator
        segment = _cross(offset[None, :, :], direction[:, None, :]) / denominator
    valid = (
        (np.abs(denominator) > 1.0e-14)
        & (ray > 0.0)
        & (segment >= -1.0e-12)
        & (segment <= 1.0 + 1.0e-12)
    )
    candidates = np.where(valid, ray, np.inf)
    radius = np.min(candidates, axis=1)
    if not np.all(np.isfinite(radius)):
        missing = int(np.count_nonzero(~np.isfinite(radius)))
        raise ValueError(f"wall does not enclose the axis on {missing} rays")
    return radius


def _contours(
    nodes: np.ndarray,
    psi_norm: np.ndarray,
    axis: np.ndarray,
    x_point: np.ndarray,
    wall: np.ndarray,
    levels: np.ndarray,
    angles: np.ndarray,
    *,
    boundary: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Sample radial roots of the piecewise-linear discrete flux map."""
    extra_nodes = [np.asarray(axis)[None, :], np.asarray(x_point).reshape(-1, 2)]
    extra_values = [np.zeros(1), np.ones(np.asarray(x_point).reshape(-1, 2).shape[0])]
    if boundary is not None:
        extra_nodes.append(np.asarray(boundary))
        extra_values.append(np.ones(len(boundary)))
    points = np.vstack((nodes, *extra_nodes))
    values = np.concatenate((psi_norm, *extra_values))
    interpolator = LinearNDInterpolator(points, values, fill_value=np.nan)

    wall_radius = _wall_radii(axis, angles, wall)
    fraction = np.linspace(0.0, 1.0, RADIAL_SAMPLE_COUNT)
    radius_grid = wall_radius[:, None] * fraction[None, :]
    direction = np.column_stack((np.cos(angles), np.sin(angles)))
    sample_points = axis + radius_grid[..., None] * direction[:, None, :]
    sampled = np.asarray(interpolator(sample_points.reshape(-1, 2))).reshape(
        len(angles), RADIAL_SAMPLE_COUNT
    )
    sampled[:, 0] = 0.0

    radii = np.full((len(levels), len(angles)), np.nan)
    radii[0] = 0.0
    for level_index, level in enumerate(levels[1:], start=1):
        reached = np.isfinite(sampled) & (sampled >= level)
        has_root = np.any(reached, axis=1)
        upper_index = np.argmax(reached, axis=1)
        rows = np.flatnonzero(has_root & (upper_index > 0))
        upper = upper_index[rows]
        lower = upper - 1
        low_value = sampled[rows, lower]
        high_value = sampled[rows, upper]
        weight = np.divide(
            level - low_value,
            high_value - low_value,
            out=np.zeros_like(low_value),
            where=np.abs(high_value - low_value) > 1.0e-15,
        )
        radii[level_index, rows] = radius_grid[rows, lower] + weight * (
            radius_grid[rows, upper] - radius_grid[rows, lower]
        )

    coordinates = axis[None, None, :] + radii[..., None] * direction[None, :, :]
    return {"radii": radii, "coordinates": coordinates}


def _distance_rows(
    levels: np.ndarray,
    state_radii: np.ndarray,
    analytic_radii: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for level, state, analytic in zip(levels, state_radii, analytic_radii, strict=True):
        valid = np.isfinite(state) & np.isfinite(analytic)
        if not np.any(valid):
            raise ValueError(f"no shared contour rays at normalized level {level}")
        difference = np.abs(state[valid] - analytic[valid])
        rows.append(
            {
                "psi_norm": float(level),
                "mean_absolute_radial_distance_m": float(np.mean(difference)),
                "maximum_absolute_radial_distance_m": float(np.max(difference)),
                "valid_ray_count": int(np.count_nonzero(valid)),
                "requested_ray_count": int(len(valid)),
            }
        )
    return rows


def _colour(level: float) -> str:
    position = np.clip(level, 0.0, 1.0) * (len(LEVEL_COLOURS) - 1)
    return LEVEL_COLOURS[int(round(position))]


def _path(points: np.ndarray, transform) -> str:
    finite = np.all(np.isfinite(points), axis=1)
    if not np.any(finite):
        return ""
    chunks: list[str] = []
    active = False
    for point, keep in zip(points, finite, strict=True):
        if not keep:
            active = False
            continue
        x, y = transform(point)
        chunks.append(f"{'L' if active else 'M'}{x:.2f},{y:.2f}")
        active = True
    if np.all(finite):
        chunks.append("Z")
    return " ".join(chunks)


def _svg(
    output: Path,
    levels: np.ndarray,
    panels: list[dict[str, Any]],
    coarse_summary: dict[str, float],
    fine_summary: dict[str, float],
) -> None:
    width, height = 1580, 650
    panel_width, panel_height = 440, 470
    lefts = (55, 545, 1035)
    top = 96
    all_wall = np.vstack([panel["wall"] for panel in panels])
    r_min, z_min = np.min(all_wall, axis=0)
    r_max, z_max = np.max(all_wall, axis=0)
    span_r = r_max - r_min
    span_z = z_max - z_min
    pad = 0.035
    r_min -= pad * span_r
    r_max += pad * span_r
    z_min -= pad * span_z
    z_max += pad * span_z
    scale = min(panel_width / (r_max - r_min), panel_height / (z_max - z_min))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Banked Solov\'ev flux-contour comparison</title>',
        '<desc id="description">Twenty-one matched normalized-flux contours compare the analytic reference, the coarse legacy forward solve, and the fine moment-blend root.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="system-ui, sans-serif" fill="#171717">',
        '<text x="55" y="32" font-size="21" font-weight="650">Discrete flux contours at 21 matched ψN levels</text>',
        '<text x="55" y="57" font-size="12.5" fill="#555">Dashed: analytic Solov\'ev reference · solid: banked forward state · radial shape distances use 720 common poloidal angles</text>',
    ]

    for panel_index, panel in enumerate(panels):
        left = lefts[panel_index]

        def transform(point, left=left):
            x = left + (point[0] - r_min) * scale
            y = top + panel_height - (point[1] - z_min) * scale
            return x, y

        lines.extend(
            [
                f'<text x="{left}" y="82" font-size="14" font-weight="650">{escape(panel["title"])}</text>',
                f'<text x="{left}" y="608" font-size="11.5" fill="#555">{escape(panel["subtitle"])}</text>',
                f'<path d="{_path(panel["wall"], transform)}" fill="none" stroke="#222" stroke-width="1.25"/>',
            ]
        )
        for level_index, level in enumerate(levels):
            if level_index == 0:
                continue
            colour = _colour(float(level))
            analytic_path = _path(panel["analytic"][level_index], transform)
            lines.append(
                f'<path d="{analytic_path}" fill="none" stroke="{colour}" stroke-width="1.05" stroke-dasharray="4 3" opacity="0.88"/>'
            )
            if panel.get("state") is not None:
                state_path = _path(panel["state"][level_index], transform)
                lines.append(
                    f'<path d="{state_path}" fill="none" stroke="{colour}" stroke-width="1.45" opacity="0.92"/>'
                )
        analytic_axis = transform(panel["analytic_axis"])
        analytic_x = transform(panel["analytic_x_point"])
        lines.extend(
            [
                f'<path d="M{analytic_axis[0] - 4:.2f},{analytic_axis[1] - 4:.2f} L{analytic_axis[0] + 4:.2f},{analytic_axis[1] + 4:.2f} M{analytic_axis[0] - 4:.2f},{analytic_axis[1] + 4:.2f} L{analytic_axis[0] + 4:.2f},{analytic_axis[1] - 4:.2f}" stroke="#b2182b" stroke-width="1.6"/>',
                f'<path d="M{analytic_x[0]:.2f},{analytic_x[1] - 5:.2f} L{analytic_x[0] + 5:.2f},{analytic_x[1]:.2f} L{analytic_x[0]:.2f},{analytic_x[1] + 5:.2f} L{analytic_x[0] - 5:.2f},{analytic_x[1]:.2f} Z" fill="none" stroke="#b2182b" stroke-width="1.4"/>',
            ]
        )
        if panel.get("state_axis") is not None:
            state_axis = transform(panel["state_axis"])
            state_x = transform(panel["state_x_point"])
            lines.extend(
                [
                    f'<circle cx="{state_axis[0]:.2f}" cy="{state_axis[1]:.2f}" r="3.7" fill="#2166ac" stroke="white" stroke-width="0.8"/>',
                    f'<path d="M{state_x[0]:.2f},{state_x[1] - 4.5:.2f} L{state_x[0] + 4.5:.2f},{state_x[1]:.2f} L{state_x[0]:.2f},{state_x[1] + 4.5:.2f} L{state_x[0] - 4.5:.2f},{state_x[1]:.2f} Z" fill="#2166ac" stroke="white" stroke-width="0.7"/>',
                ]
            )
        lines.extend(
            [
                f'<line x1="{left}" y1="{top + panel_height}" x2="{left + panel_width}" y2="{top + panel_height}" stroke="#777" stroke-width="0.7"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}" stroke="#777" stroke-width="0.7"/>',
                f'<text x="{left + panel_width / 2:.1f}" y="590" text-anchor="middle" font-size="11.5">R [m]</text>',
                f'<text x="{left - 39}" y="{top + panel_height / 2:.1f}" text-anchor="middle" font-size="11.5" transform="rotate(-90 {left - 39} {top + panel_height / 2:.1f})">Z [m]</text>',
            ]
        )

    legend_x = 1488
    legend_top = 120
    legend_height = 390
    for index in range(100):
        level = 1.0 - index / 99.0
        y = legend_top + index * legend_height / 100.0
        lines.append(
            f'<rect x="{legend_x}" y="{y:.2f}" width="13" height="{legend_height / 100 + 0.6:.2f}" fill="{_colour(level)}"/>'
        )
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = legend_top + (1.0 - level) * legend_height
        lines.append(
            f'<text x="{legend_x + 19}" y="{y + 4:.2f}" font-size="10.5">{level:.2f}</text>'
        )
    lines.extend(
        [
            f'<text x="{legend_x - 2}" y="{legend_top - 12}" font-size="11.5">ψN</text>',
            '<g font-size="10.8" fill="#444">',
            '<path d="M55,630 h22" stroke="#b2182b" stroke-width="1.5"/><path d="M58,626 l8,8 m0,-8 l-8,8" stroke="#b2182b" stroke-width="1.3"/><text x="83" y="634">analytic axis / X-point</text>',
            '<circle cx="285" cy="630" r="3.5" fill="#2166ac"/><path d="M307,625 l5,5 l-5,5 l-5,-5 Z" fill="#2166ac"/><text x="320" y="634">forward axis / X-point</text>',
            f'<text x="545" y="634">coarse: mean {coarse_summary["mean"]:.3f} m · max {coarse_summary["max"]:.3f} m</text>',
            f'<text x="1035" y="634">fine t=0.5875: mean {fine_summary["mean"]:.3f} m · max {fine_summary["max"]:.3f} m</text>',
            "</g>",
            "</g>",
            "</svg>",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def _summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    physical = [row for row in rows if row["psi_norm"] > 0.0]
    return {
        "mean": float(
            np.mean([row["mean_absolute_radial_distance_m"] for row in physical])
        ),
        "max": float(
            np.max([row["maximum_absolute_radial_distance_m"] for row in physical])
        ),
    }


def run(bank: Path, output: Path) -> dict[str, Any]:
    manifest, entries = _bank_entries(bank)
    coarse = _load_checkpoint(bank / COARSE_FILE)
    fine = _load_checkpoint(bank / FINE_FILE)
    levels = coarse["normalized_contour_levels"]
    if not np.array_equal(levels, fine["normalized_contour_levels"]):
        raise ValueError("coarse and fine checkpoints use different contour levels")
    if not np.array_equal(levels, np.linspace(0.0, 1.0, 21)):
        raise ValueError("checkpoint contour levels are not the banked 0:0.05:1 grid")

    angles = np.linspace(-math.pi, math.pi, ANGLE_COUNT, endpoint=False)
    coarse_analytic = _contours(
        coarse["state_nodes"],
        coarse["analytic_psi_norm"],
        coarse["reference_axis"],
        coarse["reference_x_point"],
        coarse["wall_contour"],
        levels,
        angles,
        boundary=coarse["reference_boundary"],
    )
    coarse_state = _contours(
        coarse["state_nodes"],
        coarse["solved_psi_norm"],
        coarse["solved_axis"],
        coarse["solved_x_point"],
        coarse["wall_contour"],
        levels,
        angles,
    )
    fine_analytic = _contours(
        fine["state_nodes"],
        fine["analytic_psi_norm"],
        fine["reference_axis"],
        fine["reference_x_point"],
        fine["wall_contour"],
        levels,
        angles,
        boundary=fine["reference_boundary"],
    )
    fine_state = _contours(
        fine["state_nodes"],
        fine["solved_psi_norm"],
        fine["solved_axis"],
        fine["solved_x_point"],
        fine["wall_contour"],
        levels,
        angles,
    )

    analytic_rows = [
        {
            "psi_norm": float(level),
            "mean_absolute_radial_distance_m": 0.0,
            "maximum_absolute_radial_distance_m": 0.0,
            "valid_ray_count": ANGLE_COUNT,
            "requested_ray_count": ANGLE_COUNT,
        }
        for level in levels
    ]
    coarse_rows = _distance_rows(
        levels, coarse_state["radii"], coarse_analytic["radii"]
    )
    fine_rows = _distance_rows(levels, fine_state["radii"], fine_analytic["radii"])
    coarse_summary = _summary(coarse_rows)
    fine_summary = _summary(fine_rows)

    coarse_receipt = entries[COARSE_FILE]["receipt"]
    fine_receipt = entries[FINE_FILE]["receipt"]
    panels = [
        {
            "title": "Analytic Solov'ev reference",
            "subtitle": f"fine bank · {len(fine['plasma_nodes']):,} plasma nodes",
            "wall": fine["wall_contour"],
            "analytic": fine_analytic["coordinates"],
            "state": None,
            "analytic_axis": fine["reference_axis"],
            "analytic_x_point": fine["reference_x_point"][0],
            "state_axis": None,
        },
        {
            "title": "Coarse legacy forward solve",
            "subtitle": f"{len(coarse['plasma_nodes']):,} plasma nodes · residual {coarse_receipt['residual']:.2e}",
            "wall": coarse["wall_contour"],
            "analytic": coarse_analytic["coordinates"],
            "state": coarse_state["coordinates"],
            "analytic_axis": coarse["reference_axis"],
            "analytic_x_point": coarse["reference_x_point"][0],
            "state_axis": coarse["solved_axis"],
            "state_x_point": coarse["solved_x_point"],
        },
        {
            "title": "Fine moment-blend root · t=0.5875",
            "subtitle": f"{len(fine['plasma_nodes']):,} plasma nodes · residual {fine_receipt['residual']:.2e}",
            "wall": fine["wall_contour"],
            "analytic": fine_analytic["coordinates"],
            "state": fine_state["coordinates"],
            "analytic_axis": fine["reference_axis"],
            "analytic_x_point": fine["reference_x_point"][0],
            "state_axis": fine["solved_axis"],
            "state_x_point": fine["solved_x_point"],
        },
    ]

    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / "solovev_contour_overlay.svg"
    findings_path = output / "solovev_contour_findings.json"
    _svg(figure_path, levels, panels, coarse_summary, fine_summary)

    findings = {
        "schema": "nova.solovev-contour-comparison.v1",
        "pure_consumer": True,
        "state_bank": {
            "manifest": str(bank / "state-bank-manifest.json"),
            "schema": manifest["schema"],
            "normalized_contour_level_count": int(len(levels)),
            "levels": levels.tolist(),
            "reference_fixture": manifest["reference_fixture"],
        },
        "metric": {
            "name": "absolute radial contour-shape distance",
            "definition": "abs(r_state(theta) - r_analytic(theta)) at matched psi_norm and 720 common poloidal angles; each radius is measured about that state's banked axis",
            "interpolation": "piecewise linear over banked state nodes; no equilibrium or source physics evaluated",
            "radial_sample_count": RADIAL_SAMPLE_COUNT,
            "angle_count": ANGLE_COUNT,
            "units": "m",
        },
        "states": {
            "analytic_solovev": {
                "label": "analytic Solov'ev reference",
                "grid": "fine bank",
                "plasma_node_count": int(len(fine["plasma_nodes"])),
                "axis": fine["reference_axis"].tolist(),
                "x_point": fine["reference_x_point"][0].tolist(),
                "contours": analytic_rows,
            },
            "coarse_legacy": {
                "label": "coarse legacy forward solve",
                "checkpoint": COARSE_FILE,
                "sha256": entries[COARSE_FILE]["sha256"],
                "residual": coarse_receipt["residual"],
                "homotopy_fraction": coarse_receipt["fraction"],
                "plasma_node_count": int(len(coarse["plasma_nodes"])),
                "wall_target_count": int(len(coarse["wall_nodes"])),
                "axis": coarse["solved_axis"].tolist(),
                "reference_axis": coarse["reference_axis"].tolist(),
                "axis_offset_m": float(
                    np.linalg.norm(coarse["solved_axis"] - coarse["reference_axis"])
                ),
                "x_point": coarse["solved_x_point"].tolist(),
                "reference_x_point": coarse["reference_x_point"][0].tolist(),
                "x_point_offset_m": float(
                    np.linalg.norm(
                        coarse["solved_x_point"] - coarse["reference_x_point"][0]
                    )
                ),
                "mean_across_nonzero_levels_m": coarse_summary["mean"],
                "maximum_across_all_levels_m": coarse_summary["max"],
                "contours": coarse_rows,
            },
            "fine_moment_blend": {
                "label": "fine last accepted moment-blend root",
                "checkpoint": FINE_FILE,
                "sha256": entries[FINE_FILE]["sha256"],
                "residual": fine_receipt["residual"],
                "homotopy_fraction": fine_receipt["fraction"],
                "plasma_node_count": int(len(fine["plasma_nodes"])),
                "wall_target_count": int(len(fine["wall_nodes"])),
                "axis": fine["solved_axis"].tolist(),
                "reference_axis": fine["reference_axis"].tolist(),
                "axis_offset_m": float(
                    np.linalg.norm(fine["solved_axis"] - fine["reference_axis"])
                ),
                "x_point": fine["solved_x_point"].tolist(),
                "reference_x_point": fine["reference_x_point"][0].tolist(),
                "x_point_offset_m": float(
                    np.linalg.norm(
                        fine["solved_x_point"] - fine["reference_x_point"][0]
                    )
                ),
                "mean_across_nonzero_levels_m": fine_summary["mean"],
                "maximum_across_all_levels_m": fine_summary["max"],
                "contours": fine_rows,
            },
        },
        "artifacts": {"figure": str(figure_path), "findings": str(findings_path)},
    }
    findings_path.write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    findings = run(args.bank, args.output)
    coarse = findings["states"]["coarse_legacy"]
    fine = findings["states"]["fine_moment_blend"]
    print(
        "contour_overlay "
        f"levels={findings['state_bank']['normalized_contour_level_count']} "
        f"coarse_mean_m={coarse['mean_across_nonzero_levels_m']:.9g} "
        f"coarse_max_m={coarse['maximum_across_all_levels_m']:.9g} "
        f"fine_mean_m={fine['mean_across_nonzero_levels_m']:.9g} "
        f"fine_max_m={fine['maximum_across_all_levels_m']:.9g}"
    )


if __name__ == "__main__":
    main()
