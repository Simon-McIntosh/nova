# ruff: noqa: E501
"""Render the MAST reference flux contours from a committed tabular source."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import contourpy
import numpy as np


SHOT = 21978
SLICE_INDEX = 46
LOBE_LEVEL = 0.675
REFERENCE_LEVELS = (0.10, 0.25, 0.50, LOBE_LEVEL, 0.80, 0.99)
LOBE_RADIUS_M = 0.5753124990151264
LOBE_HEIGHTS_M = (-1.8125, -1.75, 1.75, 1.8125)
LEVEL1_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")
SOURCE_PATH = Path(__file__).with_name("contour_source.tsv")
FIGURE_PATH = (
    Path(__file__).parents[2]
    / "docs/figures/flux-function-forward-transport/mast-flux-contours.svg"
)


@dataclass(frozen=True)
class Marker:
    name: str
    radius_m: float
    height_m: float
    psi_n: float


@dataclass(frozen=True)
class Evidence:
    metadata: dict[str, str]
    radius_m: np.ndarray
    height_m: np.ndarray
    psi_n: np.ndarray
    inside_limiter: np.ndarray
    boundary_fill: np.ndarray
    limiter: np.ndarray
    structures: dict[tuple[str, int], np.ndarray]
    markers: tuple[Marker, ...]


@dataclass(frozen=True)
class Panel:
    left: float
    top: float
    width: float
    height: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def point(self, radius: float, height: float) -> tuple[float, float]:
        x = self.left + self.width * (radius - self.x_min) / (self.x_max - self.x_min)
        y = self.top + self.height * (self.y_max - height) / (self.y_max - self.y_min)
        return x, y


def _axis_fill(mask: np.ndarray, psi_n: np.ndarray) -> np.ndarray:
    """Return the four-neighbour component containing the magnetic-axis node."""

    start = np.unravel_index(np.argmin(np.where(mask, psi_n, np.inf)), mask.shape)
    filled = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque([start])
    filled[start] = True
    while queue:
        row, column = queue.popleft()
        for row_step, column_step in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = (row + row_step, column + column_step)
            if not (
                0 <= candidate[0] < mask.shape[0] and 0 <= candidate[1] < mask.shape[1]
            ):
                continue
            if mask[candidate] and not filled[candidate]:
                filled[candidate] = True
                queue.append(candidate)
    return filled


def _polygons(geometry: object) -> Iterable[object]:
    if getattr(geometry, "geom_type", "") == "Polygon":
        yield geometry
    else:
        yield from geometry.geoms


def _format(value: float) -> str:
    return format(float(value), ".17g")


def refresh_source(path: Path) -> None:
    """Extract the immutable slice and nearby machine structures into TSV."""

    import zarr
    from shapely import from_wkb
    from shapely.geometry import Point

    from nova.catalog.mast_geometry import MachineGeometryRegistry
    from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
    from nova.equilibrium.wall_mask import inside_polygon

    group = zarr.open_group(str(LEVEL1_ROOT / f"{SHOT}.zarr"), mode="r")["efm"]
    stored_radius = np.asarray(group["gridr"][:])
    stored_height = np.asarray(group["gridz"][:])
    radius = np.linspace(
        float(stored_radius[0]), float(stored_radius[-1]), stored_radius.size
    )
    height = np.linspace(
        float(stored_height[0]), float(stored_height[-1]), stored_height.size
    )
    raw_flux = np.asarray(group["psirz"][SLICE_INDEX], dtype=np.float64)
    finite_columns = np.flatnonzero(np.all(np.isfinite(raw_flux), axis=0))
    if finite_columns.size != radius.size:
        raise ValueError("the selected flux map lacks a finite radial column")
    flux = TOTAL_FLUX_FACTOR * raw_flux[:, finite_columns]
    axis_psi = TOTAL_FLUX_FACTOR * float(group["psi_axis"][SLICE_INDEX])
    boundary_psi = TOTAL_FLUX_FACTOR * float(group["psi_boundary"][SLICE_INDEX])
    psi_n = (flux - axis_psi) / (boundary_psi - axis_psi)

    limiter_radius = np.asarray(group["limiterr"][:], dtype=np.float64)
    limiter_height = np.asarray(group["limiterz"][:], dtype=np.float64)
    finite_limiter = np.isfinite(limiter_radius) & np.isfinite(limiter_height)
    limiter = np.column_stack(
        (limiter_radius[finite_limiter], limiter_height[finite_limiter])
    )
    radius_grid, height_grid = np.meshgrid(radius, height)
    inside_limiter = np.asarray(
        inside_polygon(
            radius_grid.reshape(-1),
            height_grid.reshape(-1),
            limiter[:, 0],
            limiter[:, 1],
        )
    ).reshape(psi_n.shape)
    boundary_mask = (psi_n < 1.0) & inside_limiter
    boundary_fill = _axis_fill(boundary_mask, psi_n)

    selection = MachineGeometryRegistry.default().select(SHOT)
    machine = selection.configuration.geometry
    decoded = {
        group_name: {
            name: from_wkb(bytes.fromhex(encoded))
            for name, encoded in machine[group_name].items()
        }
        for group_name in ("active_components", "passive_components")
    }
    lobe_points = tuple(Point(LOBE_RADIUS_M, z) for z in LOBE_HEIGHTS_M)
    selected_structures: dict[str, list[np.ndarray]] = {}
    for name in ("p2_outer_lower", "p2_outer_upper"):
        selected_structures[name] = [
            np.asarray(polygon.exterior.coords, dtype=np.float64)
            for polygon in _polygons(decoded["active_components"][name])
        ]
    nearby_cases = []
    for polygon in _polygons(decoded["passive_components"]["coil_cases"]):
        if min(point.distance(polygon) for point in lobe_points) < 0.15:
            nearby_cases.append(np.asarray(polygon.exterior.coords, dtype=np.float64))
    selected_structures["coil_cases"] = nearby_cases

    markers = []
    for target_height in LOBE_HEIGHTS_M:
        row = int(np.argmin(np.abs(height - target_height)))
        column = int(np.argmin(np.abs(radius - LOBE_RADIUS_M)))
        if not math.isclose(radius[column], LOBE_RADIUS_M, rel_tol=0.0, abs_tol=2e-15):
            raise ValueError("recorded lobe radius is not a grid node")
        if not math.isclose(height[row], target_height, rel_tol=0.0, abs_tol=2e-15):
            raise ValueError("recorded lobe height is not a grid node")
        hemisphere = "lower" if target_height < 0.0 else "upper"
        position = "outer" if abs(target_height) > 1.8 else "inner"
        markers.append(
            Marker(
                name=f"lobe_{hemisphere}_{position}",
                radius_m=float(radius[column]),
                height_m=float(height[row]),
                psi_n=float(psi_n[row, column]),
            )
        )

    passive_distances = [
        point.distance(decoded["passive_components"]["coil_cases"])
        for point in lobe_points
    ]
    active_distances = []
    for point in lobe_points:
        active_distances.append(
            min(
                point.distance(decoded["active_components"]["p2_outer_lower"]),
                point.distance(decoded["active_components"]["p2_outer_upper"]),
            )
        )

    metadata = {
        "shot": str(SHOT),
        "slice_index": str(SLICE_INDEX),
        "time_s": _format(float(np.asarray(group["time"][:])[SLICE_INDEX])),
        "flux_unit": "Wb",
        "normalisation": "(psi-axis_psi)/(boundary_psi-axis_psi)",
        "axis_psi_Wb": _format(axis_psi),
        "boundary_psi_Wb": _format(boundary_psi),
        "axis_r_m": _format(float(group["magnetic_axis_r"][SLICE_INDEX])),
        "axis_z_m": _format(float(group["magnetic_axis_z"][SLICE_INDEX])),
        "grid_rows": str(height.size),
        "grid_columns": str(radius.size),
        "machine_physical_digest": selection.configuration.physical_digest,
        "machine_evidence": selection.evidence.value,
        "nearest_passive_structure": "coil_cases",
        "nearest_passive_distance_m": _format(min(passive_distances)),
        "nearest_active_structures": "p2_outer_lower,p2_outer_upper",
        "nearest_active_distance_min_m": _format(min(active_distances)),
        "nearest_active_distance_max_m": _format(max(active_distances)),
        "source": str(LEVEL1_ROOT / f"{SHOT}.zarr/efm"),
    }

    columns = (
        "record_type",
        "series",
        "part",
        "order",
        "r_m",
        "z_m",
        "psi_n",
        "inside_limiter",
        "boundary_fill",
        "value",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for name, value in metadata.items():
            writer.writerow({"record_type": "metadata", "series": name, "value": value})
        for row, z_value in enumerate(height):
            for column, r_value in enumerate(radius):
                writer.writerow(
                    {
                        "record_type": "grid",
                        "series": "normalised_flux",
                        "part": row,
                        "order": column,
                        "r_m": _format(r_value),
                        "z_m": _format(z_value),
                        "psi_n": _format(psi_n[row, column]),
                        "inside_limiter": int(inside_limiter[row, column]),
                        "boundary_fill": int(boundary_fill[row, column]),
                        "value": "-",
                    }
                )
        for order, point in enumerate(limiter):
            writer.writerow(
                {
                    "record_type": "limiter",
                    "series": "limiter",
                    "part": 0,
                    "order": order,
                    "r_m": _format(point[0]),
                    "z_m": _format(point[1]),
                    "value": "-",
                }
            )
        for name, parts in selected_structures.items():
            for part, polygon in enumerate(parts):
                for order, point in enumerate(polygon):
                    writer.writerow(
                        {
                            "record_type": "structure",
                            "series": name,
                            "part": part,
                            "order": order,
                            "r_m": _format(point[0]),
                            "z_m": _format(point[1]),
                            "value": "-",
                        }
                    )
        for marker in markers:
            writer.writerow(
                {
                    "record_type": "marker",
                    "series": marker.name,
                    "r_m": _format(marker.radius_m),
                    "z_m": _format(marker.height_m),
                    "psi_n": _format(marker.psi_n),
                    "value": "-",
                }
            )


def read_source(path: Path) -> Evidence:
    metadata: dict[str, str] = {}
    grid_rows: list[dict[str, str]] = []
    limiter_rows: list[dict[str, str]] = []
    structure_rows: dict[tuple[str, int], list[dict[str, str]]] = {}
    markers = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            record_type = row["record_type"]
            if record_type == "metadata":
                metadata[row["series"]] = row["value"]
            elif record_type == "grid":
                grid_rows.append(row)
            elif record_type == "limiter":
                limiter_rows.append(row)
            elif record_type == "structure":
                key = (row["series"], int(row["part"]))
                structure_rows.setdefault(key, []).append(row)
            elif record_type == "marker":
                markers.append(
                    Marker(
                        name=row["series"],
                        radius_m=float(row["r_m"]),
                        height_m=float(row["z_m"]),
                        psi_n=float(row["psi_n"]),
                    )
                )
            else:
                raise ValueError(f"unknown source record type {record_type!r}")

    row_count = int(metadata["grid_rows"])
    column_count = int(metadata["grid_columns"])
    if len(grid_rows) != row_count * column_count:
        raise ValueError("the tabulated grid shape disagrees with its metadata")
    radius = np.asarray([float(row["r_m"]) for row in grid_rows]).reshape(
        row_count, column_count
    )
    height = np.asarray([float(row["z_m"]) for row in grid_rows]).reshape(
        row_count, column_count
    )
    psi_n = np.asarray([float(row["psi_n"]) for row in grid_rows]).reshape(
        row_count, column_count
    )
    inside_limiter = np.asarray(
        [bool(int(row["inside_limiter"])) for row in grid_rows]
    ).reshape(row_count, column_count)
    boundary_fill = np.asarray(
        [bool(int(row["boundary_fill"])) for row in grid_rows]
    ).reshape(row_count, column_count)
    limiter = np.asarray(
        [(float(row["r_m"]), float(row["z_m"])) for row in limiter_rows]
    )
    structures = {
        key: np.asarray([(float(row["r_m"]), float(row["z_m"])) for row in rows])
        for key, rows in structure_rows.items()
    }
    return Evidence(
        metadata=metadata,
        radius_m=radius[0],
        height_m=height[:, 0],
        psi_n=psi_n,
        inside_limiter=inside_limiter,
        boundary_fill=boundary_fill,
        limiter=limiter,
        structures=structures,
        markers=tuple(markers),
    )


def _closed(points: np.ndarray) -> bool:
    return bool(np.linalg.norm(points[0] - points[-1]) < 1.0e-9)


def _contains(points: np.ndarray, point: tuple[float, float]) -> bool:
    x, y = point
    left = points[:, 0]
    vertical = points[:, 1]
    right = np.roll(left, -1)
    next_vertical = np.roll(vertical, -1)
    crossing = (vertical > y) != (next_vertical > y)
    intersect = left + (y - vertical) * (right - left) / np.where(
        next_vertical == vertical, np.inf, next_vertical - vertical
    )
    return bool(np.count_nonzero(crossing & (intersect > x)) % 2)


def validate(evidence: Evidence) -> dict[str, object]:
    if evidence.metadata["shot"] != str(SHOT):
        raise ValueError("source shot does not match the fixed reference")
    if evidence.metadata["slice_index"] != str(SLICE_INDEX):
        raise ValueError("source slice does not match the fixed reference")
    contour = contourpy.contour_generator(
        x=evidence.radius_m,
        y=evidence.height_m,
        z=evidence.psi_n,
        name="serial",
    )
    axis = (
        float(evidence.metadata["axis_r_m"]),
        float(evidence.metadata["axis_z_m"]),
    )
    core_components = {}
    for level in REFERENCE_LEVELS:
        segments = contour.lines(level)
        core = [
            segment
            for segment in segments
            if _closed(segment) and _contains(segment, axis)
        ]
        if len(core) != 1:
            raise ValueError(f"level {level:g} has {len(core)} axis-connected contours")
        core_components[level] = core[0]
    lobe_segments = [
        segment
        for segment in contour.lines(LOBE_LEVEL)
        if _closed(segment) and not _contains(segment, axis)
    ]
    if len(lobe_segments) != 2:
        raise ValueError(
            "the emphasized level does not have two closed off-axis pockets"
        )

    level_mask = (evidence.psi_n < LOBE_LEVEL) & evidence.inside_limiter
    level_core = _axis_fill(level_mask, evidence.psi_n)
    excluded = np.argwhere(level_mask & ~level_core)
    if excluded.shape != (4, 2):
        raise ValueError(
            f"expected four excluded lobe nodes, found {excluded.shape[0]}"
        )
    boundary_mask = (evidence.psi_n < 1.0) & evidence.inside_limiter
    if not np.array_equal(
        _axis_fill(boundary_mask, evidence.psi_n), evidence.boundary_fill
    ):
        raise ValueError(
            "stored boundary fill is not the axis-connected one-fill footprint"
        )
    if np.count_nonzero(boundary_mask & ~evidence.boundary_fill):
        raise ValueError("boundary-level footprint unexpectedly has disconnected nodes")

    marker_values = {marker.name: marker.psi_n for marker in evidence.markers}
    expected_markers = {
        "lobe_lower_outer": 0.6714599187524771,
        "lobe_lower_inner": 0.6611291377126922,
        "lobe_upper_inner": 0.6643434399443583,
        "lobe_upper_outer": 0.6719510697522671,
    }
    if marker_values.keys() != expected_markers.keys():
        raise ValueError("lobe marker identities changed")
    for name, expected in expected_markers.items():
        if not math.isclose(marker_values[name], expected, rel_tol=0.0, abs_tol=2e-15):
            raise ValueError(f"{name} psi_n changed from the recorded value")

    return {
        "contour": contour,
        "core_components": core_components,
        "lobe_segments": lobe_segments,
        "excluded_nodes": int(excluded.shape[0]),
        "level_core_nodes": int(level_core.sum()),
        "boundary_fill_nodes": int(evidence.boundary_fill.sum()),
    }


def _path(points: np.ndarray, panel: Panel, *, absolute_height: bool = False) -> str:
    commands = []
    for index, (radius, height) in enumerate(points):
        if absolute_height:
            height = abs(height)
        x, y = panel.point(float(radius), float(height))
        commands.append(f"{'M' if index == 0 else 'L'}{x:.2f},{y:.2f}")
    if _closed(points):
        commands.append("Z")
    return " ".join(commands)


def _text(
    x: float,
    y: float,
    value: str,
    *,
    css: str = "text",
    size: float = 16.0,
    weight: int = 400,
    anchor: str = "start",
    transform: str = "",
) -> str:
    escaped = html.escape(value)
    transform_attribute = f' transform="{transform}"' if transform else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" class="{css}" font-size="{size:.1f}" '
        f'font-weight="{weight}" text-anchor="{anchor}"{transform_attribute}>'
        f"{escaped}</text>"
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    css: str,
    *,
    width: float = 1.0,
    dash: str = "",
) -> str:
    dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'class="{css}" stroke-width="{width:.2f}"{dash_attribute}/>'
    )


def _axes(
    lines: list[str],
    panel: Panel,
    x_ticks: tuple[float, ...],
    y_ticks: tuple[float, ...],
) -> None:
    for tick in x_ticks:
        x, bottom = panel.point(tick, panel.y_min)
        lines.append(_line(x, panel.top, x, bottom, "grid", width=0.7))
        lines.append(_text(x, bottom + 18, f"{tick:g}", size=13, anchor="middle"))
    for tick in y_ticks:
        left, y = panel.point(panel.x_min, tick)
        lines.append(_line(left, y, left + panel.width, y, "grid", width=0.7))
        lines.append(_text(left - 9, y + 4, f"{tick:g}", size=13, anchor="end"))
    lines.append(
        _line(
            panel.left,
            panel.top,
            panel.left,
            panel.top + panel.height,
            "axis",
            width=1.2,
        )
    )
    lines.append(
        _line(
            panel.left,
            panel.top + panel.height,
            panel.left + panel.width,
            panel.top + panel.height,
            "axis",
            width=1.2,
        )
    )


def render_svg(evidence: Evidence, validation: dict[str, object]) -> str:
    main = Panel(48, 42, 196, 404, 0.06, 2.0, -2.0, 2.0)
    zoom = Panel(48, 514, 212, 212, 0.4, 0.8, 1.5, 1.9)
    contour = validation["contour"]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="780" '
        'viewBox="0 0 640 780" role="img" aria-labelledby="mast_title mast_desc">',
        '<title id="mast_title">MAST shot 21978 normalised-flux contours and divertor-coil pockets</title>',
        '<desc id="mast_desc">A full R-Z contour map and an absolute-Z divertor zoom show one nested axis-connected plasma family and two closed off-axis flux pockets beside the P2 divertor coils. Four lobe nodes are quoted. Reusing the boundary-level one-fill footprint, not plasma nestedness, is the extraction defect.</desc>',
        "<style>",
        ".text{fill:#202124;font-family:system-ui,sans-serif}.muted{fill:#62666a;font-family:system-ui,sans-serif}.axis{stroke:#202124}.grid{stroke:#dadce0}.contour{stroke:#70757a}.focus{stroke:#b3261e}.fill-outline{stroke:#1967d2}.limiter{stroke:#202124}.structure{stroke:#80868b;fill:#bdc1c6;fill-opacity:.28}.leader{stroke:#5f6368}.marker{stroke:#b3261e;fill:#fff}.axis-marker{stroke:#202124;fill:#202124}",
        "@media (prefers-color-scheme:dark){.text{fill:#f1f3f4}.muted{fill:#c5c9ce}.axis{stroke:#f1f3f4}.grid{stroke:#454a50}.contour{stroke:#aeb4ba}.focus{stroke:#ffb4ab}.fill-outline{stroke:#8ab4f8}.limiter{stroke:#f1f3f4}.structure{stroke:#bdc1c6;fill:#80868b;fill-opacity:.32}.leader{stroke:#c5c9ce}.marker{stroke:#ffb4ab;fill:#202124}.axis-marker{stroke:#f1f3f4;fill:#f1f3f4}}",
        "</style>",
        '<defs><clipPath id="main_clip"><rect x="48" y="42" width="196" height="404"/></clipPath><clipPath id="zoom_clip"><rect x="48" y="514" width="212" height="212"/></clipPath></defs>',
    ]
    _axes(lines, main, (0.5, 1.0, 1.5, 2.0), (-2.0, -1.0, 0.0, 1.0, 2.0))
    lines.append(
        _text(
            main.left + main.width / 2,
            482,
            "major radius R (m)",
            size=15,
            anchor="middle",
        )
    )
    lines.append(
        _text(
            15,
            main.top + main.height / 2,
            "height Z (m)",
            size=15,
            anchor="middle",
            transform=f"rotate(-90 15 {main.top + main.height / 2:.2f})",
        )
    )
    lines.append(_text(48, 26, "full field · R–Z", size=17, weight=650))

    for points in evidence.structures.values():
        lines.append(
            f'<path d="{_path(points, main)}" class="structure" stroke-width="0.9" clip-path="url(#main_clip)"/>'
        )
    lines.append(
        f'<path d="{_path(evidence.limiter, main)}" class="limiter" fill="none" stroke-width="1.35" clip-path="url(#main_clip)"/>'
    )
    mask_contour = contourpy.contour_generator(
        x=evidence.radius_m,
        y=evidence.height_m,
        z=evidence.boundary_fill.astype(float),
        name="serial",
    )
    mask_segments = mask_contour.lines(0.5)
    for points in mask_segments:
        lines.append(
            f'<path d="{_path(points, main)}" class="fill-outline" fill="none" stroke-width="1.45" stroke-dasharray="6 4" clip-path="url(#main_clip)"/>'
        )
    for level in REFERENCE_LEVELS:
        css = "focus" if level == LOBE_LEVEL else "contour"
        width = 2.35 if level == LOBE_LEVEL else 1.0
        for points in contour.lines(level):
            lines.append(
                f'<path d="{_path(points, main)}" class="{css}" fill="none" stroke-width="{width:.2f}" clip-path="url(#main_clip)"/>'
            )
    axis_x, axis_y = main.point(
        float(evidence.metadata["axis_r_m"]), float(evidence.metadata["axis_z_m"])
    )
    lines.append(
        f'<circle cx="{axis_x:.2f}" cy="{axis_y:.2f}" r="3.2" class="axis-marker"/>'
    )
    lines.append(_text(axis_x + 7, axis_y - 7, "magnetic axis", size=12.5, weight=600))
    for marker in evidence.markers:
        x, y = main.point(marker.radius_m, marker.height_m)
        lines.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.6" class="marker" stroke-width="1.7"/>'
        )

    for level, core in validation["core_components"].items():
        point = core[int(np.argmax(core[:, 1]))]
        x, y = main.point(float(point[0]), float(point[1]))
        lines.append(
            _text(
                x + 5,
                y - 4,
                f"{level:g}",
                css="text" if level == LOBE_LEVEL else "muted",
                size=12.5,
                weight=650 if level == LOBE_LEVEL else 400,
            )
        )

    limiter_point = evidence.limiter[int(np.argmax(evidence.limiter[:, 0]))]
    limiter_x, limiter_y = main.point(float(limiter_point[0]), float(limiter_point[1]))
    lines.append(
        _text(
            limiter_x - 5,
            limiter_y - 5,
            "limiter",
            size=12.5,
            weight=600,
            anchor="end",
        )
    )
    fill_segment = max(mask_segments, key=lambda points: float(np.max(points[:, 1])))
    fill_point = fill_segment[int(np.argmax(fill_segment[:, 1]))]
    fill_x, fill_y = main.point(float(fill_point[0]), float(fill_point[1]))
    lines.append(_text(fill_x + 5, fill_y - 5, "ψₙ<1 axis fill", size=12.5, weight=600))

    lines.extend(
        [
            _text(
                278, 49, "shot 21978 · slice 46 · t = 0.245 s", size=16.5, weight=650
            ),
            _text(278, 77, "At ψₙ = 0.675:", size=15.5, weight=650),
            _text(278, 100, "one closed contour surrounds the axis", size=14.5),
            _text(278, 122, "two closed contours surround off-axis pockets", size=14.5),
            _text(278, 174, "Adjudication", size=17, weight=700),
            _text(278, 201, "Closed divertor-coil flux pockets,", size=16, weight=650),
            _text(278, 224, "outside the axis-connected plasma.", size=16, weight=650),
            _text(278, 255, "The plasma contour family is nested.", size=15),
            _text(278, 278, "Invalid rule: reuse the boundary one-fill", size=15),
            _text(278, 300, "footprint at every contour level.", size=15),
            _text(
                278,
                382,
                "The dashed footprint contains 1,174 nodes.",
                css="muted",
                size=13.5,
            ),
            _text(
                278, 403, "At ψₙ<0.675, 511 core nodes remain", css="muted", size=13.5
            ),
            _text(
                278,
                424,
                "axis-connected; four lobe nodes do not.",
                css="muted",
                size=13.5,
            ),
        ]
    )

    _axes(lines, zoom, (0.4, 0.5, 0.6, 0.7, 0.8), (1.5, 1.6, 1.7, 1.8, 1.9))
    lines.append(_text(48, 498, "divertor zoom · R–|Z|", size=17, weight=650))
    lines.append(
        _text(
            zoom.left + zoom.width / 2,
            754,
            "major radius R (m)",
            size=15,
            anchor="middle",
        )
    )
    lines.append(
        _text(
            15,
            zoom.top + zoom.height / 2,
            "absolute height |Z| (m)",
            size=15,
            anchor="middle",
            transform=f"rotate(-90 15 {zoom.top + zoom.height / 2:.2f})",
        )
    )
    for (name, _part), points in evidence.structures.items():
        folded = points.copy()
        folded[:, 1] = np.abs(folded[:, 1])
        lines.append(
            f'<path d="{_path(folded, zoom)}" class="structure" stroke-width="1.0" clip-path="url(#zoom_clip)"/>'
        )
    for points in validation["lobe_segments"]:
        upper = float(np.mean(points[:, 1])) > 0.0
        dash = "" if upper else ' stroke-dasharray="6 4"'
        lines.append(
            f'<path d="{_path(points, zoom, absolute_height=True)}" class="focus" fill="none" stroke-width="2.35"{dash} clip-path="url(#zoom_clip)"/>'
        )

    marker_by_name = {marker.name: marker for marker in evidence.markers}
    for marker in evidence.markers:
        x, y = zoom.point(marker.radius_m, abs(marker.height_m))
        if "upper" in marker.name:
            lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.1" class="marker" stroke-width="1.8"/>'
            )
        else:
            lines.append(_line(x - 3.5, y - 3.5, x + 3.5, y + 3.5, "focus", width=1.8))
            lines.append(_line(x - 3.5, y + 3.5, x + 3.5, y - 3.5, "focus", width=1.8))

    inner = marker_by_name["lobe_upper_inner"]
    outer = marker_by_name["lobe_upper_outer"]
    x_inner, y_inner = zoom.point(inner.radius_m, abs(inner.height_m))
    x_outer, y_outer = zoom.point(outer.radius_m, abs(outer.height_m))
    folded_target = np.asarray((LOBE_RADIUS_M, abs(inner.height_m)))
    case_points = np.concatenate(
        [
            points
            for (name, _part), points in evidence.structures.items()
            if name == "coil_cases"
        ]
    ).copy()
    case_points[:, 1] = np.abs(case_points[:, 1])
    case_point = case_points[
        int(np.argmin(np.linalg.norm(case_points - folded_target, axis=1)))
    ]
    active_points = np.concatenate(
        [
            points
            for (name, _part), points in evidence.structures.items()
            if name in {"p2_outer_lower", "p2_outer_upper"}
        ]
    ).copy()
    active_points[:, 1] = np.abs(active_points[:, 1])
    active_point = active_points[
        int(np.argmin(np.linalg.norm(active_points - folded_target, axis=1)))
    ]
    case_x, case_y = zoom.point(float(case_point[0]), float(case_point[1]))
    active_x, active_y = zoom.point(float(active_point[0]), float(active_point[1]))
    lines.extend(
        [
            _line(x_inner, y_inner, 292, 554, "leader", width=0.8),
            _text(300, 550, "four excluded nodes · ψₙ", size=15, weight=650),
            _text(300, 574, "+Z 1.750 m  0.66434", size=14),
            _text(300, 596, "−Z 1.750 m  0.66113", size=14),
            _line(x_outer, y_outer, 292, 626, "leader", width=0.8),
            _text(300, 622, "+Z 1.8125 m  0.67195", size=14),
            _text(300, 644, "−Z 1.8125 m  0.67146", size=14),
            _text(300, 680, "nearest machine structures", size=15, weight=650),
            _line(case_x, case_y, 292, 698, "leader", width=0.8),
            _text(300, 702, "coil-case plate · 10.4 mm", size=14),
            _line(active_x, active_y, 292, 720, "leader", width=0.8),
            _text(300, 724, "P2 outer divertor coil · 21.9–34.8 mm", size=14),
            _text(300, 748, "solid +Z · dashed −Z", css="muted", size=13),
        ]
    )
    return "\n".join((*lines, "</svg>", ""))


def _summary(evidence: Evidence, validation: dict[str, object]) -> str:
    values = ", ".join(
        f"{marker.name}={marker.psi_n:.8f}" for marker in evidence.markers
    )
    return (
        f"shot={SHOT} slice={SLICE_INDEX} time_s={float(evidence.metadata['time_s']):.9f}; "
        f"psi_n={LOBE_LEVEL:g} closed_components=3 core_nodes={validation['level_core_nodes']} "
        f"excluded_lobe_nodes={validation['excluded_nodes']}; boundary_fill_nodes="
        f"{validation['boundary_fill_nodes']}; {values}; nearest=coil_cases@"
        f"{1000 * float(evidence.metadata['nearest_passive_distance_m']):.1f}mm,"
        f"p2_outer@{1000 * float(evidence.metadata['nearest_active_distance_min_m']):.1f}-"
        f"{1000 * float(evidence.metadata['nearest_active_distance_max_m']):.1f}mm; "
        "verdict=closed divertor-coil pockets outside axis-connected plasma"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="re-extract the committed TSV from the read-only MAST store",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate topology and require the committed SVG to be reproducible",
    )
    arguments = parser.parse_args()
    if arguments.refresh_source:
        refresh_source(SOURCE_PATH)
    evidence = read_source(SOURCE_PATH)
    validation = validate(evidence)
    svg = render_svg(evidence, validation)
    if arguments.check:
        if not FIGURE_PATH.exists() or FIGURE_PATH.read_text() != svg:
            raise ValueError("committed SVG is stale relative to the TSV and renderer")
    else:
        FIGURE_PATH.write_text(svg)
    print(_summary(evidence, validation))


if __name__ == "__main__":
    main()
