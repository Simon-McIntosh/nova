"""Integrate a cell's signed quadratic support by independent vertical slices."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon


def integrate(vertices, centre, pitch, coefficient, level, sigma, segments=2048):
    """Integrate area and centred first moments with two Gauss nodes per slice."""
    local = (vertices - centre) / pitch
    start, end = local, np.roll(local, -1, axis=0)
    low, high = np.min(local[:, 0]), np.max(local[:, 0])
    width = (high - low) / segments
    offsets = np.asarray((-1 / np.sqrt(3), 1 / np.sqrt(3)))
    x = (
        low + (np.arange(segments)[:, None] + 0.5) * width + offsets[None] * width / 2
    ).reshape(-1)
    dx = end[:, 0] - start[:, 0]
    fraction = (x[:, None] - start[None, :, 0]) / np.where(dx != 0, dx, 1)
    crossing = (dx != 0)[None] & (fraction >= 0) & (fraction <= 1)
    y = start[None, :, 1] + fraction * (end - start)[None, :, 1]
    bottom = np.min(np.where(crossing, y, np.inf), axis=1)
    top = np.max(np.where(crossing, y, -np.inf), axis=1)
    q = coefficient * sigma
    constant = q[0] - sigma * level + q[1] * x + q[3] * x * x
    linear = q[2] + q[4] * x
    quadratic = q[5]
    if abs(quadratic) > 1e-14:
        discriminant = linear * linear - 4 * quadratic * constant
        radical = np.sqrt(np.maximum(discriminant, 0))
        roots = np.stack(
            (
                (-linear - radical) / (2 * quadratic),
                (-linear + radical) / (2 * quadratic),
            ),
            axis=1,
        )
        roots = np.where(discriminant[:, None] >= 0, roots, bottom[:, None])
    else:
        root = -constant / np.where(np.abs(linear) > 1e-14, linear, 1)
        roots = np.stack((root, root), axis=1)
    limits = np.sort(
        np.concatenate(
            (
                bottom[:, None],
                top[:, None],
                np.clip(roots, bottom[:, None], top[:, None]),
            ),
            axis=1,
        ),
        axis=1,
    )
    lower, upper = limits[:, :-1], limits[:, 1:]
    middle = (lower + upper) / 2
    occupied = (
        (constant[:, None] + linear[:, None] * middle + quadratic * middle * middle)
        >= 0
    ) & (top > bottom)[:, None]
    height = np.sum(np.where(occupied, upper - lower, 0), axis=1)
    vertical = np.sum(
        np.where(occupied, (upper * upper - lower * lower) / 2, 0), axis=1
    )
    return (
        np.asarray(
            (
                np.sum(height) * pitch**2,
                np.sum(x * height) * pitch**3,
                np.sum(vertical) * pitch**3,
            )
        )
        * width
        / 2
    )


def run(path, cells):
    data = np.load(path)
    results = []
    for cell in cells:
        count = int(data["geometry.vertex_count"][cell])
        vertices = data["geometry.vertices"][cell, :count]
        centre, pitch = data["geometry.centre"][cell], data["geometry.pitch"][cell]
        coefficient = data["common.read.field_coefficients"][cell]
        level = float(data["common.read.boundary_flux"])
        sigma = float(data.get("common.read.sigma", -1))
        measured = integrate(vertices, centre, pitch, coefficient, level, sigma)
        refined = integrate(vertices, centre, pitch, coefficient, level, sigma, 4096)
        polygon_count = int(data["common.polygon_count"][cell])
        polygon = Polygon(data["common.polygon"][cell, :polygon_count])
        bell = np.asarray(
            (
                polygon.area,
                polygon.area * (polygon.centroid.x - centre[0]),
                polygon.area * (polygon.centroid.y - centre[1]),
            )
        )
        control = integrate(vertices, centre, pitch, np.zeros(6), -sigma, sigma)
        full = Polygon(vertices)
        expected = np.asarray(
            (
                full.area,
                full.area * (full.centroid.x - centre[0]),
                full.area * (full.centroid.y - centre[1]),
            )
        )
        row = {
            "cell": cell,
            "reference": measured.tolist(),
            "refined": refined.tolist(),
            "bell_polygon": bell.tolist(),
            "read_area": float(
                data["common.read.geometric_membership"][cell]
                * data["geometry.full_area"][cell]
            ),
            "path_weight": float(data["common.read.path_weight"][cell]),
            "full_control": control.tolist(),
            "shapely_full": expected.tolist(),
        }
        if "direct.boundary_flux" in data:
            direct_level = float(data["direct.boundary_flux"])
            row["physical_reference"] = integrate(
                vertices, centre, pitch, coefficient, direct_level, sigma
            ).tolist()
            row["physical_refined"] = integrate(
                vertices, centre, pitch, coefficient, direct_level, sigma, 4096
            ).tolist()
        results.append(row)
        print("REFERENCE", json.dumps(row), flush=True)
    path.with_name(path.stem + "-reference.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"revision={revision} tree={Path.cwd()} "
        f"command={shlex.join([sys.executable, *sys.argv])}",
        flush=True,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("cells", type=int, nargs="+")
    args = parser.parse_args()
    run(args.path, args.cells)
