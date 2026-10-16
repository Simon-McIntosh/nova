"""Compare X-point cell cuts with direct analytic flux samples."""

import argparse
import json
from pathlib import Path

import numpy as np

import nova
from nova.equilibrium.analytic_single_null import cerfon_freidberg_single_null


def inside_cell(points, vertices):
    """Classify grid points against every edge of a convex polygon."""
    next_vertex = np.roll(vertices, -1, axis=0)
    edge = next_vertex - vertices
    twice_area = np.sum(
        vertices[:, 0] * next_vertex[:, 1] - vertices[:, 1] * next_vertex[:, 0]
    )
    direction = np.sign(twice_area)
    if direction == 0:
        raise ValueError("cell has zero area")
    inside = np.ones(len(points), dtype=bool)
    for first, span in zip(vertices, edge, strict=True):
        relative = points - first
        cross = span[0] * relative[:, 1] - span[1] * relative[:, 0]
        inside &= direction * cross >= -1e-14
    return inside


def sample_cell(field, vertices, width, sigma):
    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    radial, vertical = (
        np.linspace(lower[axis], upper[axis], width, endpoint=False)
        + (upper[axis] - lower[axis]) / (2 * width)
        for axis in range(2)
    )
    x, y = np.meshgrid(radial, vertical, indexing="xy")
    points = np.column_stack((x.ravel(), y.ravel()))
    inside = inside_cell(points, vertices)
    hessian = field.hessian(field.x_point)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    if not (eigenvalues[0] < 0 < eigenvalues[1]):
        raise ValueError("analytic X-point Hessian is not a saddle")
    axis_direction = eigenvectors[:, 1]
    if np.dot(field.magnetic_axis - field.x_point, axis_direction) < 0:
        axis_direction = -axis_direction
    facing = (points[inside] - field.x_point) @ axis_direction >= 0
    flux = field.flux(points[inside])
    boundary = float(field.flux(field.x_point))
    if sigma * (float(field.flux(field.magnetic_axis)) - boundary) <= 0:
        raise ValueError("analytic axis is not on the positive flux side")
    positive = sigma * (flux - boundary) > 0
    selected = facing & positive
    return {
        "grid_width": width,
        "inside_count": int(np.sum(inside)),
        "facing_count": int(np.sum(facing)),
        "positive_count": int(np.sum(positive)),
        "selected_count": int(np.sum(selected)),
        "area_fraction": float(np.sum(selected) / np.sum(inside)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", type=int, required=True)
    args = parser.parse_args()
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    source = json.loads(args.input.read_text())
    row = next(item for item in source["rows"] if item["cell"] == args.cell)
    vertices = np.asarray(row["vertices"])
    exact = cerfon_freidberg_single_null()
    result = {"cells": source["cells"], "cell": args.cell, "rows": []}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for width in (512, 1024):
        item = sample_cell(exact, vertices, width, source["sigma"])
        result["rows"].append(item)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"ANALYTIC {json.dumps(item)}", flush=True)
    result["fraction_difference"] = abs(
        result["rows"][1]["area_fraction"] - result["rows"][0]["area_fraction"]
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"SUMMARY fraction_difference={result['fraction_difference']:.12g}")


if __name__ == "__main__":
    main()
