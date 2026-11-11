"""Hold the shell boundary fixed while refining radial and angular cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
import shapely

from measure import PROBES, dipole, dump, shielding
from nova.biot import magnetisation as iron


def annular_cells(radial_count, angular_count):
    radii = np.linspace(0.9, 1.0, radial_count + 1)
    angles = np.linspace(-np.pi / 2, np.pi / 2, 257)
    polygons, centres, areas = [], [], []
    stride = 256 // angular_count
    for inner, outer in zip(radii[:-1], radii[1:]):
        for index in range(angular_count):
            theta = angles[index * stride : (index + 1) * stride + 1]
            direction = np.column_stack([np.cos(theta), np.sin(theta)])
            direction[np.abs(direction) < 1e-15] = 0
            vertices = np.concatenate([outer * direction, inner * direction[::-1]])
            polygon = shapely.Polygon(vertices)
            polygons.append(vertices)
            centres.append(polygon.centroid.coords[0])
            areas.append(polygon.area)
    return iron._Mesh(tuple(polygons), np.array(centres), np.array(areas))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert jax.default_backend() == "gpu"
    print(
        f"DIRECTIONAL module={iron.__file__} cwd={Path.cwd()} "
        f"device={jax.devices()[0].device_kind} x64={jax.config.jax_enable_x64}",
        flush=True,
    )
    record = {"rows": []}
    # The half-circle always has 256 boundary segments in every mesh.
    for radial_count, angular_count in [(2, 64), (5, 64), (10, 64), (5, 32), (5, 128)]:
        cells = annular_cells(radial_count, angular_count)
        iron._mesh = lambda region, size: cells
        started = perf_counter()
        result = iron.solve_magnetisation(
            None,
            dipole,
            [0, 10],
            [1000, 1000],
            cell_size=0.02,
            tolerance=1e-12,
        )
        assert result.converged
        air = dipole(PROBES)
        total = air + result.field(PROBES)
        solved = np.linalg.norm(air, axis=1) / np.linalg.norm(total, axis=1)
        row = dict(
            radial_count=radial_count,
            angular_count=angular_count,
            cells=result.cell_count,
            solved=solved.tolist(),
            signed_error=(solved / shielding(1000) - 1).tolist(),
            volume=float(2 * np.pi * np.sum(cells.areas * cells.centroids[:, 0])),
            residual=result.residual_history[-1],
            seconds=perf_counter() - started,
        )
        record["rows"].append(row)
        print("DIRECTIONAL_ROW", json.dumps(row), flush=True)
        dump(args.output, record)
    print("DIRECTIONAL_SUMMARY", json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
