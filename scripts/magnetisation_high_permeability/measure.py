"""Separate shell geometry, source multipoles and cell-discretisation errors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import jax
import numpy as np
import shapely

from nova.biot import magnetisation as iron
from nova.biot.greens import greens_bz_br
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64

PROBES = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
SIZES = [0.04, 0.02, 0.01]
PERMEABILITIES = [100, 300, 1000, 3000]
MU0 = 4e-7 * np.pi


def shielding(mu, fraction=0.9**3):
    return ((2 * mu + 1) * (mu + 2) - 2 * (mu - 1) ** 2 * fraction) / (9 * mu)


def loop(points):
    bz, br = greens_bz_br(points[:, 0], points[:, 1], 0.05, 0.0)
    return np.column_stack([br, bz])


def dipole(points):
    r, z = points.T
    return (
        MU0
        * 0.05**2
        / 4
        * np.column_stack([3 * r * z, 2 * z**2 - r**2])
        / ((r**2 + z**2)[:, None] ** 2.5)
    )


def shell_mesh_volumes(cells):
    """Recover both enclosed volumes from the actual mesh boundary moments."""
    polygons = [shapely.Polygon(p) for p in cells.polygons]
    raw = shapely.union_all(polygons)
    # Stitch floating-coordinate grid seams far below the sheet operand's
    # picometre edge-identity rounding, then bound the affected material area.
    merged = shapely.union_all(polygons, grid_size=1e-14)
    difference = raw.symmetric_difference(merged).area
    print(
        "BOUNDARY_STITCH",
        json.dumps(
            dict(
                raw_type=raw.geom_type,
                raw_valid=raw.is_valid,
                raw_parts=len(shapely.get_parts(raw)),
                grid_size=1e-14,
                changed_area=difference,
            )
        ),
        flush=True,
    )
    assert difference < 1e-13
    assert isinstance(merged, shapely.Polygon) and merged.is_valid, (
        "mesh boundary remains disconnected at measurement precision"
    )
    ring = np.asarray(shapely.orient_polygons(merged).exterior.coords)
    start, end = ring[:-1], ring[1:]
    moment = (
        np.pi
        / 3
        * (start[:, 0] ** 2 + start[:, 0] * end[:, 0] + end[:, 0] ** 2)
        * (end[:, 1] - start[:, 1])
    )
    outer = np.linalg.norm((start + end) / 2, axis=1) > 0.95
    vo, vi = float(moment[outer].sum()), float(-moment[~outer].sum())
    cell_volume = 2 * np.pi * np.sum(cells.areas * cells.centroids[:, 0])
    assert abs(vo - vi - cell_volume) < 2e-14
    return vo, vi, merged


def dump(path, record):
    temporary = path.with_suffix(".pending")
    with temporary.open("w") as stream:
        json.dump(record, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.default_backend() == "gpu", jax.devices()
    record = dict(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        module=iron.__file__,
        cwd=str(Path.cwd()),
        devices=str(jax.devices()),
        x64=jax.config.jax_enable_x64,
        geometry=[],
        shell=[],
        sphere=[],
    )
    print("INSTRUMENT", json.dumps(record), flush=True)
    native_mesh, native_interaction = iron._mesh, iron._interaction
    cache = {}

    def cached_interaction(cells, points, plan):
        key = points.tobytes()
        if key not in cache:
            started = perf_counter()
            cache[key] = native_interaction(cells, points, plan)
            print(
                f"INTERACTION targets={len(points)} cells={len(cells.polygons)} "
                f"seconds={perf_counter() - started:.6f}",
                flush=True,
            )
        return cache[key]

    iron._interaction = cached_interaction
    outer = shapely.Point(0, 0).buffer(1, 512).intersection(shapely.box(0, -1, 1, 1))
    inner = shapely.Point(0, 0).buffer(0.9, 512).intersection(shapely.box(0, -1, 1, 1))
    region = outer.difference(inner)
    for size in SIZES:
        cache.clear()
        started = perf_counter()
        cells = native_mesh(region, size)
        iron._mesh = lambda region, size: cells
        meshed_volume = 2 * np.pi * np.sum(cells.areas * cells.centroids[:, 0])
        vo, vi, merged = shell_mesh_volumes(cells)
        assert merged.symmetric_difference(region).area < 1e-13
        geometry = dict(
            h=size,
            cells=len(cells.polygons),
            volume=meshed_volume,
            volume_exact=4 * np.pi / 3 * (1 - 0.9**3),
            outer_radius=(3 * vo / (4 * np.pi)) ** (1 / 3),
            inner_radius=(3 * vi / (4 * np.pi)) ** (1 / 3),
            fraction=vi / vo,
            mesh_seconds=perf_counter() - started,
        )
        record["geometry"].append(geometry)
        print("GEOMETRY", json.dumps(geometry), flush=True)
        dump(args.output, record)
        for mu in PERMEABILITIES:
            for source_name, source in [("loop", loop), ("dipole", dipole)]:
                started = perf_counter()
                result = iron.solve_magnetisation(
                    region,
                    source,
                    [0, 10],
                    [mu, mu],
                    cell_size=size,
                    tolerance=1e-12,
                )
                assert result.converged, result.residual_history
                air = source(PROBES)
                total = air + result.field(PROBES)
                solved = np.linalg.norm(air, axis=1) / np.linalg.norm(total, axis=1)
                exact = shielding(mu)
                row = dict(
                    h=size,
                    mu=mu,
                    source=source_name,
                    cells=result.cell_count,
                    exact=exact,
                    volume_implied=shielding(mu, vi / vo),
                    solved=solved.tolist(),
                    signed_error=(solved / exact - 1).tolist(),
                    residual=result.residual_history[-1],
                    iterations=result.iterations,
                    seconds=perf_counter() - started,
                )
                if source_name == "dipole":
                    r, z = cells.centroids.T
                    q = 0.05**2 / (4 * exact)
                    constant = 2 * (mu - 1) / (3 * mu) * q
                    moment = (mu + 2) / (3 * mu) * q
                    analytic = (mu - 1) * (
                        moment
                        * np.column_stack([3 * r * z, 2 * z**2 - r**2])
                        / (r**2 + z**2)[:, None] ** 2.5
                        - [0, constant]
                    )
                    row["magnetisation_relative_error"] = float(
                        np.linalg.norm(result.magnetisation - analytic)
                        / np.linalg.norm(analytic)
                    )
                    gram = cache[cells.centroids.tobytes()]
                    state = analytic.T.reshape(-1)
                    defect = state - (mu - 1) / (MU0 * mu) * (
                        source(cells.centroids).T.reshape(-1) + gram @ state
                    )
                    row["analytic_collocation_defect"] = float(
                        np.linalg.norm(defect) / np.linalg.norm(state)
                    )
                record["shell"].append(row)
                print("SHELL", json.dumps(row), flush=True)
                dump(args.output, record)
    # Refine the boundary as well as the cells: a fixed polygon has a fixed error floor.
    for size, segments in [(0.1, 64), (0.05, 128), (0.025, 256)]:
        cache.clear()
        region = (
            shapely.Point(0, 0)
            .buffer(0.5, segments)
            .intersection(shapely.box(0, -0.5, 0.5, 0.5))
        )
        cells = native_mesh(region, size)
        iron._mesh = lambda region, size: cells

        def uniform(points):
            return np.tile([0.0, 0.01], (len(points), 1))

        result = iron.solve_magnetisation(
            region, uniform, [0, 10], [1000, 1000], cell_size=size, tolerance=1e-12
        )
        assert result.converged
        points = np.array([[1.0, 0], [0, 1.0]])
        measured = (uniform(points) + result.field(points))[:, 1]
        fraction = 999 / 1002
        exact = 0.01 * np.array([1 - fraction / 8, 1 + fraction / 4])
        row = dict(
            h=size,
            segments=segments,
            cells=result.cell_count,
            error=np.abs(measured / exact - 1).tolist(),
            residual=result.residual_history[-1],
        )
        record["sphere"].append(row)
        print("SPHERE", json.dumps(row), flush=True)
        dump(args.output, record)
    summary = {"orders": [], "chi_slopes": []}
    for source in ["loop", "dipole"]:
        for mu in PERMEABILITIES:
            rows = [
                r for r in record["shell"] if r["mu"] == mu and r["source"] == source
            ]
            errors = np.abs([r["signed_error"] for r in rows])
            orders = (np.log(errors[:-1] / errors[1:]) / np.log(2)).tolist()
            summary["orders"].append(dict(source=source, mu=mu, orders=orders))
        for size in SIZES:
            rows = [
                r for r in record["shell"] if r["h"] == size and r["source"] == source
            ]
            slopes = np.polyfit(
                np.log(np.array(PERMEABILITIES) - 1),
                np.log(np.abs([r["signed_error"] for r in rows])),
                1,
            )[0]
            summary["chi_slopes"].append(
                dict(source=source, h=size, slopes=slopes.tolist())
            )
    shell_control = [
        r for r in record["shell"] if r["mu"] == 100 and r["source"] == "loop"
    ]
    assert np.all(
        np.diff(np.abs([r["signed_error"] for r in shell_control]), axis=0) < 0
    )
    assert np.all(np.diff([r["error"] for r in record["sphere"]], axis=0) < 0)
    summary["controls_pass"] = True
    record["summary"] = summary
    dump(args.output, record)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
