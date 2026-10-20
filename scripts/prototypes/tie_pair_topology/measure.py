"""Independently measure nulls and raster connectivity in the moving tie field."""

# ruff: noqa: E402

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import heapq
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import nova
from matplotlib.path import Path as PolygonPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stationary import stationary_points

from scripts.prototypes.fast_map_production import tie_probe
from tests.equilibrium.test_topology_read import (
    _census_cache,
    _census_field,
    _census_tie_setup,
)


def widest_passage(score, inside, seed):
    """Highest lower level of an eight-neighbour path from one axis seed."""
    height, width = score.shape
    passage = np.full(score.shape, -np.inf)
    passage[seed] = score[seed]
    queue = [(-score[seed], seed)]
    while queue:
        negative, (row, column) = heapq.heappop(queue)
        value = -negative
        if value < passage[row, column]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                other = (row + dr, column + dc)
                if not (
                    0 <= other[0] < height and 0 <= other[1] < width and inside[other]
                ):
                    continue
                candidate = min(value, score[other])
                if candidate > passage[other]:
                    passage[other] = candidate
                    heapq.heappush(queue, (-candidate, other))
    return passage


def nearest_index(point, r, z):
    return int(np.argmin(abs(z - point[1]))), int(np.argmin(abs(r - point[0])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=320)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    print(
        f"nova={nova.__file__} cwd={Path.cwd().resolve()} "
        f"backend={jax.default_backend()}",
        flush=True,
    )
    state, prior, _ = _census_tie_setup(_census_cache())
    geometry = state["geometry"]
    wall = np.asarray(geometry.wall_points)
    polygon = PolygonPath(wall, closed=True)
    bounds = (
        float(wall[:, 0].min()),
        float(wall[:, 0].max()),
        float(wall[:, 1].min()),
        float(wall[:, 1].max()),
    )
    sampler = tie_probe.make_sampler(state)
    sigma = float(state["convention"].sigma)
    rows = []
    arrays = {}
    for offset in (0.1732, 0.17325):
        field = _census_field(state, offset)
        values_jit = jax.jit(jax.vmap(field.value))
        jets_jit = jax.jit(field.evaluate)

        def values(points):
            chunks = []
            for start in range(0, len(points), 4096):
                chunks.append(
                    np.asarray(values_jit(jnp.asarray(points[start : start + 4096])))
                )
            return np.concatenate(chunks)

        def jets(point):
            result = jets_jit(jnp.asarray(point))
            return (
                float(result.value),
                np.asarray(result.gradient),
                np.asarray(result.hessian),
            )

        independent, grid = stationary_points(
            values, jets, bounds, polygon.contains_points, resolution=args.resolution
        )
        r, z, flux = grid["r"], grid["z"], grid["flux"]
        rr, zz = np.meshgrid(r, z, indexing="xy")
        inside = polygon.contains_points(
            np.stack((rr.ravel(), zz.ravel()), axis=1)
        ).reshape(flux.shape)
        roots, reading, booked, *_ = sampler(offset * float(state["scale"]), prior)
        root_positions = np.asarray(roots.position)
        axis = np.asarray(reading.axis)
        independent_maxima = [
            point
            for point in independent
            if point.kind == ("maximum" if sigma > 0 else "minimum")
        ]
        if not independent_maxima:
            raise RuntimeError("raster finder found no plasma-polarity extremum")
        carrier = max(independent_maxima, key=lambda item: sigma * item.level)
        passage = widest_passage(
            sigma * flux, inside, nearest_index(carrier.position, r, z)
        )
        edge = inside & (
            ~np.roll(inside, 1, axis=0)
            | ~np.roll(inside, -1, axis=0)
            | ~np.roll(inside, 1, axis=1)
            | ~np.roll(inside, -1, axis=1)
        )
        wall_level = float(np.max(passage[edge]))
        wall_index = np.unravel_index(
            np.argmax(np.where(edge, passage, -np.inf)), passage.shape
        )
        null_rows = []
        for point in independent:
            index = nearest_index(point.position, r, z)
            null_rows.append(
                {
                    "position": point.position,
                    "level": point.level,
                    "kind": point.kind,
                    "eigenvalues": point.eigenvalues,
                    "residual": point.residual,
                    "axis_path_passage_raster": float(passage[index]),
                    "persistence_from_carrier": float(
                        sigma * (carrier.level - point.level)
                    ),
                }
            )
        selected = np.flatnonzero(np.asarray(reading.membership) > 0)
        cell_centres = np.asarray(geometry.centre)[selected]
        cell_vertices = np.asarray(geometry.vertices)[selected]
        cell_counts = np.asarray(geometry.vertex_count)[selected]
        support = []
        for cell, centre, vertices, count in zip(
            selected, cell_centres, cell_vertices, cell_counts, strict=True
        ):
            path = PolygonPath(vertices[:count], closed=True)
            in_cell = path.contains_points(
                np.stack((rr.ravel(), zz.ravel()), axis=1)
            ).reshape(flux.shape)
            support.append(
                {
                    "cell": int(cell),
                    "centre": centre.tolist(),
                    "vertices": vertices[:count].tolist(),
                    "raster_points": int(in_cell.sum()),
                    "max_axis_passage": float(np.max(passage[in_cell]))
                    if in_cell.any()
                    else None,
                    "min_axis_passage": float(np.min(passage[in_cell]))
                    if in_cell.any()
                    else None,
                }
            )
        row = {
            "offset": offset,
            "sigma": sigma,
            "candidate_count": grid["candidate_count"],
            "independent": null_rows,
            "carrier": carrier.position,
            "raster_wall_passage": wall_level,
            "raster_wall_position": [float(r[wall_index[1]]), float(z[wall_index[0]])],
            "read": {
                "axis": axis.tolist(),
                "axis_flux": float(reading.axis_flux),
                "saddle": np.asarray(reading.saddle_form.position).tolist(),
                "boundary_flux": float(reading.boundary_flux),
                "boundary_class": int(reading.boundary_class),
                "wall_candidate": float(reading.wall_candidate),
                "saddle_candidate": float(reading.saddle_candidate),
                "booked_current": float(booked),
                "selected_cells": selected.tolist(),
                "root_positions": root_positions[np.asarray(roots.valid)].tolist(),
                "root_levels": np.asarray(roots.jet.value)[
                    np.asarray(roots.valid)
                ].tolist(),
                "path_weights": np.asarray(reading.path_weight)[selected].tolist(),
                "memberships": np.asarray(reading.membership)[selected].tolist(),
                "support": support,
            },
        }
        print(json.dumps(row, indent=2), flush=True)
        rows.append(row)
        key = str(offset).replace(".", "_")
        arrays[key + "_flux"] = flux
        arrays[key + "_inside"] = inside
        arrays[key + "_passage"] = passage
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "measurement.json").write_text(json.dumps(rows, indent=2) + "\n")
    np.savez_compressed(args.output / "field.npz", r=r, z=z, wall=wall, **arrays)
    print(f"RECEIPT={args.output / 'measurement.json'}", flush=True)


if __name__ == "__main__":
    main()
