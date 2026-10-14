"""Inspect quadratic ring geometry on the analytic certificate carriers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402
from shapely.geometry import LineString, Point  # noqa: E402

from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.biot.plasmagrid import PlasmaGrid  # noqa: E402
from nova.equilibrium.stencil_mesh import (  # noqa: E402
    StencilMesh,
    ring_condition,
)
from nova.frame.coilset import CoilSet  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402


assert jax.config.jax_enable_x64 is True


def inspect(kind: str, cells: int, output: Path) -> dict:
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, _source, exact = certificate._case(name, clip_mode="exact")
    wall = (
        fixture.limiter_contour(carrier, points=fixture.WALL_POINT_COUNT)
        if kind == "limited"
        else certificate._diverted_wall(exact)
    )
    coilset = CoilSet(dplasma=-500 if cells == 550 else -cells, tplasma="hex")
    coilset.firstwall.insert(wall, turn="hex")
    plasma = np.asarray(coilset.subframe.loc[:, "plasma"], dtype=bool)
    material = np.asarray(coilset.subframe.loc[:, "poly"], dtype=object)[plasma]
    centre = np.c_[
        np.asarray(coilset.subframe.loc[plasma, "x"], dtype=np.float64),
        np.asarray(coilset.subframe.loc[plasma, "z"], dtype=np.float64),
    ]
    area = np.asarray([item.poly.area for item in material])
    full_area = float(np.max(area))
    boundary = LineString(wall)
    touches_wall = np.asarray([item.poly.intersects(boundary) for item in material])
    clipped = area / full_area < 1.0 - 1.0e-9
    tri = Delaunay(centre)
    stencil, _ = PlasmaGrid.loop_neighbour_vertices(
        centre, tri.vertex_neighbor_vertices, np.flatnonzero(touches_wall)
    )
    condition = ring_condition(centre, stencil)
    clusters = centre[stencil]
    scale = np.max(np.abs(clusters - clusters[:, :1]), axis=1)
    local = (clusters - clusters[:, :1]) / scale[:, None, :]
    radial, vertical = local[..., 0], local[..., 1]
    design = np.stack(
        [
            np.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        ],
        axis=-1,
    )
    field = certificate._exact_state(name, exact, centre)
    gradient, hessian = certificate._exact_derivatives(
        name, exact, centre[stencil[:, 0]]
    )
    coefficient = np.einsum("rij,rj->ri", np.linalg.pinv(design), field[stencil])
    fitted_gradient = coefficient[:, 1:3] / scale
    fitted_operator = (
        2 * coefficient[:, 3] / scale[:, 0] ** 2
        + 2 * coefficient[:, 5] / scale[:, 1] ** 2
        - fitted_gradient[:, 0] / clusters[:, 0, 0]
    )
    exact_operator = (
        hessian[:, 0, 0] + hessian[:, 1, 1] - gradient[:, 0] / clusters[:, 0, 0]
    )
    gradient_error = np.linalg.norm(fitted_gradient - gradient, axis=1)
    operator_error = np.abs(fitted_operator - exact_operator)
    np.savez_compressed(
        output.with_suffix(".npz"),
        centre=centre,
        stencil=stencil,
        condition=condition,
        gradient_error=gradient_error,
        operator_error=operator_error,
    )
    refused = None
    try:
        StencilMesh(centre, stencil, area)
    except ValueError as exc:
        refused = str(exc)
    x_point = np.asarray(exact.x_point, dtype=float) if kind == "diverted" else None
    detail = []
    for index in np.flatnonzero(condition > 100):
        ids = stencil[index]
        cluster = centre[ids]
        local = cluster - cluster[:1]
        local /= np.max(np.abs(local), axis=0)
        r, z = local.T
        design = np.stack([np.ones(7), r, z, r * r, r * z, z * z], axis=1)
        detail.append(
            {
                "ring": int(index),
                "cells": ids.tolist(),
                "condition": float(condition[index]),
                "centroids": cluster.tolist(),
                "area_fraction": (area[ids] / full_area).tolist(),
                "clipped": clipped[ids].tolist(),
                "touches_wall": touches_wall[ids].tolist(),
                "distance_xpoint": (
                    np.linalg.norm(cluster - x_point, axis=1).tolist()
                    if x_point is not None
                    else None
                ),
                "distance_wall": [
                    float(Point(point).distance(boundary)) for point in cluster
                ],
                "singular_values": np.linalg.svd(design, compute_uv=False).tolist(),
            }
        )
    return {
        "kind": kind,
        "requested_cells": cells,
        "realised_cells": len(centre),
        "full_hex_area": full_area,
        "ring_count": len(stencil),
        "refusal": refused,
        "condition": {
            "max": float(np.max(condition)),
            "p99": float(np.percentile(condition, 99)),
            "above_20": int(np.count_nonzero(condition > 20)),
            "above_100": int(np.count_nonzero(condition > 100)),
            "above_1000": int(np.count_nonzero(condition > 1000)),
        },
        "accuracy": {
            "gradient_error_max": float(np.max(gradient_error)),
            "operator_error_max": float(np.max(operator_error)),
            "gradient_error_above_100": gradient_error[condition > 100].tolist(),
            "operator_error_above_100": operator_error[condition > 100].tolist(),
            "gradient_scale": float(np.max(np.linalg.norm(gradient, axis=1))),
            "operator_scale": float(np.max(np.abs(exact_operator))),
        },
        "detail": detail,
        "wall": wall.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.kind, args.cells, args.output)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "kind",
                    "requested_cells",
                    "realised_cells",
                    "ring_count",
                    "refusal",
                    "condition",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
