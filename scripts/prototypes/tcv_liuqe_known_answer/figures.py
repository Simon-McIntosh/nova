"""Poloidal panels for the TCV LIUQE known-answer comparison.

One panel per slice: LIUQE's psi map as unfilled line contours on one stated
level array, the per-loop TCV wall, LIUQE's own axis, X-points and boundary
outline, and nova's read over them -- its axis, its admitted saddle and the
contour at its boundary flux. LIUQE is drawn in the default ink and nova in one
feature colour, so the two null sets and the two boundaries stay separable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import imas
import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium.topology import TopologyConvention
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import (
    contour_levels,
    draw_boundary,
    draw_flux_contours,
    draw_nulls,
    draw_wall,
)

NOVA_INK = dataclasses.replace(
    DEFAULT_INK,
    axis_color="#0070c0",
    xpoint_color="#0070c0",
    separatrix_color="#0070c0",
)
SIGMA = float(TopologyConvention.from_cocos(17, 1.0).sigma)


def read_wall(path):
    db = imas.DBEntry(f"imas:hdf5?path={path}", "r", dd_version="4.1.0")
    wall = db.get("wall")
    rings = []
    for description in wall.description_2d:
        for unit in description.limiter.unit:
            radius = np.asarray(unit.outline.r, dtype=float)
            height = np.asarray(unit.outline.z, dtype=float)
            if radius.size >= 3:
                rings.append(np.column_stack((radius, height)))
    db.close()
    return tuple(rings)


def slice_view(eq, index):
    ts = eq.time_slice[index]
    profiles = ts.profiles_2d[0]
    quantities = ts.global_quantities
    return dict(
        rgrid=np.asarray(profiles.grid.dim1, dtype=float),
        zgrid=np.asarray(profiles.grid.dim2, dtype=float),
        psi=SIGMA * np.asarray(profiles.psi, dtype=float),
        axis=(float(quantities.magnetic_axis.r), float(quantities.magnetic_axis.z)),
        xpoints=np.asarray(
            [(float(point.r), float(point.z)) for point in ts.boundary.x_point],
            dtype=float,
        ).reshape(-1, 2),
        outline=np.column_stack(
            (
                np.asarray(ts.boundary.outline.r, dtype=float),
                np.asarray(ts.boundary.outline.z, dtype=float),
            )
        ),
    )


def _nearest(values, coordinate):
    return int(np.argmin(np.abs(np.asarray(values) - float(coordinate))))


def draw_panel(axes, view, row, wall_rings):
    rgrid, zgrid, psi = view["rgrid"], view["zgrid"], view["psi"]
    grid_axis = float(
        psi[_nearest(rgrid, view["axis"][0]), _nearest(zgrid, view["axis"][1])]
    )
    if len(view["xpoints"]):
        grid_boundary = float(
            np.mean(
                [
                    psi[_nearest(rgrid, point[0]), _nearest(zgrid, point[1])]
                    for point in view["xpoints"]
                ]
            )
        )
    else:
        grid_boundary = float(np.max(psi))
    levels = contour_levels(psi, 14, axis=grid_axis, boundary=grid_boundary)
    for ring in wall_rings:
        draw_wall(axes, radius=ring[:, 0], height=ring[:, 1])
    draw_flux_contours(axes, rgrid, zgrid, psi.T, levels, wall=wall_rings)
    draw_nulls(
        axes,
        magnetic_axis=view["axis"],
        x_points=view["xpoints"],
        contain=wall_rings[0],
    )
    if view["outline"].size:
        draw_boundary(
            axes, view["outline"][:, 0], view["outline"][:, 1], linestyle="--"
        )
    label = row.get("nova_label")
    if label in ("limited", "diverted"):
        boundary_flux = row.get("nova_boundary_psi")
        if boundary_flux is not None and np.isfinite(boundary_flux):
            axes.contour(
                rgrid,
                zgrid,
                psi.T,
                levels=[float(boundary_flux)],
                colors=[NOVA_INK.separatrix_color],
                linewidths=1.1,
                zorder=6,
            )
        draw_nulls(
            axes,
            magnetic_axis=row.get("nova_axis"),
            x_points=(
                np.asarray([row["nova_boundary"]], dtype=float)
                if label == "diverted"
                else None
            ),
            style=NOVA_INK,
            contain=wall_rings[0],
        )
    axes.set_title(
        f"t={row['liuqe_time']:.3f} s  LIUQE {row['liuqe_label']} / nova {label}",
        fontsize=DEFAULT_INK.label_fontsize,
    )


def build(rows, eq, wall_rings, out_path):
    columns = min(4, len(rows))
    lines = int(np.ceil(len(rows) / columns))
    figure, grid = plt.subplots(
        lines, columns, figsize=(3.6 * columns, 3.9 * lines), dpi=110
    )
    panels = np.atleast_1d(grid).ravel()
    for axes, row in zip(panels, rows):
        poloidal_axes(axes)
        draw_panel(axes, slice_view(eq, row["index"]), row, wall_rings)
        axes.set_xlim(0.55, 1.15)
        axes.set_xlabel("R [m]")
        axes.set_ylabel("Z [m]")
    for axes in panels[len(rows) :]:
        axes.set_visible(False)
    figure.tight_layout()
    figure.savefig(out_path, format="svg")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--liuqe", required=True)
    parser.add_argument("--wall", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.rows).read_text())["rows"]
    db = imas.DBEntry(f"imas:hdf5?path={args.liuqe}", "r", dd_version="3.41.0")
    eq = db.get("equilibrium")
    wall_rings = read_wall(args.wall)
    print("panels", len(rows), "wall rings", len(wall_rings), flush=True)
    build(rows, eq, wall_rings, args.out)
    db.close()
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
