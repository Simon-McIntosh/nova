"""Poloidal panels for the TCV LIUQE known-answer comparison.

One panel per slice: LIUQE's psi map as unfilled line contours on one stated
level array, the per-loop TCV wall, LIUQE's own axis, X-points and boundary
outline, and nova's read over them -- its axis, its admitted saddle and the
contour at its boundary flux. LIUQE is drawn in the default ink and nova in one
feature colour, so the two null sets and the two boundaries stay separable. A
slice nova refuses is drawn with LIUQE's geometry alone; the report caption
names its refusal so a missing nova boundary is explicit.

The archive readers are the driver's, imported rather than restated: one
definition of what a LIUQE slice and a wall IDS are, so the table and the
panels cannot disagree about the input.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import imas
import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import (
    draw_boundary,
    draw_flux_contours,
    draw_nulls,
    draw_wall,
)
from scripts.prototypes.tcv_liuqe_known_answer import measure

NOVA_INK = dataclasses.replace(
    DEFAULT_INK,
    axis_color="#0070c0",
    xpoint_color="#0070c0",
    separatrix_color="#0070c0",
)
READ_LABELS = ("limited", "diverted")


def draw_panel(axes, view, row, wall_rings):
    rgrid, zgrid, psi = view["rgrid"], view["zgrid"], view["psi"]
    grid_axis = float(row["liuqe_axis_psi_grid"])
    grid_boundary = float(row["liuqe_boundary_psi_grid"])
    levels = np.linspace(
        min(grid_axis, grid_boundary), max(grid_axis, grid_boundary), 12
    )
    for ring in wall_rings:
        draw_wall(axes, radius=ring[:, 0], height=ring[:, 1])
    draw_flux_contours(axes, rgrid, zgrid, psi.T, levels, wall=wall_rings)
    reference_contour = draw_flux_contours(
        axes,
        rgrid,
        zgrid,
        psi.T,
        [grid_boundary],
        wall=wall_rings,
        color="#666666",
        linewidth=1.0,
        zorder=5,
    )
    reference_contour.set_linestyle(":")
    draw_nulls(
        axes,
        magnetic_axis=view["axis"],
        x_points=view["xpoints"],
        contain=wall_rings[0],
    )
    if view["outline"].size:
        draw_boundary(
            axes,
            view["outline"][:, 0],
            view["outline"][:, 1],
            color=DEFAULT_INK.separatrix_color,
            linewidth=1.6,
            linestyle="--",
        )
    label = row.get("nova_label")
    if label in READ_LABELS:
        boundary_flux = row.get("nova_boundary_psi")
        if boundary_flux is not None and np.isfinite(boundary_flux):
            draw_flux_contours(
                axes,
                rgrid,
                zgrid,
                psi.T,
                [float(boundary_flux)],
                wall=wall_rings,
                color=NOVA_INK.separatrix_color,
                linewidth=1.1,
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


def build(rows, eq, wall_rings, out_path):
    columns = min(4, len(rows))
    lines = int(np.ceil(len(rows) / columns))
    figure, grid = plt.subplots(lines, columns, figsize=(14, 3.9 * lines), dpi=100)
    panels = np.atleast_1d(grid).ravel()
    for axes, row in zip(panels, rows):
        poloidal_axes(axes)
        draw_panel(axes, measure.slice_view(eq, row["index"]), row, wall_rings)
        axes.set_title(
            f"{row['index']} · {row['liuqe_time']:.3f} s", fontsize=14, pad=8
        )
    for axes in panels[len(rows) :]:
        axes.set_visible(False)
    figure.tight_layout()
    figure.savefig(out_path, format="svg")
    plt.close(figure)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--liuqe", required=True)
    parser.add_argument("--wall", required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--out")
    output.add_argument("--out-dir")
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in Path(args.rows).read_text().splitlines() if line
    ]
    db = imas.DBEntry(f"imas:hdf5?path={args.liuqe}", "r", dd_version="3.41.0")
    eq = db.get("equilibrium")
    if eq.ids_properties.homogeneous_time != 1:
        raise ValueError("equilibrium time base is not homogeneous")
    wall_rings = measure.read_wall(args.wall)
    print("nova.__file__", __import__("nova").__file__, flush=True)
    print("panels", len(rows), "wall rings", len(wall_rings), flush=True)
    if args.out_dir:
        if len(rows) != 17:
            raise ValueError("three-panel render requires all corrected slice rows")
        destination = Path(args.out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        groups = (("early", rows[:6]), ("middle", rows[6:12]), ("late", rows[12:]))
        outputs = [
            build(
                group,
                eq,
                wall_rings,
                destination / f"tcv-77549-liuqe-nova-boundary-{name}.svg",
            )
            for name, group in groups
        ]
    else:
        outputs = [build(rows, eq, wall_rings, args.out)]
    db.close()
    for path in outputs:
        print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()
