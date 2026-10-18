"""Poloidal panels of psi, both codes' boundaries, axes/saddles and the census.

One panel per record, grouped into a few multi-panel figures by machine.  The
field drawn is the record's own scipy spline (via FastField), evaluated here
independently of either nova field-swap adapter.  No axes, no gridlines, no
fill; psi is unfilled line contours; every coloured mark is in the legend.
"""

from __future__ import annotations

import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nova.media.poloidal import draw_nulls

from field import FastField
from records import parse_record

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.size": 9,
        "axes.linewidth": 0.0,
        "lines.linewidth": 1.2,
    }
)

COLOUR = {
    "efit": "#1f4e79",  # imas-efit
    "nova_a": "#b5651d",  # adapter A
    "nova_b": "#2e7d32",  # adapter B
    "wall": "#444444",
    "census": "#929292",
}

# one canvas width for every figure, so a reader meets one set of sizes; the
# many-panel band of the figure style allows the smaller label size
CANVAS_WIDTH = 14.0
PANEL_HEIGHT = 3.6
LABEL_SIZE = 12


def _assert_no_clipping(fig):
    """Refuse to write a figure whose text artists leave the canvas or overlap.

    A label or legend entry that is cut off, or that lands on another, is a
    defect in the delivered artifact however right the numbers are, so the
    check runs on every save rather than relying on a reader noticing.  The
    offending artists are named in the error.
    """
    fig.canvas.draw()
    canvas = fig.bbox
    boxes = []
    hidden_ticks = {
        id(label)
        for ax in fig.axes
        if not ax.axison
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels())
    }
    for artist in fig.findobj(matplotlib.text.Text):
        if id(artist) in hidden_ticks or not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(fig.canvas.get_renderer())
        if box.width == 0 or box.height == 0:
            continue
        if (
            box.x0 < canvas.x0 - 0.5
            or box.x1 > canvas.x1 + 0.5
            or box.y0 < canvas.y0 - 0.5
            or box.y1 > canvas.y1 + 0.5
        ):
            raise RuntimeError(
                f"text artist {artist.get_text()!r} leaves the canvas: "
                f"{box.x0:.1f},{box.y0:.1f} to {box.x1:.1f},{box.y1:.1f} "
                f"against canvas {canvas.x0:.1f},{canvas.y0:.1f} to "
                f"{canvas.x1:.1f},{canvas.y1:.1f}"
            )
        boxes.append((box.x0, box.y0, box.x1, box.y1, artist.get_text()))
    for i, first in enumerate(boxes):
        for second in boxes[i + 1 :]:
            dx = min(first[2], second[2]) - max(first[0], second[0])
            dy = min(first[3], second[3]) - max(first[1], second[1])
            if dx > 1.0 and dy > 1.0:
                raise RuntimeError(
                    f"text artists {first[4]!r} and {second[4]!r} overlap"
                )


def _contour(ax, field, record, levels):
    rs = np.linspace(record.rgrid.min(), record.rgrid.max(), 200)
    zs = np.linspace(record.zgrid.min(), record.zgrid.max(), 220)
    RR, ZZ = np.meshgrid(rs, zs, indexing="ij")
    value, _, _, _, _, _ = field.jet(RR.ravel(), ZZ.ravel())
    PSI = value.reshape(RR.shape)
    ax.contour(RR, ZZ, PSI, levels=levels, colors="#c9c9c9", linewidths=0.8)


def _contour_at(ax, field, record, level, **kwargs):
    rs = np.linspace(record.rgrid.min(), record.rgrid.max(), 200)
    zs = np.linspace(record.zgrid.min(), record.zgrid.max(), 220)
    RR, ZZ = np.meshgrid(rs, zs, indexing="ij")
    value, _, _, _, _, _ = field.jet(RR.ravel(), ZZ.ravel())
    PSI = value.reshape(RR.shape)
    lo, hi = PSI.min(), PSI.max()
    if not (lo <= level <= hi):
        return
    ax.contour(RR, ZZ, PSI, levels=[level], **kwargs)


def draw_panel(ax, row):
    record = parse_record(row["path"])
    field = FastField(record.rgrid, record.zgrid, record.psi)

    lo, hi = float(record.psi.min()), float(record.psi.max())
    span = hi - lo
    levels = np.linspace(lo, hi, 22)
    _contour(ax, field, record, levels)

    wall = record.wall_polygon
    ax.plot(wall[:, 0], wall[:, 1], color=COLOUR["wall"], lw=1.0)

    efit = row["efit"]
    a = row["adapter_a"]
    b = row["adapter_b"]

    # imas-efit boundary (dashed) and nova boundary (solid / dotted)
    _contour_at(
        ax,
        field,
        record,
        efit["boundary_level"],
        colors=COLOUR["efit"],
        linestyles="--",
        linewidths=1.0,
    )
    if a is not None and a["nova_boundary_psi"] is not None:
        _contour_at(
            ax,
            field,
            record,
            a["nova_boundary_psi"],
            colors=COLOUR["nova_a"],
            linestyles="-",
            linewidths=1.0,
        )
    if b is not None:
        level = b["nova_boundary_level"]
        _contour_at(
            ax,
            field,
            record,
            level,
            colors=COLOUR["nova_b"],
            linestyles=":",
            linewidths=1.0,
        )

    # census points, hollow by type
    for point in row["census"]:
        marker = {"maximum": "^", "minimum": "v", "saddle": "x"}[point["kind"]]
        ax.plot(
            point["r"],
            point["z"],
            marker=marker,
            markersize=3.0,
            markerfacecolor="none",
            markeredgecolor=COLOUR["census"],
            markeredgewidth=0.6,
            linestyle="none",
        )

    # imas-efit axis and X-point through the null vocabulary
    x_point = None
    if efit.get("saddle_source_position"):
        x_point = np.array([efit["saddle_source_position"]])
    draw_nulls(
        ax,
        magnetic_axis=[efit["axis_r"], efit["axis_z"]],
        x_points=x_point,
        contain=wall,
    )

    # nova axes and saddles, distinct styles
    if a is not None:
        if a["nova_axis_r"] is not None:
            ax.plot(
                a["nova_axis_r"],
                a["nova_axis_z"],
                marker="o",
                markersize=11,
                markerfacecolor="none",
                markeredgecolor=COLOUR["nova_a"],
                markeredgewidth=1.6,
                linestyle="none",
                zorder=6,
            )
        if a["nova_saddle_r"] is not None:
            ax.plot(
                a["nova_saddle_r"],
                a["nova_saddle_z"],
                marker="o",
                markeredgecolor=COLOUR["nova_a"],
                markersize=5,
                markerfacecolor=COLOUR["nova_a"],
                linestyle="none",
                zorder=6,
            )
    if b is not None:
        axis_pt = row["code_points"]["nova_b_axis"]
        if axis_pt is not None:
            ax.plot(
                axis_pt["r"],
                axis_pt["z"],
                marker="s",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor=COLOUR["nova_b"],
                markeredgewidth=1.6,
                linestyle="none",
                zorder=6,
            )
        saddle = b["nova_saddle_position"]
        if saddle:
            ax.plot(
                saddle[0],
                saddle[1],
                marker="s",
                markersize=5,
                markerfacecolor=COLOUR["nova_b"],
                markeredgecolor=COLOUR["nova_b"],
                linestyle="none",
                zorder=6,
            )

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(row["case"], fontsize=LABEL_SIZE, pad=2)


def _legend_handles():
    from matplotlib.lines import Line2D

    return [
        Line2D([], [], color="#c9c9c9", lw=0.8, label="psi contours"),
        Line2D([], [], color=COLOUR["wall"], lw=1.0, label="wall"),
        Line2D(
            [], [], color=COLOUR["efit"], ls="--", lw=1.0, label="imas-efit boundary"
        ),
        Line2D(
            [], [], color=COLOUR["nova_a"], ls="-", lw=1.0, label="adapter A boundary"
        ),
        Line2D(
            [], [], color=COLOUR["nova_b"], ls=":", lw=1.0, label="adapter B boundary"
        ),
        Line2D(
            [], [], color="red", marker="^", ls="none", ms=8, label="imas-efit axis"
        ),
        Line2D(
            [], [], color="red", marker="X", ls="none", ms=8, label="imas-efit X-point"
        ),
        Line2D(
            [],
            [],
            color=COLOUR["nova_a"],
            marker="o",
            mfc="none",
            ls="none",
            ms=9,
            label="adapter A axis",
        ),
        Line2D(
            [],
            [],
            color=COLOUR["nova_a"],
            marker="o",
            ls="none",
            ms=5,
            label="adapter A saddle",
        ),
        Line2D(
            [],
            [],
            color=COLOUR["nova_b"],
            marker="s",
            mfc="none",
            ls="none",
            ms=8,
            label="adapter B axis",
        ),
        Line2D(
            [],
            [],
            color=COLOUR["nova_b"],
            marker="s",
            ls="none",
            ms=5,
            label="adapter B saddle",
        ),
        Line2D(
            [],
            [],
            color=COLOUR["census"],
            marker="^",
            mfc="none",
            ls="none",
            ms=4,
            label="census maximum",
        ),
        Line2D(
            [],
            [],
            color=COLOUR["census"],
            marker="v",
            mfc="none",
            ls="none",
            ms=4,
            label="census minimum",
        ),
        Line2D(
            [],
            [],
            color=COLOUR["census"],
            marker="x",
            ls="none",
            ms=4,
            label="census saddle",
        ),
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    rows = json.loads(pathlib.Path(args.results).read_text())
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row["machine"], []).append(row)

    for machine, members in groups.items():
        n = len(members)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(CANVAS_WIDTH, PANEL_HEIGHT * nrows)
        )
        axes = np.atleast_1d(axes).ravel()
        for ax, row in zip(axes, members):
            draw_panel(ax, row)
        for ax in axes[n:]:
            ax.set_axis_off()
        slug = machine.lower().replace("-", "")
        legend = fig.legend(
            handles=_legend_handles(),
            loc="upper center",
            ncol=4,
            frameon=False,
            fontsize=LABEL_SIZE,
            bbox_to_anchor=(0.5, 0.99),
            borderaxespad=0.0,
        )
        fig.canvas.draw()
        legend_box = legend.get_window_extent(fig.canvas.get_renderer())
        if not fig.bbox.contains(legend_box.x0, legend_box.y0) or not fig.bbox.contains(
            legend_box.x1, legend_box.y1
        ):
            raise RuntimeError(f"legend leaves canvas for {machine}")
        fig.subplots_adjust(
            top=max(0.2, legend_box.y0 / fig.bbox.height - 0.03),
            bottom=0.02,
            left=0.01,
            right=0.99,
            wspace=0.05,
            hspace=0.10,
        )
        _assert_no_clipping(fig)
        path = outdir / f"efit-axis-{slug}.svg"
        fig.savefig(path, format="svg")
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
