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
import sys
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from efit_field_swap.crosscheck import domain_rings
from efit_field_swap.trace_record import read_trace_record

from nova.media.ink import DEFAULT_INK
from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall

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
    "obstacle": "#8a8a8a",
    "escape": "#b0182b",
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
        if (
            id(artist) in hidden_ticks
            or not artist.get_visible()
            or not artist.get_text().strip()
        ):
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


def _sample_field(field, record):
    rs = np.linspace(record.rgrid.min(), record.rgrid.max(), 200)
    zs = np.linspace(record.zgrid.min(), record.zgrid.max(), 220)
    RR, ZZ = np.meshgrid(rs, zs, indexing="ij")
    value, _, _, _, _, _ = field.jet(RR.ravel(), ZZ.ravel())
    return rs, zs, value.reshape(RR.shape).T


def _boundary_contour(ax, rs, zs, psi, level, *, color, linestyle):
    lo, hi = float(psi.min()), float(psi.max())
    if not (lo <= level <= hi):
        return
    drawn = draw_flux_contours(ax, rs, zs, psi, [level], color=color, linewidth=1.0)
    drawn.set_linestyle(linestyle)


def _hollow_axis(ax, position, *, color, marker, size):
    style = replace(
        DEFAULT_INK, axis_marker=marker, axis_color=color, axis_markersize=size
    )
    count = len(ax.lines)
    draw_nulls(ax, magnetic_axis=position, style=style)
    for line in ax.lines[count:]:
        line.set_markerfacecolor("none")
        line.set_markeredgewidth(1.6)


def _marked_saddle(ax, position, *, color, marker):
    style = replace(
        DEFAULT_INK,
        xpoint_marker=marker,
        xpoint_color=color,
        xpoint_markersize=5,
    )
    draw_nulls(ax, x_points=np.asarray([position]), style=style)


def _wall_loops_by_role(path):
    """Split the record's indexed wall loops into its vessel and its obstacles.

    The trace stores the wall as consecutive segments plus a one-based inclusive
    index per loop. Building each loop from that index, rather than drawing the
    whole segment array as one polyline, is what stops a false chord being drawn
    between the end of one loop and the start of the next on records that carry
    more than one loop. The vessel/obstacle split is the field-swap crosscheck's
    own domain split, imported rather than copied, so a panel and the decision
    it illustrates read one domain. That owner refuses an axis contained by more
    than one loop, and the refusal propagates here: a panel is not drawn on a
    domain whose vessel is undetermined.
    """
    vessel, obstacles = domain_rings(read_trace_record(path))
    return [(vessel, "vessel"), *((ring, "obstacle") for ring in obstacles)]


def _draw_wall_loops(ax, path):
    for ring, role in _wall_loops_by_role(path):
        draw_wall(
            ax,
            radius=ring[:, 0],
            height=ring[:, 1],
            color=COLOUR["wall" if role == "vessel" else "obstacle"],
            linewidth=1.0,
            linestyle="solid" if role == "vessel" else (0, (4, 3)),
        )


def _draw_escape_bottleneck(ax, escape):
    """Mark imas-efit's escape bottleneck on top of every other panel mark."""
    ax.plot(
        [escape["bottleneck"][0]],
        [escape["bottleneck"][1]],
        marker="D",
        linestyle="none",
        markersize=7,
        markerfacecolor="none",
        markeredgecolor=COLOUR["escape"],
        markeredgewidth=1.6,
        zorder=12,
    )


def draw_panel(ax, row):
    record = parse_record(row["path"])
    field = FastField(record.rgrid, record.zgrid, record.psi)

    rs, zs, psi = _sample_field(field, record)
    levels = row["contour_levels"]
    draw_flux_contours(ax, rs, zs, psi, levels, color="#c9c9c9", linewidth=0.8)

    _draw_wall_loops(ax, row["path"])

    efit = row["efit"]
    a = row["adapter_a"]
    b = row["adapter_b"]

    for kind, marker in (("maximum", "^"), ("minimum", "v"), ("saddle", "x")):
        positions = np.asarray(
            [
                [point["r"], point["z"]]
                for point in row["census"]
                if point["kind"] == kind
            ]
        )
        if positions.size:
            style = replace(
                DEFAULT_INK,
                xpoint_marker=marker,
                xpoint_color=COLOUR["census"],
                xpoint_markersize=3.0,
                xpoint_markeredgewidth=0.6,
                zorder_markers=4,
            )
            draw_nulls(ax, other_x_points=positions, style=style)

    # imas-efit's escape bottleneck belongs to the escape walk, not to nova's
    # rebuild, so it is drawn before the failed-rebuild panels return.
    if row.get("escape"):
        _draw_escape_bottleneck(ax, row["escape"])

    if row["rebuild"]["status"] != "pass":
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(
            f"{row['case']}\nrebuild failed: |grad psi|="
            f"{row['rebuild']['grad_norm_at_axis']:.3g}; comparisons excluded",
            fontsize=LABEL_SIZE,
            pad=2,
        )
        return

    # Draw each code's boundary at its own level through the flux painter.
    _boundary_contour(
        ax, rs, zs, psi, efit["boundary_level"], color=COLOUR["efit"], linestyle="--"
    )
    if a is not None and a["nova_boundary_psi"] is not None:
        _boundary_contour(
            ax,
            rs,
            zs,
            psi,
            a["nova_boundary_psi"],
            color=COLOUR["nova_a"],
            linestyle="-",
        )
    if b is not None:
        _boundary_contour(
            ax,
            rs,
            zs,
            psi,
            b["nova_boundary_level"],
            color=COLOUR["nova_b"],
            linestyle=":",
        )

    # imas-efit axis and X-point through the null vocabulary
    x_point = None
    if efit.get("saddle_source_position"):
        x_point = np.array([efit["saddle_source_position"]])
    draw_nulls(
        ax,
        magnetic_axis=[efit["axis_r"], efit["axis_z"]],
        x_points=x_point,
    )

    if a is not None:
        if a["nova_axis_r"] is not None:
            _hollow_axis(
                ax,
                [a["nova_axis_r"], a["nova_axis_z"]],
                color=COLOUR["nova_a"],
                marker="o",
                size=11,
            )
        if a["nova_saddle_r"] is not None:
            _marked_saddle(
                ax,
                [a["nova_saddle_r"], a["nova_saddle_z"]],
                color=COLOUR["nova_a"],
                marker="o",
            )
    if b is not None:
        axis_pt = row["code_points"]["nova_b_axis"]
        if axis_pt is not None:
            _hollow_axis(
                ax,
                [axis_pt["r"], axis_pt["z"]],
                color=COLOUR["nova_b"],
                marker="s",
                size=10,
            )
        saddle = b["nova_saddle_position"]
        if saddle:
            _marked_saddle(ax, saddle, color=COLOUR["nova_b"], marker="s")

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(row["case"], fontsize=LABEL_SIZE, pad=2)


def _legend_handles(include_comparisons=True):
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color="#c9c9c9", lw=0.8, label="psi contours"),
        Line2D([], [], color=COLOUR["wall"], lw=1.0, label="vessel wall"),
        Line2D(
            [],
            [],
            color=COLOUR["obstacle"],
            lw=1.0,
            ls=(0, (4, 3)),
            label="obstacle wall",
        ),
        Line2D(
            [],
            [],
            color=COLOUR["escape"],
            marker="D",
            mfc="none",
            ls="none",
            ms=7,
            label="escape bottleneck",
        ),
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
            [],
            [],
            color="red",
            marker="X",
            ls="none",
            ms=8,
            label="imas-efit saddle source grid-node proxy",
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
    if include_comparisons:
        return handles
    return [
        item
        for item in handles
        if item.get_label()
        in {"psi contours", "vessel wall", "obstacle wall", "escape bottleneck"}
        or item.get_label().startswith("census ")
    ]


def _normalise_case(case):
    return case.lower().replace("_", "-").replace(":", "-")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--escape-map", default=None)
    args = parser.parse_args()

    rows = json.loads(pathlib.Path(args.results).read_text())
    escape_map = (
        json.loads(pathlib.Path(args.escape_map).read_text()) if args.escape_map else {}
    )
    escape_map = {_normalise_case(key): value for key, value in escape_map.items()}
    for row in rows:
        entry = escape_map.get(_normalise_case(row["case"]))
        if entry:
            row["escape"] = entry
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
            handles=_legend_handles(
                include_comparisons=any(
                    row["rebuild"]["status"] == "pass" for row in members
                )
            ),
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
            top=max(
                0.2,
                legend_box.y0 / fig.bbox.height
                - (
                    0.12
                    if not any(row["rebuild"]["status"] == "pass" for row in members)
                    else 0.03
                ),
            ),
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
