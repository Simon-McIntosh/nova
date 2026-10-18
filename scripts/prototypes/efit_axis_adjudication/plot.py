"""Poloidal panels of psi, both codes' boundaries, axes/saddles and the census.

One panel per record, grouped into a few multi-panel figures by machine.  The
field drawn is the record's own scipy spline (via FastField), evaluated here
independently of either nova field-swap adapter.  No axes, no gridlines, no
fill; psi is unfilled line contours; every coloured mark is in the legend.
"""

from __future__ import annotations

import json
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

from nova.equilibrium.wall_mask import material_unit, vessel_unit
from nova.media.ink import DEFAULT_INK
from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall
from nova.media.sources.frame import inside_wall_units

from case_key import case_key
from field import FastField
from page import figure_chunks
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
    "wall": "#444444",
    "obstacle": "#8a8a8a",
    "escape": "#b0182b",
    "census": "#929292",
}

# one canvas width for every figure, so a reader meets one set of sizes; the
# many-panel band of the figure style allows the smaller label size
CANVAS_WIDTH = 14.0
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


def _typed_wall_units(rings):
    return tuple(
        vessel_unit(ring[:, 0], ring[:, 1])
        if role == "vessel"
        else material_unit(ring[:, 0], ring[:, 1])
        for ring, role in rings
    )


def _sample_field(field, record):
    rs = np.linspace(record.rgrid.min(), record.rgrid.max(), 200)
    zs = np.linspace(record.zgrid.min(), record.zgrid.max(), 220)
    RR, ZZ = np.meshgrid(rs, zs, indexing="ij")
    value, _, _, _, _, _ = field.jet(RR.ravel(), ZZ.ravel())
    return rs, zs, value.reshape(RR.shape).T


def _boundary_contour(ax, rs, zs, psi, level, wall, *, color, linestyle):
    lo, hi = float(np.nanmin(psi)), float(np.nanmax(psi))
    if not (lo <= level <= hi):
        return False
    drawn = draw_flux_contours(
        ax, rs, zs, psi, [level], wall=wall, color=color, linewidth=1.0
    )
    drawn.set_linestyle(linestyle)
    return _has_contours(drawn)


def _has_contours(drawn):
    return any(len(segment) > 1 for level in drawn.allsegs for segment in level)


def _set_panel_geometry(ax, rings):
    vessel = rings[0][0]
    rlo, rhi = float(vessel[:, 0].min()), float(vessel[:, 0].max())
    zlo, zhi = float(vessel[:, 1].min()), float(vessel[:, 1].max())
    rpad = 0.025 * (rhi - rlo)
    zpad = 0.025 * (zhi - zlo)
    ax.set_xlim(rlo - rpad, rhi + rpad)
    ax.set_ylim(zlo - zpad, zhi + zpad)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _panel_title(row):
    if not all(key in row for key in ("machine", "slice", "iteration")):
        return row["case"]
    side = row["case"].rsplit("-", 1)[-1]
    return (
        f"{row['machine']} · slice {row['slice']} · {side}\n"
        f"iteration {row['iteration']}"
    )


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
    rings = _wall_loops_by_role(row["path"])
    wall = _typed_wall_units(rings)
    points = row["census"]
    positions = np.asarray([[p["r"], p["z"]] for p in points], dtype=float).reshape(
        -1, 2
    )
    if any(not p["inside_wall"] for p in points) or (
        len(positions) and not np.all(inside_wall_units(positions, wall))
    ):
        raise ValueError("census point outside the vessel")
    efit = row["efit"]
    source = efit.get("saddle_source_position")
    saddle_mark = efit.get("saddle_plot_position")
    if (
        saddle_mark is not None
        and source is not None
        and np.allclose(saddle_mark, source, rtol=0, atol=1e-12)
    ):
        raise ValueError("grid-node saddle proxy cannot be painted")

    rs, zs, psi = _sample_field(field, record)
    levels = row["contour_levels"]
    drawn = set()
    contours = draw_flux_contours(
        ax, rs, zs, psi, levels, wall=wall, color="#c9c9c9", linewidth=0.8
    )
    if _has_contours(contours):
        drawn.add("psi contours")

    _draw_wall_loops(ax, row["path"])
    drawn.add("vessel wall")
    if any(role == "obstacle" for _, role in rings):
        drawn.add("obstacle wall")

    a = row["adapter_a"]

    for kind, marker in (("maximum", "^"), ("minimum", "v"), ("saddle", "x")):
        positions = np.asarray(
            [[point["r"], point["z"]] for point in points if point["kind"] == kind]
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
            drawn.add(f"census {kind}")

    # imas-efit's escape bottleneck belongs to the escape walk, not to nova's
    # rebuild, so it is drawn before the failed-rebuild panels return.
    if row.get("escape"):
        _draw_escape_bottleneck(ax, row["escape"])
        drawn.add("escape bottleneck")

    if row["rebuild"]["status"] != "pass":
        _set_panel_geometry(ax, rings)
        ax.set_title(
            f"{_panel_title(row)}\nrebuild failed: "
            f"|grad psi|={row['rebuild']['grad_norm_at_axis']:.3g}",
            fontsize=LABEL_SIZE,
            pad=2,
        )
        return drawn

    # Draw each code's boundary at its own level through the flux painter.
    if _boundary_contour(
        ax,
        rs,
        zs,
        psi,
        efit["boundary_level"],
        wall,
        color=COLOUR["efit"],
        linestyle="--",
    ):
        drawn.add("imas-efit boundary")
    if a is not None and a["nova_boundary_psi"] is not None:
        if _boundary_contour(
            ax,
            rs,
            zs,
            psi,
            a["nova_boundary_psi"],
            wall,
            color=COLOUR["nova_a"],
            linestyle="-",
        ):
            drawn.add("adapter A boundary")
    # The trace carries a refined saddle level, but no refined position.
    draw_nulls(
        ax,
        magnetic_axis=[efit["axis_r"], efit["axis_z"]],
        x_points=np.asarray([saddle_mark]) if saddle_mark is not None else None,
        style=replace(DEFAULT_INK, axis_color=COLOUR["efit"]),
    )
    drawn.add("imas-efit axis")

    if a is not None:
        if a["nova_axis_r"] is not None:
            _hollow_axis(
                ax,
                [a["nova_axis_r"], a["nova_axis_z"]],
                color=COLOUR["nova_a"],
                marker="o",
                size=11,
            )
            drawn.add("adapter A axis")
        if a["nova_saddle_r"] is not None:
            _marked_saddle(
                ax,
                [a["nova_saddle_r"], a["nova_saddle_z"]],
                color=COLOUR["nova_a"],
                marker="o",
            )
            drawn.add("adapter A saddle")
    _set_panel_geometry(ax, rings)
    ax.set_title(_panel_title(row), fontsize=LABEL_SIZE, pad=2)
    return drawn


def _legend_handles(drawn):
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
            [],
            [],
            color=COLOUR["efit"],
            marker="^",
            ls="none",
            ms=8,
            label="imas-efit axis",
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
    return [item for item in handles if item.get_label() in drawn]


def _canvas_height(rows):
    panel_width = (CANVAS_WIDTH - 0.8) / len(rows)
    height = 0.0
    for row in rows:
        vessel = _wall_loops_by_role(row["path"])[0][0]
        radial_span = float(np.ptp(vessel[:, 0]))
        vertical_span = float(np.ptp(vessel[:, 1]))
        height = max(height, panel_width * vertical_span / radial_span)
    return height * 1.05 + 1.5


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
    escape_map = {case_key(key): value for key, value in escape_map.items()}
    for row in rows:
        entry = escape_map.get(case_key(row["case"]))
        if entry:
            row["escape"] = entry
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list] = {}
    for row in rows:
        if row["machine"] == "JT-60SA":
            continue
        groups.setdefault(row["machine"], []).append(row)

    for machine, members in groups.items():
        for index, start, subset in figure_chunks(machine, members):
            n = len(subset)
            fig, axes = plt.subplots(
                1, n, figsize=(CANVAS_WIDTH, _canvas_height(subset))
            )
            axes = np.atleast_1d(axes).ravel()
            drawn = set()
            for ax, row in zip(axes, subset):
                drawn.update(draw_panel(ax, row))
            slug = machine.lower().replace("-", "")
            suffix = "" if index == 0 else f"-{index + 1}"
            handles = _legend_handles(drawn)
            legend = fig.legend(
                handles=handles,
                loc="upper center",
                ncol=min(4, len(handles)),
                frameon=False,
                fontsize=LABEL_SIZE,
                bbox_to_anchor=(0.5, 0.99),
                borderaxespad=0.0,
            )
            fig.canvas.draw()
            legend_box = legend.get_window_extent(fig.canvas.get_renderer())
            if not fig.bbox.contains(
                legend_box.x0, legend_box.y0
            ) or not fig.bbox.contains(legend_box.x1, legend_box.y1):
                raise RuntimeError(f"legend leaves canvas for {machine}")
            fig.subplots_adjust(
                top=max(
                    0.2,
                    (legend_box.y0 - 50.0) / fig.bbox.height,
                ),
                bottom=0.02,
                left=0.01,
                right=0.99,
                wspace=0.05,
            )
            _assert_no_clipping(fig)
            path = outdir / f"efit-axis-{slug}{suffix}.svg"
            fig.savefig(path, format="svg")
            plt.close(fig)
            print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
