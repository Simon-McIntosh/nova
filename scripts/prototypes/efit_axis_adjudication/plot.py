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
    "efit": "#1f4e79",       # imas-efit
    "nova_a": "#b5651d",     # adapter A
    "nova_b": "#2e7d32",     # adapter B
    "wall": "#444444",
    "census": "#929292",
}


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
    _contour_at(ax, field, record, efit["boundary_level"],
                colors=COLOUR["efit"], linestyles="--", linewidths=1.0)
    if a is not None and a["nova_boundary_psi"] is not None:
        _contour_at(ax, field, record, a["nova_boundary_psi"],
                    colors=COLOUR["nova_a"], linestyles="-", linewidths=1.0)
    if b is not None:
        level = b["nova_boundary_level"]
        if a is None or abs(level - a["nova_boundary_psi"]) > 1e-6:
            _contour_at(ax, field, record, level,
                        colors=COLOUR["nova_b"], linestyles=":", linewidths=1.0)

    # census points, hollow by type
    for point in row["census"]:
        marker = {"maximum": "^", "minimum": "v", "saddle": "x"}[point["kind"]]
        ax.plot(point["r"], point["z"], marker=marker, markersize=3.0,
                markerfacecolor="none", markeredgecolor=COLOUR["census"],
                markeredgewidth=0.6, linestyle="none")

    # imas-efit axis and X-point through the null vocabulary
    x_point = None
    if efit.get("saddle_source_position"):
        x_point = np.array([efit["saddle_source_position"]])
    draw_nulls(ax, magnetic_axis=[efit["axis_r"], efit["axis_z"]],
               x_points=x_point, contain=wall)

    # nova axes and saddles, distinct styles
    if a is not None:
        if a["nova_axis_r"] is not None:
            ax.plot(a["nova_axis_r"], a["nova_axis_z"], marker="o", markersize=11,
                    markerfacecolor="none", markeredgecolor=COLOUR["nova_a"],
                    markeredgewidth=1.6, linestyle="none", zorder=6)
        if a["nova_saddle_r"] is not None:
            ax.plot(a["nova_saddle_r"], a["nova_saddle_z"], marker="o", markeredgecolor=COLOUR["nova_a"],
                    markersize=5,
                    markerfacecolor=COLOUR["nova_a"],
                    linestyle="none", zorder=6)
    if b is not None:
        axis_pt = row["code_points"]["nova_b_axis"]
        if axis_pt is not None:
            ax.plot(axis_pt["r"], axis_pt["z"], marker="s", markersize=10,
                    markerfacecolor="none", markeredgecolor=COLOUR["nova_b"],
                    markeredgewidth=1.6, linestyle="none", zorder=6)
        saddle = b["nova_saddle_position"]
        if saddle:
            ax.plot(saddle[0], saddle[1], marker="s", markersize=5,
                    markerfacecolor=COLOUR["nova_b"],
                    markeredgecolor=COLOUR["nova_b"], linestyle="none", zorder=6)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(row["case"], fontsize=8, pad=2)


def _legend_handles():
    from matplotlib.lines import Line2D

    return [
        Line2D([], [], color="#c9c9c9", lw=0.8, label="psi contours"),
        Line2D([], [], color=COLOUR["wall"], lw=1.0, label="wall"),
        Line2D([], [], color=COLOUR["efit"], ls="--", lw=1.0, label="imas-efit boundary"),
        Line2D([], [], color=COLOUR["nova_a"], ls="-", lw=1.0, label="adapter A boundary"),
        Line2D([], [], color=COLOUR["nova_b"], ls=":", lw=1.0, label="adapter B boundary"),
        Line2D([], [], color="red", marker="^", ls="none", ms=8, label="imas-efit axis"),
        Line2D([], [], color="red", marker="X", ls="none", ms=8, label="imas-efit X-point"),
        Line2D([], [], color=COLOUR["nova_a"], marker="o", mfc="none", ls="none",
               ms=9, label="adapter A axis"),
        Line2D([], [], color=COLOUR["nova_a"], marker="o", ls="none", ms=5,
               label="adapter A saddle"),
        Line2D([], [], color=COLOUR["nova_b"], marker="s", mfc="none", ls="none",
               ms=8, label="adapter B axis"),
        Line2D([], [], color=COLOUR["nova_b"], marker="s", ls="none", ms=5,
               label="adapter B saddle"),
        Line2D([], [], color=COLOUR["census"], marker="^", mfc="none", ls="none",
               ms=4, label="census maximum"),
        Line2D([], [], color=COLOUR["census"], marker="v", mfc="none", ls="none",
               ms=4, label="census minimum"),
        Line2D([], [], color=COLOUR["census"], marker="x", ls="none", ms=4,
               label="census saddle"),
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
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, row in zip(axes, members):
            draw_panel(ax, row)
        for ax in axes[n:]:
            ax.set_axis_off()
        slug = machine.lower().replace("-", "")
        fig.legend(handles=_legend_handles(), loc="upper center", ncol=5,
                   frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        path = outdir / f"efit-axis-{slug}.svg"
        fig.savefig(path, format="svg")
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()