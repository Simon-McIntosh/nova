"""Plot the WEST stray poloidal field: poloidal scene above, |B|(R) below."""

from __future__ import annotations

import itertools
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nova.media.ink import DEFAULT_INK, poloidal_axes  # noqa: E402
from nova.media.poloidal import (  # noqa: E402
    draw_coils,
    draw_nulls,
    draw_wall,
)

source, stem = sys.argv[1], sys.argv[2]
d = pickle.load(open(source, "rb"))

plt.style.use("data-ink")
STYLE = DEFAULT_INK
MID = "#2b6cb0"  # midplane line: the headline series
FH = "#2e8b57"  # Fh coil plane
FB = "#c2552d"  # Fb coil plane
BAND = "#7b4fb3"  # |B| iso-lines on the poloidal panel
GREY = "#929292"

radius = d["radius"]
bmag = d["bmag"] * 1e3
edge = d["edge"]


def crossing(b, level):
    idx = np.flatnonzero((b[:-1] >= level) & (b[1:] < level))
    if not idx.size:
        return np.nan
    i = idx[-1]
    return float(np.interp(level, [b[i + 1], b[i]], [radius[i + 1], radius[i]]))


r_one = max(crossing(b, 1.0) for b in bmag)
rmax = float(np.ceil(r_one + 0.5))
zlim = 3.0

fig = plt.figure(figsize=(14, 9.6))
pol = fig.add_axes([0.05, 0.40, 0.92, 0.58])
fld = fig.add_axes([0.05, 0.08, 0.92, 0.24])
poloidal_axes(pol, STYLE)
pol.set_xlim(0, rmax)
pol.set_ylim(-zlim, zlim)

gx, gz, psi = d["gx"], d["gz"], d["psi"]
if psi.shape != (gx.size, gz.size):
    psi = psi.reshape(gx.size, gz.size)
levels = np.linspace(np.nanpercentile(psi, 1), np.nanpercentile(psi, 99), 41)
pol.contour(
    gx,
    gz,
    psi.T,
    levels=levels,
    colors=STYLE.contour_color,
    linewidths=0.6,
    zorder=STYLE.zorder_flux,
)
bnd = np.asarray(d["boundary"])
pol.plot(
    bnd[:, 0],
    bnd[:, 1],
    color=STYLE.separatrix_color,
    linewidth=STYLE.separatrix_linewidth,
    zorder=STYLE.zorder_separatrix,
)

bgrid = d["bgrid"] * 1e3
band_levels = [1, 2, 5, 10, 20]
bmask = np.where(gx[:, None] >= 3.4, bgrid, np.nan)
cs = pol.contour(
    gx,
    gz,
    bmask.T,
    levels=band_levels,
    colors=BAND,
    linewidths=1.2,
    linestyles="dashed",
    zorder=3,
)
manual = []
for level in band_levels:
    r_at = crossing(bmag[0], level)
    if np.isfinite(r_at):
        manual.append((r_at, -2.0))
pol.clabel(
    cs,
    levels=band_levels,
    inline=True,
    fontsize=12,
    fmt=lambda v: f"{v:g} mT",
    manual=manual,
)

draw_wall(
    pol, units=None, radius=d["limiter"][:, 0], height=d["limiter"][:, 1], style=STYLE
)
from matplotlib.collections import PolyCollection  # noqa: E402

pol.add_collection(
    PolyCollection(
        d["iron"], facecolors="#e4e4e4", edgecolors="#b8b8b8", linewidths=0.6, zorder=1
    )
)
for r_p, z_p in ((2.348, 0.871), (2.548, -0.703), (2.647, -0.703)):
    pol.plot(r_p, z_p, "s", color="#555555", markersize=3, zorder=4)
pol.text(4.89, 2.62, "iron", ha="center", va="bottom", fontsize=14, color="#777777")
coil_outlines = [
    np.array(
        [
            [c["r"] - c["dr"] / 2, c["z"] - c["dz"] / 2],
            [c["r"] + c["dr"] / 2, c["z"] - c["dz"] / 2],
            [c["r"] + c["dr"] / 2, c["z"] + c["dz"] / 2],
            [c["r"] - c["dr"] / 2, c["z"] + c["dz"] / 2],
        ]
    )
    for c in d["coils"]
]
draw_coils(pol, coil_outlines, STYLE, edgecolor="#333333", linewidth=1.0)
draw_nulls(
    pol,
    magnetic_axis=d["axis"],
    x_points=np.asarray(d["x_points"]).reshape(-1, 2) if len(d["x_points"]) else None,
    style=STYLE,
    contain=d["limiter"],
)
for name in ("Fh", "Fb"):
    c = next(c for c in d["coils"] if c["name"] == name)
    pol.text(
        c["r"],
        c["z"] + np.sign(c["z"]) * (c["dz"] / 2 + 0.22),
        name,
        ha="center",
        va="center",
        fontsize=14,
        color="#333333",
    )
for z, colour, lw in ((0.0, MID, 1.6), (0.637, FH, 1.2), (-0.637, FB, 1.2)):
    pol.plot([edge, rmax], [z, z], color=colour, linewidth=lw, zorder=4)

for b, colour, lw, label in (
    (bmag[1], FB, 2.6, "Fb plane, Z = −0.64 m"),
    (bmag[2], FH, 2.6, "Fh plane, Z = +0.64 m"),
    (bmag[0], MID, 3.0, "midplane, Z = 0"),
):
    fld.plot(
        radius,
        b,
        color=colour,
        linewidth=lw,
        label=label,
        zorder=5 if colour == MID else 4,
    )
iron_r = (4.80, 4.98)
fld.axvspan(*iron_r, color="#e4e4e4", zorder=0, linewidth=0)
fld.text(
    iron_r[1] + 0.08,
    30,
    "iron return limb (not in the model)",
    ha="left",
    va="center",
    fontsize=14,
    color="#777777",
)
for level in (1.0, 20.0):
    fld.axhline(level, color=GREY, linewidth=1.2, linestyle=":", zorder=1)
fld.set_yscale("log")
fld.set_ylim(0.6, 40)
fld.set_yticks([1, 2, 5, 10, 20])
fld.set_yticklabels(["1", "2", "5", "10", "20"])
fld.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fld.set_xlim(0, rmax)
fld.spines["left"].set_bounds(1, 20)
fld.spines["bottom"].set_bounds(edge, rmax)
fld.set_xticks(np.arange(np.ceil(edge), rmax + 0.01, 1.0))
fld.set_xlabel("major radius R [m]")
fld.set_ylabel("|B| [mT]")
top = fld.secondary_xaxis("top", functions=(lambda r: r - edge, lambda s: s + edge))
top.set_xticks(np.arange(0, rmax - edge + 0.01, 1.0))
top.set_xlabel("distance from Fh/Fb outer edge [m]")
top.spines["top"].set_bounds(0, rmax - edge)
fld.legend(
    loc="center left",
    bbox_to_anchor=(0.0, 0.5),
    frameon=False,
    fontsize=14,
    handlelength=2.0,
)

fig.canvas.draw()
pbox = pol.get_position()
fbox = fld.get_position()
fld.set_position([pbox.x0, fbox.y0, pbox.width, fbox.height])
fig.canvas.draw()

renderer = fig.canvas.get_renderer()
texts = [
    t
    for t in fig.findobj(matplotlib.text.Text)
    if t.get_visible() and t.get_text().strip()
]
canvas = fig.bbox
problems = []
for t in texts:
    bb = t.get_window_extent(renderer)
    if (
        bb.x0 < canvas.x0 - 1
        or bb.x1 > canvas.x1 + 1
        or bb.y0 < canvas.y0 - 1
        or bb.y1 > canvas.y1 + 1
    ):
        problems.append(f"clipped: {t.get_text()!r}")
for a, b in itertools.combinations(texts, 2):
    if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)):
        problems.append(f"overlap: {a.get_text()!r} / {b.get_text()!r}")
print("text check:", problems or "clean")
fig.savefig(f"{stem}.png", dpi=100, facecolor="white")
fig.savefig(f"{stem}.svg", facecolor="white")
print("R(1 mT) per line:", [round(crossing(b, 1.0), 2) for b in bmag])
print("R(20 mT) per line:", [round(crossing(b, 20.0), 2) for b in bmag])
