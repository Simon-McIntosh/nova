"""Plot the WEST stray poloidal field: poloidal scene above, |B| below.

Both panels share one major-radius scale, so a point on a field curve sits
directly under the same radius in the machine; the field panel's axis is read
as distance from the Fh/Fb outer edge. The flux map, the |B| iso-lines and the
field curves all come from direct Biot-Savart evaluation (psi, B_R and B_Z
each from its own Green's function). One flux map covers plasma and vacuum, so
the contours share one level spacing everywhere. The save refuses clipped or
overlapping text.

Usage: plot.py <hex solve pickle> <direct field npz> <output stem>
"""

from __future__ import annotations

import itertools
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from nova.media.ink import DEFAULT_INK, poloidal_axes  # noqa: E402
from nova.media.poloidal import draw_coils, draw_plasma_cells, draw_wall  # noqa: E402

d = pickle.load(open(sys.argv[1], "rb"))
grid = np.load(sys.argv[2])
stem = sys.argv[3]

plt.style.use("data-ink")
plt.rcParams.update(
    {"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13}
)
STYLE = DEFAULT_INK
LABEL = 13
MID, FH, FB = "#2b6cb0", "#2e8b57", "#c2552d"
BAND = "#7b4fb3"
FLUX = "#9a9a9a"
GREY = "#929292"
INK = "#333333"
IRON = "#777777"
WHITE_BOX = {"facecolor": "white", "edgecolor": "none", "pad": 2.5}

radius = d["radius"]
bmag = d["bmag"] * 1e3
edge = d["edge"]
lines = list(d["lines"])
colours = {"midplane": MID, "Fh plane": FH, "Fb plane": FB}
rmax, zlim = 12.0, 3.0


def crossing(b, level):
    """Outermost radius where ``b`` falls through ``level``."""
    idx = np.flatnonzero((b[:-1] >= level) & (b[1:] < level))
    if not idx.size:
        return np.nan
    i = idx[-1]
    return float(np.interp(level, [b[i + 1], b[i]], [radius[i + 1], radius[i]]))


fig = plt.figure(figsize=(14, 9.6))
pol = fig.add_axes([0.05, 0.40, 0.92, 0.57])
fld = fig.add_axes([0.05, 0.08, 0.92, 0.24])
poloidal_axes(pol, STYLE)
pol.set_xlim(0, rmax)
pol.set_ylim(-zlim, zlim)

# one flux map, one level spacing: the plasma holds twelve intervals
gx, gz, psi = grid["x"], grid["z"], grid["psi"]
boundary = np.asarray(d["boundary"])
flux = RegularGridInterpolator((gx, gz), psi)
psi_lcfs = float(np.mean(flux(boundary)))
inside = shapely.Polygon(boundary)
mesh = np.column_stack([a.ravel() for a in np.meshgrid(gx, gz, indexing="ij")])
in_plasma = shapely.contains_xy(inside, mesh[:, 0], mesh[:, 1]).reshape(psi.shape)
plasma_psi = psi[in_plasma]
psi_axis = float(plasma_psi[np.argmax(np.abs(plasma_psi - psi_lcfs))])
step = abs(psi_axis - psi_lcfs) / 12
low, high = np.percentile(psi, [0.5, 99.5])
levels = psi_lcfs + step * np.arange(
    np.floor((low - psi_lcfs) / step), np.ceil((high - psi_lcfs) / step) + 1
)
pol.contour(
    gx,
    gz,
    psi.T,
    levels=levels,
    colors=FLUX,
    linewidths=0.6,
    linestyles="solid",
    zorder=STYLE.zorder_plasma + 1,
)

# current-carrying hex cells, each drawn as its part inside the LCFS
clipped = []
for cell in d["plasma_cells"]:
    part = shapely.Polygon(cell).intersection(inside)
    for piece in getattr(part, "geoms", [part]):
        if piece.area > 0:
            clipped.append(np.asarray(piece.exterior.coords))
draw_plasma_cells(pol, clipped, STYLE, linewidth=0.3)
pol.plot(
    boundary[:, 0],
    boundary[:, 1],
    color=STYLE.separatrix_color,
    linewidth=STYLE.separatrix_linewidth,
    zorder=STYLE.zorder_separatrix,
)

# |B| iso-lines outboard, from the directly evaluated B_R and B_Z
bgrid = np.hypot(grid["br"], grid["bz"]) * 1e3
outboard = np.broadcast_to(gx[:, None] >= 3.4, bgrid.shape).copy()
for c in d["coils"]:  # the field nulls inside a thick winding are not stray field
    inside_coil = (np.abs(gx[:, None] - c["r"]) <= c["dr"] / 2 + 0.03) & (
        np.abs(gz[None, :] - c["z"]) <= c["dz"] / 2 + 0.03
    )
    outboard &= ~inside_coil
band_levels = [1, 2, 5, 10, 20]
cs = pol.contour(
    gx,
    gz,
    np.where(outboard, bgrid, np.nan).T,
    levels=band_levels,
    colors=BAND,
    linewidths=1.2,
    linestyles="solid",
    zorder=3,
)
manual = [
    (crossing(bmag[0], level), -2.0)
    for level in band_levels
    if np.isfinite(crossing(bmag[0], level))
]
pol.clabel(
    cs,
    levels=band_levels,
    inline=True,
    fontsize=LABEL,
    fmt=lambda v: f"{v:g} mT",
    manual=manual,
)

# machine
pol.add_collection(
    PolyCollection(
        d["iron"], facecolors="#e4e4e4", edgecolors="#b8b8b8", linewidths=0.6, zorder=1
    )
)
pol.text(4.89, 2.62, "iron", ha="center", va="bottom", fontsize=LABEL, color=IRON)
draw_wall(
    pol, units=None, radius=d["limiter"][:, 0], height=d["limiter"][:, 1], style=STYLE
)
for r_p, z_p in ((2.348, 0.871), (2.548, -0.703), (2.647, -0.703)):
    pol.plot(r_p, z_p, "s", color="#555555", markersize=3, zorder=4)
coils = {c["name"]: c for c in d["coils"]}
draw_coils(
    pol,
    [
        np.array(
            [
                [c["r"] - c["dr"] / 2, c["z"] - c["dz"] / 2],
                [c["r"] + c["dr"] / 2, c["z"] - c["dz"] / 2],
                [c["r"] + c["dr"] / 2, c["z"] + c["dz"] / 2],
                [c["r"] - c["dr"] / 2, c["z"] + c["dz"] / 2],
            ]
        )
        for c in d["coils"]
    ],
    STYLE,
    edgecolor=INK,
    linewidth=1.0,
)


def text(x, y, label, **kwargs):
    pol.text(x, y, label, fontsize=LABEL, color=INK, zorder=8, **kwargs)


for name in ("Dh", "Db", "Eh", "Eb", "Fh", "Fb"):
    text(coils[name]["r"], coils[name]["z"], name, ha="center", va="center")
for name, side in (("Fh", 1), ("Fb", -1)):
    c = coils[name]
    text(
        c["r"],
        c["z"] + side * (c["dz"] / 2 + 0.06),
        f"{d['currents'][name] * c['nturn'] / 1e6:+.3f} MA·t",
        ha="center",
        va="bottom" if side > 0 else "top",
    )
text(
    boundary[:, 0].max() + 0.12,
    0.0,
    f"Ip {d['ip'] / 1e6:+.3f} MA",
    ha="left",
    va="center",
    bbox=WHITE_BOX,
)

# sampling lines, named outboard of the 1 mT iso-line
for name, height in d["lines"].items():
    pol.plot(
        [edge, rmax],
        [height, height],
        color=colours[name],
        linewidth=1.6 if name == "midplane" else 1.2,
        zorder=4,
    )
    pol.text(
        rmax,
        height + 0.06,
        name,
        ha="right",
        va="bottom",
        fontsize=LABEL,
        color=colours[name],
        zorder=8,
    )

# field panel, read as distance from the coil edge
starts = {}
for name in ("Fb plane", "Fh plane", "midplane"):
    k = lines.index(name)
    fld.plot(
        radius,
        bmag[k],
        color=colours[name],
        linewidth=3.0 if name == "midplane" else 2.6,
        zorder=5 if name == "midplane" else 4,
    )
    starts[name] = bmag[k][0]
for name, height in {"Fh plane": 75.0, "Fb plane": 42.0, "midplane": 19.3}.items():
    fld.text(
        edge - 0.12,
        height,
        f"{name}, Z = {d['lines'][name]:+.2f} m",
        ha="right",
        va="center",
        fontsize=LABEL,
        color=colours[name],
    )
    fld.plot(
        [edge - 0.1, edge], [height, starts[name]], color=colours[name], linewidth=1.0
    )
for level in (1.0, 5.0, 20.0):
    fld.hlines(level, edge, rmax, color=GREY, linewidth=1.2, linestyle=":", zorder=1)
    fld.text(
        rmax,
        level * 1.06,
        f"{level:g} mT",
        ha="right",
        va="bottom",
        fontsize=LABEL,
        color=GREY,
    )
fld.axvspan(4.80, 4.98, color="#e4e4e4", zorder=0, linewidth=0)
fld.text(
    5.06,
    50,
    "iron return limb (not in the model)",
    ha="left",
    va="center",
    fontsize=LABEL,
    color=IRON,
)
r_five = crossing(bmag[lines.index("midplane")], 5.0)
for axes, y in ((fld, 5.0), (pol, 0.0)):
    axes.plot(
        r_five,
        y,
        "o",
        color=MID,
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=9,
    )
fld.text(
    r_five + 0.12,
    7.5,
    f"5 mT at {r_five - edge:.2f} m from the coil edge",
    ha="left",
    va="bottom",
    fontsize=LABEL,
    color=MID,
)
fld.set_yscale("log")
fld.set_ylim(0.6, 90)
fld.set_yticks([1, 2, 5, 10, 20, 50])
fld.set_yticklabels(["1", "2", "5", "10", "20", "50"])
fld.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fld.set_xlim(0, rmax)
fld.spines["left"].set_bounds(1, 50)
fld.spines["bottom"].set_bounds(edge, rmax)
distances = np.arange(0, rmax - edge + 1e-9, 1.0)
fld.set_xticks(edge + distances)
fld.set_xticklabels([f"{v:g}" for v in distances])
fld.set_xlabel("distance from Fh/Fb outer edge [m]")
fld.set_ylabel("|B| [mT]")
fig.text(
    0.012,
    0.988,
    f"WEST 62751 · t = {d['time']:.3f} s",
    ha="left",
    va="top",
    fontsize=12,
    color="#666666",
)

fig.canvas.draw()
pbox, fbox = pol.get_position(), fld.get_position()
fld.set_position([pbox.x0, fbox.y0, pbox.width, fbox.height])
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
texts = [
    t
    for t in fig.findobj(matplotlib.text.Text)
    if t.get_visible() and t.get_text().strip()
]
problems = []
for t in texts:
    bb = t.get_window_extent(renderer)
    if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.x1 + 1 or bb.y1 > fig.bbox.y1 + 1:
        problems.append(f"clipped: {t.get_text()!r}")
for a, b in itertools.combinations(texts, 2):
    if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)):
        problems.append(f"overlap: {a.get_text()!r} / {b.get_text()!r}")
if problems:
    fig.savefig(f"{stem}-refused.png", dpi=100, facecolor="white")
    raise SystemExit("text check refused: " + "; ".join(problems))
fig.savefig(f"{stem}.png", dpi=100, facecolor="white")
fig.savefig(f"{stem}.svg", facecolor="white")
print(
    f"text check: clean; {levels.size} flux levels at {step:.4f} Wb; "
    f"midplane 5 mT at {r_five - edge:.3f} m from the edge"
)
