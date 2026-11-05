"""Plot the WEST stray poloidal field: poloidal scene above, |B|(R) below.

The two panels share one major-radius scale, so a radius read off the field
panel sits directly under the same radius in the machine. The save refuses a
figure whose text leaves the canvas or overlaps other text.
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
from matplotlib.path import Path  # noqa: E402
from scipy.interpolate import RectBivariateSpline  # noqa: E402

from nova.media.ink import DEFAULT_INK, poloidal_axes  # noqa: E402
from nova.media.poloidal import (  # noqa: E402
    draw_coils,
    draw_nulls,
    draw_plasma_cells,
    draw_wall,
)

source, stem = sys.argv[1], sys.argv[2]
d = pickle.load(open(source, "rb"))

plt.style.use("data-ink")
plt.rcParams.update(
    {"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13}
)
STYLE = DEFAULT_INK
LABEL = 13
MID = "#2b6cb0"  # midplane line: the headline series
FH = "#2e8b57"  # Fh coil plane
FB = "#c2552d"  # Fb coil plane
BAND = "#7b4fb3"  # |B| iso-lines on the poloidal panel
FLUX = "#9a9a9a"  # flux contours
GREY = "#929292"
INK = "#333333"
IRON = "#777777"

radius = d["radius"]
bmag = d["bmag"] * 1e3
edge = d["edge"]
lines = list(d["lines"])  # midplane, Fb plane, Fh plane


def crossing(b, level):
    """Outermost radius where ``b`` falls through ``level``."""
    idx = np.flatnonzero((b[:-1] >= level) & (b[1:] < level))
    if not idx.size:
        return np.nan
    i = idx[-1]
    return float(np.interp(level, [b[i + 1], b[i]], [radius[i + 1], radius[i]]))


def grid_shape(field, x, z):
    field = np.asarray(field, float)
    return field if field.shape == (x.size, z.size) else field.reshape(x.size, z.size)


rmax = float(np.ceil(max(crossing(b, 1.0) for b in bmag) + 0.5))
zlim = 3.0

fig = plt.figure(figsize=(14, 9.6))
pol = fig.add_axes([0.05, 0.40, 0.92, 0.57])
fld = fig.add_axes([0.05, 0.08, 0.92, 0.24])
poloidal_axes(pol, STYLE)
pol.set_xlim(0, rmax)
pol.set_ylim(-zlim, zlim)

# flux contours: the wide map outside the plasma, the fine core map inside
boundary = np.asarray(d["boundary"])
lcfs = Path(boundary)
gx, gz = d["gx"], d["gz"]
psi = grid_shape(d["psi"], gx, gz)
mesh = np.column_stack([a.ravel() for a in np.meshgrid(gx, gz, indexing="ij")])
outside = ~lcfs.contains_points(mesh).reshape(psi.shape)
levels = np.linspace(np.nanpercentile(psi, 1), np.nanpercentile(psi, 99), 41)
pol.contour(
    gx,
    gz,
    np.where(outside, psi, np.nan).T,
    levels=levels,
    colors=FLUX,
    linewidths=0.6,
    linestyles="solid",
    zorder=STYLE.zorder_flux,
)
cx, cz = d["core_x"], d["core_z"]
psi_core = grid_shape(d["psi_core"], cx, cz)
core = RectBivariateSpline(cx, cz, psi_core)
psi_lcfs = float(np.mean(core(boundary[:, 0], boundary[:, 1], grid=False)))
psi_axis = float(core(*d["axis"], grid=False))
core_mesh = np.column_stack([a.ravel() for a in np.meshgrid(cx, cz, indexing="ij")])
inside = lcfs.contains_points(core_mesh).reshape(psi_core.shape)
core_levels = psi_axis + (psi_lcfs - psi_axis) * np.linspace(0.1, 0.9, 9)
pol.contour(
    cx,
    cz,
    np.where(inside, psi_core, np.nan).T,
    levels=np.sort(core_levels),
    colors=FLUX,
    linewidths=0.6,
    linestyles="solid",
    zorder=STYLE.zorder_plasma + 1,
)
if "plasma_cells" in d:
    # a cell straddling the LCFS is drawn as its part inside it
    region = shapely.Polygon(boundary)
    clipped = []
    for cell in d["plasma_cells"]:
        part = shapely.Polygon(cell).intersection(region)
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

# |B| iso-lines outboard, labelled below the coil planes
bgrid = grid_shape(d["bgrid"], gx, gz) * 1e3
band_levels = [1, 2, 5, 10, 20]
cs = pol.contour(
    gx,
    gz,
    np.where(gx[:, None] >= 3.4, bgrid, np.nan).T,
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

# machine: iron, limiter, stabilising plates, coils, axis
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
draw_coils(pol, coil_outlines, STYLE, edgecolor=INK, linewidth=1.0)
draw_nulls(pol, magnetic_axis=d["axis"], style=STYLE, contain=d["limiter"])


def turns_current(name):
    """Coil ampere-turns in MA."""
    return d["currents"][name] * coils[name]["nturn"] / 1e6


def text(x, y, label, **kwargs):
    pol.text(x, y, label, fontsize=LABEL, color=INK, zorder=8, **kwargs)


# names inside the coils that can hold them, currents beside every coil
for name in ("Dh", "Db", "Eh", "Eb", "Fh", "Fb"):
    c = coils[name]
    text(c["r"], c["z"], name, ha="center", va="center")
for name, side in (("Fh", 1), ("Fb", -1), ("Dh", 1), ("Db", -1)):
    c = coils[name]
    text(
        c["r"],
        c["z"] + side * (c["dz"] / 2 + 0.06),
        f"{turns_current(name):+.3f} MA·t",
        ha="center",
        va="bottom" if side > 0 else "top",
    )
for name in ("Eh", "Eb"):
    c = coils[name]
    text(
        c["r"] + c["dr"] / 2 + 0.06,
        c["z"],
        f"{turns_current(name):+.3f} MA·t",
        ha="left",
        va="center",
    )
for name in ("Bh", "Bb"):
    c = coils[name]
    text(
        c["r"] + c["dr"] / 2 + 0.06,
        c["z"],
        f"{name}\n{turns_current(name):+.3f} MA·t",
        ha="left",
        va="center",
    )
text(
    coils["A"]["r"],
    coils["A"]["z"] - coils["A"]["dz"] / 2 - 0.06,
    f"A\n{turns_current('A'):+.3f} MA·t",
    ha="center",
    va="top",
)
for group, side in (("top", 1), ("bottom", -1)):
    names = [name for name in coils if name.startswith(f"Divertor_{group}")]
    total = sum(turns_current(name) for name in names)
    r_start = min(coils[name]["r"] for name in names)
    z_edge = side * max(abs(coils[name]["z"]) for name in names)
    text(
        r_start,
        z_edge + side * 0.08,
        f"div. {total:+.3f} MA·t",
        ha="left",
        va="bottom" if side > 0 else "top",
    )
text(
    d["axis"][0],
    d["axis"][1] - 0.16,
    f"Ip {d['ip'] / 1e6:+.3f} MA",
    ha="center",
    va="top",
)

# the three sampling lines, named at their outboard ends
colours = {"midplane": MID, "Fh plane": FH, "Fb plane": FB}
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

# field panel
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
label_heights = {"Fh plane": 75.0, "Fb plane": 42.0, "midplane": 19.3}
for name, height in label_heights.items():
    z_line = d["lines"][name]
    fld.text(
        edge - 0.12,
        height,
        f"{name}, Z = {z_line:+.2f} m",
        ha="right",
        va="center",
        fontsize=LABEL,
        color=colours[name],
    )
    fld.plot(
        [edge - 0.1, edge],
        [height, starts[name]],
        color=colours[name],
        linewidth=1.0,
    )
for level in (1.0, 5.0, 20.0):
    fld.hlines(level, edge, rmax, color=GREY, linewidth=1.2, linestyle=":", zorder=1)
iron_r = (4.80, 4.98)
fld.axvspan(*iron_r, color="#e4e4e4", zorder=0, linewidth=0)
fld.text(
    iron_r[1] + 0.08,
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
fld.set_xticks(np.arange(np.ceil(edge), rmax + 0.01, 1.0))
fld.set_xlabel("major radius R [m]")
fld.set_ylabel("|B| [mT]")
top = fld.secondary_xaxis("top", functions=(lambda r: r - edge, lambda s: s + edge))
top.set_xticks(np.arange(0, rmax - edge + 0.01, 1.0))
top.set_xlabel("distance from Fh/Fb outer edge [m]")
top.spines["top"].set_bounds(0, rmax - edge)

fig.text(
    0.012,
    0.988,
    f"WEST {d.get('shot', 62751)} · t = {d['time']:.3f} s",
    ha="left",
    va="top",
    fontsize=12,
    color="#666666",
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
if problems:
    fig.savefig(f"{stem}-refused.png", dpi=100, facecolor="white")
    raise SystemExit("text check refused: " + "; ".join(problems))
print("text check: clean")
fig.savefig(f"{stem}.png", dpi=100, facecolor="white")
fig.savefig(f"{stem}.svg", facecolor="white")
print(f"midplane 5 mT at R = {r_five:.3f} m, {r_five - edge:.3f} m from the edge")
print("R(1 mT) per line:", [round(crossing(b, 1.0), 2) for b in bmag])
