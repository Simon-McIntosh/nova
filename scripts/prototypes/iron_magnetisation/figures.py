"""Figures for the iron-magnetisation plan, from the prototype receipts.

Usage: figures.py <research dir> <hex solve pickle> <efit probe pickle> <out dir>
Writes west-midplane-iron.svg, west-iron-permeability.svg and, when the shell
receipt is complete, shell-shielding-convergence.svg.
"""

from __future__ import annotations

import itertools
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from nova.media.ink import DEFAULT_INK, poloidal_axes  # noqa: E402
from nova.media.poloidal import draw_coils, draw_wall  # noqa: E402

research, solve_path, efit_path, out = map(Path, sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
plt.style.use("data-ink")
plt.rcParams.update(
    {"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13}
)
LABEL = 13
SUBJECT = "#7b4fb3"  # nova edge-sheet iron, nonlinear: the subject
LINEAR = "#b49ad8"  # the same, linear mu_r = 215
EFIT = "#2e8b57"  # imas-efit with its iron model
NEUTRAL = "#929292"  # imas-efit without iron
MODEL = "#222222"  # nova air core, dashed


def checked_save(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [
        t
        for t in fig.findobj(matplotlib.text.Text)
        if t.get_visible() and t.get_text().strip()
    ]
    problems = [
        f"clipped {t.get_text()!r}"
        for t in texts
        if (bb := t.get_window_extent(renderer)).x0 < -1
        or bb.y0 < -1
        or bb.x1 > fig.bbox.x1 + 1
        or bb.y1 > fig.bbox.y1 + 1
    ]
    problems += [
        f"overlap {a.get_text()!r}/{b.get_text()!r}"
        for a, b in itertools.combinations(texts, 2)
        if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))
    ]
    if problems:
        fig.savefig(out / f"{stem}-refused.png", dpi=100)
        raise SystemExit("; ".join(problems))
    fig.savefig(out / f"{stem}.svg", facecolor="white")
    fig.savefig(out / f"{stem}.png", dpi=100, facecolor="white")
    print("wrote", stem)


solve = pickle.load(open(solve_path, "rb"))
edge = solve["edge"]
nonlinear = pickle.load(open(research / "west-008-table.pkl", "rb"))
linear = pickle.load(open(research / "west-008-215.pkl", "rb"))
efit = pickle.load(open(efit_path, "rb"))
mid = list(solve["lines"]).index("midplane")
radius = solve["radius"]
distance = radius - edge

# midplane |B| for every arm
fig, ax = plt.subplots(figsize=(14, 7.2))
fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.12)
select = [k for k, p in enumerate(efit["points"]) if p[0] == "midplane"]
efit_r = np.array([efit["points"][k][1] for k in select]) - edge
arms = [
    (distance, solve["bmag"][mid] * 1e3, MODEL, "--", 2.4, "nova, air core"),
    (
        efit_r,
        efit["noiron"]["b"][select] * 1e3,
        NEUTRAL,
        "-",
        2.6,
        "imas-efit, no iron",
    ),
    (efit_r, efit["iron"]["b"][select] * 1e3, EFIT, "-", 2.6, "imas-efit, iron model"),
    (
        distance,
        linear["with_iron"][mid] * 1e3,
        LINEAR,
        "-",
        2.6,
        "nova sheets, mu_r = 215",
    ),
    (
        distance,
        nonlinear["with_iron"][mid] * 1e3,
        SUBJECT,
        "-",
        3.0,
        "nova sheets, CEA B-H",
    ),
]
for x, y, colour, style, width, label in arms:
    keep = (x > 0.05) & ~((x > 4.80 - edge - 0.02) & (x < 4.98 - edge + 0.02))
    y = np.where(keep, y, np.nan)  # the field inside the iron limb is not stray field
    keep = x > 0.05
    ax.plot(
        x[keep],
        y[keep],
        color=colour,
        linestyle=style,
        linewidth=width,
        zorder=5 if colour == SUBJECT else 4,
        label=label,
    )
fig.legend(
    loc="upper center", ncol=3, frameon=False, fontsize=LABEL, bbox_to_anchor=(0.5, 1.0)
)
ax.axvspan(4.80 - edge, 4.98 - edge, color="#e4e4e4", linewidth=0, zorder=0)
ax.text(
    4.98 - edge + 0.06,
    30,
    "outer return limb",
    fontsize=LABEL,
    color="#777777",
    va="center",
)
ax.set_yscale("log")
ax.set_xlim(0, 7.5)
ax.set_xticks(np.arange(0, 7.6, 1.0))
ax.set_ylim(0.02, 60)
ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50])
ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50"])
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("distance from Fh/Fb outer edge on the midplane [m]")
ax.set_ylabel("|B| [mT]")
checked_save(fig, "west-midplane-iron")
plt.close(fig)

# converged relative permeability of the iron cells
fig = plt.figure(figsize=(14, 7.2))
ax = fig.add_axes([0.25, 0.02, 0.45, 0.96])
poloidal_axes(ax)
cells = PolyCollection(
    nonlinear["polygons"],
    array=nonlinear["mu_r"],
    cmap="viridis",
    edgecolors="none",
    zorder=1,
)
ax.add_collection(cells)
draw_wall(
    ax,
    units=None,
    radius=solve["limiter"][:, 0],
    height=solve["limiter"][:, 1],
    style=DEFAULT_INK,
)
draw_coils(
    ax,
    [
        np.array(
            [
                [c["r"] - c["dr"] / 2, c["z"] - c["dz"] / 2],
                [c["r"] + c["dr"] / 2, c["z"] - c["dz"] / 2],
                [c["r"] + c["dr"] / 2, c["z"] + c["dz"] / 2],
                [c["r"] - c["dr"] / 2, c["z"] + c["dz"] / 2],
            ]
        )
        for c in solve["coils"]
    ],
    DEFAULT_INK,
    edgecolor="#333333",
    linewidth=1.0,
)
ax.set_xlim(-0.1, 5.3)
ax.set_ylim(-3.8, 3.8)
bar = fig.colorbar(cells, cax=fig.add_axes([0.72, 0.12, 0.018, 0.76]))
bar.set_ticks([250, 500, 750, 1000, 1250, 1500])
bar.set_label("relative permeability", fontsize=15)
bar.ax.tick_params(labelsize=13)
bar.outline.set_visible(False)
checked_save(fig, "west-iron-permeability")
plt.close(fig)

# shell shielding convergence, once the receipt is complete
shell_path = research / "shell.json"
if shell_path.is_file():
    shell = json.load(open(shell_path))
    fine = research / "shell-mu1000.json"
    if fine.is_file():
        shell.update(json.load(open(fine)))
    fig, ax = plt.subplots(figsize=(14, 7.2))
    fig.subplots_adjust(left=0.09, right=0.86, top=0.95, bottom=0.12)
    shades = {"10": "#c9b6e6", "100": "#9a78cc", "1000": SUBJECT}
    for mu, colour in shades.items():
        rows = sorted(
            (float(k.split("_h")[1]), float(np.max(v["error"])))
            for k, v in shell.items()
            if k.startswith(f"mu{mu}_")
        )
        if not rows:
            continue
        h, err = np.array(rows).T
        ax.plot(h * 100, err, color=colour, marker="o", linewidth=2.6)
        ax.text(
            h[-1] * 100 * 1.06,
            err[-1],
            f"mu_r = {mu}",
            fontsize=LABEL,
            color=colour,
            va="center",
            ha="left",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 6)
    ax.set_ylim(5e-5, 0.15)
    ax.set_xticks([1, 2, 4])
    ax.set_xticklabels(["1", "2", "4"])
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["0.01 %", "0.1 %", "1 %", "10 %"])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("cell size [cm]")
    ax.set_ylabel("shielding-factor error")
    checked_save(fig, "shell-shielding-convergence")
