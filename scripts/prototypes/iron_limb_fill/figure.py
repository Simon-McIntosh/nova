"""Draw the limb stand-in comparison: WEST permeability, control and stray field.

Usage: figure.py <evidence dir> <source hex pickle> <figure dir>
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
import numpy as np  # noqa: E402

from scripts.edge_sheet.figure import save_checked  # noqa: E402
from scripts.west_iron_solve.figure import draw_machine  # noqa: E402

SMEAR, FILL, TENSOR, NEUTRAL = "#3b6fb6", "#2e8b57", "#b8327e", "#929292"
MODELS = (
    ("stored", "smear", SMEAR, "section smear"),
    ("fill", "fill", FILL, "fill fraction"),
    ("tensor", "tensor", TENSOR, "fill fraction, cross-field"),
)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checked(fig, str(path))
    svg = re.sub(
        r'width="[^\"]+" height="[^\"]+"',
        'width="31.1111cm" height="16cm"',
        path.read_text(),
        count=1,
    )
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    print(f"FIGURE {path}", flush=True)


def permeability(root, source, out):
    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    data = {name: np.load(root / "west" / f"west-{name}.npz") for name, *_ in MODELS}
    high = max(float(d["mu"].max()) for d in data.values())
    norm = matplotlib.colors.Normalize(vmin=1, vmax=high)
    cmap = matplotlib.colormaps["viridis"]
    for index, (name, _, _, label) in enumerate(MODELS):
        axis = fig.add_axes([0.01 + 0.295 * index, 0.1, 0.29, 0.88])
        with (root / "west" / f"west-{name}-polygons.pkl").open("rb") as stream:
            polygons = pickle.load(stream)
        axis.add_collection(
            PolyCollection(
                polygons,
                facecolors=cmap(norm(data[name]["mu"])),
                edgecolors="none",
                zorder=2,
            )
        )
        draw_machine(axis, source)
        axis.set_xlim(-0.1, 6.2)
        axis.set_ylim(-3.8, 3.8)
        fig.text(0.155 + 0.295 * index, 0.04, label, ha="center")
    colour_axis = fig.add_axes([0.905, 0.25, 0.012, 0.5])
    bar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colour_axis
    )
    bar.set_label("iron relative permeability")
    bar.outline.set_visible(False)
    save(fig, out / "west-62751-permeability.svg")
    plt.close(fig)


def control(root, out):
    summary = json.loads((root / "comparison.json").read_text())
    probes = np.asarray(summary["stray_probes"])
    behind = (probes[:, 1] == 0) & (probes[:, 0] > 6)
    finest = min(row["size"] for row in summary["axisymmetric"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.86, wspace=0.12)
    for axis, mu in zip(axes, (50.0, 200.0), strict=True):
        axis.axhline(1, color=NEUTRAL, linewidth=1.2, linestyle=":", zorder=1)
        for _, key, colour, label in MODELS:
            row = next(
                r
                for r in summary["axisymmetric"]
                if r["model"] == key and r["mu"] == mu and r["size"] == finest
            )
            axis.plot(
                probes[behind, 0],
                np.asarray(row["stray_total_ratio"])[behind],
                color=colour,
                linewidth=2.6,
                marker="o",
                markersize=7,
                label=label if mu == 50 else None,
                zorder=3,
            )
        axis.set_xlabel("major radius on the midplane [m]")
        axis.set_xlim(6.1, 10.2)
        axis.set_xticks([6.5, 7, 8, 9, 10])
        axis.text(9.9, 0.72, f"iron $\\mu_r$ = {mu:g}", ha="right")
    axes[0].set_ylabel("stand-in field / six-limb mean field")
    axes[0].set_ylim(0.7, 1.35)
    fig.legend(loc="upper center", ncol=3, frameon=False)
    save(fig, out / "control-stray-ratio.svg")
    plt.close(fig)


def stray(root, out):
    grid = np.load(root / "west" / "grid.npy").reshape(121, 92, 2)
    air = np.load(root / "west" / "grid-air.npy").reshape(121, 92, 2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.86, wspace=0.12)
    lines = (
        (60, grid[60, :, 0] > 6.05, "major radius on the midplane [m]"),
        (102, grid[102, :, 0] < 6.0, "major radius 0.6 m above the upper arms [m]"),
    )
    for axis, (row, keep, label) in zip(axes, lines, strict=True):
        radius = grid[row, keep, 0]
        axis.plot(
            radius,
            np.linalg.norm(air[row, keep], axis=1) * 1e3,
            color=NEUTRAL,
            linewidth=2.4,
            label="no iron" if row == 60 else None,
            zorder=2,
        )
        for name, _, colour, text in MODELS:
            iron = np.load(root / "west" / f"west-{name}.npz")["grid_iron"]
            total = air[row, keep] + iron.reshape(121, 92, 2)[row, keep]
            axis.plot(
                radius,
                np.linalg.norm(total, axis=1) * 1e3,
                color=colour,
                linewidth=2.6,
                label=text if row == 60 else None,
                zorder=3,
            )
        axis.set_yscale("log")
        axis.set_ylim(0.03, 40)
        axis.set_yticks([0.1, 1, 10], labels=["0.1", "1", "10"])
        axis.minorticks_off()
        axis.set_xlim(radius.min(), radius.max())
        axis.set_xticks(np.arange(np.ceil(radius.min()), radius.max() + 0.01))
        axis.set_xlabel(label)
    axes[0].set_ylabel("poloidal field [mT]")
    fig.legend(loc="upper center", ncol=4, frameon=False)
    save(fig, out / "west-62751-stray-lines.svg")
    plt.close(fig)


def main():
    root, source_path, out = (Path(a) for a in sys.argv[1:4])
    with source_path.open("rb") as stream:
        source = pickle.load(stream)
    plt.style.use("data-ink")
    plt.rcParams.update({"svg.fonttype": "none"})
    permeability(root, source, out)
    control(root, out)
    stray(root, out)


if __name__ == "__main__":
    main()
