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

STORED, TRUE, NEUTRAL = "#3b6fb6", "#2e8b57", "#929292"
MODELS = (
    ("stored", "stored", STORED, "stored tapered outline, unit fill"),
    ("true", "true", TRUE, "true flat outline, fill fraction"),
)
SIZES = (0.08, 0.04)
STYLE = "scripts/magnetisation_high_permeability/data-ink.mplstyle"


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
    data = list(MODELS)
    fine = min(SIZES)
    high = max(
        float(np.load(root / f"west-{name}-{fine:g}.npz")["permeability_relative"].max())
        for name, *_ in data
    )
    norm = matplotlib.colors.Normalize(vmin=1, vmax=high)
    cmap = matplotlib.colormaps["viridis"]
    for index, (name, _, _, label) in enumerate(MODELS):
        axis = fig.add_axes([0.02 + 0.49 * index, 0.08, 0.46, 0.84])
        with (root / f"west-{name}-{fine:g}-polygons.pkl").open("rb") as stream:
            polygons = pickle.load(stream)
        values = np.load(root / f"west-{name}-{fine:g}.npz")["permeability_relative"]
        axis.add_collection(
            PolyCollection(
                polygons,
                facecolors=cmap(norm(values)),
                edgecolors="none",
                zorder=2,
            )
        )
        draw_machine(axis, source)
        axis.set_xlim(-0.1, 6.2)
        axis.set_axis_off()
        fig.text(0.25 + 0.49 * index, 0.03, label, ha="center")
    colour_axis = fig.add_axes([0.905, 0.25, 0.012, 0.5])
    bar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colour_axis
    )
    bar.set_label("iron relative permeability")
    bar.outline.set_visible(False)
    save(fig, out / "west-62751-permeability.svg")
    plt.close(fig)


def stray(root, source, out):
    """Draw the probe fields of both models at both sizes."""
    summary = json.loads((root / "summary.json").read_text())
    fine, coarse = min(SIZES), max(SIZES)
    panels = (
        ("behind_limbs", 0, 6.05, "Z [m] outboard of the 6.0 m limb face"),
        ("above_arms", 1, 3.7, "R [m] above the upper arms"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.16, top=0.82, wspace=0.30)
    for axis, (name, column, face, label) in zip(axes, panels):
        points = np.asarray(summary["probe_groups"][name])
        keep = np.isclose(points[:, column], face)
        order = np.argsort(points[keep, 1 - column])
        for key, _, colour, label_model in MODELS:
            fine_field = np.load(root / f"west-{key}-{fine:g}.npz")["field"]
            coarse_field = np.load(root / f"west-{key}-{coarse:g}.npz")["field"]
            offsets = summary["group_offsets"][name]
            local = np.flatnonzero(keep)[order]
            index = offsets + local
            magnitude = np.linalg.norm(fine_field[index], axis=1) * 1e3
            coarse_magnitude = np.linalg.norm(coarse_field[index], axis=1) * 1e3
            axis.plot(points[local, 1 - column], magnitude, color=colour, linewidth=2.6, label=label_model)
            axis.plot(points[local, 1 - column], coarse_magnitude, color=colour, linewidth=1.2, linestyle=":", label=None)
        axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, prune='both'))
        axis.tick_params(labelsize=11)
    axes[0].set_ylabel("poloidal field [mT]")
    fig.legend(loc="upper center", ncol=2, frameon=False)
    save(fig, out / "west-62751-stray-lines.svg")
    plt.close(fig)


def main():
    root, source_path, out = (Path(argument) for argument in sys.argv[1:4])
    with source_path.open("rb") as stream:
        source = pickle.load(stream)
    style = Path(__file__).resolve().parents[2] / STYLE
    if style.is_file():
        plt.style.use(style)
    plt.rcParams.update({"svg.fonttype": "none"})
    permeability(root, source, out)
    stray(root, source, out)


if __name__ == "__main__":
    main()
