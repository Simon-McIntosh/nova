"""Show the deterministically selected cells and the reference polygon geometry."""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nova.media.ink import DEFAULT_INK, poloidal_axes

output = Path(__file__).resolve().parent / "bernstein-reference"
arrays = np.load(output / "production-patch-reference-arrays.npz")
figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
for ax in axes:
    poloidal_axes(ax)
for index, cell in enumerate(arrays["cell_ids"]):
    polygon = arrays["atomic_vertices"][index, : arrays["counts"][index]]
    closed = np.vstack((polygon, polygon[:1]))
    reference = arrays[f"reference_polygon_{cell}"].astype(float)
    ref_closed = np.vstack((reference, reference[:1]))
    axes[0].plot(
        *closed.T,
        color=DEFAULT_INK.coil_edgecolor,
        linewidth=DEFAULT_INK.coil_linewidth,
    )
    axes[0].fill(
        *reference.T,
        facecolor=DEFAULT_INK.plasma_facecolor,
        alpha=DEFAULT_INK.plasma_alpha,
    )
    axes[0].plot(
        *ref_closed.T,
        color=DEFAULT_INK.flux_color,
        linewidth=DEFAULT_INK.flux_linewidth,
    )
    axes[0].text(*arrays["centres"][index], str(cell), fontsize=6, ha="center")
    if index == 0:
        axes[1].plot(
            *closed.T,
            color=DEFAULT_INK.coil_edgecolor,
            linewidth=1,
            linestyle="--",
            label="atomic cell",
        )
        axes[1].fill(
            *reference.T,
            facecolor=DEFAULT_INK.plasma_facecolor,
            alpha=DEFAULT_INK.plasma_alpha,
        )
        axes[1].plot(
            *ref_closed.T,
            color=DEFAULT_INK.flux_color,
            linewidth=1,
            label="reference polygon",
        )
        roots = arrays["reference_crossings"][0].astype(float)
        axes[1].plot(
            *roots.T,
            "o",
            color=DEFAULT_INK.separatrix_color,
            markersize=4,
            label="edge roots",
        )
        axes[1].plot(
            *reference[:129].T, ".", color=DEFAULT_INK.flux_color, markersize=2
        )
        axes[1].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8
        )
axes[0].set_title("First twenty live weak-110 cut cells", fontsize=11)
axes[1].set_title(
    f"Cell {arrays['cell_ids'][0]}: 128 straight arc segments", fontsize=11
)
figure.suptitle("Frozen-coefficient reference geometry", fontsize=13)
figure.savefig(output / "reference-cells.svg")
figure.savefig(output / "reference-cells.png", dpi=150)
