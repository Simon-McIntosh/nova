"""Render the measured polygon identity and shared clip execution structure."""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nova.media.ink import DEFAULT_INK, poloidal_axes

OUTPUT = Path(__file__).resolve().parent / "clip-body"


def render():
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax in axes:
        poloidal_axes(ax)
    with np.load(OUTPUT / "weak-rotation-reactor-static-110.npz") as arrays:
        live = arrays["after_included"] & arrays["after_boundary"]
        cell = int(np.flatnonzero(live)[0])
        for arm, color, style, label in (
            ("before", "#888888", "-", "baseline"),
            ("after", DEFAULT_INK.flux_color, "--", "mapped cell body"),
        ):
            count = int(arrays[arm + "_vertex_count"][cell])
            polygon = arrays[arm + "_support_vertices"][cell, :count]
            closed = np.vstack((polygon, polygon[:1]))
            axes[0].plot(
                *closed.T, color=color, linestyle=style, linewidth=1.5, label=label
            )
        axes[0].set_title(f"Weak 110, clipped cell {cell}", fontsize=11)
        figure.legend(
            *axes[0].get_legend_handles_labels(),
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
            ncol=2,
        )
    with np.load(OUTPUT / "diverted-single-null-110.npz") as arrays:
        live = np.flatnonzero(arrays["after_wedge_saddle"])
        if len(live):
            cell = int(live[0])
            for wedge, color in enumerate(("#0072B2", "#009E73", "#D55E00", "#CC79A7")):
                for arm, style in (("before", "-"), ("after", "--")):
                    count = int(arrays[arm + "_wedge_vertex_count"][cell, wedge])
                    polygon = arrays[arm + "_wedge_support_vertices"][
                        cell, wedge, :count
                    ]
                    closed = np.vstack((polygon, polygon[:1]))
                    axes[1].plot(*closed.T, color=color, linestyle=style, linewidth=1.3)
            saddle = arrays["after_wedge_saddle_vertex"][cell]
            axes[1].plot(*saddle, "x", color=DEFAULT_INK.separatrix_color)
            axes[1].set_title(
                f"Single-null 110, four wedges in cell {cell}", fontsize=11
            )
        else:
            axes[1].text(
                0.5,
                0.5,
                "No four-root saddle cell on this row",
                ha="center",
                transform=axes[1].transAxes,
            )
    figure.suptitle("Baseline and mapped polygons on the same coordinates", fontsize=13)
    figure.savefig(OUTPUT / "polygon-identity.svg")
    figure.savefig(OUTPUT / "polygon-identity.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    render()
