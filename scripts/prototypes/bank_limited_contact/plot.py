"""Show the whole wall and its contact edge on the same analytic flux scale."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from nova.media.ink import DEFAULT_INK, trace_axes
from scripts.prototypes.bank_solve_lanes.plot import POPULATION, save_checked


def render(report, arcs, values, output, preview=None, *, overlap_control=False):
    """Save after the shared clipping, text collision, and data overlap checks."""
    plt.style.use("data-ink")
    feature = DEFAULT_INK.flux_color
    style = DEFAULT_INK.variant(label_fontsize=14)
    row = report["rows"][0]
    perimeter = row["perimeter_m"]
    centered = (
        arcs - row["extremum"]["arc_m"] + perimeter / 2
    ) % perimeter - perimeter / 2
    order = np.argsort(centered)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100)
    for index, ax in enumerate(axes):
        trace_axes(ax, style)
        selected = order if index == 0 else order[abs(centered[order]) <= 0.145]
        ax.plot(
            centered[selected],
            values[selected],
            color=POPULATION,
            linewidth=2.6,
            zorder=1,
        )
        ax.set_xlabel("wall arc from contact [m]", fontsize=16)
        ax.set_ylabel("analytic flux [Wb]", fontsize=16)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.spines[["left", "bottom"]].set_linewidth(1.2)
    axes[0].set_xlim(-perimeter / 2, perimeter / 2)
    axes[0].plot(
        0, row["extremum"]["flux_wb"], "o", color=feature, markersize=7, zorder=5
    )
    axes[1].set_xlim(-0.145, 0.145)
    axes[1].set_ylim(-0.075, 0.013)
    axes[1].set_xticks([-0.1, 0, 0.1])
    axes[1].set_yticks([-0.06, -0.04, -0.02, 0])
    for measured, marker in zip(report["rows"], ("s", "^"), strict=True):
        point = measured["route"]
        axes[1].plot(
            point["arc_from_contact_m"],
            point["flux_wb"],
            marker=marker,
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=feature,
            markeredgewidth=1.6,
            markersize=14 if marker == "^" else 8,
            zorder=4,
        )
    archived = row["saved_oracle"]
    axes[1].plot(
        archived["arc_from_contact_m"],
        archived["flux_wb"],
        "x",
        color="#555555",
        markersize=9,
        markeredgewidth=2,
        zorder=5,
    )
    axes[1].plot(
        0, row["extremum"]["flux_wb"], "o", color=feature, markersize=5, zorder=6
    )
    handles = [
        Line2D([], [], color=POPULATION, linewidth=2.6, label="analytic wall flux"),
        Line2D(
            [],
            [],
            color=feature,
            marker="o",
            linestyle="none",
            label="maximum / designed contact",
        ),
        Line2D(
            [],
            [],
            color="#555555",
            marker="x",
            linestyle="none",
            label="saved oracle vertex",
        ),
        Line2D(
            [],
            [],
            color=feature,
            marker="s",
            markerfacecolor="none",
            linestyle="none",
            label="route, 550 cells",
        ),
        Line2D(
            [],
            [],
            color=feature,
            marker="^",
            markerfacecolor="none",
            linestyle="none",
            label="route, 2000 cells",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.80, wspace=0.30)
    if overlap_control:
        axes[1].text(-0.005, -0.0001, "deliberate data overlap", fontsize=14)
    save_checked(fig, output)
    # Keep the vector artifact compact without changing its drawing commands.
    output.write_text(" ".join(output.read_text().splitlines()) + "\n")
    if preview:
        fig.savefig(Path(preview), dpi=100)
    plt.close(fig)
