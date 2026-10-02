"""Render measured route residual histories on paired trace axes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import DEFAULT_INK, trace_axes


HERE = Path(__file__).resolve().parent
DATA = HERE / "residual-traces.json"
OUTPUT = HERE / "route-residual-before-after"


def _finite_trace(values) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(values, dtype=np.float64)
    step = np.arange(residual.size)
    selected = np.isfinite(residual) & (residual > 0.0)
    return step[selected], residual[selected]


def render() -> None:
    """Write PNG and SVG twins from the persisted trace receipt."""
    receipt = json.loads(DATA.read_text())
    style = DEFAULT_INK
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.8),
        sharey=True,
        constrained_layout=True,
        facecolor=style.figure_facecolor,
    )
    route_style = {
        "picard": (style.flux_color, "Picard"),
        "newton_krylov": (style.trace_cursor_color, "Newton–Krylov"),
    }
    for axes_item, state_name, title in zip(
        axes, ("before", "after"), ("Before", "After"), strict=True
    ):
        trace_axes(axes_item, style)
        for route, (color, label) in route_style.items():
            step, residual = _finite_trace(receipt[state_name][route])
            axes_item.plot(
                step,
                residual,
                color=color,
                linewidth=style.trace_linewidth,
                label=label,
            )
            if residual.size:
                axes_item.plot(
                    step[-1],
                    residual[-1],
                    marker="o",
                    markersize=style.trace_markersize,
                    color=color,
                )
        axes_item.axhline(
            receipt["residual_tolerance"],
            color=style.contour_color,
            linewidth=style.contour_linewidth,
            linestyle="--",
            label="registered tolerance",
        )
        axes_item.set_yscale("log")
        axes_item.set_xlabel("map evaluation", fontsize=style.label_fontsize)
        axes_item.set_title(title, fontsize=style.label_fontsize + 1)
    axes[0].set_ylabel("relative fixed-point residual", fontsize=style.label_fontsize)
    axes[1].legend(frameon=False, fontsize=style.label_fontsize, loc="upper right")
    figure.suptitle(
        "Shared forward-map convergence by accelerated route",
        fontsize=style.label_fontsize + 2,
    )
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=style.figure_dpi)
    figure.savefig(OUTPUT.with_suffix(".svg"))
    plt.close(figure)


if __name__ == "__main__":
    render()
