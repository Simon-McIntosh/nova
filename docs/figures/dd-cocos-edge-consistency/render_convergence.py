"""Render the edge-consistency convergence figure.

The measurement is the one ``tests/test_edge_consistency.py`` asserts, imported
from that module rather than restated here, so the figure cannot drift from the
gate it illustrates.  Run from the repository root:

    JAX_PLATFORMS=cpu python docs/figures/dd-cocos-edge-consistency/render_convergence.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import DEFAULT_INK, trace_axes
from nova.imas.edge_consistency import EDGE_BAND_RELATIVE_THRESHOLD

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).with_name("edge-consistency-convergence.svg")
SIZES = (33, 65, 129, 257)


def _fixture():
    spec = importlib.util.spec_from_file_location(
        "edge_consistency_fixture", ROOT / "tests" / "test_edge_consistency.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    fixture = _fixture()
    state = fixture.cerfon_freidberg_single_null()
    spacings = np.array(
        [
            (fixture.RADIAL_BOUNDS[1] - fixture.RADIAL_BOUNDS[0]) / (size - 1)
            for size in SIZES
        ]
    )
    band = np.array([fixture._measure(state, size).band_relative for size in SIZES])

    order = float(np.polyfit(np.log(spacings), np.log(band), 1)[0])
    fit = band[0] * (spacings / spacings[0]) ** order

    plt.style.use("data-ink")
    figure, axes = plt.subplots(figsize=(31.1 / 2.54, 16.0 / 2.54), layout="constrained")
    trace_axes(axes, DEFAULT_INK)
    axes.set_xscale("log")
    axes.set_yscale("log")

    axes.plot(
        spacings,
        band,
        color=DEFAULT_INK.flux_color,
        linewidth=3.0,
        marker="o",
        markersize=5.0,
        zorder=4,
        label="band residual",
    )
    axes.plot(
        spacings,
        fit,
        color="#111111",
        linewidth=2.4,
        linestyle="--",
        zorder=3,
        label=f"fitted order {order:.2f}",
    )
    axes.axhline(
        EDGE_BAND_RELATIVE_THRESHOLD,
        color="#929292",
        linewidth=1.2,
        linestyle=":",
        zorder=2,
    )

    axes.legend(
        frameon=False,
        fontsize=14,
        loc="lower right",
    )
    axes.annotate(
        f"threshold {EDGE_BAND_RELATIVE_THRESHOLD:.1e}",
        xy=(spacings[-1], EDGE_BAND_RELATIVE_THRESHOLD),
        xytext=(spacings[-1] * 1.2, EDGE_BAND_RELATIVE_THRESHOLD * 1.6),
        fontsize=14,
        color="#929292",
        annotation_clip=False,
    )

    axes.set_xlabel("radial grid spacing [m]", fontsize=16)
    axes.set_ylabel("band residual / profile current", fontsize=16)
    axes.tick_params(axis="both", which="both", labelsize=14)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(14)

    figure.savefig(OUTPUT)
    print(f"order={order:.4f}")
    for size, spacing, value in zip(SIZES, spacings, band):
        print(f"size={size} spacing={spacing:.6f} band_relative={value:.6e}")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()