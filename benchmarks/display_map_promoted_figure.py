"""Draw the manufactured display-map mismatch from the shared test fixture."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import jax
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.text import Text
import numpy as np

from nova.equilibrium.display_flux import _flux, build_display_flux_map
from nova.jax.config import configure_dtypes
from nova.media.ink import poloidal_axes
from scripts.prototypes.bank_solve_lanes.plot import POPULATION, save_checked

ROOT = Path(__file__).resolve().parents[1]
FIGURE = (
    ROOT
    / "docs/figures/separatrix-spline-extraction/display-map-solve-currents"
    / "promoted-node-mismatch.svg"
)
PROMOTED = "#c0392b"
TEXT = "#52514e"


def measure():
    """Evaluate both fields with the fixture's currents and promoted state."""
    configure_dtypes()
    assert jax.config.jax_enable_x64
    spec = importlib.util.spec_from_file_location(
        "display_fixture", ROOT / "tests/test_display_flux.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    profile, lattice = fixture._profile()
    state, residual = fixture._solve_with_compensation(profile, lattice.node_count)
    promoted = np.zeros(lattice.node_count, dtype=bool)
    promoted[[10, 11, 20]] = True
    trial = state.copy()
    trial[promoted] += 0.05
    coupling = fixture._coupling(profile)
    moments = fixture._moment_law(state, lattice.node_count)
    coordinate = np.asarray(lattice.coordinate)
    deviations, displays = [], []
    for compensation in (fixture._compensation, None):
        display = build_display_flux_map(
            profile,
            trial,
            coupling=coupling,
            moments=moments,
            promoted=promoted,
            compensation=compensation,
            chunk=128,
        )
        # Use the builder's evaluator at the lattice nodes, without resampling.
        deviation = np.abs(
            _flux(coupling, coordinate, moments, 128, compensation) - trial
        )
        np.testing.assert_allclose(
            [deviation.max(), deviation[~promoted].max()],
            [display.max_node_mismatch_wb, display.max_solved_node_mismatch_wb],
            rtol=1e-12,
            atol=0,
        )
        deviations.append(deviation)
        displays.append(display)
    assert residual < 1e-9
    assert displays[0].promoted_node_count == 3
    assert deviations[0][~promoted].max() <= 10 * residual
    assert deviations[1][~promoted].max() > 10 * residual
    metrics = {
        "nodes": lattice.node_count,
        "solved_nodes": int((~promoted).sum()),
        "promoted_nodes": displays[0].promoted_node_count,
        "residual_wb": residual,
        "solved_mismatch_wb": displays[0].max_solved_node_mismatch_wb,
        "all_node_mismatch_wb": displays[0].max_node_mismatch_wb,
        "withheld_solved_mismatch_wb": displays[1].max_solved_node_mismatch_wb,
        "reproduction_relative_error": displays[0].reproduction_relative_error,
    }
    print("MEASURE " + json.dumps(metrics, sort_keys=True), flush=True)
    return coordinate, promoted, deviations, residual


def draw(coordinate, promoted, deviations, residual):
    """Separate population names from their marks with one shared legend."""
    plt.style.use("data-ink")
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "display-mismatch"})
    figure, (spatial, mismatch) = plt.subplots(
        1, 2, figsize=(14, 7.2), dpi=100, gridspec_kw={"width_ratios": [1, 1.5]}
    )
    poloidal_axes(spatial)
    spatial.set_xticks([])
    spatial.set_yticks([])
    spatial.set_xlim(0.52, 1.48)
    spatial.set_ylim(-0.52, 0.52)
    spatial.plot(
        *coordinate[~promoted].T,
        "o",
        color=POPULATION,
        markersize=5,
        linestyle="none",
        zorder=2,
    )
    spatial.plot(
        *coordinate[promoted].T,
        "D",
        color=PROMOTED,
        markersize=8,
        linestyle="none",
        zorder=4,
    )
    (seed,) = spatial.plot(
        1.0,
        0.0,
        "x",
        color=TEXT,
        markersize=8,
        markeredgewidth=2,
        zorder=5,
        label="X-point seed",
    )
    node = np.arange(coordinate.shape[0])
    (withheld,) = mismatch.plot(
        node[~promoted],
        deviations[1][~promoted] * 1000,
        "o",
        markerfacecolor="none",
        markeredgecolor=POPULATION,
        markersize=7,
        markeredgewidth=1.2,
        linestyle="none",
        zorder=2,
        label="compensation withheld (solved nodes)",
    )
    (solved,) = mismatch.plot(
        node[~promoted],
        deviations[0][~promoted] * 1000,
        "o",
        color=POPULATION,
        markersize=4,
        linestyle="none",
        zorder=3,
        label="solved nodes",
    )
    (highlighted,) = mismatch.plot(
        node[promoted],
        deviations[0][promoted] * 1000,
        "D",
        color=PROMOTED,
        markersize=8,
        linestyle="none",
        zorder=4,
        label="promoted",
    )
    threshold = 10 * residual * 1000
    mismatch.plot(
        [0, node[-1]],
        [threshold, threshold],
        color=POPULATION,
        linestyle=":",
        linewidth=1.2,
        zorder=1,
    )
    mismatch.annotate(
        "ten residuals",
        (45, threshold),
        xytext=(0, 6),
        textcoords="offset points",
        color=TEXT,
        fontsize=14,
        va="bottom",
    )
    # A linear interval includes exact zero without clipping or moving a sample.
    mismatch.set_yscale("symlog", linthresh=1e-10)
    mismatch.set_ylim(-1e-10, 200)
    mismatch.set_yticks([0, 1e-9, 1e-6, 1e-3, 1, 100])
    mismatch.set_xlim(-3, node[-1] + 3)
    mismatch.set_xticks([0, 20, 40, 60, 80])
    mismatch.set_xlabel("lattice node")
    mismatch.set_ylabel("absolute map − state [mWb]")
    mismatch.spines[["top", "right"]].set_visible(False)
    mismatch.grid(False)
    figure.legend(
        handles=[solved, highlighted, seed, withheld],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        labelcolor=TEXT,
        columnspacing=2,
        handletextpad=0.6,
    )
    figure.subplots_adjust(left=0.04, right=0.97, bottom=0.13, top=0.82, wspace=0.3)
    return figure


def save(figure, output):
    """Require readable text as well as the repository's checked geometry."""
    contrast = []
    for text in figure.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        rgb = np.asarray(to_rgb(text.get_color()))
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        ratio = 1.05 / (float(linear @ [0.2126, 0.7152, 0.0722]) + 0.05)
        assert ratio >= 4.5, f"low-contrast text: {text.get_text()!r}: {ratio:.3f}:1"
        contrast.append(ratio)
    save_checked(figure, output)
    output.write_text(
        "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
    )
    print(
        f"CHECK text_contrast_min={min(contrast):.3f}:1 clipped=0 "
        "text_overlap=0 text_over_data=0",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURE)
    args = parser.parse_args()
    figure = draw(*measure())
    try:
        save(figure, args.output)
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
