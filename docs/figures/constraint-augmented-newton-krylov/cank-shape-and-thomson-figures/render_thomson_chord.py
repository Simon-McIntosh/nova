"""Regenerate the Thomson chord comparison with a per-chord residual panel.

The receipt's verdict is a measured disagreement reported without tuning, so the
figure must read as disagreement: a residual panel carries the per-chord
prediction minus measurement, and the title carries the receipt's rms and maximum
for both series. The maximum is persisted into the receipt beside the rms so the
plotted numbers are re-derivable from the receipt's own fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
FIGURE_DIR = ROOT / "docs/figures/constraint-augmented-newton-krylov/thomson-forward"
RECEIPT = FIGURE_DIR / "receipt.json"

MEASURED_COLOR = "#1a1a1a"
PREDICTION_COLOR = "#3366cc"
ZERO_COLOR = "#9e9e9e"
DENSITY_SCALE = 1e19


def series(receipt):
    trace = receipt["trace"]
    radius = np.asarray([c["position_m"][0] for c in trace], float)
    measured_t = np.asarray([c["measured_temperature_ev"] for c in trace], float)
    predicted_t = np.asarray([c["predicted_temperature_ev"] for c in trace], float)
    measured_n = (
        np.asarray([c["measured_density_m3"] for c in trace], float) / DENSITY_SCALE
    )
    predicted_n = (
        np.asarray([c["predicted_density_m3"] for c in trace], float) / DENSITY_SCALE
    )
    return radius, measured_t, predicted_t, measured_n, predicted_n


def persist_maximum(receipt, temperature_max, density_max):
    disagreement = receipt["disagreement"]
    disagreement["temperature_max_absolute_error_ev"] = temperature_max
    disagreement["density_max_absolute_error_m3"] = density_max
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")


def comparison(axes, radius, measured, predicted, ylabel):
    axes.plot(
        radius,
        measured,
        color=MEASURED_COLOR,
        linewidth=3.0,
        marker="o",
        markersize=5,
        label="measured",
    )
    axes.plot(
        radius,
        predicted,
        color=PREDICTION_COLOR,
        linewidth=2.6,
        linestyle="--",
        marker="s",
        markersize=5,
        label="untuned prediction",
    )
    axes.set_ylabel(ylabel)
    axes.set_xlabel("MAST scattering radius [m]")


def residual(axes, radius, values, rms, maximum, ylabel):
    axes.axhline(0.0, color=ZERO_COLOR, linewidth=1.2, linestyle=":", zorder=1)
    axes.plot(
        radius,
        values,
        color=PREDICTION_COLOR,
        linewidth=2.6,
        marker="o",
        markersize=5,
        zorder=3,
    )
    axes.set_ylabel(ylabel)
    axes.set_xlabel("MAST scattering radius [m]")
    axes.text(
        0.03,
        0.95,
        "rms %.3g\nmax %.3g %s" % (rms, maximum["value"], maximum["unit"]),
        transform=axes.transAxes,
        va="top",
        ha="left",
        fontsize=15,
        color=PREDICTION_COLOR,
    )


def main():
    receipt = json.loads(RECEIPT.read_text())
    radius, measured_t, predicted_t, measured_n, predicted_n = series(receipt)
    residual_t = predicted_t - measured_t
    residual_n = predicted_n - measured_n
    rms_t = float(np.sqrt(np.mean(residual_t**2)))
    rms_n = float(np.sqrt(np.mean(residual_n**2)))
    max_t = float(np.max(np.abs(residual_t)))
    max_n = float(np.max(np.abs(residual_n)))
    persist_maximum(receipt, max_t, max_n * DENSITY_SCALE)

    plt.style.use("data-ink")
    figure, grid = plt.subplots(
        2, 2, figsize=(14, 8.5), dpi=100, constrained_layout=True
    )
    top_left, top_right = grid[0]
    bottom_left, bottom_right = grid[1]

    comparison(top_left, radius, measured_t, predicted_t, "T_e [eV]")
    top_left.legend(loc="upper right", fontsize=16)
    residual(
        top_right,
        radius,
        residual_t,
        rms_t,
        {"value": max_t, "unit": "eV"},
        "residual [eV]",
    )

    comparison(bottom_left, radius, measured_n, predicted_n, "n_e [1e19 m^-3]")
    residual(
        bottom_right,
        radius,
        residual_n,
        rms_n,
        {"value": max_n, "unit": "1e19 m^-3"},
        "residual [1e19 m^-3]",
    )

    figure.suptitle(
        "MAST 22086/43 Thomson chords: T_e rms %.1f eV (max %.1f); n_e rms %.3f (max %.3f) [1e19 m^-3]"  # noqa: E501
        % (rms_t, max_t, rms_n, max_n),
        fontsize=16,
    )
    figure.savefig(FIGURE_DIR / "chord-comparison.png")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
