"""Render the band residual per sampled slice, in both flux-function senses.

The rows are the measure's own output over the sampled slices of the three
conforming sources, recorded once per sense: as the module measures them, and
with the flux functions negated, which is the sense collinear with the flux map
the stored files carry.  The threshold is drawn as the reference the module
declares, so a reader can see it cleared on every slice in both senses.

Run from the repository root::

    JAX_PLATFORMS=cpu python \
        docs/figures/dd-cocos-ingest/edge-consistency-table/render_band_residual.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import DEFAULT_INK, trace_axes
from nova.imas.edge_consistency import EDGE_BAND_RELATIVE_THRESHOLD

DIRECTORY = Path(__file__).resolve().parent
ROWS = DIRECTORY / "band-residual-rows.json"
OUTPUT = DIRECTORY / "edge-consistency-band-residual.svg"

AS_MEASURED = "#929292"
COLLINEAR = DEFAULT_INK.flux_color
THRESHOLD = "#929292"


def main() -> None:
    sources = json.loads(ROWS.read_text())
    plt.style.use("data-ink")
    figure, panels = plt.subplots(
        1, len(sources), figsize=(31.1 / 2.54, 16.0 / 2.54), layout="constrained"
    )
    for panel, source in zip(panels, sources):
        trace_axes(panel, DEFAULT_INK)
        panel.set_yscale("log")
        index = np.array([row["index"] for row in source["rows"]], dtype=float)
        as_measured = np.array([row["band_as_read"] for row in source["rows"]])
        collinear = np.array([row["band_negated"] for row in source["rows"]])
        panel.plot(
            index,
            as_measured,
            color=AS_MEASURED,
            linewidth=2.6,
            marker="o",
            markersize=4.0,
            zorder=3,
            label="as the module measures",
        )
        panel.plot(
            index,
            collinear,
            color=COLLINEAR,
            linewidth=3.0,
            marker="o",
            markersize=4.0,
            zorder=4,
            label="collinear with the flux map",
        )
        panel.axhline(
            EDGE_BAND_RELATIVE_THRESHOLD,
            color=THRESHOLD,
            linewidth=1.2,
            linestyle=":",
            zorder=2,
        )
        panel.set_title(source["source"], fontsize=14)
        panel.set_xlabel("time slice index", fontsize=16)
        panel.tick_params(axis="both", which="both", labelsize=14)
        for label in panel.get_xticklabels() + panel.get_yticklabels():
            label.set_fontsize(14)
    panels[0].set_ylabel("band residual / band profile current", fontsize=16)
    panels[0].annotate(
        f"threshold {EDGE_BAND_RELATIVE_THRESHOLD:.1e}",
        xy=(float(panels[0].get_xlim()[1]), EDGE_BAND_RELATIVE_THRESHOLD),
        xytext=(0.05, 0.35),
        textcoords="axes fraction",
        fontsize=14,
        color=THRESHOLD,
    )
    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        frameon=False,
        fontsize=14,
        ncol=2,
    )
    figure.savefig(OUTPUT)
    for panel, source in zip(panels, sources):
        band = np.array([row["band_as_read"] for row in source["rows"]])
        collinear = np.array([row["band_negated"] for row in source["rows"]])
        print(
            f"{source['source']}: as_measured {band.min():.4g}-{band.max():.4g}, "
            f"collinear {collinear.min():.4g}-{collinear.max():.4g}"
        )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()