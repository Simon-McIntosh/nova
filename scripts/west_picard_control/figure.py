"""Draw the residual against iteration for Picard and Newton on WEST 62751.

Both histories use the solver's own normalisation,
||F(M)|| / max(||M||, ||B_ext||/mu0), so the two lines are comparable and the
tolerance line is the one Newton was held to. Rendered through
:func:`save_checked`, which refuses to write a figure whose text is clipped or
overlapping.

    python scripts/west_picard_control/figure.py --run <run-dir> --destination <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from nova.media.ink import DEFAULT_INK, trace_axes  # noqa: E402
from scripts.edge_sheet.figure import (  # noqa: E402
    HEIGHT_INCHES,
    WIDTH_INCHES,
    save_checked,
)

PICARD = "#2e8b57"
NEWTON = "#1a1a1a"
LIMIT = "#929292"


def render(run: Path, destination: Path):
    record = json.loads((run / "measurement.json").read_text())
    assert record["complete"]
    picard = record["picard"]
    newton = record["newton"]
    tolerance = float(record["newton_tolerance"])
    picard_history = np.asarray(picard["residual_history"])
    newton_history = np.asarray(newton["residual_history"])
    plt.style.use("data-ink")
    plt.rcParams["svg.fonttype"] = "none"
    fig = plt.figure(figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=100)
    axis = fig.add_axes([0.085, 0.13, 0.80, 0.70])
    trace_axes(axis, DEFAULT_INK.variant(label_fontsize=16))
    axis.semilogy(
        np.arange(picard_history.size),
        picard_history,
        color=PICARD,
        linewidth=3.0,
        zorder=5,
        label=f"Picard, {picard['iterations']} iterations",
    )
    axis.semilogy(
        np.arange(newton_history.size),
        newton_history,
        color=NEWTON,
        linestyle=(0, (6, 2)),
        linewidth=2.6,
        marker="o",
        markersize=6,
        markerfacecolor=NEWTON,
        markeredgecolor="none",
        zorder=6,
        label=f"Newton, {newton['iterations']} corrections",
    )
    axis.axhline(tolerance, color=LIMIT, linewidth=1.2, linestyle=(0, (1, 2)), zorder=1)
    axis.plot(
        [picard_history.size - 1],
        [picard_history[-1]],
        marker="o",
        markersize=7,
        markerfacecolor="none",
        markeredgecolor=PICARD,
        markeredgewidth=2.0,
        zorder=7,
    )
    axis.set_yscale("log")
    floor = 10 ** np.floor(np.log10(min(picard_history.min(), tolerance)))
    axis.set_ylim(floor, 2.0)
    axis.set_xlim(-1, picard_history.size + 1)
    axis.set_xlabel("iteration")
    axis.set_ylabel("||F(M)|| / max(||M||, ||B||/µ0)")
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D([], [], color=LIMIT, linewidth=1.2, linestyle=(0, (1, 2))))
    labels.append(f"tolerance {tolerance:.0e}")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        ncol=3,
        handlelength=2.0,
        fontsize=14,
    )
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "west-62751-picard-residual.svg"
    preview = run / "west-62751-picard-residual.png"
    artists = save_checked(fig, str(output), str(preview))
    tree = ET.parse(output)
    root = tree.getroot()
    root.set("width", f"{16 * WIDTH_INCHES / HEIGHT_INCHES:.6f}cm")
    root.set("height", "16cm")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(
        f"FIGURE_TEXT_LAYOUT_PASS artists={len(artists)} SVG={output} PNG={preview}",
        flush=True,
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("docs/figures/iron-core-magnetisation/icm-west-picard-control"),
    )
    arguments = parser.parse_args()
    render(arguments.run, arguments.destination)
