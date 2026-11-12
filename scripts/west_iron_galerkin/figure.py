"""Compare the weighted spectrum and drive continuation of iron formulations."""

import argparse
from io import StringIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nova.media.ink import DEFAULT_INK, trace_axes  # noqa: E402
from scripts.edge_sheet.figure import (  # noqa: E402
    WIDTH_INCHES,
    HEIGHT_INCHES,
    save_checked,
)

GALERKIN = "#7b4fb3"
COLLOCATION = "#929292"


def render(run):
    record = json.loads((run / "measurement.json").read_text())
    prior = json.loads(
        (
            run.parent / "r-20261112T054008905425-icm-west-root-branch/measurement.json"
        ).read_text()
    )
    assert record["ramp"][-1]["drive"] == 1
    plt.style.use("scripts/magnetisation_high_permeability/data-ink.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=100)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.16, top=0.86, wspace=0.32)
    for axis in axes:
        trace_axes(axis, DEFAULT_INK.variant(label_fontsize=16))
    for name, color, dash, width, layer in [
        ("collocation-saved", COLLOCATION, "--", 2.6, 2),
        ("galerkin-saved", GALERKIN, "-", 3.0, 4),
    ]:
        values = np.load(run / (name + "-eigenvalues.npy"))
        axes[0].plot(
            np.linspace(0, 1, len(values)),
            values,
            color=color,
            linestyle=dash,
            linewidth=width,
            zorder=layer,
            label="collocation" if name.startswith("collocation") else "Galerkin",
        )
    for limit in (0, 1):
        axes[0].axhline(
            limit, color=COLLOCATION, linewidth=1.2, linestyle=":", zorder=1
        )
    axes[0].set_xlabel("sorted eigenvalue fraction")
    axes[0].set_ylabel("eigenvalue of volume-weighted G / μ₀")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-0.16, 1.16)
    axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1])
    for rows, color, dash, key, width, layer in [
        (prior["steps"], COLLOCATION, "--", "max_M_A_per_m", 2.6, 2),
        (record["ramp"], GALERKIN, "-", "maximum_M_A_per_m", 3.0, 4),
    ]:
        axes[1].plot(
            [row["drive"] for row in rows],
            [row[key] / 1e6 for row in rows],
            color=color,
            linestyle=dash,
            linewidth=width,
            zorder=layer,
        )
    axes[1].set_xlabel("drive fraction")
    axes[1].set_ylabel("maximum |M| [MA/m]")
    axes[1].set_xlim(0, 1)
    axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=False,
        ncol=2,
        handlelength=3,
    )
    for axis in axes:
        for coordinate in (axis.xaxis, axis.yaxis):
            low, high = sorted(coordinate.get_view_interval())
            coordinate.set_ticks(
                [tick for tick in coordinate.get_majorticklocs() if low <= tick <= high]
            )
    destination = Path("docs/figures/iron-core-magnetisation/icm-galerkin-interaction")
    destination.mkdir(parents=True, exist_ok=True)
    buffer = StringIO()
    artists = save_checked(fig, buffer, str(run / "spectrum-ramp.png"))
    root = ET.fromstring(buffer.getvalue())
    root.set("width", f"{16 * WIDTH_INCHES / HEIGHT_INCHES:.6f}cm")
    root.set("height", "16cm")
    output = destination / "spectrum-ramp.svg"
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    print(
        f"FIGURE_TEXT_LAYOUT_PASS artists={len(artists)} SVG={output} "
        f"PNG={run / 'spectrum-ramp.png'}",
        flush=True,
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    render(parser.parse_args().run)
