"""Render drive history and saved-root strip profiles from a completed receipt."""

from __future__ import annotations

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
    HEIGHT_INCHES,
    WIDTH_INCHES,
    save_checked,
)

# Retain the iron study's candidate purple and neutral comparison vocabulary.
RAMP = "#7b4fb3"
ROOT_STYLES = {"A": ("#929292", "--"), "B": ("#555555", ":"), "C": (RAMP, "-")}


def render(run, partial=False):
    record = json.loads((run / "measurement.json").read_text())
    if partial:
        saved = run.parent / "r-20261112T000127065672-icm-magnetisation-smoothness"
        column = np.load(saved / "strip-column.npz")
        profiles = {
            "A": column["baseline"],
            "B": column["homotopy"],
            "C": np.load(run / "ramp-latest.npy")[column["cells"]],
        }
        labels = {
            "A": "root A",
            "B": "root B",
            "C": f"ramp at {record['steps'][-1]['drive']:.6f}",
        }
    else:
        assert record["complete"]
        column = np.load(run / "strip-roots.npz")
        profiles = {name: column[name] for name in ROOT_STYLES}
        labels = {name: "root " + name for name in ROOT_STYLES}
    plt.style.use("scripts/magnetisation_high_permeability/data-ink.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=100)
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.11, top=0.87, wspace=0.32, hspace=0.47
    )
    for axis in axes.flat:
        trace_axes(axis, DEFAULT_INK.variant(label_fontsize=16))
        axis.tick_params(width=1.2)
        for side in ("left", "bottom"):
            axis.spines[side].set_linewidth(1.2)
    drive = [row["drive"] for row in record["steps"]]
    axes[0, 0].plot(
        drive,
        [row["max_M_A_per_m"] / 1e6 for row in record["steps"]],
        color=RAMP,
        linewidth=3.0,
        zorder=4,
    )
    axes[1, 0].plot(
        drive,
        [row["iterations"] for row in record["steps"]],
        color=RAMP,
        linewidth=3.0,
        zorder=4,
    )
    rejected = [row for row in record["attempts"] if not row["converged"]]
    if rejected:
        axes[1, 0].plot(
            [row["drive"] for row in rejected],
            [row["iterations"] for row in rejected],
            color="#b33a3a",
            marker="x",
            linestyle="none",
            markersize=7,
            label="iteration budget",
            zorder=6,
        )
    axes[0, 0].set_ylabel("maximum |M| [MA/m]")
    axes[1, 0].set_ylabel("Newton corrections")
    for axis in axes[:, 0]:
        axis.set_xlabel("drive fraction")
        axis.set_xlim(0, 1)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
    for component, axis in enumerate(axes[:, 1]):
        for name, (colour, dash) in ROOT_STYLES.items():
            axis.plot(
                column["z"],
                profiles[name][:, component] / 1e6,
                color=colour,
                linestyle=dash,
                linewidth=3.0 if name == "C" else 2.6,
                zorder=5 if name == "C" else 2,
                label=labels[name],
            )
        axis.set_xlabel("height in strip [m]")
        axis.set_ylabel(("radial M", "vertical M")[component] + " [MA/m]")
        axis.set_xticks([-2, -1.5, -1, -0.5])
    handles, labels = axes[0, 1].get_legend_handles_labels()
    failure_handles, failure_labels = axes[1, 0].get_legend_handles_labels()
    handles.extend(failure_handles)
    labels.extend(failure_labels)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=4 if rejected else 3,
        handlelength=3,
    )
    for axis in axes.flat:
        for coordinate in (axis.xaxis, axis.yaxis):
            low, high = sorted(coordinate.get_view_interval())
            coordinate.set_ticks(
                [tick for tick in coordinate.get_majorticklocs() if low <= tick <= high]
            )
    output = Path(
        "docs/figures/iron-core-magnetisation/icm-west-root-branch/root-branch.svg"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = StringIO()
    artists = save_checked(fig, buffer, str(run / "root-branch.png"))
    root = ET.fromstring(buffer.getvalue())
    root.set("width", f"{16 * WIDTH_INCHES / HEIGHT_INCHES:.6f}cm")
    root.set("height", "16cm")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    print(
        f"FIGURE_TEXT_LAYOUT_PASS artists={len(artists)} "
        f"SVG={output} PNG={run / 'root-branch.png'}",
        flush=True,
    )
    (run / "figure-summary.json").write_text(
        json.dumps(
            {
                "partial": partial,
                "drive": record["steps"][-1]["drive"],
                "labels": labels,
                "checked_text_artists": len(artists),
            },
            indent=2,
        )
        + "\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if not args.partial:
        from scripts.west_root_branch.origin import audit_origin

        audit_origin(args.run)
    render(args.run, args.partial)
