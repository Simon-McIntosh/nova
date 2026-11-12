"""Draw converged iron permeability and the mesh-pair far-field error bar.

Left panel: the 0.04 m mesh as cell edges coloured by the converged relative
permeability, unfilled, with the wall and coil outlines for scale. Right panel:
the far field at the plan's probes for both meshes, the separation between the
pair being the discretisation error bar.

    python scripts/west_iron_galerkin_pair/figure.py --run <run-dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import xml.etree.ElementTree as ET

import imas
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from nova.equilibrium.wall_mask import wall_unit_from_ids  # noqa: E402
from nova.media.ink import DEFAULT_INK, poloidal_axes, trace_axes  # noqa: E402
from nova.media.poloidal import draw_coils, draw_wall  # noqa: E402
from scripts.edge_sheet.figure import (  # noqa: E402
    HEIGHT_INCHES,
    WIDTH_INCHES,
    save_checked,
)
from scripts.west_iron_galerkin_pair.measure import (  # noqa: E402
    SOURCE,
    machine_source,
)

FINE = "#2e6fb7"
COARSE = "#929292"
DIFFERENCE = "#1a1a1a"


def draw_machine(axis, source):
    """Draw unit-faithful wall units and coil outlines on a poloidal axis."""
    poloidal_axes(axis)
    uri, dd_version = machine_source()
    with imas.DBEntry(uri, "r", dd_version=dd_version) as entry:
        stored = entry.get("wall", autoconvert=False).description_2d[0].limiter.unit
        wall = tuple(wall_unit_from_ids(unit, index=i) for i, unit in enumerate(stored))
    assert wall
    draw_wall(axis, units=wall, style=DEFAULT_INK)
    outlines = []
    for coil in source["coils"]:
        r, z, dr, dz = (coil[key] for key in ("r", "z", "dr", "dz"))
        outlines.append(
            np.array(
                [
                    [r - dr / 2, z - dz / 2],
                    [r + dr / 2, z - dz / 2],
                    [r + dr / 2, z + dz / 2],
                    [r - dr / 2, z + dz / 2],
                ]
            )
        )
    draw_coils(axis, outlines, DEFAULT_INK, edgecolor=COARSE, linewidth=1.0)
    axis.set_xlim(-0.1, 5.3)
    axis.set_ylim(-3.8, 3.8)


def render(run: Path):
    record = json.loads((run / "measurement.json").read_text())
    assert record["complete"]
    sizes = sorted(record["meshes"], key=float)
    coarse, fine = (record["meshes"][size] for size in sizes)
    with (run / sizes[1] / "solution.pkl").open("rb") as stream:
        solution = pickle.load(stream)
    with SOURCE.open("rb") as stream:
        source = pickle.load(stream)
    plt.style.use("data-ink")
    plt.rcParams["svg.fonttype"] = "none"
    fig = plt.figure(figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=100)
    geometry = fig.add_axes([0.015, 0.06, 0.46, 0.84])
    trace = fig.add_axes([0.63, 0.20, 0.34, 0.66])
    trace_axes(trace, DEFAULT_INK.variant(label_fontsize=16))
    table = np.asarray(record["table_mu_relative"])
    norm = Normalize(vmin=float(table.min()), vmax=float(table.max()))
    geometry.add_collection(
        PolyCollection(
            solution._mesh.polygons,
            facecolors="none",
            edgecolors=plt.get_cmap("viridis")(norm(solution.permeability_relative)),
            linewidths=0.55,
            zorder=3,
        )
    )
    draw_machine(geometry, source)
    scale = fig.add_axes([0.505, 0.24, 0.012, 0.52])
    colour_bar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), cax=scale)
    colour_bar.set_label("converged μr", fontsize=16)
    colour_bar.ax.tick_params(labelsize=14)
    colour_bar.outline.set_visible(False)
    colour_bar.ax.spines["left"].set_visible(False)
    colour_bar.ax.spines["right"].set_visible(False)
    distances = np.asarray(record["probe_distance_m"])
    coarse_field = np.asarray(coarse["magnitude_T"]) * 1e6
    fine_field = np.asarray(fine["magnitude_T"]) * 1e6
    trace.plot(
        distances,
        coarse_field,
        color=COARSE,
        linestyle="none",
        marker="o",
        markersize=8,
        markeredgecolor=COARSE,
        markerfacecolor="none",
        markeredgewidth=2.0,
        zorder=3,
        label=f"{sizes[0]} m mesh, {coarse['cells']} cells",
    )
    trace.plot(
        distances,
        fine_field,
        color=FINE,
        linestyle="none",
        marker="o",
        markersize=8,
        zorder=5,
        label=f"{sizes[1]} m mesh, {fine['cells']} cells",
    )
    for distance, upper, lower in zip(distances, fine_field, coarse_field, strict=True):
        trace.errorbar(
            distance,
            (upper + lower) / 2,
            yerr=abs(upper - lower) / 2,
            color=DIFFERENCE,
            linewidth=1.2,
            capsize=6,
            capthick=1.2,
            zorder=4,
        )
    trace.set_xlabel("distance from outer iron edge [m]")
    trace.set_ylabel("|B| at the probe [µT]")
    trace.set_xlim(0.4, 4.6)
    trace.set_xticks(distances)
    trace.set_ylim(0, 300)
    trace.set_yticks([0, 50, 100, 150, 200, 250, 300])
    difference = (fine_field - coarse_field).tolist()
    trace.text(
        0.02,
        0.96,
        "mesh-pair difference "
        + ", ".join(f"{value:.1f}" for value in difference)
        + " µT",
        transform=trace.transAxes,
        fontsize=14,
        color=DIFFERENCE,
        va="top",
        ha="left",
    )
    handles, labels = trace.get_legend_handles_labels()
    handles.append(
        Line2D([], [], color=DIFFERENCE, linewidth=1.2, label="mesh-pair difference")
    )
    labels.append("mesh-pair difference")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=3,
        handlelength=1.6,
        fontsize=14,
    )
    destination = Path(
        "docs/figures/iron-core-magnetisation/icm-west-62751-galerkin-pair"
    )
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "west-62751-mesh-pair.svg"
    preview = run / "west-62751-mesh-pair.png"
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
    render(parser.parse_args().run)
