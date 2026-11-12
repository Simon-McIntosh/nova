"""Render converged WEST iron cell edges and the mesh-pair field difference."""

import argparse
import json
from pathlib import Path
import pickle
import re

import imas
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
import numpy as np  # noqa: E402

from nova.equilibrium.wall_mask import wall_unit_from_ids  # noqa: E402
from nova.media.ink import DEFAULT_INK, poloidal_axes  # noqa: E402
from nova.media.poloidal import draw_coils, draw_wall  # noqa: E402
from scripts.edge_sheet.figure import save_checked  # noqa: E402
from scripts.west_iron_solve.measure import MACHINE  # noqa: E402

IRON = "#7b4fb3"


def draw_machine(axis, source):
    """Draw unit-faithful wall and coil outlines on a poloidal axis."""
    poloidal_axes(axis)
    axis.set_xticks([])
    axis.set_yticks([])
    with imas.DBEntry(f"imas:hdf5?path={MACHINE}", "r", dd_version="4.1.0") as entry:
        stored = entry.get("wall", autoconvert=False).description_2d[0].limiter.unit
        outline = stored[0].outline
        np.testing.assert_allclose(
            np.column_stack((np.asarray(outline.r), np.asarray(outline.z))),
            source["limiter"],
            rtol=0,
            atol=1e-12,
        )
        wall = tuple(wall_unit_from_ids(unit, index=i) for i, unit in enumerate(stored))
    assert wall
    draw_wall(axis, units=wall, style=DEFAULT_INK)
    outlines = []
    for coil in source["coils"]:
        r, z, dr, dz = (coil[k] for k in ("r", "z", "dr", "dz"))
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
    draw_coils(axis, outlines, DEFAULT_INK, edgecolor="#929292", linewidth=1.0)
    axis.set_xlim(-0.1, 5.3)
    axis.set_ylim(-3.8, 3.8)
    return len(wall)


def save_figure(fig, output, preview):
    """Check text geometry and declare the common slide canvas."""
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    labels = save_checked(fig, str(output), str(preview))
    svg = output.read_text()
    svg = re.sub(
        r'width="[^\"]+" height="[^\"]+"',
        'width="31.1111cm" height="16cm"',
        svg,
        count=1,
    )
    svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    output.write_text(svg)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.receipt.read_text())
    coarse, fine = summary["rows"]
    assert coarse["converged"] and fine["converged"]
    directory = args.receipt.parent / f"{fine['cell_size_m']:.4g}"
    with (directory / "solution.pkl").open("rb") as stream:
        solution = pickle.load(stream)
    with Path(fine["source"]).open("rb") as stream:
        source = pickle.load(stream)
    plt.style.use("data-ink")
    plt.rcParams.update({"svg.fonttype": "none"})
    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    mesh_axis = fig.add_axes([0.02, 0.04, 0.41, 0.92])
    poloidal_axes(mesh_axis)
    mesh_axis.set_xticks([])
    mesh_axis.set_yticks([])
    norm = matplotlib.colors.Normalize(
        vmin=solution.permeability_relative.min(),
        vmax=solution.permeability_relative.max(),
    )
    cmap = matplotlib.colormaps["viridis"]
    edges = PolyCollection(
        solution._mesh.polygons,
        facecolors="none",
        edgecolors=cmap(norm(solution.permeability_relative)),
        linewidths=0.35,
        zorder=2,
    )
    mesh_axis.add_collection(edges)
    wall_count = draw_machine(mesh_axis, source)
    colour_axis = fig.add_axes([0.445, 0.2, 0.014, 0.6])
    bar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colour_axis
    )
    bar.set_label("relative permeability")
    bar.outline.set_visible(False)
    axis = fig.add_axes([0.66, 0.18, 0.31, 0.66])
    delta = np.asarray(summary["discretisation_error_vector_norm_T"])
    scale, unit = (1e3, "mT") if delta.max() >= 1e-3 else (1e6, "µT")
    axis.plot(
        fine["probe_distance_m"],
        delta * scale,
        color=IRON,
        linewidth=3.0,
        marker="o",
        markersize=7,
        zorder=3,
    )
    axis.set_xlabel("distance from coil edge [m]")
    axis.set_ylabel(f"mesh field difference [{unit}]")
    axis.set_xticks(fine["probe_distance_m"])
    axis.set_xlim(0.7, 4.3)
    axis.set_ylim(bottom=0)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    labels = save_figure(fig, args.output, args.preview)
    print(
        "FIGURE "
        + json.dumps(
            dict(
                output=str(args.output),
                preview=str(args.preview),
                labels=labels,
                cells=fine["cells"],
                cell_size_m=fine["cell_size_m"],
                field_difference_unit=unit,
                no_direct_labels=True,
                wall_units=wall_count,
            )
        )
    )


if __name__ == "__main__":
    main()
