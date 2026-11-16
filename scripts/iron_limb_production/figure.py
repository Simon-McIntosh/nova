"""Draw the limb stand-in comparison: WEST permeability and stray field.

Usage: figure.py <evidence dir> <source hex pickle> <figure dir>
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.tri  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

from scripts.edge_sheet.figure import save_checked  # noqa: E402
from scripts.west_iron_solve.figure import draw_machine  # noqa: E402

STORED, TRUE, NEUTRAL = "#3b6fb6", "#2e8b57", "#929292"
MODELS = (
    ("stored", "stored", STORED, "stored tapered outline, unit fill"),
    ("true", "true", TRUE, "true flat outline, fill fraction"),
)
SIZES = (0.08, 0.04)
STYLE = "scripts/magnetisation_high_permeability/data-ink.mplstyle"
EXTENT = (-0.1, 6.25, -3.7, 3.7)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checked(fig, str(path))
    svg = re.sub(
        r'width="[^\"]+" height="[^\"]+"',
        'width="31.1111cm" height="16cm"',
        path.read_text(),
        count=1,
    )
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    print(f"FIGURE {path}", flush=True)


def iron_regions(polygons, centroids):
    """Return one polygon per disconnected iron region and each cell's region.

    The cells of one model are a union of region polygons separated by air
    gaps, so the connected components of their union are the iron regions.
    A cell whose centroid falls outside every component is refused rather
    than assigned, because a contour drawn over it would span iron that is
    not there.
    """
    union = shapely.union_all([Polygon(corners) for corners in polygons])
    parts = [part for part in shapely.get_parts(union) if part.area > 1e-9]
    region = np.full(len(centroids), -1)
    for index, part in enumerate(parts):
        region[shapely.contains_xy(part, centroids[:, 0], centroids[:, 1])] = index
    assert (region >= 0).all(), f"{int((region < 0).sum())} cells outside a region"
    return parts, region


def outline_regions(axis, parts):
    """Draw every iron region boundary in neutral grey under the contours."""
    for part in parts:
        ring = np.asarray(part.exterior.coords)
        axis.plot(ring[:, 0], ring[:, 1], color=NEUTRAL, linewidth=0.9, zorder=1)


def region_contours(axis, part, centroids, values, levels, norm, cmap):
    """Contour one iron region on its own triangulation, gaps masked out.

    A triangle is kept only when the region polygon covers it whole, so no
    contour chord crosses a concavity, a limb-to-arm gap or the air outside
    the region.
    """
    triangulation = matplotlib.tri.Triangulation(centroids[:, 0], centroids[:, 1])
    triangles = shapely.polygons(centroids[triangulation.triangles])
    triangulation.set_mask(~shapely.covers(part, triangles))
    axis.tricontour(
        triangulation,
        values,
        levels=levels,
        cmap=cmap,
        norm=norm,
        linewidths=0.8,
        zorder=2,
    )
    return int(np.count_nonzero(~triangulation.mask))


def permeability(root, source, out):
    """Draw both models' relative permeability at the fine mesh."""
    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    fine = min(SIZES)
    high = max(
        float(
            np.load(root / f"west-{name}-{fine:g}.npz")["permeability_relative"].max()
        )
        for name, *_ in MODELS
    )
    norm = matplotlib.colors.Normalize(vmin=1, vmax=high)
    cmap = matplotlib.colormaps["viridis"]
    levels = np.linspace(1.0, high, 7)
    for index, (name, _, _, label) in enumerate(MODELS):
        axis = fig.add_axes([0.03 + 0.44 * index, 0.10, 0.41, 0.82])
        arrays = np.load(root / f"west-{name}-{fine:g}.npz")
        with (root / f"west-{name}-{fine:g}-polygons.pkl").open("rb") as stream:
            polygons = pickle.load(stream)
        centroids = arrays["centroids"]
        parts, region = iron_regions(polygons, centroids)
        outline_regions(axis, parts)
        kept = []
        for index_region, part in enumerate(parts):
            selected = region == index_region
            kept.append(
                region_contours(
                    axis,
                    part,
                    centroids[selected],
                    arrays["permeability_relative"][selected],
                    levels,
                    norm,
                    cmap,
                )
            )
        assert all(count > 0 for count in kept), kept
        print(f"REGIONS {name} kept_triangles={kept}", flush=True)
        draw_machine(axis, source)
        axis.set_xlim(EXTENT[0], EXTENT[1])
        axis.set_ylim(EXTENT[2], EXTENT[3])
        axis.set_axis_off()
        fig.text(0.235 + 0.44 * index, 0.04, label, ha="center", fontsize=13)
    colour_axis = fig.add_axes([0.895, 0.28, 0.012, 0.46])
    bar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colour_axis
    )
    bar.set_label("iron relative permeability")
    bar.outline.set_visible(False)
    save(fig, out / "west-62751-permeability.svg")
    plt.close(fig)


def ring_angle(points):
    """Return the poloidal angle in degrees, 0 to 360, about the ring centre."""
    centre = points.mean(axis=0)
    return (
        np.degrees(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))
        % 360.0
    )


def stray(root, source, out):
    """Draw the probe fields of both models at both sizes, one panel a group."""
    del source
    summary = json.loads((root / "summary.json").read_text())
    fine, coarse = min(SIZES), max(SIZES)
    fields = {
        key: {
            size: np.load(root / f"west-{key}-{size:g}.npz")["field"] for size in SIZES
        }
        for key, *_ in MODELS
    }
    offsets = summary["group_offsets"]
    points = {
        name: np.asarray(value) for name, value in summary["probe_groups"].items()
    }

    def magnitude(key, size, index):
        return np.linalg.norm(fields[key][size][index], axis=1) * 1e3

    fig = plt.figure(figsize=(15, 5.8), dpi=100)
    axes = [
        fig.add_axes([0.065, 0.20, 0.25, 0.58]),
        fig.add_axes([0.385, 0.20, 0.25, 0.58]),
        fig.add_axes([0.705, 0.20, 0.25, 0.58]),
    ]
    panels = (
        (axes[0], "in_vessel", "in-vessel ring", "poloidal angle [deg]"),
        (axes[1], "behind_limbs", "behind the limbs", "Z [m]"),
        (axes[2], "above_arms", "above the arms", "R [m]"),
    )
    for axis, name, title, xlabel in panels:
        group = points[name]
        index = offsets[name] + np.arange(len(group))
        segments = probe_segments(name, group)
        for key, _, colour, _ in MODELS:
            for order in segments:
                local = index[order]
                abscissa = segment_x(name, group[order])
                if name == "in_vessel":
                    abscissa[-1] = 360.0
                axis.plot(
                    abscissa,
                    magnitude(key, fine, local),
                    color=colour,
                    linewidth=2.4,
                    solid_capstyle="round",
                )
                axis.plot(
                    abscissa,
                    magnitude(key, coarse, local),
                    color=NEUTRAL,
                    linewidth=1.2,
                    linestyle=(0, (2, 2)),
                )
        axis.set_xlabel(xlabel, fontsize=14)
        axis.set_title(title, fontsize=14, loc="left", pad=8)
        axis.tick_params(labelsize=12)
        axis.margins(x=0.04, y=0.10)
        axis.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=5, prune="both")
        )
        axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("poloidal field [mT]", fontsize=14)
    handles = [
        Line2D([], [], color=colour, linewidth=2.4, label=label)
        for _, _, colour, label in MODELS
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=NEUTRAL,
            linewidth=1.2,
            linestyle=(0, (2, 2)),
            label="0.08 m mesh",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.88),
    )
    save(fig, out / "west-62751-stray-lines.svg")
    plt.close(fig)


def segment_x(name, group):
    """Return the plotted abscissa of one probe segment."""
    if name == "in_vessel":
        return ring_angle(group)
    if name == "behind_limbs":
        return group[:, 1]
    return group[:, 0]


def probe_segments(name, group):
    """Return the probe rows of each line, ordered along its own abscissa.

    The in-vessel ring is one closed loop; the behind-the-limb and
    above-the-arm groups are split into their upper and lower segments and
    then into one line per fixed R or Z, so no line crosses the air gap
    between Z = -2.5 m and Z = +2.5 m where there are no probes.
    """
    if name == "in_vessel":
        order = np.argsort(ring_angle(group))
        return [np.concatenate([order, order[:1]])]
    fixed, along = (0, 1) if name == "behind_limbs" else (1, 0)
    segments = []
    for sign in (1, -1):
        rows = np.flatnonzero(np.sign(group[:, along]) == sign)
        for value in np.unique(group[rows, fixed]):
            line = rows[group[rows, fixed] == value]
            segments.append(line[np.argsort(group[line, along])])
    return segments


def main():
    root, source_path, out = (Path(argument) for argument in sys.argv[1:4])
    with source_path.open("rb") as stream:
        source = pickle.load(stream)
    style = Path(__file__).resolve().parents[2] / STYLE
    if style.is_file():
        plt.style.use(style)
    plt.rcParams.update({"svg.fonttype": "none"})
    permeability(root, source, out)
    stray(root, source, out)


if __name__ == "__main__":
    main()
