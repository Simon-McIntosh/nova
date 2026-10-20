"""Show where the contour-tree carrier stops before the physical wall."""

from __future__ import annotations

# Precision must be configured before array-valued field modules are imported.
# ruff: noqa: E402

import argparse
from dataclasses import replace
import json
from pathlib import Path
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import ConvexHull

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import nova

assert jax.config.jax_enable_x64 is True
from nova.equilibrium import topology
from nova.equilibrium.wall_mask import WallUnit
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.prototypes.efit_field_swap.crosscheck import (
    domain_rings,
    lattice,
    read_trace_record,
)
from scripts.prototypes.efit_maximin_certificate.run import ROWS

REFERENCE = "#1f6f8b"
TREE = "#333333"
TREE_NULL = "#b46b2d"
HULL = "#929292"


def null_artist(axes, label, **kwargs):
    """Name the artist created by the shared null painter."""
    previous = len(axes.lines)
    poloidal.draw_nulls(axes, **kwargs)
    artist = axes.lines[previous]
    artist.set_label(label)
    return artist


def render(receipt, output, preview):
    print(f"nova.__file__={nova.__file__} backend={jax.default_backend()}", flush=True)
    rows = json.loads(receipt.read_text())
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 11), gridspec_kw={"height_ratios": [1.6, 1]}
    )
    handles = []
    for column, spec in enumerate(ROWS[1:]):
        row = next(
            r for r in rows if r["case"] == spec["case"] and r["resolution"] == 16
        )
        record = read_trace_record(spec["source"])
        boundary, obstacles = domain_rings(record)
        cells, samples = lattice(boundary, 16, obstacles)
        geometry = topology.TopologyGeometry.from_cells(
            cells, samples, (boundary, *obstacles)
        )
        mesh = geometry.contour_mesh
        points = np.asarray(mesh.vertex_rz)
        live = np.asarray(mesh.vertex_valid)
        triangles = np.asarray(mesh.triangles)[np.asarray(mesh.triangle_valid)]
        spline = RectBivariateSpline(record.rgrid, record.zgrid, record.psi)
        centre_values = spline.ev(
            np.asarray(geometry.centre)[:, 0], np.asarray(geometry.centre)[:, 1]
        )
        values = np.sum(
            centre_values[np.asarray(geometry.contour_sources)]
            * np.asarray(geometry.contour_weights),
            axis=1,
        )
        triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
        hull = ConvexHull(points[live])
        edge = points[live][hull.vertices]
        edge = np.vstack((edge, edge[0]))
        r = np.linspace(boundary[:, 0].min(), boundary[:, 0].max(), 300)
        z = np.linspace(boundary[:, 1].min(), boundary[:, 1].max(), 300)
        rr, zz = np.meshgrid(r, z)
        flux = spline.ev(rr.ravel(), zz.ravel()).reshape(rr.shape)
        units = (WallUnit(boundary[:, 0], boundary[:, 1], kind="vessel", closed=True),)
        axis_node = row["tree_nodes"][row["path"][0]]
        saddle = row["reached"]
        levels = [spec["efit_level"], 0.03, 0.1, 0.3, 0.6]
        for band, ax in enumerate(axes[:, column]):
            poloidal_axes(ax)
            chosen = levels if band == 0 else levels[:1]
            ax.contour(
                r, z, flux, levels=chosen, colors=REFERENCE, linewidths=1.8, zorder=3
            )
            ax.tricontour(
                triangulation,
                values,
                levels=chosen,
                colors=TREE,
                linewidths=1.8,
                linestyles="--",
                zorder=2,
            )
            poloidal.draw_wall(ax, units=units)
            (hull_artist,) = ax.plot(
                edge[:, 0],
                edge[:, 1],
                color=HULL,
                linewidth=1.2,
                linestyle=":",
                label="tree domain",
                zorder=4,
            )
            ref_axis = null_artist(
                ax,
                "physical axis",
                magnetic_axis=spec["axis"],
                style=replace(DEFAULT_INK, axis_color=REFERENCE),
                contain=units,
            )
            ref_saddle = null_artist(
                ax,
                "physical saddle",
                other_x_points=np.asarray([saddle["polish_position"]]),
                style=replace(DEFAULT_INK, xpoint_color=REFERENCE),
            )
            tree_axis = null_artist(
                ax,
                "tree axis vertex",
                magnetic_axis=axis_node["position"],
                style=replace(DEFAULT_INK, axis_color=TREE_NULL),
                contain=units,
            )
            tree_saddle = null_artist(
                ax,
                "tree saddle vertex",
                other_x_points=np.asarray([saddle["position"]]),
                style=replace(DEFAULT_INK, xpoint_color=TREE_NULL),
            )
            (contact,) = ax.plot(
                0.85,
                spec["axis"][1],
                marker="o",
                color=REFERENCE,
                markerfacecolor="white",
                markersize=7,
                linestyle="none",
                label="wall contact",
                zorder=8,
            )
            if band == 0:
                ax.set_xlim(0.80, 2.20)
                ax.set_ylim(-0.60, 0.90)
            else:
                ax.set_xlim(0.825, 1.01)
                ax.set_ylim(0.06, 0.34)
            if column == band == 0:
                (reference_line,) = ax.plot(
                    [], [], color=REFERENCE, linewidth=1.8, label="physical field"
                )
                (tree_line,) = ax.plot(
                    [],
                    [],
                    color=TREE,
                    linewidth=1.8,
                    linestyle="--",
                    label="tree field",
                )
                handles = [
                    reference_line,
                    tree_line,
                    hull_artist,
                    contact,
                    ref_axis,
                    ref_saddle,
                    tree_axis,
                    tree_saddle,
                ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.98, bottom=0.11, wspace=0.08, hspace=0.08
    )
    checker = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "tie-state-current-gap"
            / "render_tie_pair.py"
        )
    )["check_text"]
    checker(fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg")
    output.write_text(
        "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
    )
    fig.savefig(preview, dpi=100)
    plt.close(fig)
    print(f"FIGURE={output}")
    print(f"PREVIEW={preview}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    render(args.receipt, args.output, args.preview)
