"""Render device cell-ladder traces and selected carrier support panels."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "fixed-shape"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import topology
from nova.media.ink import DEFAULT_INK, poloidal_axes, trace_axes
from nova.media.poloidal import (
    draw_nulls,
    draw_plasma_cells,
    draw_scattered_contours,
    draw_wall,
)
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs
from nova.equilibrium.solve_request import TopologyPolicy


COLOURS = {
    "limited": DEFAULT_INK.flux_color,
    "diverted": DEFAULT_INK.thomson_secondary_color,
}


def _trace(rows, output, metric, threshold, ylabel):
    figure, axes = plt.subplots(figsize=(14, 6), dpi=100)
    trace_axes(axes)
    for kind, colour in COLOURS.items():
        values = sorted(
            (row for row in rows if row["kind"] == kind and metric in row),
            key=lambda row: row["realised_cells"],
        )
        if not values:
            continue
        x = np.asarray([row["realised_cells"] for row in values])
        y = np.asarray([row[metric] for row in values])
        axes.plot(x, y, color=colour, linewidth=3, marker="o", markersize=8)
        offset = 0
        if metric == "map_wall_ms":
            offset = 13 if kind == "limited" else -13
        axes.annotate(
            kind,
            (x[-1], y[-1]),
            xytext=(9, offset),
            textcoords="offset points",
            color=colour,
            va="center",
            fontsize=20,
        )
    axes.axhline(threshold, color="#777777", linestyle=":", linewidth=1.2)
    axes.annotate(
        f"{threshold:g}",
        (min(row["realised_cells"] for row in rows), threshold),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=20,
        color="#666666",
    )
    axes.set_yscale("log")
    axes.set_xlabel("plasma cells", fontsize=22)
    axes.set_ylabel(ylabel, fontsize=22)
    axes.tick_params(labelsize=20)
    axes.margins(x=0.12, y=0.15)
    figure.savefig(output, format="svg", bbox_inches="tight")
    plt.close(figure)


def _support_panel(kind, cells, run, output):
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    machine, key, _ = _machine(run / "cache", carrier, cells)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(field, kind)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(
        lambda: topology.read(field, geometry, convention, TopologyPolicy())
    )()
    assert bool(reading.valid) and bool(reading.qualified)
    fraction = np.asarray(reading.membership)
    count = np.asarray(geometry.vertex_count)
    full = (fraction >= 1 - 1e-14) & (count == 6)
    cut = (fraction > 0) & ~full
    roots = np.vstack(
        (
            np.asarray(reading.axis)[None],
            np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        )
    )
    centre = np.asarray(geometry.centre)
    pitch = float(np.median(np.asarray(geometry.pitch)))
    near_root = (
        np.min(np.linalg.norm(centre[:, None] - roots[None], axis=-1), axis=1)
        <= 4 * pitch
    )
    live_edges = np.arange(geometry.neighbour.shape[1])[None] < count[:, None]
    wall_band = np.any((np.asarray(geometry.neighbour) < 0) & live_edges, axis=1)
    band = cut | wall_band | near_root
    polygons = [
        np.asarray(vertices)[: int(size)]
        for vertices, size in zip(geometry.vertices, count, strict=True)
    ]
    figure, axes = plt.subplots(figsize=(14, 8), dpi=100)
    poloidal_axes(axes)
    draw_plasma_cells(
        axes,
        [polygon for polygon, selected in zip(polygons, band, strict=True) if selected],
        facecolor="none",
        edgecolor=DEFAULT_INK.plasma_edgecolor,
        linewidth=0.8,
        alpha=1.0,
    )
    draw_plasma_cells(
        axes,
        [polygon for polygon, selected in zip(polygons, cut, strict=True) if selected],
        facecolor=DEFAULT_INK.plasma_facecolor,
        edgecolor=DEFAULT_INK.plasma_edgecolor,
        linewidth=0.5,
        alpha=0.7,
    )
    draw_wall(axes, machine.wall_node[:, 0], machine.wall_node[:, 1])
    analytic = certificate._exact_state(name, exact, np.asarray(machine.node))
    levels = np.linspace(float(reading.axis_flux), float(reading.boundary_flux), 9)[
        1:-1
    ]
    draw_scattered_contours(
        axes,
        machine.node[:, 0],
        machine.node[:, 1],
        analytic,
        levels,
        machine.wall_node,
        color="#666666",
        linewidth=1.2,
    )
    reference_style = DEFAULT_INK.variant(axis_color="#555555", xpoint_color="#555555")
    draw_nulls(
        axes,
        reading.axis,
        np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        style=reference_style,
        contain=machine.wall_node,
    )
    terminal = run / f"forward-{kind}-{cells}-terminal-moments.npy"
    terminal_roots = run / f"forward-{kind}-{cells}-terminal-roots.npy"
    terminal_available = terminal.is_file() and terminal_roots.is_file()
    if terminal_available:
        moments = jnp.asarray(np.load(terminal))
        operator = fixture.forward_operator(source, machine)
        exterior, _ = _exterior(run / "cache", key)
        solved = np.asarray(exterior + _image(operator, moments))[: len(machine.node)]
        draw_scattered_contours(
            axes,
            machine.node[:, 0],
            machine.node[:, 1],
            solved,
            levels,
            machine.wall_node,
            color=DEFAULT_INK.flux_color,
            linewidth=2,
        )
        solved_roots = np.load(terminal_roots)
        draw_nulls(
            axes,
            solved_roots[0],
            solved_roots[1:],
            style=DEFAULT_INK,
            contain=machine.wall_node,
        )
    handles = [
        Patch(
            facecolor=DEFAULT_INK.plasma_facecolor,
            edgecolor=DEFAULT_INK.plasma_edgecolor,
            label="cut",
        ),
        Patch(
            facecolor="none",
            edgecolor=DEFAULT_INK.plasma_edgecolor,
            label="gather band",
        ),
        Line2D([], [], color="#666666", label="analytic flux"),
    ]
    if terminal_available:
        handles.append(
            Line2D([], [], color=DEFAULT_INK.flux_color, label="terminal flux")
        )
    axes.legend(
        handles=handles,
        frameon=False,
        fontsize=16,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    axes.autoscale_view()
    figure.savefig(output, format="svg", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--cells", type=int, required=True)
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads((args.run / f"values-{kind}-{cells}.json").read_text())
        for cells in (250, 500, 1000, 2000)
        for kind in ("limited", "diverted")
    ]
    _trace(
        rows,
        args.figures / "wall-by-cells.svg",
        "map_wall_ms",
        1.0,
        "map evaluation [ms]",
    )
    _trace(
        rows,
        args.figures / "error-by-cells.svg",
        "map_error",
        1e-3,
        "relative sup error",
    )
    for kind in ("limited", "diverted"):
        _support_panel(kind, args.cells, args.run, args.figures / f"{kind}-support.svg")


if __name__ == "__main__":
    main()
