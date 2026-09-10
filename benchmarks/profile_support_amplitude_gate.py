"""Evaluate the reinstated chord clip's unit-amplitude current on the
commitments.

The clip against the signed flux of the current iterate (the production
chord mode) is evaluated on the persisted terminal states of the
weak/moderate/strong -110 Solov'ev certificate rows: the total per-cell
current at unit amplitude against the analytic total over the analytic
plasma region, the participation of every analytic cut cell, and the six
worst per-cell booked-minus-analytic differences, with one line-contour
panel per row.

The before column is the whole-cell ratio implied by the banked terminal
amplitude: ``amplitude = target / unit-amplitude-total`` and the target is
the analytic total, so the whole-cell unit-amplitude ratio equals
``1 / amplitude`` on the same committed state (the prior census measured
0.9526 on weak).  When the state's read boundary sits inside the analytic
separatrix and its amplitude departs from unity -- the committed rows were
converged under whole-cell booking -- the after ratio moves toward unity
but carries that state-level residual; the freshly-boundary-consistent
acceptance is where the re-solved rows are judged.

Receipt and panels land under ``docs/figures/cut-cell-current-attribution/
rung-a``.  CPU-only x64: run with ``JAX_PLATFORMS=cpu`` under the shared
environment interpreter, never on login, one allocation for all three rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np

from nova.jax.config import configure_dtypes

#: The persisted certificate parts (read-only reference data in main).
PARTS = Path(
    "/home/ITER/mcintos/Code/nova/docs/figures/gs-absolute-accuracy/solovev/"
    "production-route-parts"
)


def _row_config():
    """Return (row label, case suffix, part stem) in gate order."""
    return [
        ("weak", "weak-rotation-reactor", "weak-rotation-reactor-static"),
        (
            "moderate",
            "moderate-rotation-conventional",
            "moderate-rotation-conventional-static",
        ),
        ("strong", "strong-rotation-compact", "strong-rotation-compact-static"),
    ]


def _analytic_region_polygon(case, nodes: int = 240):
    """Return the analytic plasma polygon at the exact boundary flux."""
    from shapely.geometry import Polygon

    radius, half_height, _weight, _offset = case._surface_nodes(0.0, nodes)
    upper = np.c_[radius, half_height]
    lower = np.c_[radius[::-1], -half_height[::-1]]
    return Polygon(np.vstack((upper, lower)))


def _atomic_cell_polygons(machine):
    """Return every atomic cell as a shapely polygon."""
    from shapely.geometry import Polygon

    atomic = machine.moment_geometry.atomic_mesh
    nodes = np.asarray(atomic.node_coordinates)
    cells = np.asarray(atomic.cell_nodes)
    counts = np.asarray(atomic.cell_vertex_count)
    return [Polygon(nodes[row[: counts[index]]]) for index, row in enumerate(cells)]


def _cell_analytic_regions(case, machine, density):
    """Return per-cell analytic intersection area fraction and current.

    Each atomic cell is clipped against the analytic plasma polygon and the
    exact density ``J_phi`` is integrated over the intersection with the
    same fixed Duffy-style rule the fixture uses on clipped polygons.  The
    analytic plasma is the level ``psi == 0`` of the case's own flux, so the
    classification uses the sign of the exact flux at the cell's vertices;
    the clipped-region integral is computed with shapely for the current.
    """
    from scripts.analytic_oracle_fixtures.measure import _polygon_rule

    region = _analytic_region_polygon(case)
    cells = _atomic_cell_polygons(machine)
    atomic = machine.moment_geometry.atomic_mesh
    nodes = np.asarray(atomic.node_coordinates)
    cell_nodes = np.asarray(atomic.cell_nodes)
    counts = np.asarray(atomic.cell_vertex_count)

    area_fraction = np.zeros(len(cells))
    current = np.zeros(len(cells))
    for index, polygon in enumerate(cells):
        if not polygon.is_valid or polygon.area <= 0.0:
            continue
        intersection = polygon.intersection(region)
        if intersection.is_empty or intersection.area <= 0.0:
            continue
        area_fraction[index] = intersection.area / polygon.area
        coords = intersection.exterior.coords[:]
        points, weights = _polygon_rule(np.asarray(coords, dtype=np.float64))
        current[index] = float(np.sum(weights * density(points)))
    vertex_flux_min = np.zeros(len(cells))
    vertex_flux_max = np.zeros(len(cells))
    for index, row in enumerate(cell_nodes):
        vertex_flux = np.asarray(
            case.flux(nodes[row[: counts[index]], 0], nodes[row[: counts[index]], 1])
        )
        vertex_flux_min[index] = vertex_flux.min()
        vertex_flux_max[index] = vertex_flux.max()
    return {
        "area_fraction": area_fraction,
        "current": current,
        "vertex_flux_min": vertex_flux_min,
        "vertex_flux_max": vertex_flux_max,
    }


def _current_density_evaluator(case):
    """Return ``case.toroidal_current_density`` bound to one array call."""

    def density(points):
        return np.asarray(case.toroidal_current_density(points[:, 0], points[:, 1]))

    return density


def _terminal_state(part) -> np.ndarray:
    return np.asarray(part["render_data"]["terminal_flux_wb"], dtype=np.float64)


def _terminal_amplitude(part) -> float:
    history = part["solver"]["lambda_amplitude_history"]["samples"]
    for sample in history:
        if sample["state"] == "terminal":
            return float(sample["amplitude"])
    raise KeyError("no terminal amplitude in the row's history")


def _measure_row(part):
    """Return every measured quantity for one committed terminal state."""
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture
    from tests.rotating_equilibrium_references import reference_cases

    configure_dtypes()
    import jax.numpy as jnp

    case_name = part["case"][: part["case"].index("-static")]
    case = reference_cases()[case_name].static_limit()
    machine = oracle_fixture.cached_machine(
        case, -110, wall_nodes=oracle_fixture.WALL_POINT_COUNT
    )
    operator = oracle_fixture.forward_operator(case, machine)
    state = _terminal_state(part)
    moments = operator.cell_current_moments(jnp.asarray(state))
    booked = np.asarray(moments.cell_current, dtype=np.float64)
    _masks, _topology, _sample, support = operator._support_partition(
        jnp.asarray(state)
    )
    included = np.asarray(support.included)
    density = _current_density_evaluator(case)
    regions = _cell_analytic_regions(case, machine, density)
    return {
        "case": case,
        "machine": machine,
        "operator": operator,
        "booked": booked,
        "included": included,
        "area_fraction": regions["area_fraction"],
        "vertex_flux_min": regions["vertex_flux_min"],
        "vertex_flux_max": regions["vertex_flux_max"],
        "analytic_current": regions["current"],
        "analytic_total": case.plasma_current(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)

    receipt_rows = []
    for label, _case_suffix, part_stem in _row_config():
        part = json.loads(
            (PARTS / f"{part_stem}-production-route-reduced.json").read_text()
        )
        measured = _measure_row(part)
        analytic_total = measured["analytic_total"]
        after_sum = float(np.sum(measured["booked"]))
        after_ratio = after_sum / analytic_total
        before_ratio = 1.0 / _terminal_amplitude(part)

        # A cell is cut when the exact analytic flux changes sign across its
        # vertices (the analytic separatrix enters and leaves it).
        cut = (measured["vertex_flux_min"] < 0.0) & (measured["vertex_flux_max"] > 0.0)
        participating = cut & (measured["booked"] != 0.0) & measured["included"]
        difference = measured["booked"] - measured["analytic_current"]
        worst = np.argsort(np.abs(difference))[::-1][:6]

        receipt_rows.append(
            {
                "row": label,
                "ratio_before": before_ratio,
                "ratio_after": after_ratio,
                "sum_current_a": after_sum,
                "analytic_total_a": analytic_total,
                "amplitude_terminal": _terminal_amplitude(part),
                "cut_cell_count": int(np.sum(cut)),
                "cut_cells_participating": int(np.sum(participating)),
                "worst_cells": [
                    {
                        "cell": int(index),
                        "booked_a": float(measured["booked"][index]),
                        "analytic_a": float(measured["analytic_current"][index]),
                        "difference_a": float(difference[index]),
                    }
                    for index in worst
                ],
            }
        )
        _render_panel(
            measured,
            cut,
            difference,
            part,
            label,
            output / f"{label}-booked-minus-analytic.png",
        )

    receipt = {
        "gate": "profile-support-amplitude",
        "cells": -110,
        "measured_on": "committed terminal states",
        "rows": receipt_rows,
        "acceptance": "ratio within one percent of unity; every analytic "
        "cut cell participating",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    (output / "receipt.md").write_text(_receipt_markdown(receipt))
    return 0


def _render_panel(measured, cut, difference, part, label, target: Path):
    """Draw one line-contour panel of booked-minus-analytic cell current."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from nova.media.ink import poloidal_axes
    from nova.media.poloidal import draw_boundary, draw_nulls, draw_wall

    machine = measured["machine"]
    operator = measured["operator"]
    atomic = operator.moment_geometry.atomic_mesh
    node = np.asarray(atomic.node_coordinates)
    cells = np.asarray(atomic.cell_nodes)
    counts = np.asarray(atomic.cell_vertex_count)
    wall = np.asarray(machine.wall_node)
    node_flux = np.asarray(measured["case"].flux(node[:, 0], node[:, 1]))

    figure, axes = plt.subplots(figsize=(9, 9))
    poloidal_axes(axes)
    axes.tricontour(
        node[:, 0],
        node[:, 1],
        node_flux,
        levels=np.linspace(float(np.min(node_flux)), float(np.max(node_flux)), 12),
        colors="#16537e",
        linewidths=0.6,
        zorder=2,
    )
    centroid = np.asarray(atomic.centroids)
    diff = np.asarray(difference)
    diff_levels = (
        np.linspace(float(np.min(diff)), float(np.max(diff)), 8)
        if np.ptp(diff) > 0.0
        else np.asarray([0.0])
    )
    axes.tricontour(
        centroid[:, 0],
        centroid[:, 1],
        diff,
        levels=diff_levels,
        colors="#b3541e",
        linewidths=1.1,
        zorder=3,
    )
    polygons = [node[cells[index][: counts[index]]] for index in np.flatnonzero(cut)]
    if polygons:
        axes.add_collection(
            PolyCollection(
                polygons,
                facecolors="none",
                edgecolors="#c9a227",
                linewidths=1.4,
                zorder=4,
            )
        )
    radius, half_height, _weight, _offset = measured["case"]._surface_nodes(0.0, 240)
    boundary = np.vstack(
        (np.c_[radius, half_height], np.c_[radius[::-1], -half_height[::-1]])
    )
    draw_boundary(axes, boundary[:, 0], boundary[:, 1])
    draw_wall(axes, wall[:, 0], wall[:, 1])
    draw_nulls(
        axes,
        magnetic_axis=np.asarray(part["render_data"]["analytic_topology"]["axis_rz_m"]),
        x_points=np.asarray(part["render_data"]["analytic_topology"]["x_point_rz_m"]),
    )
    axes.text(
        0.99,
        0.01,
        (
            f"{label}: booked-minus-analytic per-cell current [A]\n"
            f"flux levels {levels_str(node_flux)} Wb; "
            f"current levels {levels_str(diff)} A"
        ),
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    figure.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(figure)


def levels_str(values: np.ndarray) -> str:
    return f"{float(np.min(values)):.2e}..{float(np.max(values)):.2e}"


def _receipt_markdown(receipt) -> str:
    lines = [
        "# Profile-support amplitude gate",
        "",
        "Measured on the committed terminal states of the "
        f"{receipt['cells']}-cell Solov'ev certificate rows; ratios are "
        "the unit-amplitude current total over the analytic total of the "
        "exact density.",
        "",
        "| row | before | after | analytic cut cells | participating |",
        "|---|---|---|---|---|",
    ]
    for row in receipt["rows"]:
        lines.append(
            f"| {row['row']} | {row['ratio_before']:.6f} | "
            f"{row['ratio_after']:.6f} | {row['cut_cell_count']} | "
            f"{row['cut_cells_participating']} |"
        )
    for row in receipt["rows"]:
        lines.extend(
            [
                "",
                f"## {row['row']}",
                "",
                "- before ratio (whole-cell, banked amplitude): "
                f"{row['ratio_before']:.6f}",
                f"- after ratio (signed-flux clip): {row['ratio_after']:.6f}",
                f"- terminal amplitude: {row['amplitude_terminal']:.6f}",
                f"- analytic cut cells: {row['cut_cell_count']}, "
                f"participating: {row['cut_cells_participating']}",
                "",
                "Six worst per-cell booked-minus-analytic:",
                "",
                "| cell | booked [A] | analytic [A] | difference [A] |",
                "|---|---|---|---|",
            ]
        )
        for entry in row["worst_cells"]:
            lines.append(
                f"| {entry['cell']} | {entry['booked_a']:.1f} | "
                f"{entry['analytic_a']:.1f} | {entry['difference_a']:.1f} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
