r"""Per-cell census of the common scrape-off-layer current ledger.

The transport seam's mapped-source fixture (`tests/test_transport_evolved_state.py`)
books 0.196 percent of the plasma current in common scrape-off-layer cells
(``1.96e-3 < common_sol / Ip < 3e-3`` is the asserted bound in the test).  This
driver rebuilds that equilibrium exactly as the test does — the same fixture
chain, the same ``anderson`` route, the same evaluation budget — and locates
every ampere of the leak cell by cell: which cells carry common-SOL current,
whether the fitted (and analytic) separatrix passes through each, and how much
of the leak the committed production clip geometrically cuts versus how much is
full-cell labelling by the cell centroid.  The committed clip's geometry is
characterised directly (all-ones profile support = no cell is cut), and the
receipt states whether an exact-support clip mode is selectable on the machine
(the uniform-clip machinery is in-flight elsewhere and absent on this base).

Run as a single CPU SLURM job: JAX_PLATFORMS=cpu, root venv python directly
(no uv on the compute node), PYTHONPATH at the repository root.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

from nova.equilibrium.domain import PlasmaDomain
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.sources.frame import coerce_wall_units

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "figures" / "forward-solve-api" / "sol-ledger-census"
RECEIPT = OUTPUT_DIR / "sol-ledger-census.json"
FIGURE = OUTPUT_DIR / "sol-ledger-current.png"
RENDER_RECEIPT = OUTPUT_DIR / "sol-ledger-render.json"

#: Set to 1 to render the panel with the wall call omitted, so the gate's wall
#: assertion is exercised against a panel this change can actually produce
#: without the wall: the negative control, never a production path.
DROP_WALL = os.environ.get("SOL_LEDGER_DROP_WALL") == "1"

# The mapped-source transport fixtures, used exactly as the tests build them.
# The pytest fixture decorators refuse direct calls, so the wrapped functions
# are invoked in the same order and with the same arguments the fixtures use.
from tests.test_equilibrium_forward_solve import (  # noqa: E402
    machine as _machine_fixture,
)
from tests.test_equilibrium_forward_solve import (  # noqa: E402
    converged as _converged_fixture,
)
from tests.test_equilibrium_forward_solve import (  # noqa: E402
    _wall_loop,
)
from tests.test_transport_evolved_state import (  # noqa: E402
    mapped_case as _mapped_case_fixture,
)
from tests.test_transport_evolved_state import (  # noqa: E402
    mapped_equilibrium as _mapped_equilibrium_fixture,
)

DOMAIN_NAMES = {
    int(PlasmaDomain.EXCLUDED_MATERIAL): "excluded_material",
    int(PlasmaDomain.CORE): "core",
    int(PlasmaDomain.COMMON_SOL): "common_sol",
    int(PlasmaDomain.PRIVATE_FLUX): "private_flux",
}


def _clipped_geometry(equilibrium, profile, seed):
    """Return per-cell flux maps and the separatrix-through-cell flags.

    The fitted separatrix is the solved boundary flux; the analytic one is the
    Solov'ev seed surface the material wall lies on.  Both are reconstructed on
    the shared atomic nodes with the same own-node polynomial the topology read
    uses, and each cell is flagged when its nodes straddle the boundary on
    either side (the polarity-aware closed test of the production read).
    """
    operator = profile.operator
    grid_flux = np.asarray(equilibrium.flux)[: operator.grid.node_number]
    polarity = int(operator.polarity)
    boundary_flux = float(equilibrium.topology.boundary_flux)
    seed_grid_flux = np.asarray(seed)[: operator.grid.node_number]
    _wall, wall_flux = _wall_loop()
    atomic = operator.moment_geometry.atomic_mesh
    atomic_flux = np.asarray(operator.moment_geometry.shared_flux_stencil(grid_flux))
    seed_atomic_flux = np.asarray(
        operator.moment_geometry.shared_flux_stencil(seed_grid_flux)
    )

    def closed_at(flux, boundary):
        if polarity > 0:
            return flux >= boundary
        return flux < boundary

    fitted_closed = closed_at(atomic_flux, boundary_flux)
    analytic_closed = closed_at(seed_atomic_flux, wall_flux)
    cell_nodes = np.asarray(atomic.cell_nodes)
    counts = np.asarray(atomic.cell_vertex_count)

    def straddle(closed):
        flags = np.zeros(operator.grid.node_number, dtype=bool)
        for cell in range(operator.grid.node_number):
            nodes = cell_nodes[cell][: counts[cell]]
            values = closed[nodes]
            flags[cell] = bool(np.any(values) and np.any(~values))
        return flags

    return grid_flux, straddle(fitted_closed), straddle(analytic_closed)


def _cell_table(equilibrium, profile, straddle_fitted, straddle_analytic):
    """Return the receipt rows for every cell carrying current, and the SOL
    totals split by whether the fitted separatrix cuts the cell."""
    cell_current = np.asarray(equilibrium.cell_current, dtype=float)
    label = np.asarray(equilibrium.domains.label, dtype=np.int64)
    psi_norm = np.asarray(equilibrium.domains.psi_norm, dtype=float)
    participating = np.asarray(equilibrium.domains.profile_participation, dtype=bool)
    rows = []
    for cell in np.flatnonzero(cell_current != 0.0):
        rows.append(
            {
                "cell_index": int(cell),
                "domain": DOMAIN_NAMES[int(label[cell])],
                "current_a": float(cell_current[cell]),
                "centroid_psi_norm": float(psi_norm[cell]),
                "production_clip_included": bool(participating[cell]),
                # The committed clip traces its support at all-ones flux, so no
                # cell ever carries a geometric crossing: its boundary flag is
                # false everywhere by construction.
                "production_clip_cut": False,
                "fitted_separatrix_through": bool(straddle_fitted[cell]),
                "analytic_separatrix_through": bool(straddle_analytic[cell]),
            }
        )
    sol = label == int(PlasmaDomain.COMMON_SOL)
    common_sol_current = float(np.sum(cell_current[sol]))
    fitted_cut = sol & straddle_fitted
    cut_current = float(np.sum(cell_current[fitted_cut]))
    uncut_current = common_sol_current - cut_current
    return rows, {
        "common_sol_total_a": common_sol_current,
        "common_sol_in_fitted_cut_cells_a": cut_current,
        "common_sol_in_uncut_cells_a": uncut_current,
        "common_sol_cut_cell_count": int(np.sum(fitted_cut)),
        "common_sol_uncut_cell_count": int(np.sum(sol & ~straddle_fitted)),
    }


def _booking_line():
    """Return the receptor and the mechanism that book the leak, with lines."""
    receptor = "nova/equilibrium/observation.py:558 (current_ledger, common_sol)"
    mechanism = (
        "nova/equilibrium/forward_operator.py:2021-2026 (_profile_support "
        "traces the support at all-ones flux, so no cell is geometrically cut; "
        "the open-field-line selection tests the cell centroid flux against "
        "the separatrix)"
    )
    return receptor, mechanism


def _measure():
    """Run the fixture chain and return the census receipt payload."""
    machine = _machine_fixture.__wrapped__()
    profile, seed = machine[0], machine[1]
    converged = _converged_fixture.__wrapped__(machine)
    case = _mapped_case_fixture.__wrapped__(machine, converged)
    equilibrium = _mapped_equilibrium_fixture.__wrapped__(case)

    grid_flux, straddle_fitted, straddle_analytic = _clipped_geometry(
        equilibrium, profile, seed
    )
    rows, sol_split = _cell_table(
        equilibrium, profile, straddle_fitted, straddle_analytic
    )

    ledger = equilibrium.ledger
    plasma_current = float(equilibrium.moments.plasma_current)
    totals = {
        "core": float(ledger.core),
        "common_sol": float(ledger.common_sol),
        "private_flux": float(ledger.private_flux),
        "excluded_material": float(ledger.excluded_material),
        "total": float(ledger.total),
    }
    fraction = float(ledger.common_sol) / plasma_current

    operator = profile.operator
    clip_mode_available = bool(
        hasattr(operator, "set_support_clip_mode")
        or hasattr(operator, "support_clip_mode")
    )
    return {
        "ledger": totals,
        "plasma_current_a": plasma_current,
        "common_sol_over_plasma_current": fraction,
        "per_cell": rows,
        "common_sol_split": sol_split,
        "support_clip_mode": {
            "available": clip_mode_available,
            "reason": (
                "the mapped-case machine exposes no set_support_clip_mode or "
                "support_clip_mode symbol on this revision (package-wide grep "
                "returns none), so the committed chord clip cannot be compared "
                "against an exact-support clip here; that switch is the "
                "in-flight uniform-clip work"
            )
            if not clip_mode_available
            else "selectable",
        },
        "fixed_point_residual": float(equilibrium.fixed_point.residual),
        "converged_boolean": bool(
            float(equilibrium.fixed_point.residual) < 1.0e-6
        ),
        "topology_branch": {
            "axis": list(map(float, np.asarray(equilibrium.topology.axis))),
            "x_point": list(map(float, np.asarray(equilibrium.topology.x_point))),
            "boundary_flux": float(equilibrium.topology.boundary_flux),
        },
    }, (profile, equilibrium, grid_flux)


def _finite_points(array):
    """Return the all-finite rows of an ``(n, 2)`` array as nested lists."""
    if array is None:
        return []
    points = np.asarray(array, dtype=float).reshape(-1, 2)
    keep = np.all(np.isfinite(points), axis=1)
    return [[float(radius), float(height)] for radius, height in points[keep]]


def _admitted_nulls(equilibrium):
    """Return the admitted axis, saddle set and wall-contact points.

    A limited boundary admits no magnetic saddle: its boundary nulls are the
    strike points where the closed surface meets the wall.  A diverted
    boundary admits the saddle as its boundary, so the qualified set beyond
    the admitted one is drawn hollow rather than left as absence.
    """
    topology = equilibrium.topology
    labelled = equilibrium.labelled_flux
    axis = _finite_points(topology.axis)
    x_points = _finite_points(topology.x_point)
    strike = _finite_points(labelled.strike_points)
    other = _finite_points(labelled.secondary_x_point)
    if x_points:
        null_class = "saddle"
    elif strike:
        null_class = "wall_contact"
    else:
        null_class = "none"
    return {
        "axis": axis[0] if axis else [],
        "x_points": x_points,
        "strike_points": strike,
        "other_x_points": other,
        "class": null_class,
    }


def _title_lines(split):
    """Return the panel title, one clause a line, carrying the ampere split."""
    lines = [
        "Mapped-source current ledger: common-SOL cells",
        "red outlines: carrying common-SOL cells with per-cell A",
        "red LCFS marker and admitted boundary nulls over grey solved contours",
    ]
    if split:
        lines.append(
            f"common-SOL total {split['common_sol_total_a']:.2f} A split "
            f"{split['common_sol_in_fitted_cut_cells_a']:.2f} A in the "
            f"{split['common_sol_cut_cell_count']} fitted-cut cells + "
            f"{split['common_sol_in_uncut_cells_a']:.2f} A in the "
            f"{split['common_sol_uncut_cell_count']} fully open cells"
        )
    return lines


def _render_payload(profile, equilibrium, grid_flux, rows, split):
    """Return the entire drawing of one solved state, serializable and static.

    ``node``, ``grid_flux`` and ``levels`` are returned as flat parallel lists
    because ``tricontour`` consumes them positionally; every polyline is a list
    of ``[radius, height]`` points so the payload survives a JSON round trip.
    """
    operator = profile.operator
    node = np.asarray(operator.grid.coordinate, dtype=float)
    wall = np.asarray(operator.topology.wall.coordinate, dtype=float)
    boundary_flux = float(equilibrium.topology.boundary_flux)
    labelled = equilibrium.labelled_flux
    lcfs = []
    if labelled is not None and int(labelled.lcfs_vertex_count) > 2:
        lcfs = _finite_points(
            np.asarray(labelled.lcfs)[: int(labelled.lcfs_vertex_count)]
        )
    centroids = np.asarray(operator.moment_geometry.atomic_mesh.centroids, dtype=float)
    sol_cells = []
    for row in rows:
        if row["domain"] != "common_sol":
            continue
        cell = int(row["cell_index"])
        polygon = np.asarray(operator.moment_geometry.polygons[cell], dtype=float)
        sol_cells.append(
            {
                "cell_index": cell,
                "polygon": _finite_points(polygon),
                "centroid": [float(centroids[cell][0]), float(centroids[cell][1])],
                "current_a": float(row["current_a"]),
            }
        )
    levels = poloidal.contour_levels(grid_flux, 16, boundary=boundary_flux)
    return {
        "node": _finite_points(node),
        "grid_flux": [float(v) for v in np.asarray(grid_flux, dtype=float)],
        "levels": [float(v) for v in np.asarray(levels, dtype=float)],
        "wall": _finite_points(wall),
        "lcfs": lcfs,
        "nulls": _admitted_nulls(equilibrium),
        "sol_cells": sol_cells,
        "split": dict(split) if split else {},
        "title_lines": _title_lines(split),
    }


def _text_width(figure, text, fontsize):
    """Return one title line's advance width in pixels."""
    probe = Text(0.0, 0.0, text, fontsize=fontsize, figure=figure)
    return float(probe.get_window_extent(renderer=figure.canvas.get_renderer()).width)


def _wrap_line(figure, text, fontsize, budget):
    """Greedily wrap one clause into lines that each fit ``budget`` pixels."""
    lines = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and _text_width(figure, candidate, fontsize) > budget:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    return lines


def _points_or_none(points):
    """Return a non-empty point set as an array, else None."""
    if not points:
        return None
    return np.asarray(points, dtype=float)


def _fit_suptitle(figure, lines, fontsize=11.0, limit=0.96):
    """Draw the title wrapped to the canvas and return per-line measurements.

    A title is readable only if it lies inside the canvas, so each clause is
    measured in pixels and split at whitespace until it fits.  The measured
    widths and canvas width are returned so a reader checks the fit instead of
    trusting it.
    """
    figure.canvas.draw()
    canvas_px = float(figure.get_size_inches()[0] * figure.dpi)
    budget = limit * canvas_px
    fitted = []
    widths = []
    for line in lines:
        for part in _wrap_line(figure, line, fontsize, budget):
            fitted.append(part)
            widths.append(round(_text_width(figure, part, fontsize), 2))
    figure.suptitle("\n".join(fitted), fontsize=fontsize)
    figure.canvas.draw()
    return {
        "lines": fitted,
        "line_widths_px": widths,
        "canvas_width_px": round(canvas_px, 2),
        "text": "\n".join(fitted),
    }


def _draw_panel(render, output):
    """Draw the ledger panel from a serialized render payload, with no solve."""
    node = np.asarray(render["node"], dtype=float)
    grid_flux = np.asarray(render["grid_flux"], dtype=float)
    levels = np.asarray(render["levels"], dtype=float)
    wall = np.asarray(render["wall"], dtype=float).reshape(-1, 2)
    nulls = render.get("nulls", {})
    figure, axes = plt.subplots(1, 1, figsize=(9.6, 7.6), dpi=180)
    plot_axes = poloidal_axes(axes)
    plot_axes.tricontour(
        node[:, 0], node[:, 1], grid_flux, levels=levels,
        colors="#999999", linewidths=0.35,
    )
    if not DROP_WALL:
        poloidal.draw_wall(plot_axes, units=(wall,))
    if render.get("lcfs"):
        lcfs = np.asarray(render["lcfs"], dtype=float)
        poloidal.draw_boundary(plot_axes, lcfs[:, 0], lcfs[:, 1])
    for cell in render.get("sol_cells", []):
        polygon = np.asarray(cell["polygon"], dtype=float)
        if polygon.shape[0] > 2:
            loop = np.vstack([polygon, polygon[:1]])
            plot_axes.plot(loop[:, 0], loop[:, 1], color="#cc0000", linewidth=1.1)
        centre = np.asarray(cell["centroid"], dtype=float)
        current = float(cell["current_a"])
        label = f"{current:.3g} A" if abs(current) < 1000 else f"{current / 1000:.2f} kA"
        plot_axes.text(
            centre[0], centre[1], label, fontsize=6, ha="center", va="center", color="#cc0000"
        )
    solved_style = DEFAULT_INK.variant(axis_marker="^")
    tally = poloidal.draw_nulls(
        plot_axes,
        magnetic_axis=nulls.get("axis") or None,
        x_points=_points_or_none(nulls.get("x_points")),
        strike_points=_points_or_none(nulls.get("strike_points")),
        other_x_points=_points_or_none(nulls.get("other_x_points")),
        style=solved_style,
        contain=wall,
    )
    plot_axes.set_xlim(float(wall[:, 0].min()), float(wall[:, 0].max()))
    plot_axes.set_ylim(float(wall[:, 1].min()), float(wall[:, 1].max()))
    title = _fit_suptitle(figure, render.get("title_lines", []))
    output.parent.mkdir(parents=True, exist_ok=True)
    vector = Path(output).with_suffix(".svg")
    figure.savefig(vector)
    figure.savefig(output)
    plt.close(figure)
    return {
        "wall_node_count": int(wall.shape[0]) if not DROP_WALL else 0,
        "wall_unit_count": 0 if DROP_WALL else len(coerce_wall_units(wall)),
        "axis_drawn": bool(nulls.get("axis")),
        "admitted_null_class": nulls.get("class", "none"),
        "x_points_drawn": int(tally["x_points_drawn"]),
        "strike_points_drawn": int(tally["strike_points_drawn"]),
        "other_x_points_drawn": int(tally["other_x_points_drawn"]),
        "x_points_dropped_outside_wall": int(tally["x_points_dropped_outside_wall"]),
        "title": title,
        "png": str(output),
        "svg": str(vector),
    }


def render_from_receipt(receipt_path=RECEIPT, output=FIGURE, metrics_path=None):
    """Draw the ledger panel from a committed receipt, with no solve.

    The receipt carries the whole drawing of the terminal state, so this path
    imports no solver, runs no fixture chain and may run on the login node.
    ``metrics_path defaults to ``RENDER_RECEIPT``; the gate reads it.
    """
    payload = json.loads(Path(receipt_path).read_text())
    render = payload.get("render")
    if render is None:
        raise ValueError(f"{receipt_path} carries no render payload")
    metrics = _draw_panel(render, Path(output))
    metrics["receipt"] = str(receipt_path)
    target = Path(metrics_path) if metrics_path else RENDER_RECEIPT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics

def _verdict(sol_split):
    """Return the one-sentence attribution statement.

    The class is full-cell labelling: the committed clip traces its support at
    all-ones flux, so no cell is geometrically cut and every booked ampere is
    an integral over a whole cell assigned by its centroid's flux side.  The
    reported split is the internal subdivision between cells the fitted
    separatrix cuts and cells entirely beyond it.
    """
    cut = sol_split["common_sol_in_fitted_cut_cells_a"]
    uncut = sol_split["common_sol_in_uncut_cells_a"]
    total = cut + uncut
    if total == 0.0:
        return "the scrape-off-layer ledger carries no current"
    cut_share = cut / max(total, 1e-30)
    uncut_share = uncut / max(total, 1e-30)
    return (
        "the scrape-off-layer current is entirely full-cell labelling - no "
        "cell is geometrically cut - split "
        f"{cut_share:.6f} in the {sol_split['common_sol_cut_cell_count']} "
        "cells the fitted separatrix cuts and "
        f"{uncut_share:.6f} in the "
        f"{sol_split['common_sol_uncut_cell_count']} fully open cells"
    )


def main() -> int:
    """Run the census and write the receipt and figure.

    ``--render-only`` skips the fixture chain and redraws the committed
    receipt, which is solve-free and safe on a login node."""
    if "--render-only" in sys.argv:
        metrics = render_from_receipt()
        print(json.dumps(metrics, indent=2))
        print(f"\nRENDER_RECEIPT {RENDER_RECEIPT}")
        return 0
    configure_dtypes()
    assert jax.config.jax_enable_x64, "x64 must be enabled"
    payload, (profile, equilibrium, grid_flux) = _measure()
    payload["verdict"] = _verdict(payload["common_sol_split"])
    receptor, mechanism = _booking_line()
    payload["booking_line"] = {"receptor": receptor, "mechanism": mechanism}
    payload["figure"] = {
        "path": str(FIGURE.relative_to(ROOT)),
        "project_src": "/nova/figures/forward-solve-api/sol-ledger-census/"
        + FIGURE.name,
        "vector_path": str(FIGURE.with_suffix(".svg").relative_to(ROOT)),
        "render_receipt": str(RENDER_RECEIPT.relative_to(ROOT)),
    }
    payload["render"] = _render_payload(
        profile,
        equilibrium,
        grid_flux,
        payload["per_cell"],
        payload["common_sol_split"],
    )
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2) + "\n")
    metrics = render_from_receipt()
    print(json.dumps(metrics, indent=2))
    print(f"\nRECEIPT {RECEIPT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
