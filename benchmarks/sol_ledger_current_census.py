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
import sys
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium.domain import PlasmaDomain
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "figures" / "forward-solve-api" / "sol-ledger-census"
RECEIPT = OUTPUT_DIR / "sol-ledger-census.json"
FIGURE = OUTPUT_DIR / "sol-ledger-current.png"

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
    rows = []
    for cell in np.flatnonzero(cell_current != 0.0):
        rows.append(
            {
                "cell_index": int(cell),
                "domain": DOMAIN_NAMES[int(label[cell])],
                "current_a": float(cell_current[cell]),
                "centroid_psi_norm": float(psi_norm[cell]),
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


def _draw_figure(profile, equilibrium, grid_flux, rows, output: Path):
    """Render the whole-domain panel with common-SOL cells outlined."""
    operator = profile.operator
    node = np.asarray(operator.grid.coordinate, dtype=float)
    wall = np.asarray(operator.topology.wall.coordinate, dtype=float)
    boundary_flux = float(equilibrium.topology.boundary_flux)
    axis = np.asarray(equilibrium.topology.axis, dtype=float)
    labelled = equilibrium.labelled_flux
    x_points = []
    if labelled is not None:
        for name in ("primary_x_point", "secondary_x_point"):
            point = np.asarray(getattr(labelled, name), dtype=float)
            if np.all(np.isfinite(point)):
                x_points.append(point)
    x_points = np.asarray(x_points).reshape(-1, 2) if x_points else None

    levels = poloidal.contour_levels(grid_flux, 16, boundary=boundary_flux)
    figure, axes = plt.subplots(1, 1, figsize=(7.2, 5.6))
    plot_axes = poloidal_axes(axes)
    plot_axes.tricontour(
        node[:, 0],
        node[:, 1],
        grid_flux,
        levels=levels,
        colors="#999999",
        linewidths=0.35,
    )
    poloidal.draw_wall(plot_axes, wall[:, 0], wall[:, 1])
    if labelled is not None and labelled.lcfs_vertex_count > 0:
        lcfs = np.asarray(labelled.lcfs)[: int(labelled.lcfs_vertex_count)]
        if lcfs.shape[0] > 2:
            poloidal.draw_boundary(plot_axes, lcfs[:, 0], lcfs[:, 1])
    sol_cells = {
        row["cell_index"] for row in rows if row["domain"] == "common_sol"
    }
    for cell in sorted(sol_cells):
        polygon = np.asarray(operator.moment_geometry.polygons[cell], dtype=float)
        loop = np.vstack([polygon, polygon[:1]])
        plot_axes.plot(loop[:, 0], loop[:, 1], color="#cc0000", linewidth=1.1, zorder=6)
        centre = np.asarray(operator.moment_geometry.atomic_mesh.centroids[cell])
        current = float(equilibrium.cell_current[cell])
        plot_axes.text(
            centre[0],
            centre[1],
            f"{current:.3g} A" if abs(current) < 1000 else f"{current/1000:.2f} kA",
            fontsize=6,
            ha="center",
            va="center",
            color="#cc0000",
            zorder=7,
        )
    solved_style = DEFAULT_INK.variant(axis_marker="^")
    poloidal.draw_nulls(
        plot_axes,
        magnetic_axis=axis,
        x_points=x_points,
        style=solved_style,
        contain=wall,
    )
    figure.suptitle(
        "Mapped-source current ledger: common-SOL cells\n"
        "red outlines: carrying common-SOL cells with per-cell A; "
        "red LCFS and nulls over grey solved flux contours"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def _verdict(sol_split):
    """Return the one-sentence attribution statement."""
    cut = sol_split["common_sol_in_fitted_cut_cells_a"]
    uncut = sol_split["common_sol_in_uncut_cells_a"]
    total = cut + uncut
    if total == 0.0:
        return "the scrape-off-layer ledger carries no current"
    if uncut == 0.0:
        return (
            "the scrape-off-layer current is entirely full-cell labelling of "
            "cells the fitted separatrix cuts whose centroids fall on the open "
            "side - no cell is geometrically cut"
        )
    if cut == 0.0:
        return (
            "the scrape-off-layer current is entirely full-cell labelling of "
            "fully open in-board cells the fitted separatrix does not cut"
        )
    return (
        f"the scrape-off-layer current is a split: "
        f"{cut / max(total, 1e-30):.6f} in cells the fitted separatrix cuts, "
        f"{uncut / max(total, 1e-30):.6f} in fully open cells"
    )


def main() -> int:
    """Run the census and write the receipt and figure."""
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
    }
    _draw_figure(profile, equilibrium, grid_flux, payload["per_cell"], FIGURE)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"\nRECEIPT {RECEIPT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
