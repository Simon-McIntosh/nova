"""Regenerate the diverted-oracle census panel and the receipt it is read from.

The panel is drawn through the committed ``nova.media`` painters so the null
vocabulary is the one the repository's rule fixes: a solid triangle for the
read magnetic axis, a filled cross for an admitted saddle, hollow markers for
the other qualified nulls. The wall is the fixture's own limiter contour, the
same coordinate set that enters the census computation, drawn closed by
``draw_wall``. The exact diverted flux is drawn as unfilled line contours on
one stated level array.

The receipt beside the PNG carries every number the caption quotes, each under
a named JSON path, so a reader recovers them without the picture.

Run on a compute allocation with the shared environment interpreter and
``PYTHONPATH`` pointed at the worktree; the analytic fixture needs no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp

from nova.media.layout import poloidal_view
from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall

CASE = "diverted-jump-bearing"
RUNG_CELLS = -300
STATED_LEVELS = 25


def _census_route(requested_cells, oracle_fixture, certificate, exact, carrier, source):
    """Return the fine-rung mesh, read state, and topology census table."""
    machine = oracle_fixture.cached_machine(
        carrier, requested_cells, wall_nodes=oracle_fixture.WALL_POINT_COUNT
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    exact_state = certificate._exact_state(CASE, exact, coordinates)
    empty = oracle_fixture.forward_operator(source, machine)
    physical = oracle_fixture.exact_current_moments(
        source, empty, exact_state, analytic=exact
    )
    coefficients = empty.coupling_current_moments(physical)
    interior = oracle_fixture._internal_flux_image(empty, coefficients)
    operator = oracle_fixture.forward_operator(source, machine, exact_state - interior)
    _masks, read = operator.read(jnp.asarray(exact_state))
    flux_pool = operator.null_flux_pool(jnp.asarray(exact_state))
    census = operator._fixed_design_topology.grid.candidate_table_status(flux_pool)
    return machine, read, census


def _finite_list(values, digits=6):
    array = np.asarray(values, dtype=np.float64).reshape(-1)[:2]
    if not np.all(np.isfinite(array)):
        return [None, None]
    return [round(float(v), digits) for v in array]


def main(out_dir: str) -> int:
    from benchmarks import solovev_certificate as certificate
    from benchmarks.solovev_certificate import AXIS_M, X_POINT_M
    from nova.jax.config import configure_dtypes
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture

    configure_dtypes()

    carrier, source, exact = certificate._case(CASE)
    machine, read, census = _census_route(
        RUNG_CELLS, oracle_fixture, certificate, exact, carrier, source
    )

    wall = np.asarray(machine.wall_node, dtype=np.float64)
    grid = np.asarray(machine.node, dtype=np.float64)
    extent = (
        float(min(wall[:, 0].min(), grid[:, 0].min())),
        float(max(wall[:, 0].max(), grid[:, 0].max())),
        float(min(wall[:, 1].min(), grid[:, 1].min())),
        float(max(wall[:, 1].max(), grid[:, 1].max())),
    )
    radius = np.linspace(extent[0], extent[1], 321)
    height = np.linspace(extent[2], extent[3], 321)
    mesh_r, mesh_z = np.meshgrid(radius, height)
    points = np.column_stack((mesh_r.ravel(), mesh_z.ravel()))
    psi = np.asarray(exact.flux(points), dtype=np.float64).reshape(mesh_r.shape)
    levels = np.unique(np.append(np.linspace(psi.min(), psi.max(), STATED_LEVELS), 0.0))

    read_axis = np.asarray(read.axis, dtype=np.float64).reshape(-1)[:2]
    read_saddle = np.asarray(read.x_point, dtype=np.float64).reshape(-1)[:2]
    reference = np.vstack(
        (
            np.asarray(AXIS_M, dtype=np.float64).reshape(-1)[:2],
            np.asarray(X_POINT_M, dtype=np.float64).reshape(-1)[:2],
        )
    )

    view = poloidal_view(extent)
    draw_flux_contours(view.poloidal, radius, height, psi, levels)
    draw_wall(view.poloidal, wall[:, 0], wall[:, 1])
    tally = draw_nulls(
        view.poloidal,
        magnetic_axis=read_axis,
        x_points=read_saddle[None, :],
        other_x_points=reference,
        contain=wall,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png_path = out / "diverted-oracle-census-fine.png"
    view.figure.savefig(png_path, dpi=160)

    crossing = np.asarray(census["ring_crossing_count"])
    receipt = {
        "case": CASE,
        "rung": {
            "requested_cells": int(RUNG_CELLS),
            "realised_cells": int(len(machine.node)),
            "pitch_m": round(float(np.sqrt(np.median(np.asarray(machine.area)))), 6),
        },
        "flux": {
            "psi_wb": {
                "min": round(float(psi.min()), 6),
                "max": round(float(psi.max()), 6),
            },
            "levels": {
                "count": int(levels.size),
                "stated": int(STATED_LEVELS),
                "zero_added": bool(0.0 in levels),
                "levels_wb": [round(float(v), 6) for v in levels.tolist()],
            },
        },
        "axis": {"read_rz_m": _finite_list(read_axis)},
        "x_point": {"read_rz_m": _finite_list(read_saddle)},
        "reference_null": {
            "axis_rz_m": _finite_list(reference[0]),
            "x_point_rz_m": _finite_list(reference[1]),
        },
        "diverted": bool(read.diverted),
        "wall": {
            "nodes": int(len(wall)),
            "source": "carrier limiter contour (oracle_fixture.limiter_contour)",
            "units": 1,
            "closed": True,
        },
        "null_vocabulary": {
            "axis_marker": "solid triangle",
            "saddle_marker": "filled cross",
            "other_marker": "hollow cross",
            "axis_drawn": bool(np.all(np.isfinite(read_axis))),
            "saddle_drawn": bool(np.all(np.isfinite(read_saddle))),
            "other_drawn": int(tally["other_x_points_drawn"]),
            "x_points_drawn": int(tally["x_points_drawn"]),
            "x_points_dropped_outside_wall": int(
                tally["x_points_dropped_outside_wall"]
            ),
        },
        "topology": {
            "raw_ring_count_o_x": np.asarray(census["raw_ring_count"]).tolist(),
            "candidate_count_o_x": np.asarray(census["candidate_count"]).tolist(),
            "capacity_o_x": np.asarray(census["capacity"]).tolist(),
            "overflow": int(np.asarray(census["overflow"]).sum()),
            "value_histogram": {
                str(value): int(np.count_nonzero(crossing == value))
                for value in np.unique(crossing)
            },
        },
        "figure": {
            "path": png_path.name,
            "painters": [
                "nova.media.poloidal.draw_flux_contours",
                "nova.media.poloidal.draw_wall",
                "nova.media.poloidal.draw_nulls",
            ],
        },
    }
    (out / "diverted-oracle-census.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        "CENSUS_RECEIPT "
        + json.dumps({"path": str(out / "diverted-oracle-census.json")})
    )
    print(
        "FIGURE "
        + json.dumps(
            {
                "path": str(png_path),
                "level_count": int(levels.size),
                "psi_min": float(psi.min()),
                "psi_max": float(psi.max()),
                "read_axis_rz_m": read_axis.tolist(),
                "read_saddle_rz_m": read_saddle.tolist(),
                "reference_nulls_rz_m": reference.tolist(),
                "tally": tally,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
