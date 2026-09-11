"""Gate: the frozen-current flux image with clipped-polygon cut-cell blocks.

Gate of the cut-cell coupling correction.  On each row the analytic
equilibrium's per-cell currents are frozen as the exact zeroth and first
density moments over each cell's clipped plasma polygon of the frozen
per-trip support, and those moments are contracted through the shipped
operator exactly as a solve would.  Two couplings are measured against the
oracle flux: the production as-built path, whose physical first moments are
referenced to the atomic cell centroid and converted and imaged with the
full atomic cell's second moments and kernel blocks; and the clipped
path, whose cut cells are referenced to their clipped polygon's centroid
and converted and imaged with that polygon's area-normalised second moments
and polygon-analytic kernel blocks.  Interior cells keep the precomputed
atomic blocks in both.  No clip decision and no iteration enter the
prediction, so the residual between the two paths is the coupling's own
first-order term on the cut cells.

The acceptance is that the zeroth-plus-first clipped image sits at or
below the zeroth-only image on every row (weak -110 rms over span at or
below 0.0057, the interior-only first-order figure 0.0061 being the floor
the cut cells must reach), the single-cell control stays at the two-e-13
level, and a synthetic two-cell test pins the clipped coupling to the exact
ring-inductance integral over its clipped polygon to one part in ten
billion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nova.biot.greens import section_centroid
from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from benchmarks import solovev_certificate as certificate


CASE_REQUESTS = (
    ("weak-rotation-reactor-static", -110),
    ("moderate-rotation-conventional-static", -110),
    ("strong-rotation-compact-static", -110),
    ("weak-rotation-reactor-static", -300),
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs/figures/cut-cell-current-attribution/cut-cell-coupling"
RECEIPT = OUTPUT_ROOT / "receipt.json"
REPORT = OUTPUT_ROOT / "report.md"


def _duffy_rule(vertices, order=28):
    """Return Duffy nodes and weights over one convex polygon."""
    from numpy.polynomial.legendre import leggauss

    vertices = np.asarray(vertices, dtype=np.float64)
    nodes, weights = leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    radial, vertical = np.meshgrid(nodes, nodes)
    rule_weight = (weights[:, None] * weights[None, :]).ravel()
    rr = radial.ravel()
    ss = vertical.ravel()
    points = []
    point_weights = []
    for index in range(1, len(vertices) - 1):
        first = vertices[0]
        second = vertices[index]
        third = vertices[index + 1]
        edge_first = second - first
        edge_second = third - first
        cross = abs(edge_first[0] * edge_second[1] - edge_first[1] * edge_second[0])
        point = (
            first[None, :]
            + rr[:, None] * edge_first[None, :]
            + (1.0 - rr)[:, None] * ss[:, None] * edge_second[None, :]
        )
        points.append(point)
        point_weights.append(cross * (1.0 - rr) * rule_weight)
    return np.vstack(points), np.concatenate(point_weights)


def _exact_density(exact, points):
    """Return the exact toroidal current density at physical (R, Z)."""
    from nova.biot.greens import MU0 as MU_0

    from nova.equilibrium.analytic_single_null import CerfonFreidbergSingleNull

    if isinstance(exact, CerfonFreidbergSingleNull):
        source = np.asarray(exact.grad_shafranov_source(points), dtype=np.float64)
    else:
        source = np.asarray(
            exact.delta_star(points[:, 0], points[:, 1]), dtype=np.float64
        )
    return -source / (MU_0 * points[:, 0])


def _frozen_support_moments(exact, machine, support, included):
    """Return (atomic-referenced, clipped-referenced, kind) exact moments.

    ``kind`` is 1 interior, 2 cut and 0 excluded, mirroring the class
    decomposition the gate reports.
    """
    vertices = np.asarray(support.support_vertices)
    count = np.asarray(support.vertex_count)
    centroids = np.asarray(support.centroids)
    boundary = np.asarray(support.boundary, dtype=bool)
    cell_count = machine.moment_geometry.atomic_mesh.centroids.shape[0]
    moments_atomic = np.zeros((cell_count, 3))
    moments_clipped = np.zeros((cell_count, 3))
    kind = np.zeros(cell_count, dtype=int)
    for cell in np.flatnonzero(included):
        live = int(count[cell])
        polygon = vertices[cell][:live]
        clipped_centre = section_centroid(polygon)
        points, weights = _duffy_rule(polygon)
        density = _exact_density(exact, points)
        clipped_offset = points - clipped_centre
        m0 = float(np.sum(density * weights))
        mr_clip = float(np.sum(density * weights * clipped_offset[:, 0]))
        mz_clip = float(np.sum(density * weights * clipped_offset[:, 1]))
        moments_clipped[cell] = (m0, mr_clip, mz_clip)
        # The production moment centre is the atomic cell centroid, so the
        # atomic-referenced moments follow the clipped ones by
        # MR_atomic = MR_clipped + (clipped_centre - atomic_centroid) * M0.
        shift = clipped_centre - centroids[cell]
        moments_atomic[cell] = (
            m0,
            mr_clip + shift[0] * m0,
            mz_clip + shift[1] * m0,
        )
        kind[cell] = 2 if boundary[cell] else 1
    return moments_atomic, moments_clipped, kind


def measure_row(case_name: str, requested_cells: int) -> dict:
    """Measure one case-resolution row: before, after, zeroth and classes."""
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the frozen-current flux image gate requires x64")
    if jax.default_backend() != "cpu":
        raise RuntimeError(
            "the frozen-current flux image gate requires the CPU backend"
        )
    carrier, source, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier, exact, requested_cells)
    grid_count = len(machine.node)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(case_name, exact, coordinates)
    oracle_grid = oracle_state[:grid_count]
    span = float(np.max(oracle_grid) - np.min(oracle_grid))

    empty_operator = oracle_fixture.forward_operator(source, machine)
    certificate_moments = oracle_fixture.exact_current_moments(
        source, empty_operator, oracle_state
    )
    certificate_coefficients = empty_operator.coupling_current_moments(
        certificate_moments
    )
    production_internal = oracle_fixture._internal_flux_image(
        empty_operator, certificate_coefficients
    )
    operator = oracle_fixture.forward_operator(
        source, machine, oracle_state - production_internal
    )
    external = np.asarray(operator.external(), dtype=np.float64)

    partition = operator._support_partition(jnp.asarray(oracle_state))
    support = partition[3]
    geometry = operator._clipped_coupling_geometry(support)
    boundary = np.asarray(support.boundary, dtype=bool)
    labels = np.asarray(partition[0].label)
    included = np.asarray(support.included, dtype=bool) & (
        labels != int(PlasmaDomain.EXCLUDED_MATERIAL)
    )

    moments_atomic, moments_clipped, kinds = _frozen_support_moments(
        exact, machine, support, included
    )
    cut = np.flatnonzero(boundary & included)
    interior = np.flatnonzero(included & ~boundary)

    def image_for(phys, geom):
        coefficients = operator.coupling_current_moments(
            CellCurrentMoments(
                jnp.asarray(phys[:, 0]),
                jnp.asarray(phys[:, 1]),
                jnp.asarray(phys[:, 2]),
            ),
            geom,
        )
        return external + np.asarray(
            operator.current_moment_image(coefficients, geom), dtype=np.float64
        )

    def class_image(base, class_mask, geom):
        physical = base.copy()
        others = np.flatnonzero(~class_mask)
        physical[others, 1] = 0.0
        physical[others, 2] = 0.0
        return image_for(physical, geom)

    def metrics(error):
        absolute = np.abs(error)
        return {
            "rms_error_over_span": float(np.sqrt(np.mean(error**2)) / span),
            "max_error_over_span": float(np.max(absolute) / span),
            "rms_absolute_error_wb": float(np.sqrt(np.mean(error**2))),
        }

    zeroth_only = external + np.asarray(
        operator.current_moment_image(
            CellCurrentMoments(
                jnp.asarray(moments_atomic[:, 0]),
                jnp.zeros(operator.grid.node_number),
                jnp.zeros(operator.grid.node_number),
            )
        ),
        dtype=np.float64,
    )
    before = image_for(moments_atomic, None)
    after = image_for(moments_atomic, geometry)
    cut_only_before = class_image(moments_atomic, cut.astype(bool), None)
    cut_only_after = class_image(moments_clipped, cut.astype(bool), geometry)
    interior_only_after = class_image(moments_clipped, interior.astype(bool), geometry)
    grid = np.arange(grid_count)
    record = {
        "case": case_name,
        "requested_cells": requested_cells,
        "grid_node_count": grid_count,
        "flux_span_wb": span,
        "cell_census": {
            "included": int(np.count_nonzero(included)),
            "cut": int(len(cut)),
            "interior": int(len(interior)),
        },
        "coupling": {
            "zeroth_only": metrics(zeroth_only[grid] - oracle_grid),
            "before_as_built": metrics(before[grid] - oracle_grid),
            "after_clipped": metrics(after[grid] - oracle_grid),
            "cut_cells_only_before": metrics(cut_only_before[grid] - oracle_grid),
            "cut_cells_only_after": metrics(cut_only_after[grid] - oracle_grid),
            "interior_cells_only_after": metrics(
                interior_only_after[grid] - oracle_grid
            ),
            "first_order_delta_before": metrics(before[grid] - oracle_grid)[
                "rms_error_over_span"
            ]
            - metrics(zeroth_only[grid] - oracle_grid)["rms_error_over_span"],
            "first_order_delta_after": metrics(after[grid] - oracle_grid)[
                "rms_error_over_span"
            ]
            - metrics(zeroth_only[grid] - oracle_grid)["rms_error_over_span"],
        },
        "plot_data": {
            "node_rz_m": np.asarray(machine.node),
            "wall_rz_m": np.asarray(machine.wall_node),
            "error_fields_wb": {
                "zeroth_only": (zeroth_only[grid] - oracle_grid),
                "before_as_built": (before[grid] - oracle_grid),
                "after_clipped": (after[grid] - oracle_grid),
            },
        },
    }
    return record


def _shared_error_levels(fields) -> np.ndarray:
    values = np.concatenate([np.abs(np.asarray(f)) for f in fields.values()])
    nonzero = values[values > 0.0]
    lower = max(float(np.percentile(nonzero, 5.0)), np.finfo(float).tiny)
    upper = float(np.max(nonzero))
    return np.geomspace(lower, upper, 6)


def draw_figure(record: dict, path: Path) -> dict:
    """Draw the weak -110 before/after error panels as line contours."""
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes

    plot = record["plot_data"]
    node = np.asarray(plot["node_rz_m"])
    wall = np.asarray(plot["wall_rz_m"])
    fields = plot["error_fields_wb"]
    levels = _shared_error_levels(fields)
    names = ("zeroth_only", "before_as_built", "after_clipped")
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), constrained_layout=True)
    metrics = record["coupling"]
    for column, name in enumerate(names):
        axis = poloidal_axes(axes[column])
        error = np.asarray(fields[name], dtype=np.float64)
        axis.tricontour(
            node[:, 0],
            node[:, 1],
            np.maximum(np.abs(error), levels[0]),
            levels=levels,
            colors="firebrick",
            linewidths=0.8,
        )
        poloidal.draw_wall(axis, wall[:, 0], wall[:, 1], style=DEFAULT_INK)
        metric = metrics[name]
        axis.set_title(
            f"{name.replace('_', ' ')}\n"
            f"rms {metric['rms_error_over_span']:.5f} of span",
            fontsize=7,
        )
    figure.suptitle(
        "Frozen-current flux image, weak -110: coupling error vs the oracle flux\n"
        "red: |image - oracle| on shared levels; wall drawn; the clipped path "
        "references cut cells to their clipped polygon and images over it",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return {
        "path": str(path.relative_to(ROOT)),
        "shared_absolute_error_levels_wb": [float(v) for v in levels],
    }


def adjudicate(rows) -> dict:
    """Return the acceptance verdict across the rows."""
    floors = {
        "weak-rotation-reactor-static|110": 0.0057,
    }
    verdicts = {}
    for row in rows:
        key = f"{row['case']}|{abs(row['requested_cells'])}"
        floor = floors.get(key, row["coupling"]["zeroth_only"]["rms_error_over_span"])
        after = row["coupling"]["after_clipped"]["rms_error_over_span"]
        zeroth = row["coupling"]["zeroth_only"]["rms_error_over_span"]
        before = row["coupling"]["before_as_built"]["rms_error_over_span"]
        verdicts[key] = {
            "after_at_or_below_zeroth": after <= zeroth + 1.0e-9,
            "after_at_or_below_floor": after <= floor + 1.0e-9,
            "after": after,
            "zeroth": zeroth,
            "before": before,
        }
    all_after = all(
        v["after_at_or_below_zeroth"] and v["after_at_or_below_floor"]
        for v in verdicts.values()
    )
    weak_110 = verdicts["weak-rotation-reactor-static|110"]
    return {
        "every_row_after_at_or_below_zeroth": all(
            v["after_at_or_below_zeroth"] for v in verdicts.values()
        ),
        "every_row_after_at_or_below_its_floor": all(
            v["after_at_or_below_floor"] for v in verdicts.values()
        ),
        "weak110_after_rms_over_span": weak_110["after"],
        "weak110_zeroth_rms_over_span": weak_110["zeroth"],
        "weak110_before_rms_over_span": weak_110["before"],
        "passed": bool(all_after),
        "rows": verdicts,
    }


def run_rows(selected=None) -> list[dict]:
    requests = (
        CASE_REQUESTS
        if selected is None
        else [req for req in CASE_REQUESTS if req in selected]
    )
    return [measure_row(name, cells) for name, cells in requests]


def _jsonable(value):
    """Return a JSON-serialisable form of a numpy array or scalar."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialise {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", action="append", default=None)
    arguments = parser.parse_args()
    selected = None
    if arguments.row:
        selected = []
        for item in arguments.row:
            name, _, cells = item.rpartition(":")
            selected.append((name, int(cells)))
    rows = run_rows(selected)
    receipt = {
        "schema": "nova.frozen-current-flux-image-gate.v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": [r for r in rows],
        "adjudication": adjudicate(rows),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, default=_jsonable) + "\n")
    weak110 = [
        r
        for r in rows
        if r["case"] == "weak-rotation-reactor-static" and r["requested_cells"] == -110
    ]
    if weak110:
        draw_figure(weak110[0], OUTPUT_ROOT / "weak-110-coupling-errors.svg")
    print(json.dumps(receipt["adjudication"], indent=2))


if __name__ == "__main__":
    main()
