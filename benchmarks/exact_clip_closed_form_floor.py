"""Evaluate the closed-form polygon moments against the locked per-row budget.

The route under test integrates a local quadratic density over the clipped
polygon's own polygon with Green's theorem, per straight edge, with no
quadrature node. Its only approximation is the number of straight segments the
traced level arc is resolved into, so the sweep reports, per row and per arc
vertex count, the moment agreement with the retained fan arm and the margin
against the row's locked budget: one tenth of the smallest other error term the row
carries.

Each row is one measurement, and the driver runs one measurement per process:
the CLI takes ``--case`` and ``--cells``, so a lane submits one job per row and each
job builds its machine once.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.exact_clip_moment_floor import (
    CASES,
    CELL_REQUESTS,
    MOMENT_NAMES,
    REFINED_FAN_ORDER,
    _build,
    _fan_quadratic_density_moments,
    _relative_difference,
)
from nova.equilibrium.clip_quadrature import (
    _quadratic_coefficients,
    _quadratic_sample_field,
)
from nova.equilibrium.polygon_moments import polygon_density_moments
from nova.jax.config import configure_dtypes

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = Path(
    os.environ.get(
        "NOVA_EXACT_CLOSED_FORM_REPORT_ROOT",
        "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local/exact-closed-form",
    )
)
ARC_VERTEX_COUNTS = (8, 16, 32, 64, 128)
SWEEP_CASES = CASES[:3]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def probe(case_name: str, requested_cells: int) -> None:
    """Report the support buffer layout a row actually carries."""
    operator, support, field, bank_capacity, flux_span = _build(
        case_name, requested_cells
    )
    vertices = np.asarray(support.support_vertices, dtype=np.float64)
    count = np.asarray(support.vertex_count, dtype=np.intp)
    included = np.asarray(support.included)
    boundary = np.asarray(support.boundary)
    print("polygon_moments_probe_start")
    print("vertices", vertices.shape, "flux_span", flux_span)
    print("count_min", int(count.min()), "count_max", int(count.max()))
    print("boundary_cells", int(np.count_nonzero(included & boundary)))
    print("counts", np.unique(count, return_counts=True))
    for cell in np.flatnonzero(included & boundary)[:3]:
        print("cell", int(cell), "count", int(count[cell]))
        print(vertices[cell, : count[cell]])
    print("support_fields", sorted(support._fields))
    print("polygon_moments_probe_end")


def run(
    case_name: str, requested_cells: int, probe_only: bool = False
) -> dict[str, Any] | None:
    """Measure one row in this process and write its receipt.

    The row is built once and both the sweep and the cost study read that one
    build, so a lane's job ends with exactly one machine construction behind it.
    """
    configure_dtypes()
    assert jax.config.jax_enable_x64
    if probe_only:
        probe(case_name, requested_cells)
        return None
    built = _build(case_name, requested_cells)
    operator, support, field = built[0], built[1], built[2]
    vertices = np.asarray(support.support_vertices, dtype=np.float64)
    count = np.asarray(support.vertex_count, dtype=np.intp)
    polygon, live = _arc_resolved_polygons(vertices, count, ARC_VERTEX_COUNTS[-1])
    receipt = sweep(case_name, requested_cells, built)
    receipt["cost_study"] = cost_study(operator, support, field, polygon, live)
    smallest = next(
        (
            segments
            for segments in ARC_VERTEX_COUNTS
            if receipt["counts"][str(segments)]["meets_budget"]
        ),
        None,
    )
    receipt["smallest_arc_vertex_count_meeting_budget"] = smallest
    _write_json(
        REPORT_ROOT / "parts" / f"{case_name}-{abs(requested_cells)}.json", receipt
    )
    print("closed_form_smallest", smallest)
    return receipt


#: Leading support slots the traced level arc occupies on the exact-clip route:
#: both endpoints plus one interior sample per arc segment.
_ARC_POINTS = 129


def _arc_resolved_polygons(
    vertices, count, segments: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve each cell's traced arc into a fixed number of straight segments.

    The support buffer carries the arc as a sampled polyline in its leading
    slots and the straight cell boundary follows. A coarser resolution keeps the
    arc's two endpoints and takes evenly spaced samples between them, so the
    polygon's area, first and second moments are those of a region whose arc is
    that polyline and nothing else changes.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    count = np.asarray(count, dtype=np.intp)
    cells = vertices.shape[0]
    index = np.rint(np.linspace(0.0, _ARC_POINTS - 1, segments + 1)).astype(np.intp)
    arc = vertices[:, index, :]
    base_slots = _ARC_POINTS + np.arange(vertices.shape[1], dtype=np.intp)
    base_live = base_slots[None, :] < count[:, None]
    capacity = segments + 1 + int(base_slots.size)
    polygon = np.zeros((cells, capacity, 2), dtype=np.float64)
    polygon[:, : segments + 1, :] = arc
    base = np.take_along_axis(
        vertices, np.minimum(base_slots, vertices.shape[1] - 1)[None, :, None], axis=1
    )
    polygon[:, segments + 1 :, :] = np.where(base_live[:, :, None], base, 0.0)
    live = (segments + 1) + np.sum(base_live, axis=1)
    return polygon, live.astype(np.intp)


def _hlo_instruction_count(lowered) -> int:
    """Count the instructions in a lowered program's textual module."""
    return sum(
        1
        for line in lowered.as_text().splitlines()
        if line.startswith("  ") and not line.startswith("  //")
    )


def _closed_form_moments(
    operator, support, field, polygon: np.ndarray, live: np.ndarray
) -> np.ndarray:
    """Reduce the arc-resolved polygons to the three cut-moment series."""
    cell_index = jnp.arange(len(live), dtype=jnp.int32)
    sample, psi_norm, _radial, _vertical, centre, scale = _quadratic_sample_field(
        field, cell_index
    )
    sampled = operator.source.core.current_density(sample[..., 0], psi_norm)
    fitted = np.asarray(_quadratic_coefficients(sampled), dtype=np.float64)
    centre = np.asarray(centre, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    local = (polygon - centre[:, None, :]) / scale[:, None, :]
    moments = polygon_density_moments(
        jnp.asarray(local), jnp.asarray(live), jnp.asarray(fitted)
    )
    area_scale = scale[:, 0] * scale[:, 1]
    current = area_scale * np.asarray(moments.area)
    offset = centre - np.asarray(support.centroids, dtype=np.float64)
    radial = (
        area_scale * scale[:, 0] * np.asarray(moments.radial) + current * offset[:, 0]
    )
    vertical = (
        area_scale * scale[:, 1] * np.asarray(moments.vertical) + current * offset[:, 1]
    )
    return np.stack((current, radial, vertical), axis=0)


#: The locked budget of each row: one tenth of the smallest other error term the
#: row carries, in the units the row-margin table states them. The three
#: coupling rows carry the second-order coupling frozen-image error of their
#: own component.
#: The fallback rows carry the fan refinement floor, reported per moment, and
#: apply to every row the coupling table did not reach. That floor sits at
#: round-off scale, so its tenth is below any polyline's own round-off floor and
#: a fallback row is reported as met by no arc vertex count rather than by
#: a count that would read as a verdict on the route. A row is met when every
#: moment series it gates sits at or below its budget.
COUPLING_BUDGET = {
    ("weak-rotation-reactor-static", -110): 1.147e-5,
    ("weak-rotation-reactor-static", -300): 3.244e-6,
    ("moderate-rotation-conventional-static", -110): 1.040e-5,
    ("moderate-rotation-conventional-static", -300): 3.348e-6,
}
FALLBACK_BUDGET = {
    "current": 2.924e-17,
    "radial": 5.109e-16,
    "vertical": 5.064e-16,
}


def _row_budget(case_name: str, requested_cells: int) -> dict[str, float]:
    """Return the locked per-moment budget of one row."""
    coupling = COUPLING_BUDGET.get((case_name, requested_cells))
    if coupling is not None:
        return {name: coupling for name in MOMENT_NAMES}
    return dict(FALLBACK_BUDGET)


def sweep(
    case_name: str, requested_cells: int, built: tuple | None = None
) -> dict[str, Any]:
    """Sweep the arc vertex count over one row and score each against its budget."""
    operator, support, field, _bank, flux_span = (
        _build(case_name, requested_cells) if built is None else built
    )
    profile = operator.source.core
    boundary = np.asarray(support.included) & np.asarray(support.boundary)
    vertices = np.asarray(support.support_vertices, dtype=np.float64)
    count = np.asarray(support.vertex_count, dtype=np.intp)
    reference = _fan_quadratic_density_moments(
        support, field, profile, order=REFINED_FAN_ORDER
    )
    budget = _row_budget(case_name, requested_cells)
    counts: dict[str, Any] = {}
    for segments in ARC_VERTEX_COUNTS:
        polygon, live = _arc_resolved_polygons(vertices, count, segments)
        observed = _closed_form_moments(operator, support, field, polygon, live)
        relative = _relative_difference(observed[:, boundary], reference[:, boundary])
        series = {name: float(value) for name, value in zip(MOMENT_NAMES, relative)}
        margin = {
            name: budget[name] / value if value > 0.0 else None
            for name, value in series.items()
        }
        counts[str(segments)] = {
            "moment_relative_l2_against_fan": series,
            "budget": dict(budget),
            "margin_over_budget": margin,
            "meets_budget": all(
                value <= budget[name] for name, value in series.items()
            ),
        }
    return {
        "schema": "nova.exact-clip-closed-form-floor.v1",
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": int(len(count)),
        "cut_cells": int(np.count_nonzero(boundary)),
        "flux_span": flux_span,
        "arc_vertex_counts": list(ARC_VERTEX_COUNTS),
        "counts": counts,
    }


def cost_study(
    operator, support, field, polygon: np.ndarray, live: np.ndarray
) -> dict[str, Any]:
    """Report the closed-form route's program size and peak temporary."""
    cell_index = jnp.arange(len(live), dtype=jnp.int32)
    sample, psi_norm, _r, _v, centre, scale = _quadratic_sample_field(field, cell_index)
    sampled = operator.source.core.current_density(sample[..., 0], psi_norm)
    fitted = np.asarray(_quadratic_coefficients(sampled), dtype=np.float64)
    local = (polygon - np.asarray(centre)[:, None, :]) / np.asarray(scale)[:, None, :]
    lowered = jax.jit(polygon_density_moments).lower(
        jnp.asarray(local), jnp.asarray(live), jnp.asarray(fitted)
    )
    compiled = lowered.compile()
    memory = compiled.memory_analysis()
    cost = compiled.cost_analysis()
    return {
        "buffer_capacity": int(local.shape[1]),
        "live_edges_per_cut_cell": float(np.mean(live)),
        "quadrature_nodes_per_cut_cell": 0,
        "table_terms_per_live_edge": 15 * 6 * 5,
        "instruction_count": _hlo_instruction_count(lowered),
        "peak_temporary_bytes": int(getattr(memory, "temp_size_in_bytes", 0)),
        "flops": None if cost is None else float(cost.get("flops", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=SWEEP_CASES)
    parser.add_argument("--cells", type=int, choices=CELL_REQUESTS)
    parser.add_argument("--probe", action="store_true")
    arguments = parser.parse_args()
    case_name = arguments.case or SWEEP_CASES[0]
    requested_cells = arguments.cells or CELL_REQUESTS[0]
    run(case_name, requested_cells, probe_only=arguments.probe)


if __name__ == "__main__":
    main()
