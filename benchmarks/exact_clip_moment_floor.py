"""Measure boundary-integrated cut moments against the retained fan arm."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks import solovev_certificate as certificate
from nova.equilibrium.clip_quadrature import (
    _compact_chord_polygon,
    _quadratic_coefficients,
    _quadratic_sample_field,
    clipped_support_current_moments,
    cut_cell_bank_capacity,
    cut_cell_moment_evaluation_bound,
)
from nova.equilibrium.stencil_mesh import CellCurrentMoments, flux_field_polynomial
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-codex/exact-moments"
)
FIGURE_ROOT = ROOT / "docs/figures/exact-clip-moment-quadrature"
CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
    "diverted-single-null",
)
CELL_REQUESTS = (-110, -300, -1000)
MOMENT_NAMES = ("current", "radial", "vertical")
FAN_ORDER = 8
REFINED_FAN_ORDER = 16
FAN_POINTS_PER_CUT_CELL = 196_480


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


def _case_key(case_name: str, requested_cells: int) -> str:
    return f"{case_name}-{abs(requested_cells)}"


def _build(case_name: str, requested_cells: int):
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    state = certificate._exact_state(case_name, exact, coordinates)
    operator = fixture.forward_operator(source_case, machine)
    support = fixture._analytic_profile_support(exact, operator, state)
    physical = jnp.asarray(state[: operator.physical_node_number])
    grid_flux, _wall_flux = operator.topology.split_flux_map(physical)
    axis_flux = jnp.asarray(fixture._analytic_axis_flux(exact), dtype=grid_flux.dtype)
    flux_span = -axis_flux
    centroid_flux = (grid_flux - axis_flux) / flux_span
    sample_flux = (
        operator.sample_node_flux(jnp.asarray(state)) - axis_flux
    ) / flux_span
    field = flux_field_polynomial(
        operator._support_moment_stencils, centroid_flux, sample_flux
    )
    ring_centres = np.concatenate(
        [stencil.ring_centre for stencil in operator._support_moment_stencils]
    )
    bank_capacity = cut_cell_bank_capacity(
        operator.moment_geometry.atomic_mesh.centroids, ring_centres
    )
    return operator, support, field, bank_capacity, float(flux_span)


def _fan_cut_moments(support, field, profile, order: int) -> np.ndarray:
    count = np.asarray(support.vertex_count, dtype=np.intp)
    vertices = np.asarray(support.support_vertices, dtype=np.float64)
    centres = np.asarray(support.centroids, dtype=np.float64)
    boundary = np.asarray(support.included) & np.asarray(support.boundary)
    coefficient = np.asarray(field.coefficient, dtype=np.float64)
    sample_centre = np.asarray(field.centre, dtype=np.float64)
    scale = np.asarray(field.scale, dtype=np.float64)
    values = np.zeros((3, len(count)), dtype=np.float64)
    for cell in np.flatnonzero(boundary):
        polygon = vertices[cell, : count[cell]]
        points, weights = fixture._polygon_rule(polygon, order=order)
        local = (points - sample_centre[cell]) / scale[cell]
        radial, vertical = local[:, 0], local[:, 1]
        design = np.column_stack(
            (
                np.ones(len(points)),
                radial,
                vertical,
                radial**2,
                radial * vertical,
                vertical**2,
            )
        )
        psi_norm = design @ coefficient[cell]
        density = np.asarray(
            profile.current_density(jnp.asarray(points[:, 0]), jnp.asarray(psi_norm)),
            dtype=np.float64,
        )
        weighted = density * weights
        offset = points - centres[cell]
        values[:, cell] = (
            np.sum(weighted),
            np.sum(weighted * offset[:, 0]),
            np.sum(weighted * offset[:, 1]),
        )
    return values


def _boundary_exact_density_moments(support, field, profile, order: int) -> np.ndarray:
    """Integrate exact density by a boundary homotopy over the sampled polygon."""
    node, weight = np.polynomial.legendre.leggauss(order)
    unit_node = 0.5 * (node + 1.0)
    unit_weight = 0.5 * weight
    count = np.asarray(support.vertex_count, dtype=np.intp)
    vertices = np.asarray(support.support_vertices, dtype=np.float64)
    centres = np.asarray(support.centroids, dtype=np.float64)
    boundary = np.asarray(support.included) & np.asarray(support.boundary)
    coefficient = np.asarray(field.coefficient, dtype=np.float64)
    sample_centre = np.asarray(field.centre, dtype=np.float64)
    scale = np.asarray(field.scale, dtype=np.float64)
    values = np.zeros((3, len(count)), dtype=np.float64)
    for cell in np.flatnonzero(boundary):
        polygon = vertices[cell, : count[cell]]
        anchor = polygon[0]
        edge_first = polygon[1:-1] - anchor
        edge_second = polygon[2:] - anchor
        direction = (1.0 - unit_node)[None, :, None] * edge_first[
            :, None, :
        ] + unit_node[None, :, None] * edge_second[:, None, :]
        points = (
            anchor[None, None, None, :]
            + unit_node[None, :, None, None] * direction[:, None, :, :]
        )
        local = (points - sample_centre[cell]) / scale[cell]
        radial, vertical = local[..., 0], local[..., 1]
        psi_norm = (
            coefficient[cell, 0]
            + coefficient[cell, 1] * radial
            + coefficient[cell, 2] * vertical
            + coefficient[cell, 3] * radial**2
            + coefficient[cell, 4] * radial * vertical
            + coefficient[cell, 5] * vertical**2
        )
        density = np.asarray(
            profile.current_density(jnp.asarray(points[..., 0]), jnp.asarray(psi_norm)),
            dtype=np.float64,
        )
        jacobian = np.abs(
            edge_first[:, 0] * edge_second[:, 1] - edge_first[:, 1] * edge_second[:, 0]
        )
        area_weight = (
            jacobian[:, None, None]
            * unit_node[None, :, None]
            * unit_weight[None, :, None]
            * unit_weight[None, None, :]
        )
        offset = points - centres[cell]
        weighted = density * area_weight
        values[:, cell] = (
            np.sum(weighted),
            np.sum(weighted * offset[..., 0]),
            np.sum(weighted * offset[..., 1]),
        )
    return values


def _fan_quadratic_density_moments(support, field, profile, order: int) -> np.ndarray:
    """Integrate the production six-sample quadratic density on the fan region."""
    cell_index = jnp.arange(len(support.vertex_count), dtype=jnp.int32)
    points, psi_norm, _radial, _vertical, polynomial_centre, coordinate_scale = (
        _quadratic_sample_field(field, cell_index)
    )
    sampled_density = profile.current_density(points[..., 0], psi_norm)
    fitted = np.asarray(_quadratic_coefficients(sampled_density), dtype=np.float64)
    polynomial_centre = np.asarray(polynomial_centre, dtype=np.float64)
    coordinate_scale = np.asarray(coordinate_scale, dtype=np.float64)
    count = np.asarray(support.vertex_count, dtype=np.intp)
    vertices = np.asarray(support.support_vertices, dtype=np.float64)
    centres = np.asarray(support.centroids, dtype=np.float64)
    boundary = np.asarray(support.included) & np.asarray(support.boundary)
    values = np.zeros((3, len(count)), dtype=np.float64)
    for cell in np.flatnonzero(boundary):
        polygon = vertices[cell, : count[cell]]
        fan_points, fan_weights = fixture._polygon_rule(polygon, order=order)
        local = (fan_points - polynomial_centre[cell]) / coordinate_scale[cell]
        radial, vertical = local[:, 0], local[:, 1]
        design = np.column_stack(
            (
                np.ones(len(fan_points)),
                radial,
                vertical,
                radial**2,
                radial * vertical,
                vertical**2,
            )
        )
        density = design @ fitted[cell]
        weighted = density * fan_weights
        offset = fan_points - centres[cell]
        values[:, cell] = (
            np.sum(weighted),
            np.sum(weighted * offset[:, 0]),
            np.sum(weighted * offset[:, 1]),
        )
    return values


def _relative_difference(observed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    scale = np.linalg.norm(reference, axis=1)
    return np.linalg.norm(observed - reference, axis=1) / np.maximum(
        scale, np.finfo(np.float64).tiny
    )


def _replace_cut(base: CellCurrentMoments, cut: np.ndarray, values: np.ndarray):
    return CellCurrentMoments(
        *(
            jnp.asarray(original).at[cut].set(values[index, cut])
            for index, original in enumerate(base)
        )
    )


def _frozen_image(operator, base, boundary, values) -> np.ndarray:
    moments = _replace_cut(base, boundary, values)
    return np.asarray(
        operator.current_moment_image(operator.coupling_current_moments(moments))
    )


def discriminate() -> dict[str, Any]:
    """Separate sampled-region and quadratic-density effects on the weak row."""
    case_name = CASES[0]
    requested_cells = CELL_REQUESTS[0]
    operator, support, field, bank_capacity, flux_span = _build(
        case_name, requested_cells
    )
    boundary = np.asarray(support.included) & np.asarray(support.boundary)
    production = jax.jit(
        lambda carried_support, carried_field: clipped_support_current_moments(
            carried_support,
            carried_support.included,
            carried_field,
            operator.source.core,
            cut_cell_capacity=bank_capacity,
        )
    )(support, field)
    jax.block_until_ready(production)
    production_array = np.stack([np.asarray(value) for value in production])
    fan = _fan_cut_moments(support, field, operator.source.core, FAN_ORDER)
    exact_boundary = _boundary_exact_density_moments(
        support, field, operator.source.core, FAN_ORDER
    )
    quadratic_fan = _fan_quadratic_density_moments(
        support, field, operator.source.core, FAN_ORDER
    )
    fan_image = _frozen_image(operator, production, boundary, fan)

    def arm(name: str, description: str, values: np.ndarray) -> dict[str, Any]:
        image = _frozen_image(operator, production, boundary, values)
        return {
            "name": name,
            "description": description,
            "moment_relative_l2_against_fan": dict(
                zip(
                    MOMENT_NAMES,
                    _relative_difference(values[:, boundary], fan[:, boundary]),
                    strict=True,
                )
            ),
            "frozen_image_delta_sup_over_span": float(
                np.max(np.abs(image - fan_image)) / abs(flux_span)
            ),
        }

    arms = [
        arm(
            "sampled_polygon_exact_density_boundary",
            "Boundary homotopy over all 131 sampled vertices with pointwise "
            "profile density.",
            exact_boundary,
        ),
        arm(
            "sampled_polygon_quadratic_density_fan",
            "Retained fan region with the production six-sample quadratic density fit.",
            quadratic_fan,
        ),
    ]
    production_arm = arm(
        "chord_sagitta_quadratic_density",
        "Production chord-plus-sagitta region with the six-sample quadratic "
        "density fit.",
        production_array,
    )
    production_image = _frozen_image(operator, production, boundary, production_array)
    quadratic_image = _frozen_image(operator, production, boundary, quadratic_fan)
    exact_fan_norm = np.linalg.norm(fan[:, boundary], axis=1)
    region_difference = np.linalg.norm(
        production_array[:, boundary] - quadratic_fan[:, boundary], axis=1
    ) / np.maximum(exact_fan_norm, np.finfo(np.float64).tiny)
    capacity = int(np.asarray(support.support_vertices).shape[1])
    fan_points = (capacity - 2) * FAN_ORDER**2
    if fan_points != FAN_POINTS_PER_CUT_CELL:
        raise RuntimeError(
            f"fan allocation changed: measured {fan_points}, expected "
            f"{FAN_POINTS_PER_CUT_CELL}"
        )
    cut_vertex_count = np.asarray(support.vertex_count, dtype=np.intp)[boundary]
    live_points = (cut_vertex_count - 2) * FAN_ORDER**2
    padding_points = fan_points - live_points
    census = []
    for vertex_count in np.unique(cut_vertex_count):
        selected = cut_vertex_count == vertex_count
        census.append(
            {
                "vertex_count": int(vertex_count),
                "cut_cells": int(np.count_nonzero(selected)),
                "live_evaluations_per_cut_cell": int(live_points[selected][0]),
                "exact_zero_padding_per_cut_cell": int(padding_points[selected][0]),
            }
        )
    payload = {
        "schema": "nova.exact-clip-density-region-discriminator.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": int(len(support.vertex_count)),
        "cut_cells": int(np.count_nonzero(boundary)),
        "fan_order": FAN_ORDER,
        "arms": arms,
        "production_context": production_arm,
        "region_increment_at_quadratic_density": {
            "moment_relative_l2_over_exact_fan_norm": dict(
                zip(
                    MOMENT_NAMES,
                    region_difference,
                    strict=True,
                )
            ),
            "frozen_image_delta_sup_over_span": float(
                np.max(np.abs(production_image - quadratic_image)) / abs(flux_span)
            ),
        },
        "fan_allocation": {
            "fixed_evaluations_per_cut_cell": fan_points,
            "census": census,
            "total_live_evaluations": int(np.sum(live_points)),
            "total_exact_zero_padding": int(np.sum(padding_points)),
            "live_fraction": float(
                np.sum(live_points) / (fan_points * len(live_points))
            ),
        },
        "lane": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "host": socket.gethostname(),
            "jax_platform": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
    }
    _write_json(REPORT_ROOT / "density-region-discriminator.json", payload)
    return payload


def measure(case_name: str, requested_cells: int) -> dict[str, Any]:
    operator, support, field, bank_capacity, flux_span = _build(
        case_name, requested_cells
    )
    boundary = np.asarray(support.included) & np.asarray(support.boundary)
    reduced = jax.jit(
        lambda carried_support, carried_field: clipped_support_current_moments(
            carried_support,
            carried_support.included,
            carried_field,
            operator.source.core,
            cut_cell_capacity=bank_capacity,
        )
    )(support, field)
    jax.block_until_ready(reduced)
    reduced_array = np.stack([np.asarray(value) for value in reduced])
    if np.any(~np.isfinite(reduced_array[:, boundary])):
        compact = _compact_chord_polygon(support.support_vertices, support.vertex_count)
        supported = np.asarray(compact[-1], dtype=bool)
        raise RuntimeError(
            "boundary reduction refused cells "
            f"{np.flatnonzero(boundary & ~supported).tolist()} with counts "
            f"{np.asarray(support.vertex_count)[boundary & ~supported].tolist()}"
        )
    fan = _fan_cut_moments(support, field, operator.source.core, FAN_ORDER)
    relative = _relative_difference(reduced_array[:, boundary], fan[:, boundary])

    refined = None
    floor = None
    image = None
    if case_name == CASES[0] and requested_cells == CELL_REQUESTS[0]:
        refined = _fan_cut_moments(
            support, field, operator.source.core, REFINED_FAN_ORDER
        )
        refinement_delta = _relative_difference(refined[:, boundary], fan[:, boundary])
        floor = (4.0 / 3.0) * refinement_delta
        fan_moments = _replace_cut(reduced, boundary, fan)
        refined_moments = _replace_cut(reduced, boundary, refined)
        reduced_image = np.asarray(
            operator.current_moment_image(
                operator.coupling_current_moments(CellCurrentMoments(*reduced_array))
            )
        )
        fan_image = np.asarray(
            operator.current_moment_image(
                operator.coupling_current_moments(fan_moments)
            )
        )
        refined_image = np.asarray(
            operator.current_moment_image(
                operator.coupling_current_moments(refined_moments)
            )
        )
        image = {
            "boundary_minus_fan_sup_over_span": float(
                np.max(np.abs(reduced_image - fan_image)) / abs(flux_span)
            ),
            "fan_refinement_floor_sup_over_span": float(
                (4.0 / 3.0) * np.max(np.abs(refined_image - fan_image)) / abs(flux_span)
            ),
        }

    receipt = {
        "schema": "nova.exact-clip-moment-floor.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": int(len(support.vertex_count)),
        "cut_cells": int(np.count_nonzero(boundary)),
        "fan_order": FAN_ORDER,
        "refined_fan_order": REFINED_FAN_ORDER if refined is not None else None,
        "moment_relative_l2_boundary_minus_fan": dict(
            zip(MOMENT_NAMES, relative, strict=True)
        ),
        "fan_refinement_floor_relative_l2": (
            None if floor is None else dict(zip(MOMENT_NAMES, floor, strict=True))
        ),
        "frozen_current_image": image,
        "evaluation_points_per_cut_cell": cut_cell_moment_evaluation_bound(),
        "chord_polygon_vertex_capacity": 24,
        "lane": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "host": socket.gethostname(),
            "jax_platform": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
    }
    _write_json(
        REPORT_ROOT / "parts" / f"{_case_key(case_name, requested_cells)}.json", receipt
    )
    return receipt


def finalize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    floor = next(
        row["fan_refinement_floor_relative_l2"]
        for row in rows
        if row["fan_refinement_floor_relative_l2"] is not None
    )
    for row in rows:
        row["floor_ratio"] = {
            name: row["moment_relative_l2_boundary_minus_fan"][name] / floor[name]
            for name in MOMENT_NAMES
        }
    payload = {
        "schema": "nova.exact-clip-moment-floor-summary.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "fan_refinement_floor_relative_l2": floor,
        "rows": rows,
        "maximum_floor_ratio": max(
            value for row in rows for value in row["floor_ratio"].values()
        ),
    }
    _write_json(REPORT_ROOT / "summary.json", payload)
    _write_json(FIGURE_ROOT / "comparison.json", payload)

    labels = [
        f"{row['case'].split('-')[0]} {abs(row['requested_cells'])}" for row in rows
    ]
    position = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    for offset, name in zip((-0.24, 0.0, 0.24), MOMENT_NAMES, strict=True):
        axis.bar(
            position + offset,
            [row["floor_ratio"][name] for row in rows],
            width=0.22,
            label=name,
        )
    axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_yscale("log")
    axis.set_ylabel("boundary-minus-fan / measured fan floor")
    axis.set_xticks(position, labels, rotation=35, ha="right")
    axis.legend(frameon=False, ncols=3)
    axis.set_title("Exact clipped moments against the retained fan arm")
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_ROOT / "comparison.svg")
    plt.close(figure)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cells", type=int, choices=CELL_REQUESTS)
    parser.add_argument("--discriminate", action="store_true")
    arguments = parser.parse_args()
    configure_dtypes()
    if not jax.config.jax_enable_x64 or jax.default_backend() != "cpu":
        raise RuntimeError("this measurement requires binary64 on the CPU backend")
    if arguments.discriminate:
        payload = discriminate()
        for row in payload["arms"]:
            print(
                "ARM",
                row["name"],
                row["moment_relative_l2_against_fan"],
                row["frozen_image_delta_sup_over_span"],
                flush=True,
            )
        print("FAN_ALLOCATION", payload["fan_allocation"], flush=True)
        return
    requests = (
        [(arguments.case, arguments.cells)]
        if arguments.case is not None and arguments.cells is not None
        else [(case_name, cells) for case_name in CASES for cells in CELL_REQUESTS]
    )
    rows = []
    for case_name, requested_cells in requests:
        row = measure(case_name, requested_cells)
        rows.append(row)
        print(
            "ROW",
            _case_key(case_name, requested_cells),
            row["moment_relative_l2_boundary_minus_fan"],
            flush=True,
        )
    if len(rows) == len(CASES) * len(CELL_REQUESTS):
        summary = finalize(rows)
        print("MAX_FLOOR_RATIO", summary["maximum_floor_ratio"], flush=True)


if __name__ == "__main__":
    main()
