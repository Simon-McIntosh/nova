"""Measure boundary-integrated cut moments against the retained fan arm."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
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
    clipped_support_current_moments,
    cut_cell_bank_capacity,
)

try:
    from nova.equilibrium.clip_quadrature import (
        _ARC_EDGE_NODE,
        _ARC_EDGE_ORDER,
        _ARC_EDGE_WEIGHT,
        _DENSITY_POWERS,
        _DENSITY_SAMPLE_LOCAL,
        _compact_chord_polygon,
        _quadratic_coefficients,
        _quadratic_sample_field,
        cut_capacity_edge_bound,
        cut_cell_moment_evaluation_bound,
    )
except ImportError:
    _compact_chord_polygon = None
    _quadratic_coefficients = None
    _quadratic_sample_field = None
    cut_capacity_edge_bound = None
    cut_cell_moment_evaluation_bound = None
    _ARC_EDGE_NODE = None
    _ARC_EDGE_WEIGHT = None
    _ARC_EDGE_ORDER = None
    _DENSITY_SAMPLE_LOCAL = None
    _DENSITY_POWERS = None
from nova.equilibrium.stencil_mesh import CellCurrentMoments, flux_field_polynomial
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local/exact-gauss"
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
PLAN_FAN_POINTS_PER_CUT_CELL = 196_480


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


def _require_boundary_route() -> None:
    if any(
        item is None
        for item in (
            _quadratic_coefficients,
            _quadratic_sample_field,
            cut_capacity_edge_bound,
            cut_cell_moment_evaluation_bound,
            _ARC_EDGE_NODE,
            _ARC_EDGE_WEIGHT,
        )
    ):
        raise RuntimeError("the boundary-reduction implementation is unavailable")


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


def edge_order_study() -> dict[str, Any]:
    """Find the lowest per-edge Gauss order exact on the density model's integral.

    The boundary rule integrates, along each straight edge, the radial
    antiderivative of the local density model. That integrand is a polynomial of
    degree five in the edge parameter, so a Gauss rule is exact on it once
    2 * order - 1 reaches five. This checks the claim directly on the model's
    own monomials rather than on the reduced moments.

    The reference is the monomial's antiderivative in closed form. A sampled rule
    cannot serve as it: its own truncation sits near 1e-6 on these integrands,
    which floors the measured defect above the exactness threshold and makes
    every order read as inexact.
    """

    def exact_line_integral(exponent: int, vertical_power: int) -> float:
        """Integrate t**exponent * (1 - 2 t)**vertical_power along the unit edge."""
        return float(
            sum(
                math.comb(vertical_power, term) * (-2.0) ** term / (exponent + term + 1)
                for term in range(vertical_power + 1)
            )
        )

    result: dict[str, Any] = {}
    for order in (1, 2, 3, 4):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        defect = 0.0
        for radial_power, vertical_power in _DENSITY_POWERS:
            exponent = radial_power + 1
            quadrature = np.sum(
                weights * nodes**exponent * (1.0 - 2.0 * nodes) ** vertical_power
            )
            exact = exact_line_integral(exponent, vertical_power)
            defect = max(defect, abs(quadrature - exact) / max(abs(exact), 1e-300))
        result[str(order)] = defect
    exact_orders = [order for order in (1, 2, 3, 4) if result[str(order)] <= 1e-12]
    if not exact_orders:
        raise ValueError(
            "no tested per-edge Gauss order is exact on the model "
            f"antiderivative: {result}"
        )
    lowest = exact_orders[0]
    return {
        "relative_line_integral_defect_by_order": result,
        "lowest_order_exact_on_the_model_antiderivative": lowest,
        "integrand_degree_in_edge_parameter": 5,
    }


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


def discriminate(
    case_name: str = CASES[0],
    requested_cells: int = CELL_REQUESTS[0],
    built: tuple | None = None,
) -> dict[str, Any]:
    """Separate sampled-region and quadratic-density effects on one row.

    The receipt is the row's other error term: the retained fan region carrying
    the production quadratic density fit, measured against the fan's own
    pointwise profile, alongside an exact-density arm that must reproduce the
    fan identically. ``built`` carries a ``_build`` result forward so a row
    report does not rebuild the machine and support.
    """
    _require_boundary_route()
    operator, support, field, bank_capacity, flux_span = (
        _build(case_name, requested_cells) if built is None else built
    )
    boundary = np.asarray(support.included) & np.asarray(support.boundary)
    production = jax.jit(
        lambda carried_support, carried_field: clipped_support_current_moments(
            carried_support,
            carried_support.included,
            carried_field,
            operator.source.core,
            cut_cell_capacity=bank_capacity,
            boundary_reduction=True,
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
            "Boundary homotopy over every live sampled vertex with pointwise "
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
        "sampled_arc_density_model",
        "Production sampled-arc boundary route: the clip's own 128-segment arc "
        "region carrying a per-edge Gauss rule of the local degree-four density "
        "model.",
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
    cut_vertex_count = np.asarray(support.vertex_count, dtype=np.intp)[boundary]
    live_points = (cut_vertex_count - 2) * FAN_ORDER**2
    padding_points = fan_points - live_points
    plan_padding_points = PLAN_FAN_POINTS_PER_CUT_CELL - live_points
    census = []
    for vertex_count in np.unique(cut_vertex_count):
        selected = cut_vertex_count == vertex_count
        census.append(
            {
                "vertex_count": int(vertex_count),
                "cut_cells": int(np.count_nonzero(selected)),
                "live_evaluations_per_cut_cell": int(live_points[selected][0]),
                "exact_zero_padding_per_cut_cell": int(padding_points[selected][0]),
                "plan_reference_zero_padding_per_cut_cell": int(
                    plan_padding_points[selected][0]
                ),
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
            "plan_reference_fixed_evaluations_per_cut_cell": (
                PLAN_FAN_POINTS_PER_CUT_CELL
            ),
            "actual_support_capacity": capacity,
            "actual_fixed_evaluations_per_cut_cell": fan_points,
            "census": census,
            "total_live_evaluations": int(np.sum(live_points)),
            "total_exact_zero_padding": int(np.sum(padding_points)),
            "plan_reference_total_exact_zero_padding": int(np.sum(plan_padding_points)),
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
    suffix = (
        ""
        if (case_name, requested_cells)
        == (
            CASES[0],
            CELL_REQUESTS[0],
        )
        else f"-{_case_key(case_name, requested_cells)}"
    )
    _write_json(REPORT_ROOT / f"density-region-discriminator{suffix}.json", payload)
    return payload


def row_report(case_name: str, requested_cells: int) -> dict[str, Any]:
    """Measure one row's route error and its other error term from one build."""
    built = _build(case_name, requested_cells)
    row = measure(case_name, requested_cells, built)
    discriminator = discriminate(case_name, requested_cells, built)
    return {
        "row": row,
        "discriminator": discriminator,
        "budget_one_tenth": {
            name: 0.1 * value
            for name, value in discriminator["arms"][1][
                "moment_relative_l2_against_fan"
            ].items()
        },
    }


def measure(
    case_name: str, requested_cells: int, built: tuple | None = None
) -> dict[str, Any]:
    _require_boundary_route()
    operator, support, field, bank_capacity, flux_span = (
        _build(case_name, requested_cells) if built is None else built
    )
    boundary = np.asarray(support.included) & np.asarray(support.boundary)
    reduced = jax.jit(
        lambda carried_support, carried_field: clipped_support_current_moments(
            carried_support,
            carried_support.included,
            carried_field,
            operator.source.core,
            cut_cell_capacity=bank_capacity,
            boundary_reduction=True,
        )
    )(support, field)
    jax.block_until_ready(reduced)
    reduced_array = np.stack([np.asarray(value) for value in reduced])
    refused = boundary & ~np.all(np.isfinite(reduced_array), axis=0)
    if np.any(refused):
        raise RuntimeError(
            "boundary reduction refused cells "
            f"{np.flatnonzero(refused).tolist()} with polygon counts "
            f"{np.asarray(support.vertex_count)[refused].tolist()}"
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
        "fixed_edges_per_cut_cell": cut_capacity_edge_bound(),
        "per_edge_gauss_order": _ARC_EDGE_ORDER,
        "live_evaluations_per_cut_cell": (
            len(_DENSITY_SAMPLE_LOCAL) + cut_capacity_edge_bound() * _ARC_EDGE_ORDER
        ),
        "per_edge_gauss_order_study": (
            edge_order_study()
            if case_name == CASES[0] and requested_cells == CELL_REQUESTS[0]
            else None
        ),
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


def default_route_snapshot(path: Path, revision: str) -> dict[str, Any]:
    """Persist the default weak-row moments and frozen image without rounding."""
    operator, support, field, bank_capacity, _flux_span = _build(
        CASES[0], CELL_REQUESTS[0]
    )
    moments = jax.jit(
        lambda carried_support, carried_field: clipped_support_current_moments(
            carried_support,
            carried_support.included,
            carried_field,
            operator.source.core,
            cut_cell_capacity=bank_capacity,
        )
    )(support, field)
    jax.block_until_ready(moments)
    moment_array = np.stack([np.asarray(value) for value in moments])
    image = np.asarray(
        operator.current_moment_image(operator.coupling_current_moments(moments))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, revision=np.asarray(revision), moments=moment_array, image=image)
    return {
        "revision": revision,
        "path": str(path),
        "moment_shape": list(moment_array.shape),
        "image_shape": list(image.shape),
    }


def compare_default_route_snapshots(base_path: Path, current_path: Path):
    """Require last-bit identity between base and current default-route arrays."""
    with (
        np.load(base_path, allow_pickle=False) as base,
        np.load(current_path, allow_pickle=False) as current,
    ):
        result = {}
        for name in ("moments", "image"):
            base_value = np.asarray(base[name])
            current_value = np.asarray(current[name])
            result[name] = {
                "array_equal": bool(np.array_equal(base_value, current_value)),
                "maximum_absolute_difference": float(
                    np.max(np.abs(base_value - current_value))
                ),
                "sha256": hashlib.sha256(base_value.tobytes()).hexdigest(),
            }
        payload = {
            "schema": "nova.exact-clip-default-route-identity.v1",
            "created_at": datetime.now(UTC).isoformat(),
            "base_revision": str(base["revision"]),
            "current_revision": str(current["revision"]),
            "arrays": result,
        }
    _write_json(REPORT_ROOT / "default-route-bit-identity.json", payload)
    if not all(value["array_equal"] for value in result.values()):
        raise RuntimeError("default exact-clip route differs from the base revision")
    return payload


def finalize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    floor = next(
        row["fan_refinement_floor_relative_l2"]
        for row in rows
        if row["fan_refinement_floor_relative_l2"] is not None
    )
    budget = {name: 0.1 * floor[name] for name in MOMENT_NAMES}
    for row in rows:
        row["floor_ratio"] = {
            name: row["moment_relative_l2_boundary_minus_fan"][name] / floor[name]
            for name in MOMENT_NAMES
        }
        row["one_tenth_of_the_fan_floor"] = dict(budget)
        row["budget_ratio_against_one_tenth_of_the_fan_floor"] = {
            name: row["moment_relative_l2_boundary_minus_fan"][name] / budget[name]
            for name in MOMENT_NAMES
        }
    payload = {
        "schema": "nova.exact-clip-moment-floor-summary.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "fan_refinement_floor_relative_l2": floor,
        "one_tenth_of_the_fan_floor_relative_l2": budget,
        "rows": rows,
        "maximum_floor_ratio": max(
            value for row in rows for value in row["floor_ratio"].values()
        ),
        "maximum_budget_ratio": max(
            value
            for row in rows
            for value in row["budget_ratio_against_one_tenth_of_the_fan_floor"].values()
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
    parser.add_argument(
        "--row-report",
        action="store_true",
        help="measure the requested row and its density/region discriminator once",
    )
    parser.add_argument("--default-route-snapshot", type=Path)
    parser.add_argument("--snapshot-revision")
    parser.add_argument(
        "--compare-default-route-snapshots",
        type=Path,
        nargs=2,
        metavar=("BASE", "CURRENT"),
    )
    arguments = parser.parse_args()
    configure_dtypes()
    if not jax.config.jax_enable_x64 or jax.default_backend() != "cpu":
        raise RuntimeError("this measurement requires binary64 on the CPU backend")
    if arguments.default_route_snapshot is not None:
        if arguments.snapshot_revision is None:
            parser.error("--default-route-snapshot requires --snapshot-revision")
        snapshot = default_route_snapshot(
            arguments.default_route_snapshot, arguments.snapshot_revision
        )
        print("DEFAULT_ROUTE_SNAPSHOT", snapshot, flush=True)
        return
    if arguments.compare_default_route_snapshots is not None:
        comparison = compare_default_route_snapshots(
            *arguments.compare_default_route_snapshots
        )
        print("DEFAULT_ROUTE_IDENTITY", comparison, flush=True)
        return
    if arguments.row_report:
        if arguments.case is None or arguments.cells is None:
            parser.error("--row-report requires --case and --cells")
        report = row_report(arguments.case, arguments.cells)
        row = report["row"]
        print(
            "ROW",
            _case_key(arguments.case, arguments.cells),
            row["moment_relative_l2_boundary_minus_fan"],
            flush=True,
        )
        for arm in report["discriminator"]["arms"]:
            print(
                "ARM",
                arm["name"],
                arm["moment_relative_l2_against_fan"],
                arm["frozen_image_delta_sup_over_span"],
                flush=True,
            )
        print("BUDGET", report["budget_one_tenth"], flush=True)
        print(
            "SHAPE",
            {
                "cut_cells": row["cut_cells"],
                "realised_cells": row["realised_cells"],
                "per_edge_gauss_order": row["per_edge_gauss_order"],
                "fixed_edges_per_cut_cell": row["fixed_edges_per_cut_cell"],
                "live_evaluations_per_cut_cell": row["live_evaluations_per_cut_cell"],
                "evaluation_points_per_cut_cell": row["evaluation_points_per_cut_cell"],
            },
            flush=True,
        )
        return
    if arguments.discriminate:
        payload = discriminate(
            arguments.case or CASES[0], arguments.cells or CELL_REQUESTS[0]
        )
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
