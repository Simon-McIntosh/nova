"""Diagnose vessel containment in the terminal MAST saddle selector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from matplotlib.path import Path as PolygonPath
import numpy as np
from scipy import ndimage

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.diiid_forward_gs_match import _margin_graded_newton_krylov
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    GMRES_ITERATIONS,
    NEWTON_STEPS,
    RELAXATION,
    STEP_CAP,
    WARMUP_SWEEPS,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


TARGET = (21983, 35)
ARMS = ("pure_arm", "mixed_arm")
BANKED = Path("docs/figures/dual-branch-selection/pinned-branch-contrast.json")
OUTPUT = Path(
    "docs/figures/primary-xpoint-evidence/out-of-vessel-saddle-selection.json"
)
OFFSET_FRACTION = 2.0e-4


def _terminal_states(
    profile: Any, seed: jax.Array, target_current: float
) -> dict[str, jax.Array]:
    initial = jnp.stack((seed, seed))
    portfolio = profile.solve_portfolio(
        initial,
        route="newton_krylov",
        target_current=target_current,
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
        warmup=WARMUP_SWEEPS,
        relaxation=RELAXATION,
        step_cap=STEP_CAP,
    )
    portfolio.branches.equilibrium.flux.block_until_ready()
    pure = portfolio.branches.equilibrium.flux[int(TopologyClass.DIVERTED)]
    mapped = profile.flux_map(
        requested_class=TopologyClass.DIVERTED,
        target_current=target_current,
    )
    mixed = _margin_graded_newton_krylov(
        mapped,
        profile.operator.topology_margin,
        seed,
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
    ).state
    mixed.block_until_ready()
    return {"pure_arm": pure, "mixed_arm": mixed}


def _nearest_masked_label(
    radius: np.ndarray,
    height: np.ndarray,
    labels: np.ndarray,
    coordinate: np.ndarray,
) -> tuple[int, float]:
    distance = np.hypot(
        radius[None, :] - coordinate[0], height[:, None] - coordinate[1]
    )
    eligible = labels != 0
    candidate = np.where(eligible, distance, np.inf)
    flat = int(np.argmin(candidate))
    vertical, radial = np.unravel_index(flat, candidate.shape)
    return int(labels[vertical, radial]), float(candidate[vertical, radial])


def _component_measure(
    profile: Any,
    state: jax.Array,
    boundary_flux: float,
    wall_point: np.ndarray,
) -> dict[str, Any]:
    operator = profile.operator
    physical = jnp.asarray(state)[: operator.physical_node_number]
    grid_flux, _wall_flux = operator.topology.split_flux_map(physical)
    radius, height, shape = operator.connectivity_grid_axes()
    radial_count, vertical_count = shape
    flux = np.asarray(grid_flux).reshape((radial_count, vertical_count)).T
    radius = np.asarray(radius, dtype=float)
    height = np.asarray(height, dtype=float)
    inside = (
        np.asarray(operator.inside_material)
        .reshape((radial_count, vertical_count))
        .T.astype(bool)
    )
    _masks, topology = operator.read(state)
    axis = np.asarray(topology.axis, dtype=float)
    axis_vertical = int(np.argmin(np.abs(height - axis[1])))
    axis_radial = int(np.argmin(np.abs(radius - axis[0])))
    axis_flux = float(flux[axis_vertical, axis_radial])
    outward_sign = float(np.sign(boundary_flux - axis_flux) or 1.0)
    outward_flux = outward_sign * (flux - axis_flux)
    boundary_level = outward_sign * (boundary_flux - axis_flux)
    flux_range = max(float(np.ptp(outward_flux)), np.finfo(float).eps)
    inward_offset = max(
        OFFSET_FRACTION * flux_range,
        32.0 * np.finfo(float).eps * flux_range,
    )
    confined = inside & (outward_flux <= boundary_level - inward_offset)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labels, component_count = ndimage.label(confined, structure=structure)
    axis_label, axis_distance = _nearest_masked_label(radius, height, labels, axis)
    wall_label, wall_distance = _nearest_masked_label(
        radius, height, labels, wall_point
    )
    private_labels = np.unique(labels[(labels != 0) & (labels != axis_label)])
    private_mask = (labels != 0) & (labels != axis_label)
    cell_area = float(np.mean(np.diff(radius)) * np.mean(np.diff(height)))
    return {
        "boundary_flux_wb": boundary_flux,
        "classification_offset_fraction_of_flux_range": OFFSET_FRACTION,
        "classification_offset_wb": inward_offset,
        "confined_component_count": int(component_count),
        "axis_component_label": axis_label,
        "axis_to_classified_cell_distance_m": axis_distance,
        "wall_component_label": wall_label,
        "wall_to_classified_cell_distance_m": wall_distance,
        "wall_in_axis_component": bool(wall_label != 0 and wall_label == axis_label),
        "wall_in_private_component": bool(wall_label != 0 and wall_label != axis_label),
        "private_component_count": int(private_labels.size),
        "private_component_area_m2": float(np.sum(private_mask) * cell_area),
    }


def _candidate_table(
    diagnostics: dict[str, Any], wall: np.ndarray
) -> tuple[list[dict[str, Any]], int]:
    polygon = PolygonPath(wall, closed=True)
    candidates = diagnostics["typed_saddle_candidates"]
    points = np.asarray([row["coordinate_m"] for row in candidates], dtype=float)
    inside = polygon.contains_points(points, radius=1.0e-12)
    eligible_indices = [
        index
        for index, (row, contained) in enumerate(zip(candidates, inside, strict=True))
        if contained
        and np.all(np.isfinite(row["coordinate_m"]))
        and np.isfinite(row["flux_wb"])
        and np.isfinite(row["normalized_flux_operand"])
        and row["fitted_null_type"] == 0.0
    ]
    if not eligible_indices:
        raise RuntimeError("the typed saddle table contains no in-vessel candidate")
    in_vessel_index = min(
        eligible_indices,
        key=lambda index: candidates[index]["normalized_flux_operand"],
    )
    table = []
    for index, (row, contained) in enumerate(zip(candidates, inside, strict=True)):
        finite = bool(
            np.all(np.isfinite(row["coordinate_m"]))
            and np.isfinite(row["flux_wb"])
            and np.isfinite(row["normalized_flux_operand"])
        )
        saddle = bool(row["fitted_null_type"] == 0.0)
        table.append(
            {
                "index": index,
                "coordinate_m": row["coordinate_m"],
                "flux_wb": row["flux_wb"],
                "normalized_flux_operand": row["normalized_flux_operand"],
                "fitted_null_type": row["fitted_null_type"],
                "finite_fields_valid": finite,
                "fitted_as_saddle": saddle,
                "inside_operator_wall_polygon": bool(contained),
                "production_selector_eligible": bool(finite and saddle),
                "in_vessel_selector_eligible": bool(finite and saddle and contained),
                "selected_by_production": bool(row["selected"]),
                "selected_after_vessel_filter": index == in_vessel_index,
            }
        )
    return table, in_vessel_index


def _arm_record(
    profile: Any,
    state: jax.Array,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    wall = np.asarray(profile.operator.wall.coordinate, dtype=float)
    table, in_vessel_index = _candidate_table(diagnostics, wall)
    production_index = next(
        row["index"] for row in table if row["selected_by_production"]
    )
    in_vessel = table[in_vessel_index]
    production = table[production_index]
    wall_record = diagnostics["wall_operand"]
    wall_point = np.asarray(wall_record["coordinate_m"], dtype=float)
    production_heights = np.asarray(
        [row["coordinate_m"][1] for row in table if row["production_selector_eligible"]]
    )
    in_vessel_heights = np.asarray(
        [row["coordinate_m"][1] for row in table if row["in_vessel_selector_eligible"]]
    )
    production_band = [
        float(np.min(production_heights)),
        float(np.max(production_heights)),
    ]
    in_vessel_band = [
        float(np.min(in_vessel_heights)),
        float(np.max(in_vessel_heights)),
    ]
    production_inside_band = bool(
        production_band[0] <= wall_point[1] <= production_band[1]
    )
    in_vessel_inside_band = bool(
        in_vessel_band[0] <= wall_point[1] <= in_vessel_band[1]
    )
    current_components = _component_measure(
        profile, state, float(production["flux_wb"]), wall_point
    )
    in_vessel_components = _component_measure(
        profile, state, float(in_vessel["flux_wb"]), wall_point
    )
    return {
        "selection_status": diagnostics["selection_status"],
        "connectivity_admission": diagnostics["connectivity_admission"],
        "wall_coordinate_m": wall_point.tolist(),
        "wall_flux_wb": wall_record["flux_wb"],
        "wall_normalized_flux_before_shadow": wall_record[
            "normalized_flux_before_shadow"
        ],
        "candidate_count": len(table),
        "candidate_table": table,
        "production_selection": {
            "index": production_index,
            "coordinate_m": production["coordinate_m"],
            "normalized_flux_operand": production["normalized_flux_operand"],
            "inside_operator_wall_polygon": production["inside_operator_wall_polygon"],
            "rule": "argmin(normalized_flux_operand) over finite typed saddles",
            "lower_than_selected_in_vessel_by": float(
                in_vessel["normalized_flux_operand"]
                - production["normalized_flux_operand"]
            ),
        },
        "in_vessel_selection": {
            "index": in_vessel_index,
            "coordinate_m": in_vessel["coordinate_m"],
            "flux_wb": in_vessel["flux_wb"],
            "normalized_flux_operand": in_vessel["normalized_flux_operand"],
        },
        "production_height_band_m": production_band,
        "wall_inside_production_height_band": production_inside_band,
        "production_shadow_verdict": (
            "ADMITTED" if production_inside_band else "SHADOWED"
        ),
        "in_vessel_height_band_m": in_vessel_band,
        "wall_inside_in_vessel_height_band": in_vessel_inside_band,
        "in_vessel_shadow_verdict": (
            "ADMITTED" if in_vessel_inside_band else "SHADOWED"
        ),
        "verdict_flips_to_shadowed": bool(
            production_inside_band and not in_vessel_inside_band
        ),
        "components_at_production_selected_level": current_components,
        "components_at_in_vessel_selected_level": in_vessel_components,
    }


def run() -> dict[str, Any]:
    configure_dtypes()
    receipt = json.loads(BANKED.read_text())
    reference = next(
        row
        for row in receipt["references"]
        if (row["reference"]["shot"], row["reference"]["slice_index"]) == TARGET
    )
    response_cache, _evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    selected_row, qualification = selected[TARGET]
    case, context = _mast_case_from_selection(SHOT_STORE, selected_row, qualification)
    passive_case, profile, policy = _passive_inclusive_case(
        case, context, response_cache
    )
    if policy["section_kernel_evaluations_this_shot"] != 0:
        raise RuntimeError("diagnostic reconstruction entered the response builder")
    seed = jnp.asarray(passive_case["state"])
    target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
    states = _terminal_states(profile, seed, target_current)
    wall = np.asarray(profile.operator.wall.coordinate, dtype=float)
    arms = {
        arm: _arm_record(
            profile,
            states[arm],
            reference[arm]["terminal_xpoint_diagnostics"],
        )
        for arm in ARMS
    }
    flip_count = sum(row["verdict_flips_to_shadowed"] for row in arms.values())
    zero_private_count = sum(
        row["components_at_production_selected_level"]["private_component_count"] == 0
        and row["components_at_production_selected_level"]["private_component_area_m2"]
        == 0.0
        for row in arms.values()
    )
    payload = {
        "artifact": "MAST terminal typed-saddle vessel-containment diagnosis",
        "shot": TARGET[0],
        "slice_index": TARGET[1],
        "response_carrier_direct_builder_entries": policy[
            "section_kernel_evaluations_this_shot"
        ],
        "operator_wall_coordinate": {
            "node_count": int(wall.shape[0]),
            "radial_extent_m": [float(np.min(wall[:, 0])), float(np.max(wall[:, 0]))],
            "vertical_extent_m": [float(np.min(wall[:, 1])), float(np.max(wall[:, 1]))],
            "first_equals_last": bool(np.array_equal(wall[0], wall[-1])),
            "containment_method": (
                "matplotlib.path.Path.contains_points over operator.wall.coordinate "
                "with a 1e-12 m boundary radius"
            ),
        },
        "source_audit": {
            "typed_selector": (
                "nova/equilibrium/connectivity_boundary.py selects supplied typed "
                "candidates with argmin(normalized level); no wall polygon or "
                "inside_material operand participates"
            ),
            "typed_candidate_producer": (
                "nova/equilibrium/forward_operator.py _FixedDesignNull2D receives "
                "only sampled flux and fixed grid stencils; it has no vessel mask"
            ),
            "separate_connectivity_detector": (
                "nova/equilibrium/connectivity_boundary.py xpoint_candidates does "
                "receive inside_limiter, but that connectivity-local table does not "
                "replace the supplied typed candidates used by the class selector"
            ),
            "typed_selector_polygon_containment_found": False,
        },
        "arms": arms,
        "summary": {
            "arms_with_genuine_in_vessel_pinch_candidate": sum(
                any(
                    row["in_vessel_selector_eligible"]
                    and 1.0 <= abs(row["coordinate_m"][1]) <= 1.4
                    for row in arm["candidate_table"]
                )
                for arm in arms.values()
            ),
            "arms_where_detector_produced_but_selector_passed_over_pinch": sum(
                any(
                    row["in_vessel_selector_eligible"]
                    and 1.0 <= abs(row["coordinate_m"][1]) <= 1.4
                    for row in arm["candidate_table"]
                )
                and not arm["production_selection"]["inside_operator_wall_polygon"]
                for arm in arms.values()
            ),
            "arms_where_admission_flips_to_shadowed": flip_count,
            "arms_with_zero_private_region_at_production_level": zero_private_count,
            "conclusion": (
                "Both detectors produced genuine in-vessel pinch saddles, but the "
                "uncontained normalized-level argmin selected a lower-level "
                "out-of-vessel saddle. Filtering the typed table by the operator "
                "wall polygon narrows both height bands and flips both wall extrema "
                "from ADMITTED to SHADOWED. The banked CORRECT_ADMISSION control was "
                "therefore admitted only because a spurious out-of-vessel saddle "
                "widened the height band."
            ),
            "zero_private_region_explanation": (
                "At the spuriously low out-of-vessel saddle level, the inward-offset "
                "confined mask has only the magnetic-axis component. With no other "
                "nonzero component label, private_component_count is 0 and its "
                "summed cell area is exactly 0.0 m2."
            ),
        },
    }
    if flip_count != len(ARMS):
        raise RuntimeError("vessel filtering did not shadow both banked wall extrema")
    if zero_private_count != len(ARMS):
        raise RuntimeError("the production saddle level unexpectedly has private flux")
    if any(
        arm["production_selection"]["inside_operator_wall_polygon"]
        for arm in arms.values()
    ):
        raise RuntimeError("a banked production saddle unexpectedly lies in-vessel")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
