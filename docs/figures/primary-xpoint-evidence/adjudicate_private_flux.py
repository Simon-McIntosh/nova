"""Reconstruct terminal separatrices and adjudicate wall-shadow geometry."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.diiid_forward_gs_match import (
    _margin_graded_newton_krylov,
    _terminal_xpoint_diagnostics,
)
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


HERE = Path(__file__).resolve().parent
BANKED = Path("docs/figures/dual-branch-selection/pinned-branch-contrast.json")
OUTPUT_JSON = HERE / "private-flux-shadow-adjudication.json"
OUTPUT_PNG = HERE / "private-flux-shadow-adjudication.png"
TARGETS = ((21978, 35), (21983, 35), (22086, 43))
ARMS = ("pure_arm", "mixed_arm")


def _terminal_states(
    profile, seed: jax.Array, target_current: float
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


def _nearest_masked_cell(
    radius: np.ndarray,
    height: np.ndarray,
    mask: np.ndarray,
    coordinate: np.ndarray,
) -> tuple[int, int, float]:
    distance = np.hypot(
        radius[None, :] - coordinate[0], height[:, None] - coordinate[1]
    )
    candidate = np.where(mask, distance, np.inf)
    flat = int(np.argmin(candidate))
    vertical, radial = np.unravel_index(flat, candidate.shape)
    return vertical, radial, float(candidate[vertical, radial])


def _geometry(profile, state: jax.Array, banked: dict) -> tuple[dict, dict]:
    operator = profile.operator
    physical = jnp.asarray(state)[: operator.physical_node_number]
    grid_flux, _wall_flux = operator.topology.split_flux_map(physical)
    radius, height, shape = operator.connectivity_grid_axes()
    radial_count, vertical_count = shape
    flux = np.asarray(grid_flux).reshape((radial_count, vertical_count)).T
    radius = np.asarray(radius)
    height = np.asarray(height)
    inside = (
        np.asarray(operator.inside_material)
        .reshape((radial_count, vertical_count))
        .T.astype(bool)
    )
    _masks, topology = operator.read(state)
    axis = np.asarray(topology.axis, dtype=float)
    wall = np.asarray(operator.wall.coordinate, dtype=float)

    diagnostics = _terminal_xpoint_diagnostics(profile, state, topology)
    wall_record = diagnostics["wall_operand"]
    selected_x = np.asarray(diagnostics["selected_x_coordinate_m"], dtype=float)
    selected_x_flux = float(diagnostics["selected_x_flux_wb"])
    wall_point = np.asarray(wall_record["coordinate_m"], dtype=float)
    axis_flux = float(
        flux[
            np.argmin(np.abs(height - axis[1])),
            np.argmin(np.abs(radius - axis[0])),
        ]
    )
    outward_sign = np.sign(selected_x_flux - axis_flux) or 1.0
    normalized = outward_sign * (flux - axis_flux)
    boundary_level = outward_sign * (selected_x_flux - axis_flux)
    cell_scale = max(float(np.ptp(normalized)), np.finfo(float).eps)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    sensitivity = []
    selected_masks = None
    for offset_fraction in (5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3):
        inward_offset = max(
            offset_fraction * cell_scale,
            32.0 * np.finfo(float).eps * cell_scale,
        )
        confined_mask = inside & (normalized <= boundary_level - inward_offset)
        labels, component_count = ndimage.label(confined_mask, structure=structure)
        axis_i, axis_j, axis_cell_distance = _nearest_masked_cell(
            radius, height, confined_mask, axis
        )
        axis_label = int(labels[axis_i, axis_j])
        wall_i, wall_j, wall_cell_distance = _nearest_masked_cell(
            radius, height, confined_mask, wall_point
        )
        wall_label = int(labels[wall_i, wall_j])
        wall_in_private_component = bool(wall_label != 0 and wall_label != axis_label)
        wall_in_axis_component = bool(wall_label != 0 and wall_label == axis_label)
        private_mask = (labels != 0) & (labels != axis_label)
        sensitivity.append(
            {
                "offset_fraction_of_flux_range": offset_fraction,
                "offset_wb": inward_offset,
                "wall_in_private_flux_component": wall_in_private_component,
                "wall_to_classified_cell_distance_m": wall_cell_distance,
                "private_component_count": int(len(np.unique(labels[private_mask]))),
            }
        )
        if offset_fraction == 2.0e-4:
            selected_masks = (
                private_mask,
                component_count,
                wall_i,
                wall_j,
                wall_cell_distance,
                axis_cell_distance,
                wall_in_private_component,
                wall_in_axis_component,
                inward_offset,
            )
    assert selected_masks is not None
    (
        private_mask,
        component_count,
        wall_i,
        wall_j,
        wall_cell_distance,
        axis_cell_distance,
        wall_in_private_component,
        wall_in_axis_component,
        inward_offset,
    ) = selected_masks
    private_area = float(
        np.sum(private_mask) * np.mean(np.diff(radius)) * np.mean(np.diff(height))
    )
    typed_heights = np.asarray(
        [
            candidate["coordinate_m"][1]
            for candidate in diagnostics["typed_saddle_candidates"]
        ],
        dtype=float,
    )
    finite_heights = typed_heights[np.isfinite(typed_heights)]
    wall_level = float(wall_record["normalized_flux_before_shadow"])
    selected_x_level = float(diagnostics["selected_x_normalized_flux_operand"])
    wall_inside_selected_flux_level = bool(wall_level < selected_x_level)
    record = {
        "wall_status": wall_record["status"],
        "wall_shadowed": bool(wall_record["shadowed"]),
        "wall_coordinate_m": wall_point.tolist(),
        "wall_normalized_flux_before_shadow": wall_level,
        "selected_x_coordinate_m": selected_x.tolist(),
        "selected_x_flux_wb": selected_x_flux,
        "selected_x_normalized_flux_operand": selected_x_level,
        "wall_inside_selected_flux_level": wall_inside_selected_flux_level,
        "wall_outside_selected_flux_level": not wall_inside_selected_flux_level,
        "selected_x_height_band_m": [
            float(np.min(finite_heights)),
            float(np.max(finite_heights)),
        ],
        "wall_beyond_x_height_band": bool(
            wall_point[1] < np.min(finite_heights)
            or wall_point[1] > np.max(finite_heights)
        ),
        "wall_in_private_flux_component": wall_in_private_component,
        "wall_in_axis_connected_component": wall_in_axis_component,
        "nearest_confined_cell_to_wall_m": [
            float(radius[wall_j]),
            float(height[wall_i]),
        ],
        "wall_to_classified_cell_distance_m": wall_cell_distance,
        "axis_to_classified_cell_distance_m": axis_cell_distance,
        "confined_component_count": int(component_count),
        "private_component_area_m2": private_area,
        "classification_offset_wb": inward_offset,
        "offset_sensitivity": sensitivity,
        "private_flux_verdict_stable_across_offsets": bool(
            len({row["wall_in_private_flux_component"] for row in sensitivity}) == 1
        ),
        "banked_coordinate_exact": bool(
            np.array_equal(
                wall_point, np.asarray(banked["wall_operand"]["coordinate_m"])
            )
            and np.array_equal(
                selected_x, np.asarray(banked["selected_x_coordinate_m"])
            )
        ),
    }
    plot = {
        "radius": radius,
        "height": height,
        "flux": flux,
        "inside": inside,
        "private": private_mask,
        "wall": wall,
        "wall_point": wall_point,
        "axis": axis,
        "selected_x": selected_x,
        "selected_x_flux": selected_x_flux,
    }
    return record, plot


def run() -> dict:
    configure_dtypes()
    banked_receipt = json.loads(BANKED.read_text())
    banked = {
        (int(row["reference"]["shot"]), int(row["reference"]["slice_index"])): row
        for row in banked_receipt["references"]
    }
    response_cache, _evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }

    results: list[dict] = []
    plots: list[dict] = []
    for key in TARGETS:
        selected_row, qualification = selected[key]
        case, context = _mast_case_from_selection(
            SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        if policy["section_kernel_evaluations_this_shot"] != 0:
            raise RuntimeError(
                "geometry reconstruction entered the direct response builder"
            )
        seed = jnp.asarray(passive_case["state"])
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        states = _terminal_states(profile, seed, target_current)
        for arm_name in ARMS:
            record, plot = _geometry(
                profile,
                states[arm_name],
                banked[key][arm_name]["terminal_xpoint_diagnostics"],
            )
            record.update(
                {
                    "shot": key[0],
                    "slice_index": key[1],
                    "arm": arm_name,
                    "verdict": (
                        "CORRECT_PRIVATE_FLUX_REJECTION"
                        if record["wall_shadowed"]
                        and record["wall_in_private_flux_component"]
                        else "CORRECT_ADMISSION"
                        if not record["wall_shadowed"]
                        and record["wall_outside_selected_flux_level"]
                        else "GEOMETRY_PREDICATE_DISAGREEMENT"
                    ),
                }
            )
            plot.update(record)
            results.append(record)
            plots.append(plot)

    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.4), constrained_layout=True)
    for axis_plot, plot in zip(axes.flat, plots, strict=True):
        radius = plot["radius"]
        height = plot["height"]
        axis_plot.contourf(
            radius,
            height,
            plot["private"].astype(float),
            levels=[0.5, 1.5],
            colors=["#d8b365"],
            alpha=0.55,
        )
        axis_plot.contour(
            radius,
            height,
            plot["flux"],
            levels=[plot["selected_x_flux"]],
            colors=["#1f5a85"],
            linewidths=1.8,
        )
        axis_plot.plot(plot["wall"][:, 0], plot["wall"][:, 1], color="0.35", lw=1.0)
        axis_plot.scatter(*plot["axis"], marker="o", s=24, color="#222222", zorder=5)
        axis_plot.scatter(
            *plot["selected_x"],
            marker="x",
            s=58,
            linewidths=2.0,
            color="#1f5a85",
            zorder=6,
        )
        wall_color = "#b2182b" if plot["wall_shadowed"] else "#2f7d32"
        axis_plot.scatter(
            *plot["wall_point"], marker="D", s=36, color=wall_color, zorder=7
        )
        arm_label = "pure" if plot["arm"] == "pure_arm" else "mixed"
        verdict_label = (
            "rejected: private flux" if plot["wall_shadowed"] else "admitted"
        )
        axis_plot.set_title(
            f"{plot['shot']}/{plot['slice_index']} {arm_label}\n{verdict_label}",
            fontsize=9.5,
        )
        axis_plot.set_aspect("equal", adjustable="box")
        axis_plot.set_xlim(0.18, 1.75)
        axis_plot.set_ylim(-2.05, 2.05)
        axis_plot.set_xlabel("R [m]")
        axis_plot.set_ylabel("Z [m]")
    fig.suptitle(
        "MAST wall extrema against the terminal separatrix\n"
        "ochre: private-flux component · blue: selected-X separatrix · "
        "diamond: wall extremum",
        fontsize=11,
    )
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    payload = {
        "artifact": "geometric adjudication of the private-flux wall shadow",
        "project_absolute_src": (
            "/nova/figures/primary-xpoint-evidence/private-flux-shadow-adjudication.png"
        ),
        "method": (
            "At a flux level displaced inward from the selected-X separatrix by "
            "2e-4 of the terminal flux range, 4-connected in-vessel cells are "
            "labelled. A wall extremum is private flux iff its nearest confined "
            "cell belongs to a component disconnected from the magnetic-axis component."
        ),
        "terminal_count": len(results),
        "rejected_terminal_count": sum(row["wall_shadowed"] for row in results),
        "private_flux_supported_rejection_count": sum(
            row["verdict"] == "CORRECT_PRIVATE_FLUX_REJECTION" for row in results
        ),
        "admitted_terminal_count": sum(not row["wall_shadowed"] for row in results),
        "admission_outside_private_flux_count": sum(
            row["verdict"] == "CORRECT_ADMISSION" for row in results
        ),
        "terminals": results,
        "selection_fallback": (
            "The exact margin is undefined when the only confined-most wall extremum "
            "is private flux. Dual-branch selection therefore needs a total "
            "observable: "
            "retain exact margin where finite and use the already-defined always-fork "
            "policy on these shadow-rejected frames."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
