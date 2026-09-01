"""Test whether a discrete topology switch drives a Newton period-two cycle.

The benchmark replays the banked unpenalised merit-ranked trajectory for the
cycling and converged MAST references, retains the final promoted states, and
compares every discrete topology operand named by the diagnosis contract.  It
does not alter the solver or production topology implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.contraction_discriminator import (
    BANKED_CONTRAST,
    REPRODUCTION_ABSOLUTE_TOLERANCE,
    TARGET_REFERENCES,
    _banked_rows,
    _relative_sup,
    _replay_iteration,
)
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import (
    _persisted_response_cache,
    _source_revision,
)
from nova.equilibrium import fixed_point as fixed_point_solver
from nova.equilibrium.connectivity_boundary import (
    _flood_fill,
    traced_boundary_read,
    traced_margin_candidate_diagnostics,
)
from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.topology import TopologyClass
from nova.geometry import select
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    HERE / "docs/figures/discrete-operator-analytic-error/newton-cycle-mask-switch.json"
)
PARITY_STATE_INDICES = (-2, -1)
CONTROL_STATE_INDICES = (-4, -3, -2, -1)


def _sha256(path: Path) -> str:
    """Return the content identity of one evidence input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    """Return a stable identity for one fixed-shape array."""

    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _finite_list(values: Any) -> list[float | None]:
    """Serialize a numeric vector without non-JSON floating sentinels."""

    return [float(value) if np.isfinite(value) else None for value in np.ravel(values)]


def _mask_record(mask: Any, radius: np.ndarray, height: np.ndarray) -> dict[str, Any]:
    """Serialize one Boolean cell mask compactly and reproducibly."""

    values = np.asarray(mask, dtype=bool)
    indices = np.argwhere(values)
    return {
        "shape_vertical_radial": [int(value) for value in values.shape],
        "true_cell_count": int(np.count_nonzero(values)),
        "sha256": _array_sha256(values),
        "true_cell_bounds": (
            None
            if indices.size == 0
            else {
                "radial_index": [int(indices[:, 1].min()), int(indices[:, 1].max())],
                "vertical_index": [
                    int(indices[:, 0].min()),
                    int(indices[:, 0].max()),
                ],
                "radius_m": [
                    float(radius[indices[:, 1].min()]),
                    float(radius[indices[:, 1].max()]),
                ],
                "height_m": [
                    float(height[indices[:, 0].min()]),
                    float(height[indices[:, 0].max()]),
                ],
            }
        ),
    }


def _mask_difference(
    left: Any, right: Any, radius: np.ndarray, height: np.ndarray
) -> dict[str, Any]:
    """Report a cell-by-cell Boolean-mask comparison."""

    first = np.asarray(left, dtype=bool)
    second = np.asarray(right, dtype=bool)
    changed = np.argwhere(first != second)
    return {
        "identical": bool(changed.size == 0),
        "differing_cell_count": int(changed.shape[0]),
        "differing_cells": [
            {
                "vertical_index": int(vertical),
                "radial_index": int(radial),
                "radius_m": float(radius[radial]),
                "height_m": float(height[vertical]),
                "first": bool(first[vertical, radial]),
                "second": bool(second[vertical, radial]),
            }
            for vertical, radial in changed
        ],
    }


def _topology_snapshot(
    profile, state: Any
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read every discrete topology operand for one promoted flux state."""

    operator = profile.operator
    physical = jnp.asarray(state)[: operator.physical_node_number]
    grid_flux, wall_flux = operator.topology.split_flux_map(physical)
    grid_o, grid_x = operator._fixed_design_topology.grid(grid_flux)
    axis_data = operator._fixed_design_topology.o_point_data(grid_o, operator.polarity)
    wall_data = operator._fixed_design_topology.wall(wall_flux, operator.polarity)
    radius_device, height_device, connectivity_shape = operator.connectivity_grid_axes()
    radius = np.asarray(radius_device, dtype=np.float64)
    height = np.asarray(height_device, dtype=np.float64)
    radial_count, vertical_count = connectivity_shape
    flux_2d = jnp.asarray(grid_flux).reshape((radial_count, vertical_count)).T
    inside_2d = (
        jnp.asarray(operator.inside_material).reshape((radial_count, vertical_count)).T
    )
    classification_wall = jnp.concatenate((wall_data[:2], wall_data[2:3]))
    wall_cluster_index, wall_cluster_roll = select.traced_wall_index(
        operator.polarity * wall_flux
    )

    reading = traced_boundary_read(
        flux_2d,
        radius_device,
        height_device,
        inside_2d,
        axis_data[0],
        axis_data[1],
        96,
        18,
        2,
        jnp.empty((0,), dtype=radius_device.dtype),
        jnp.asarray(1.0, dtype=grid_flux.dtype),
        operator.wall.coordinate[:, 0],
        operator.wall.coordinate[:, 1],
        wall_flux,
        classification_x=grid_x,
        classification_wall=classification_wall,
    )
    candidate = traced_margin_candidate_diagnostics(
        flux_2d,
        radius_device,
        height_device,
        inside_2d,
        axis_data[0],
        axis_data[1],
        96,
        18,
        operator.wall.coordinate[:, 0],
        operator.wall.coordinate[:, 1],
        wall_flux,
        grid_x,
        classification_wall,
    )

    psi_axis = reading["psi_axis"]
    psi_out = reading["psi_out"]
    span = psi_out - psi_axis
    span_safe = jnp.where(jnp.abs(span) > 0.0, span, jnp.asarray(1.0, span.dtype))
    normalized = (flux_2d - psi_axis) / span_safe
    confined = (normalized <= reading["s_star"]) & inside_2d
    axis_vertical = jnp.argmin(jnp.abs(height_device - axis_data[1]))
    axis_radial = jnp.argmin(jnp.abs(radius_device - axis_data[0]))
    seed = jnp.zeros_like(confined, dtype=bool).at[axis_vertical, axis_radial].set(True)
    core = (
        _flood_fill(
            confined,
            seed,
            int(radial_count + vertical_count),
            True,
        )
        > 0.5
    )

    emergent_masks, emergent = operator.read(state)
    pinned_masks, pinned = operator.read(state, requested_class=TopologyClass.DIVERTED)
    selected_index = int(candidate["selected_typed_candidate_index"])
    selected_present = bool(candidate["selected_typed_candidate_present"])
    selected_candidate = np.asarray(
        candidate["selected_typed_candidate"], dtype=np.float64
    )
    wall_candidate = np.asarray(candidate["wall_candidate"], dtype=np.float64)
    labels = np.asarray(pinned_masks.label)
    masks = {
        "connectivity_confined": np.asarray(confined, dtype=bool),
        "connectivity_core": np.asarray(core, dtype=bool),
        "production_core": (
            labels.reshape((radial_count, vertical_count)).T == int(PlasmaDomain.CORE)
        ),
        "production_private_flux": (
            labels.reshape((radial_count, vertical_count)).T
            == int(PlasmaDomain.PRIVATE_FLUX)
        ),
        "production_open_field_line": np.asarray(pinned_masks.open_field_line)
        .reshape((radial_count, vertical_count))
        .T,
    }
    record = {
        "state_sha256": _array_sha256(np.asarray(state, dtype=np.float64)),
        "connectivity_boundary": {
            "found": bool(reading["found"]),
            "s_star": float(reading["s_star"]),
            "s_flood": float(reading["s_flood"]),
            "psi_bnd_wb": float(reading["psi_bnd"]),
            "n_core_cells_reported": int(reading["n_core_cells"]),
            "selected_typed_saddle_index": selected_index,
            "selected_typed_saddle_present": selected_present,
            "selected_typed_saddle_r_z_psi_type": _finite_list(selected_candidate),
            "typed_saddle_candidate_count": int(candidate["typed_candidate_count"]),
            "selected_wall_extremum_r_z_psi": _finite_list(wall_candidate),
            "selected_wall_cluster_index": int(wall_cluster_index),
            "selected_wall_cluster_roll": int(wall_cluster_roll),
            "wall_extremum_present": bool(candidate["wall_candidate_present"]),
            "private_flux_shadowed": bool(candidate["wall_shadowed"]),
            "wall_normalized_flux_before_shadow": float(
                candidate["wall_normalized_flux_operand_before_shadow"]
            ),
            "wall_normalized_flux_after_shadow": (
                float(candidate["wall_normalized_flux_operand"])
                if np.isfinite(float(candidate["wall_normalized_flux_operand"]))
                else None
            ),
            "selected_x_normalized_flux": (
                float(candidate["selected_x_normalized_flux_operand"])
                if np.isfinite(float(candidate["selected_x_normalized_flux_operand"]))
                else None
            ),
        },
        "achieved_topology_class": (
            "diverted" if bool(emergent.diverted) else "limited"
        ),
        "pinned_residual_topology_class": (
            "diverted" if bool(pinned.diverted) else "limited"
        ),
        "production_boundary": {
            "emergent_axis_r_z": _finite_list(emergent.axis),
            "emergent_axis_flux_wb": float(emergent.axis_flux),
            "emergent_boundary_r_z": _finite_list(emergent.boundary),
            "emergent_boundary_flux_wb": float(emergent.boundary_flux),
            "emergent_x_point_r_z": _finite_list(emergent.x_point),
            "emergent_x_point_flux_wb": float(emergent.x_point_flux),
            "emergent_wall_point_r_z": _finite_list(emergent.wall_point),
            "emergent_wall_point_flux_wb": float(emergent.wall_point_flux),
            "pinned_boundary_r_z": _finite_list(pinned.boundary),
            "pinned_boundary_flux_wb": float(pinned.boundary_flux),
        },
        "masks": {
            name: _mask_record(mask, radius, height) for name, mask in masks.items()
        },
        "emergent_domain_cell_counts": {
            domain.name.lower(): int(
                np.count_nonzero(np.asarray(emergent_masks.label) == int(domain))
            )
            for domain in PlasmaDomain
        },
    }
    return record, {**masks, "radius": radius, "height": height}


def _compare_snapshots(
    first: dict[str, Any],
    second: dict[str, Any],
    first_masks: dict[str, np.ndarray],
    second_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare discrete topology selections and masks between two states."""

    radius = first_masks["radius"]
    height = first_masks["height"]
    mask_names = sorted(set(first_masks) - {"radius", "height"})
    mask_differences = {
        name: _mask_difference(first_masks[name], second_masks[name], radius, height)
        for name in mask_names
    }
    first_boundary = first["connectivity_boundary"]
    second_boundary = second["connectivity_boundary"]
    first_saddle = np.asarray(
        first_boundary["selected_typed_saddle_r_z_psi_type"], dtype=np.float64
    )
    second_saddle = np.asarray(
        second_boundary["selected_typed_saddle_r_z_psi_type"], dtype=np.float64
    )
    first_wall = np.asarray(
        first_boundary["selected_wall_extremum_r_z_psi"], dtype=np.float64
    )
    second_wall = np.asarray(
        second_boundary["selected_wall_extremum_r_z_psi"], dtype=np.float64
    )
    discrete_selections = {
        "selected_typed_saddle_index_changes": (
            first_boundary["selected_typed_saddle_index"]
            != second_boundary["selected_typed_saddle_index"]
        ),
        "selected_typed_saddle_presence_changes": (
            first_boundary["selected_typed_saddle_present"]
            != second_boundary["selected_typed_saddle_present"]
        ),
        "selected_wall_extremum_cluster_changes": (
            first_boundary["selected_wall_cluster_index"]
            != second_boundary["selected_wall_cluster_index"]
            or first_boundary["selected_wall_cluster_roll"]
            != second_boundary["selected_wall_cluster_roll"]
        ),
        "private_flux_shadow_verdict_flips": (
            first_boundary["private_flux_shadowed"]
            != second_boundary["private_flux_shadowed"]
        ),
        "achieved_topology_class_changes": (
            first["achieved_topology_class"] != second["achieved_topology_class"]
        ),
        "pinned_residual_topology_class_changes": (
            first["pinned_residual_topology_class"]
            != second["pinned_residual_topology_class"]
        ),
    }
    masks_identical = all(item["identical"] for item in mask_differences.values())
    selections_identical = not any(discrete_selections.values())
    return {
        "state_relative_sup_difference": None,
        "mask_differences": mask_differences,
        "binding_level_absolute_difference": abs(
            first_boundary["s_star"] - second_boundary["s_star"]
        ),
        "boundary_flux_absolute_difference_wb": abs(
            first_boundary["psi_bnd_wb"] - second_boundary["psi_bnd_wb"]
        ),
        "discrete_selections": discrete_selections,
        "continuous_selected_operand_changes": {
            "typed_saddle_maximum_absolute_change": float(
                np.nanmax(np.abs(first_saddle - second_saddle))
            ),
            "wall_extremum_maximum_absolute_change": float(
                np.nanmax(np.abs(first_wall - second_wall))
            ),
        },
        "all_masks_identical": masks_identical,
        "all_discrete_selections_identical": selections_identical,
        "any_discrete_residual_definition_switch": bool(
            not masks_identical or not selections_identical
        ),
    }


def run(
    *,
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
    banked_contrast: Path = BANKED_CONTRAST,
    output: Path = DEFAULT_OUTPUT,
    carrier: Path = response_carrier.DEFAULT_CARRIER,
    carrier_receipt: Path = response_carrier.DEFAULT_RECEIPT,
) -> dict[str, Any]:
    """Replay retained states and bank their discrete topology comparison."""

    configure_dtypes()
    banked = _banked_rows(banked_contrast)
    response_cache, carrier_evidence = _persisted_response_cache(
        carrier, carrier_receipt
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(bank)
    }
    records = []
    for key in TARGET_REFERENCES:
        selected_row, qualification = selected[key]
        case, context = _mast_case_from_selection(store, selected_row, qualification)
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        if int(policy["section_kernel_evaluations_this_shot"]) != 0:
            raise RuntimeError("replay entered the direct response builder")
        seed = jnp.asarray(passive_case["state"])
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        mapped = profile.flux_map(
            requested_class=TopologyClass.DIVERTED,
            target_current=target_current,
        )
        replay = _replay_iteration(mapped, profile.operator.topology_margin, seed)
        replay.state.block_until_ready()
        measured = np.asarray(replay.residuals, dtype=np.float64)
        expected = np.asarray(
            banked[key]["mixed_arm"]["residual_sequence"], dtype=np.float64
        )
        terminal = float(
            fixed_point_solver._relative_residual(mapped(replay.state), replay.state)
        )
        maximum_difference = float(
            max(
                np.max(np.abs(measured - expected)),
                abs(terminal - float(banked[key]["mixed_arm"]["terminal_residual"])),
            )
        )
        if maximum_difference > REPRODUCTION_ABSOLUTE_TOLERANCE:
            raise AssertionError(
                f"{key} did not reproduce the banked trajectory: {maximum_difference}"
            )

        state_indices = (
            PARITY_STATE_INDICES
            if key == TARGET_REFERENCES[0]
            else CONTROL_STATE_INDICES
        )
        snapshots = []
        masks = []
        for index in state_indices:
            snapshot, state_masks = _topology_snapshot(profile, replay.states[index])
            snapshot["promotion_state_index"] = int(replay.states.shape[0] + index)
            snapshots.append(snapshot)
            masks.append(state_masks)
        comparisons = []
        for index in range(len(snapshots) - 1):
            comparison = _compare_snapshots(
                snapshots[index], snapshots[index + 1], masks[index], masks[index + 1]
            )
            comparison["first_promotion_state_index"] = snapshots[index][
                "promotion_state_index"
            ]
            comparison["second_promotion_state_index"] = snapshots[index + 1][
                "promotion_state_index"
            ]
            comparison["state_relative_sup_difference"] = _relative_sup(
                replay.states[state_indices[index]],
                replay.states[state_indices[index + 1]],
            )
            comparisons.append(comparison)
        records.append(
            {
                "reference": {"shot": key[0], "slice_index": key[1]},
                "banked_reproduction": {
                    "passes": True,
                    "absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
                    "maximum_residual_absolute_difference": maximum_difference,
                    "banked_tail": expected[-4:].tolist(),
                    "replayed_tail": measured[-4:].tolist(),
                },
                "snapshots": snapshots,
                "adjacent_final_promotion_comparisons": comparisons,
                "final_state_repetition": {
                    "same_parity_relative_sup": [
                        _relative_sup(replay.states[-1], replay.states[-3]),
                        _relative_sup(replay.states[-2], replay.states[-4]),
                    ],
                    "maximum_same_parity_relative_sup": max(
                        _relative_sup(replay.states[-1], replay.states[-3]),
                        _relative_sup(replay.states[-2], replay.states[-4]),
                    ),
                    "opposite_parity_relative_sup": _relative_sup(
                        replay.states[-1], replay.states[-2]
                    ),
                },
            }
        )

    cycling = records[0]
    control = records[1]
    cycling_comparison = cycling["adjacent_final_promotion_comparisons"][0]
    control_stable = all(
        comparison["all_masks_identical"]
        and comparison["all_discrete_selections_identical"]
        for comparison in control["adjacent_final_promotion_comparisons"]
    )
    supports = bool(cycling_comparison["any_discrete_residual_definition_switch"])
    receipt = {
        "artifact": "discrete topology operands across a Newton period-two cycle",
        "source_commit": _source_revision(),
        "driver_sha256": _sha256(Path(__file__)),
        "evidence_inputs": {
            "banked_contrast": str(banked_contrast.relative_to(HERE)),
            "banked_contrast_sha256": _sha256(banked_contrast),
            "response_carrier": carrier_evidence,
        },
        "measurement_contract": {
            "cycling_reference": list(TARGET_REFERENCES[0]),
            "control_reference": list(TARGET_REFERENCES[1]),
            "state_source": (
                "retained states from one exact replay of the banked twelve-promotion "
                "unpenalised merit-ranked trajectory"
            ),
            "cycling_states": "last two promoted states, one from each parity",
            "control_states": "last four promoted states",
            "mask_comparison": (
                "cell-by-cell on the connectivity confined and flood-filled core "
                "masks and on every production domain mask"
            ),
        },
        "references": records,
        "verdict": {
            "name": (
                "discrete_residual_definition_switch_supports_newton_cycle"
                if supports
                else "discrete_mask_and_selection_switch_hypothesis_refuted"
            ),
            "supports_discrete_mask_switch_mechanism": supports,
            "cycling_all_masks_identical": cycling_comparison["all_masks_identical"],
            "cycling_all_discrete_selections_identical": cycling_comparison[
                "all_discrete_selections_identical"
            ],
            "cycling_binding_level_absolute_difference": cycling_comparison[
                "binding_level_absolute_difference"
            ],
            "cycling_boundary_flux_absolute_difference_wb": cycling_comparison[
                "boundary_flux_absolute_difference_wb"
            ],
            "control_masks_and_selections_stable_across_final_promotions": (
                control_stable
            ),
            "interpretation": (
                "the exact Newton linearisation holds a discrete residual definition "
                "fixed while the full step crosses its switching surface"
                if supports
                else (
                    "the cycling states use identical masks and discrete selections; "
                    "the cycle is smooth in the residual definition and its mechanism "
                    "must lie in the continuously varying residual terms"
                )
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    """Run the discrete topology switch diagnosis from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--bank", type=Path, default=DECOMPOSITION_BANK)
    parser.add_argument("--banked-contrast", type=Path, default=BANKED_CONTRAST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    parser.add_argument(
        "--carrier-receipt", type=Path, default=response_carrier.DEFAULT_RECEIPT
    )
    arguments = parser.parse_args()
    result = run(
        store=arguments.store,
        bank=arguments.bank,
        banked_contrast=arguments.banked_contrast,
        output=arguments.output,
        carrier=arguments.carrier,
        carrier_receipt=arguments.carrier_receipt,
    )
    print(json.dumps(result["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
