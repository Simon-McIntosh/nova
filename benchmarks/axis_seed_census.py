"""Census continuous-axis ownership and flood occupancy on frozen MAST arms.

The production topology grid is centre-first: one coordinate owns each plasma
cell. This driver reconstructs both frozen solve arms, checks the continuous
axis against an independent control-cell containment test, and compares the
connectivity flood with the stored material mask versus the mask that admits
exactly the axis-owning cell. The 21985/51 mixed arm additionally retains every
promoted Newton state so an empty terminal cannot hide an earlier transition.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.contraction_discriminator import _replay_iteration
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from benchmarks.out_of_vessel_saddle_selection import _terminal_states
from nova.equilibrium.connectivity_boundary import _flood_fill, traced_boundary_read
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "docs/figures/profile-owned-solve-domain"
OUTPUT_JSON = OUTPUT_DIRECTORY / "axis-seed-census.json"
OUTPUT_FIGURE = OUTPUT_DIRECTORY / "axis-seed-census.png"
TARGET_REFERENCE = (21985, 51)
ARM_NAMES = ("pure_arm", "mixed_arm")


def _finite(value: Any) -> float | None:
    """Return a strict-JSON scalar or null for a non-finite value."""
    scalar = float(value)
    return scalar if np.isfinite(scalar) else None


def _sha256(path: Path) -> str:
    """Return the stable byte identity of one evidence input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _control_cell_contains(
    radius: np.ndarray,
    height: np.ndarray,
    radial_index: int,
    vertical_index: int,
    axis: np.ndarray,
) -> bool:
    """Check axis containment against independent midpoint cell boundaries."""

    def bounds(values: np.ndarray, index: int) -> tuple[float, float]:
        lower = (
            -np.inf
            if index == 0
            else 0.5 * (float(values[index - 1]) + float(values[index]))
        )
        upper = (
            np.inf
            if index + 1 == values.size
            else 0.5 * (float(values[index]) + float(values[index + 1]))
        )
        return lower, upper

    radial_bounds = bounds(radius, radial_index)
    vertical_bounds = bounds(height, vertical_index)
    return bool(
        radial_bounds[0] <= axis[0] <= radial_bounds[1]
        and vertical_bounds[0] <= axis[1] <= vertical_bounds[1]
    )


def _connectivity_snapshot(
    profile: Any,
    state: jax.Array,
    *,
    admit_axis_cell: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read one state with or without the axis-owning material admission."""
    operator = profile.operator
    physical = jnp.asarray(state)[: operator.physical_node_number]
    grid_flux, wall_flux = operator.topology.split_flux_map(physical)
    _masks, topology, _connected = (
        operator._fixed_design_topology.read_with_connectivity(
            physical,
            operator.polarity,
            operator.inside_material,
            TopologyClass.DIVERTED,
        )
    )
    seed, repaired_material = operator.connectivity_axis_seed(topology.axis)
    material = repaired_material if admit_axis_cell else operator.inside_material
    radius_device, height_device, shape = operator.connectivity_grid_axes()
    radius = np.asarray(radius_device, dtype=np.float64)
    height = np.asarray(height_device, dtype=np.float64)
    radial_count, vertical_count = shape
    flux = grid_flux.reshape((radial_count, vertical_count)).T
    material_2d = material.reshape((radial_count, vertical_count)).T
    _vmap_o, vmap_x = operator._fixed_design_topology.grid(grid_flux)
    classification_wall = jnp.concatenate(
        (topology.wall_point, topology.wall_point_flux[None])
    )
    reading = traced_boundary_read(
        flux,
        radius_device,
        height_device,
        material_2d,
        topology.axis[0],
        topology.axis[1],
        96,
        18,
        2,
        jnp.empty((0,), dtype=radius_device.dtype),
        jnp.asarray(1.0, dtype=grid_flux.dtype),
        operator.wall.coordinate[:, 0],
        operator.wall.coordinate[:, 1],
        wall_flux,
        classification_x=vmap_x,
        classification_wall=classification_wall,
    )

    seed_host = np.asarray(seed, dtype=bool)
    seed_flat = int(np.flatnonzero(seed_host)[0])
    radial_index, vertical_index = np.unravel_index(
        seed_flat, (radial_count, vertical_count)
    )
    axis = np.asarray(topology.axis, dtype=np.float64)
    seed_coordinate = np.asarray(operator.grid.coordinate[seed_flat], dtype=np.float64)
    original_material = np.asarray(operator.inside_material, dtype=bool)
    span = reading["psi_out"] - reading["psi_axis"]
    safe_span = jnp.where(jnp.abs(span) > 0.0, span, jnp.asarray(1.0, span.dtype))
    normalized = (flux - reading["psi_axis"]) / safe_span
    confined = (normalized <= reading["s_star"]) & material_2d
    seed_2d = seed.reshape((radial_count, vertical_count)).T
    core = (
        _flood_fill(
            confined,
            seed_2d,
            radial_count + vertical_count,
            True,
        )
        > 0.5
    )
    record = {
        "axis_coordinate_m": axis.tolist(),
        "axis_cell_coordinate_m": seed_coordinate.tolist(),
        "axis_to_cell_centre_distance_m": float(np.linalg.norm(seed_coordinate - axis)),
        "axis_cell_flat_index": seed_flat,
        "axis_cell_radial_index": int(radial_index),
        "axis_cell_vertical_index": int(vertical_index),
        "cell_contains_continuous_axis": _control_cell_contains(
            radius,
            height,
            radial_index,
            vertical_index,
            axis,
        ),
        "axis_cell_in_original_material": bool(original_material[seed_flat]),
        "axis_cell_in_effective_material": bool(np.asarray(material)[seed_flat]),
        "material_cells_admitted": int(
            np.count_nonzero(np.asarray(material, dtype=bool) & ~original_material)
        ),
        "confined_component_cell_count": int(reading["n_core_cells"]),
        "boundary_found": bool(reading["found"]),
        "binding_level": _finite(reading["s_star"]),
        "class_margin": _finite(reading["class_margin"]),
    }
    arrays = {
        "radius": radius,
        "height": height,
        "original_material": original_material.reshape(
            (radial_count, vertical_count)
        ).T,
        "effective_material": np.asarray(material_2d, dtype=bool),
        "core": np.asarray(core, dtype=bool),
        "axis": axis,
        "seed_coordinate": seed_coordinate,
    }
    return record, arrays


def _arm_census(profile: Any, state: jax.Array) -> tuple[dict[str, Any], dict]:
    """Compare original and repaired connectivity for one frozen arm."""
    original, _original_arrays = _connectivity_snapshot(
        profile, state, admit_axis_cell=False
    )
    repaired, repaired_arrays = _connectivity_snapshot(
        profile, state, admit_axis_cell=True
    )
    return {
        "cell_contains_continuous_axis": repaired["cell_contains_continuous_axis"],
        "axis_coordinate_m": repaired["axis_coordinate_m"],
        "axis_cell_coordinate_m": repaired["axis_cell_coordinate_m"],
        "axis_to_cell_centre_distance_m": repaired["axis_to_cell_centre_distance_m"],
        "axis_cell_in_original_material": original["axis_cell_in_original_material"],
        "material_cells_admitted": repaired["material_cells_admitted"],
        "original_confined_component_cell_count": original[
            "confined_component_cell_count"
        ],
        "repaired_confined_component_cell_count": repaired[
            "confined_component_cell_count"
        ],
        "original_boundary_found": original["boundary_found"],
        "repaired_boundary_found": repaired["boundary_found"],
        "repaired_class_margin": repaired["class_margin"],
    }, repaired_arrays


def _earlier_window(
    profile: Any,
    seed: jax.Array,
    pure_terminal: jax.Array,
    target_current: float,
) -> dict:
    """Retain the 21985/51 pre-terminal window and count recovered floods."""
    mapped = profile.flux_map(
        requested_class=TopologyClass.DIVERTED,
        target_current=target_current,
    )
    replay = _replay_iteration(mapped, profile.operator.topology_margin, seed)
    replay.state.block_until_ready()

    mixed_rows = []
    for index, state in enumerate(replay.states):
        row, _arrays = _arm_census(profile, state)
        mixed_rows.append(
            {
                "promotion_index": index,
                "original_confined_component_cell_count": row[
                    "original_confined_component_cell_count"
                ],
                "repaired_confined_component_cell_count": row[
                    "repaired_confined_component_cell_count"
                ],
                "axis_cell_in_original_material": row["axis_cell_in_original_material"],
                "cell_contains_continuous_axis": row["cell_contains_continuous_axis"],
            }
        )

    pure_rows = []
    for name, state in (("initial", seed), ("terminal", pure_terminal)):
        row, _arrays = _arm_census(profile, state)
        pure_rows.append(
            {
                "window_position": name,
                "original_confined_component_cell_count": row[
                    "original_confined_component_cell_count"
                ],
                "repaired_confined_component_cell_count": row[
                    "repaired_confined_component_cell_count"
                ],
                "axis_cell_in_original_material": row["axis_cell_in_original_material"],
                "cell_contains_continuous_axis": row["cell_contains_continuous_axis"],
            }
        )

    all_rows = mixed_rows + pure_rows
    recovered = [
        row
        for row in all_rows
        if row["original_confined_component_cell_count"] == 0
        and row["repaired_confined_component_cell_count"] > 0
    ]
    return {
        "mixed_promotions": mixed_rows,
        "pure_endpoints": pure_rows,
        "states_compared": len(all_rows),
        "empty_original_to_nonempty_repaired_count": len(recovered),
        "all_repaired_components_nonempty": all(
            row["repaired_confined_component_cell_count"] > 0 for row in all_rows
        ),
        "attribution": (
            "The original centre-derived material mask suppresses the continuous-axis "
            "cell on the recovered states; admitting that one owning cell changes an "
            "empty flood into a non-empty axis component. This attributes the empty-"
            "component mechanism, not convergence of the later residual-domain solve."
        ),
    }


def _draw_target_arms(target_arrays: dict[str, dict[str, np.ndarray]], path: Path):
    """Draw material ownership and the recovered axis components."""
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), constrained_layout=True)
    colour = ListedColormap(["#d9d9d9", "#9ecae1", "#2171b5"])
    for axis_plot, arm in zip(axes, ARM_NAMES, strict=True):
        arrays = target_arrays[arm]
        code = arrays["effective_material"].astype(int)
        code[arrays["core"]] = 2
        axis_plot.pcolormesh(
            arrays["radius"],
            arrays["height"],
            code,
            shading="nearest",
            cmap=colour,
            vmin=0,
            vmax=2,
            rasterized=True,
        )
        axis_plot.scatter(
            *arrays["axis"], marker="x", s=52, linewidths=1.8, color="#b2182b"
        )
        axis_plot.scatter(
            *arrays["seed_coordinate"],
            marker="o",
            s=34,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.2,
        )
        axis_plot.set_title(arm.replace("_", " "))
        axis_plot.set_xlabel("R [m]")
        axis_plot.set_ylabel("Z [m]")
        axis_plot.set_aspect("equal", adjustable="box")
        axis_plot.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "MAST 21985/51: continuous axis (red) and admitted owning cell (ring)"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(output_json: Path = OUTPUT_JSON, output_figure: Path = OUTPUT_FIGURE) -> dict:
    """Run the frozen twelve-arm seed census and write its evidence artifacts."""
    configure_dtypes()
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER,
        response_carrier.DEFAULT_RECEIPT,
    )
    selected = select_slices_by_shot(DECOMPOSITION_BANK)
    rows = []
    target_arrays: dict[str, dict[str, np.ndarray]] = {}
    earlier_window = None
    for selected_row, qualification in selected:
        shot = int(selected_row["shot"])
        slice_index = int(selected_row["slice_index"])
        identity = f"{shot}/{slice_index}"
        print(f"reconstructing MAST {identity}", flush=True)
        case, context = _mast_case_from_selection(
            SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        if int(policy["section_kernel_evaluations_this_shot"]) != 0:
            raise RuntimeError("seed census entered the direct response builder")
        seed = jnp.asarray(passive_case["state"])
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        states = _terminal_states(profile, seed, target_current)
        for arm in ARM_NAMES:
            record, arrays = _arm_census(profile, states[arm])
            rows.append(
                {
                    "shot": shot,
                    "slice_index": slice_index,
                    "identity": identity,
                    "arm": arm,
                    **record,
                }
            )
            if (shot, slice_index) == TARGET_REFERENCE:
                target_arrays[arm] = arrays
        if (shot, slice_index) == TARGET_REFERENCE:
            earlier_window = _earlier_window(
                profile, seed, states["pure_arm"], target_current
            )
        del states, profile, passive_case, case, context
        jax.clear_caches()
        gc.collect()

    containing = sum(row["cell_contains_continuous_axis"] for row in rows)
    target_rows = [
        row for row in rows if (row["shot"], row["slice_index"]) == TARGET_REFERENCE
    ]
    target_nonempty = sum(
        row["repaired_confined_component_cell_count"] > 0 for row in target_rows
    )
    if len(rows) != 12 or containing != 12:
        raise AssertionError(f"axis-owning cells passed {containing} of {len(rows)}")
    if len(target_rows) != 2 or target_nonempty != 2:
        raise AssertionError(
            f"21985/51 non-empty repaired components {target_nonempty} of 2"
        )
    if earlier_window is None:
        raise AssertionError("21985/51 earlier-window census was not produced")
    if not earlier_window["all_repaired_components_nonempty"]:
        raise AssertionError("a repaired 21985/51 earlier-window component is empty")
    if earlier_window["empty_original_to_nonempty_repaired_count"] == 0:
        raise AssertionError(
            "the earlier window contains no attributable seed recovery"
        )

    payload = {
        "artifact": "continuous magnetic-axis to plasma-cell seed census",
        "driver_sha256": _sha256(Path(__file__)),
        "evidence_inputs": {
            "decomposition_bank": str(DECOMPOSITION_BANK),
            "decomposition_bank_sha256": _sha256(DECOMPOSITION_BANK),
            "response_carrier": carrier_evidence,
        },
        "measurement_contract": {
            "cell_ownership": (
                "production nearest-centre seed independently checked against "
                "tensor control-cell midpoint bounds"
            ),
            "material_repair": (
                "logical OR of the immutable material mask with exactly the "
                "continuous-axis owning cell"
            ),
            "component": (
                "production traced_boundary_read axis flood on the saddle-aware "
                "hex partition"
            ),
        },
        "rows": rows,
        "earlier_window_21985_51": earlier_window,
        "headline": {
            "axis_cells_containing_continuous_axis": containing,
            "frozen_arm_count": len(rows),
            "target_nonempty_components": target_nonempty,
            "target_component_count": len(target_rows),
            "passes": True,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _draw_target_arms(target_arrays, output_figure)
    print(
        f"PASS: {containing} of {len(rows)} cells contain the continuous axis; "
        f"21985/51 non-empty components {target_nonempty} of {len(target_rows)}",
        flush=True,
    )
    return payload


def main() -> None:
    """Run from the command line with optional artifact destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-figure", type=Path, default=OUTPUT_FIGURE)
    args = parser.parse_args()
    run(args.output_json, args.output_figure)


if __name__ == "__main__":
    main()
