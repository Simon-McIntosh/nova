"""Receipt the MAST parity lane on its reference-native spatial default."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import matplotlib

from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    REFERENCE_NATIVE_GRID_POINTS,
    _mast_case_from_selection,
    select_slices_by_shot,
)
from benchmarks.efit_parity_tared_external_field import (
    OUTPUT_RECEIPT as COARSE_CONTROL_RECEIPT,
    _solve_row,
    build_tare,
)
from nova.frame.frame import Frame
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

OUTPUT_DIRECTORY = Path("docs/figures/forward-operator-refinement")
OUTPUT_RECEIPT = OUTPUT_DIRECTORY / "reference-native-resolution-default.json"
OUTPUT_FIGURE = OUTPUT_DIRECTORY / "reference-native-resolution-default.png"
MEASUREMENT_SOURCE = Path("benchmarks/forward_resolution_default.py")
CONTROL_SHOT = 21978
INTERMEDIATE_GRID_POINTS = 65
BANKED_CPU_BUILD_WALL_HOURS = 8.8
BANKED_DEVICE_BUILD_WALL_MINUTES = 2.4
BANKED_COUPLING_MEMORY_GB = 0.44


def _sha256(path: Path) -> str:
    """Return the hexadecimal digest of one evidence input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coupling_memory(profile) -> dict[str, Any]:
    """Count active and resident dense response blocks without double-counting."""
    blocks = {}
    for target_name in ("grid", "wall"):
        target = getattr(profile.operator, target_name)
        for block_name in (
            "source_target",
            "plasma_target",
            "plasma_target_r",
            "plasma_target_z",
        ):
            block = getattr(target, block_name)
            blocks[f"{target_name}.{block_name}"] = {
                "shape": list(block.shape),
                "dtype": str(block.dtype),
                "bytes": int(block.nbytes),
            }
            block.block_until_ready()
    active_names = {
        "grid.source_target",
        "grid.plasma_target",
        "wall.source_target",
        "wall.plasma_target",
    }
    active_bytes = sum(
        item["bytes"] for name, item in blocks.items() if name in active_names
    )
    resident_bytes = sum(item["bytes"] for item in blocks.values())
    return {
        "blocks": blocks,
        "active_coupling_bytes": active_bytes,
        "active_coupling_gb_decimal": active_bytes / 1.0e9,
        "resident_coupling_bytes": resident_bytes,
        "resident_coupling_gb_decimal": resident_bytes / 1.0e9,
        "resident_includes_zero_linear_moment_companions": True,
        "linear_moments_enabled": bool(profile.operator.use_linear_moments),
    }


def _metric_row(result: dict[str, Any], mesh: dict[str, Any]) -> dict[str, Any]:
    """Flatten the residual and geometry quantities used in resolution comparisons."""
    metrics = result["raw_registered_rows"]["metrics"]
    controlled = result["instrument_controlled_rows"]
    return {
        "grid_shape": mesh["kind"],
        "lattice_cell_count": mesh["realised_cells"],
        "stored_lcfs_interior_cell_count": mesh["stored_lcfs_interior_cells"],
        "radial_step_m": mesh["radial_step_m"],
        "vertical_step_m": mesh["vertical_step_m"],
        "terminal_fixed_point_residual": result["solve"]["terminal_residual"],
        "flux_sup_fraction_of_reference_span": metrics["flux_map"][
            "sup_fraction_of_reference_span"
        ],
        "flux_rms_fraction_of_reference_span": metrics["flux_map"][
            "rms_fraction_of_reference_span"
        ],
        "magnetic_axis_distance_m": metrics["magnetic_axis"]["distance_m"],
        "lcfs_symmetric_mean_distance_m": metrics["lcfs"]["symmetric_mean_distance_m"],
        "x_point_distance_m": metrics["x_point"]["distance_m"],
        "topology_class_agreement": metrics["topology"]["agreement"],
        "controlled_lcfs_status": controlled["lcfs_closed_branch"]["status"],
        "controlled_lcfs_distance": controlled["lcfs_closed_branch"]["distance"],
        "measured": True,
    }


def _coarse_row(receipt: dict[str, Any]) -> dict[str, Any]:
    """Normalize the protected 33-point control beside fresh measurements."""
    result = next(
        row
        for row in receipt["per_shot"]
        if int(row["reference"]["shot"]) == CONTROL_SHOT
    )
    metrics = result["raw_registered_rows"]["metrics"]
    controlled = result["instrument_controlled_rows"]
    support = result["reference"]["source_coordinate"]["support_nodes"]
    return {
        "grid_shape": "33 by 33 rectangular benchmark lattice",
        "lattice_cell_count": 33 * 33,
        "stored_lcfs_interior_cell_count": int(support),
        "radial_step_m": 0.125,
        "vertical_step_m": 0.125,
        "terminal_fixed_point_residual": result["solve"]["terminal_residual"],
        "flux_sup_fraction_of_reference_span": metrics["flux_map"][
            "sup_fraction_of_reference_span"
        ],
        "flux_rms_fraction_of_reference_span": metrics["flux_map"][
            "rms_fraction_of_reference_span"
        ],
        "magnetic_axis_distance_m": metrics["magnetic_axis"]["distance_m"],
        "lcfs_symmetric_mean_distance_m": metrics["lcfs"]["symmetric_mean_distance_m"],
        "x_point_distance_m": metrics["x_point"]["distance_m"],
        "topology_class_agreement": metrics["topology"]["agreement"],
        "controlled_lcfs_status": controlled["lcfs_closed_branch"]["status"],
        "controlled_lcfs_distance": controlled["lcfs_closed_branch"]["distance"],
        "measured": True,
        "measurement_source": str(COARSE_CONTROL_RECEIPT),
        "measurement_source_sha256": _sha256(COARSE_CONTROL_RECEIPT),
    }


def _fresh_row(
    store: Path,
    selected: dict[str, Any],
    qualification: dict[str, Any],
    grid_points: int,
) -> dict[str, Any]:
    """Build and solve one tared control at a fixed spatial resolution."""
    build_started = perf_counter()
    case, context = _mast_case_from_selection(
        store,
        selected,
        qualification,
        grid_points=grid_points,
    )
    memory = _coupling_memory(context["profile"])
    build_wall_seconds = perf_counter() - build_started

    tare_started = perf_counter()
    tare = build_tare(
        context["profile"],
        case["state"],
        context["reference_flux"],
    )
    tare_wall_seconds = perf_counter() - tare_started
    solve_started = perf_counter()
    result = _solve_row({"case": case, "context": context, "tare": tare})
    solve_wall_seconds = perf_counter() - solve_started
    return {
        "metrics": _metric_row(result, case["mesh"]),
        "coupling_memory": memory,
        "cost": {
            "operator_build_wall_seconds": build_wall_seconds,
            "tare_construction_wall_seconds": tare_wall_seconds,
            "fresh_solve_wall_seconds": solve_wall_seconds,
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "measured_this_run": True,
        },
        "tare_closure": tare["closure"],
        "solve": result["solve"],
    }


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot residual, geometry and interior population across the three meshes."""
    labels = ["33×33", "65×65", "95×95"]
    residual = [row["terminal_fixed_point_residual"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), constrained_layout=True)
    axes[0].plot(labels, residual, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Terminal fixed-point residual")
    axes[0].set_title("Tared current-constrained solve")
    for name, label in (
        ("magnetic_axis_distance_m", "axis"),
        ("lcfs_symmetric_mean_distance_m", "LCFS"),
        ("x_point_distance_m", "X point"),
    ):
        axes[1].plot(labels, [row[name] for row in rows], marker="o", label=label)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Distance [m]")
    axes[1].set_title("Raw geometry rows")
    axes[1].legend(fontsize=8)
    axes[2].bar(
        labels,
        [row["stored_lcfs_interior_cell_count"] for row in rows],
    )
    axes[2].axhline(2141, color="black", ls="--", lw=0.9, label="reference ≈2141")
    axes[2].set_ylabel("Stored-LCFS interior cells")
    axes[2].set_title("Scored plasma support")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", which="both", alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
    output: Path = OUTPUT_RECEIPT,
    figure: Path = OUTPUT_FIGURE,
) -> dict[str, Any]:
    """Measure the promoted parity resolution and serialize its evidence."""
    configure_dtypes()
    selected = {
        int(row["shot"]): (row, qualification)
        for row, qualification in select_slices_by_shot(bank)
    }
    control, qualification = selected[CONTROL_SHOT]
    if int(control["slice_index"]) != 35:
        raise RuntimeError("the resolution control no longer selects slice 35")
    if jax.default_backend() != "gpu":
        raise RuntimeError("the fresh reference-native measurement requires a GPU")

    banked = json.loads(COARSE_CONTROL_RECEIPT.read_text())
    coarse = _coarse_row(banked)
    intermediate = _fresh_row(store, control, qualification, INTERMEDIATE_GRID_POINTS)
    gc.collect()
    reference_native = _fresh_row(
        store, control, qualification, REFERENCE_NATIVE_GRID_POINTS
    )
    rows = [coarse, intermediate["metrics"], reference_native["metrics"]]
    if reference_native["metrics"]["stored_lcfs_interior_cell_count"] < 2141:
        raise RuntimeError("the promoted grid is not reference-native over the LCFS")
    _plot(rows, figure)

    global_default = Frame.__dataclass_fields__["dplasma"].default
    receipt = {
        "receipt": "MAST parity reference-native resolution default",
        "status": "complete",
        "selection": {
            "shot": CONTROL_SHOT,
            "slice_index": int(control["slice_index"]),
            "qualification_passes": bool(qualification["passes"]),
        },
        "default_decision": {
            "scored_parity_axis_points": REFERENCE_NATIVE_GRID_POINTS,
            "scored_parity_grid_shape": [
                REFERENCE_NATIVE_GRID_POINTS,
                REFERENCE_NATIVE_GRID_POINTS,
            ],
            "stored_lcfs_interior_cells": reference_native["metrics"][
                "stored_lcfs_interior_cell_count"
            ],
            "global_frame_dplasma": global_default,
            "global_frame_default_retained": bool(global_default == -500),
            "reason": (
                "General CoilSet construction retains dplasma=-500; changing it would "
                "invalidate semantic caches for every implicit plasma mesh. Only the "
                "parity lane scored against the reference map is promoted."
            ),
            "forward_wrapper_factory_added": False,
        },
        "resolution_rows": {
            "banked_33_point": coarse,
            "fresh_65_point": intermediate,
            "fresh_95_point_reference_native": reference_native,
        },
        "cost_summary": {
            "fresh_device_backend": jax.default_backend(),
            "fresh_device": str(jax.devices()[0]),
            "reference_native_device_build_wall_seconds_measured": reference_native[
                "cost"
            ]["operator_build_wall_seconds"],
            "reference_native_device_solve_wall_seconds_measured": reference_native[
                "cost"
            ]["fresh_solve_wall_seconds"],
            "reference_native_active_coupling_gb_measured": reference_native[
                "coupling_memory"
            ]["active_coupling_gb_decimal"],
            "reference_native_resident_coupling_gb_measured": reference_native[
                "coupling_memory"
            ]["resident_coupling_gb_decimal"],
            "banked_projection": {
                "coupling_memory_gb": BANKED_COUPLING_MEMORY_GB,
                "cpu_nine_block_build_wall_hours": BANKED_CPU_BUILD_WALL_HOURS,
                "device_build_wall_minutes": BANKED_DEVICE_BUILD_WALL_MINUTES,
                "measured_this_run": False,
                "reading": (
                    "These are the carried reference-native projections. The fresh "
                    "device build and actual resident parity arrays are reported "
                    "separately and are not substituted by the projections."
                ),
            },
        },
        "forward_suite": {
            "status": "pending post-measurement validation",
            "passed": None,
            "passed_count": None,
            "failed_count": None,
        },
        "figure": {
            "path": str(figure),
            "src": (
                "/nova/figures/forward-operator-refinement/"
                "reference-native-resolution-default.png"
            ),
            "sha256": _sha256(figure),
        },
        "evidence_inputs": {
            "coarse_control": str(COARSE_CONTROL_RECEIPT),
            "coarse_control_sha256": _sha256(COARSE_CONTROL_RECEIPT),
            "decomposition_bank": str(bank),
            "decomposition_bank_sha256": _sha256(bank),
            "measurement_source": str(MEASUREMENT_SOURCE),
            "measurement_source_sha256": _sha256(MEASUREMENT_SOURCE),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    """Parse evidence paths, execute both fresh rows and print the headline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--bank", type=Path, default=DECOMPOSITION_BANK)
    parser.add_argument("--output", type=Path, default=OUTPUT_RECEIPT)
    parser.add_argument("--figure", type=Path, default=OUTPUT_FIGURE)
    arguments = parser.parse_args()
    receipt = run(arguments.store, arguments.bank, arguments.output, arguments.figure)
    metrics = receipt["resolution_rows"]["fresh_95_point_reference_native"]["metrics"]
    print(
        "REFERENCE_NATIVE_RESOLUTION "
        f"interior_cells={metrics['stored_lcfs_interior_cell_count']} "
        f"residual={metrics['terminal_fixed_point_residual']:.9g} "
        f"active_coupling_gb="
        f"{receipt['cost_summary']['reference_native_active_coupling_gb_measured']:.6g}"
    )


if __name__ == "__main__":
    main()
