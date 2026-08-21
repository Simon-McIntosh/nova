"""Measure diverted DIII-D roots with shipped and recovered conductor currents.

Five polarity-screened frames and their five directly recovered conductor
currents are read from the landed recovery receipt.  Each frame is solved twice
through the same topology-pinned Newton--Krylov path: once with the nineteen
shipped poloidal conductors (the twentieth shipped channel supplies the
toroidal-field function), and once with the five recovered poloidal conductors
appended.  The recovered values are immutable inputs; this benchmark performs
no fit, current adjustment, or profile adjustment.

Distance from the labelled map is deliberately diagnostic.  A root passes only
when its fixed-point residual is at most 1e-6, every result is finite, and the
unpinned terminal topology read is diverted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.diiid_boundary_current_recovery import (
    NETCDF_DD_VERSION,
    NETCDF_ENTRY,
    OMITTED_COILS,
    POLARITY_RECEIPT,
    RECEIPT_NAME as RECOVERY_RECEIPT_NAME,
    DEFAULT_OUTPUT as RECOVERY_OUTPUT,
    _rectangle_vertices,
)
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    REGISTERED_ACCELERATED_GMRES_ITERATIONS,
    REGISTERED_ACCELERATED_NEWTON_STEPS,
    REGISTERED_ACCELERATED_RELAXATION,
    REGISTERED_ACCELERATED_STEP_CAP,
    REGISTERED_ACCELERATED_WARMUP,
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _plasma_mask,
    _read,
    build_profile,
)
from nova.biot.polygon import polygon_greens
from nova.equilibrium.forward import ForwardProfile, SaddleSeedGeometry
from nova.equilibrium.topology import TopologyClass
from nova.imas.diiid_description import POLOIDAL_CONDUCTORS
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/diverted-root")
PREREGISTRATION_NAME = "diverted_root_full_currents_preregistration.json"
RECEIPT_NAME = "diverted_root_full_currents_receipt.json"
CHECKPOINT_NAME = "diverted_root_full_currents_frames.jsonl"
FIGURE_NAME = "diverted_root_full_currents.png"
FRAME_COUNT = 5
POLARITY_AFFECTED_SHOT_COUNT = 603
FIXED_POINT_CRITERION = 1.0e-6
LABEL_REPRESENTABILITY_CEILING = 0.0429
CURRENT_ARM_NAMES = ("shipped_20_only", "shipped_20_plus_recovered_5")


@dataclass(frozen=True)
class FrameInput:
    """One immutable frame and its landed recovered-current vector."""

    shot: str
    frame: int
    recovered_currents_a: tuple[float, ...]


def preregistration() -> dict[str, Any]:
    """Return the fixed input, solve, topology, and diagnostic declaration."""

    return {
        "measurement": "diverted free-boundary root with complete current set",
        "selection": {
            "frames": FRAME_COUNT,
            "source": str(RECOVERY_OUTPUT / RECOVERY_RECEIPT_NAME),
            "rule": (
                "the five distinct-shot replacement frames already banked by the "
                "boundary-current recovery measurement"
            ),
            "polarity_screen": (
                "every shot is absent from the landed 603-shot affected population"
            ),
        },
        "current_arms": {
            "control": (
                "nineteen shipped poloidal conductors plus the shipped bcoil "
                "channel used by the toroidal-field source"
            ),
            "full": (
                "the same twenty shipped channels plus ECOILB, E567UP, E567DN, "
                "E89UP and E89DN at the directly recovered values banked per frame"
            ),
            "poloidal_conductor_count": len(POLOIDAL_CONDUCTORS) + len(OMITTED_COILS),
            "coefficients_fitted": 0,
            "current_adjustments": 0,
        },
        "solver": {
            "entry_point": "ForwardProfile.solve_branch",
            "route": "newton_krylov",
            "requested_class": "diverted",
            "relative_residual_criterion": FIXED_POINT_CRITERION,
            "newton_steps": REGISTERED_ACCELERATED_NEWTON_STEPS,
            "gmres_iterations": REGISTERED_ACCELERATED_GMRES_ITERATIONS,
            "warmup": REGISTERED_ACCELERATED_WARMUP,
            "relaxation": REGISTERED_ACCELERATED_RELAXATION,
            "step_cap": REGISTERED_ACCELERATED_STEP_CAP,
            "seed": (
                "production topology-anchored cold seed constructed from plasma "
                "current, current centroid, and the label-read axis/saddle geometry; "
                "no stored flux sample is copied into the seed"
            ),
        },
        "pass_criterion": (
            "finite terminal receipt AND relative residual <= 1e-6 AND unpinned "
            "terminal topology classified diverted"
        ),
        "label_distance": {
            "measure": (
                "gauge-aligned RMS on the labelled LCFS interior divided by the "
                "whole labelled-map range"
            ),
            "representability_ceiling": LABEL_REPRESENTABILITY_CEILING,
            "role": "diagnostic only; never included in the pass criterion",
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_preregistration(output: Path) -> Path:
    """Persist the complete declaration before any solve runs."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / PREREGISTRATION_NAME
    encoded = json.dumps(preregistration(), indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("on-disk diverted-root preregistration differs from policy")
    path.write_text(encoded)
    return path


def selected_inputs(
    recovery_receipt: dict[str, Any], affected_shots: set[str]
) -> list[FrameInput]:
    """Read exactly the banked, distinct-shot, polarity-screened frame inputs."""

    root = recovery_receipt["root_existence"]["replacement_polarity_screened"]
    if (
        root["frame_count"] < FRAME_COUNT
        or not root["all_shots_screened_free_of_affected_population"]
    ):
        raise RuntimeError("landed recovery receipt lacks five screened frames")
    selected = []
    for record in root["frames"][:FRAME_COUNT]:
        shot = str(record["shot"])
        if shot in affected_shots:
            raise RuntimeError(f"selected shot {shot} is polarity affected")
        currents = record["recovered_currents_a"]
        selected.append(
            FrameInput(
                shot=shot,
                frame=int(record["frame"]),
                recovered_currents_a=tuple(
                    float(currents[name]) for name in OMITTED_COILS
                ),
            )
        )
    if len({item.shot for item in selected}) != FRAME_COUNT:
        raise RuntimeError("the diverted-root cohort must use distinct shots")
    return selected


def _omitted_vertices() -> dict[str, tuple[tuple[np.ndarray, float], ...]]:
    """Read the independent geometry and signed turns for omitted conductors."""

    import imas

    result: dict[str, tuple[tuple[np.ndarray, float], ...]] = {}
    with imas.DBEntry(NETCDF_ENTRY, "r", dd_version=NETCDF_DD_VERSION) as entry:
        active = entry.get("pf_active", autoconvert=False)
        coils = {str(coil.name): coil for coil in active.coil}
        for name in OMITTED_COILS:
            elements = []
            for element in coils[name].element:
                geometry = element.geometry
                geometry_type = int(geometry.geometry_type)
                if geometry_type == 1:
                    vertices = np.column_stack(
                        [
                            np.asarray(geometry.outline.r, dtype=float),
                            np.asarray(geometry.outline.z, dtype=float),
                        ]
                    )
                elif geometry_type == 2:
                    vertices = _rectangle_vertices(geometry)
                else:
                    raise ValueError(
                        f"unsupported geometry type {geometry_type} for {name}"
                    )
                elements.append((vertices, float(element.turns_with_sign)))
            result[name] = tuple(elements)
    return result


def omitted_response(
    coordinates: np.ndarray,
    geometry: dict[str, tuple[tuple[np.ndarray, float], ...]],
) -> np.ndarray:
    """Return total-flux response at arbitrary targets for the five conductors."""

    target = np.asarray(coordinates, dtype=float)
    columns = []
    for name in OMITTED_COILS:
        response = np.zeros(len(target), dtype=float)
        for vertices, turns in geometry[name]:
            response += turns * polygon_greens(target[:, 0], target[:, 1], vertices)[0]
        columns.append(response)
    return np.column_stack(columns)


def append_recovered_conductors(
    profile: ForwardProfile,
    geometry: dict[str, tuple[tuple[np.ndarray, float], ...]],
) -> ForwardProfile:
    """Append five response columns without changing any plasma-side operator."""

    grid_response = omitted_response(profile.operator.grid.coordinate, geometry)
    wall_response = omitted_response(profile.operator.wall.coordinate, geometry)
    grid = replace(
        profile.operator.grid,
        source_target=jnp.column_stack(
            (profile.operator.grid.source_target, jnp.asarray(grid_response))
        ),
    )
    wall = replace(
        profile.operator.wall,
        source_target=jnp.column_stack(
            (profile.operator.wall.source_target, jnp.asarray(wall_response))
        ),
    )
    current = jnp.r_[
        profile.operator.external_current,
        jnp.zeros(len(OMITTED_COILS), dtype=profile.operator.external_current.dtype),
    ]
    operator = replace(
        profile.operator,
        grid=grid,
        wall=wall,
        external_current=current,
    )
    return replace(profile, operator=operator)


def current_arms(profile: ForwardProfile, recovered: tuple[float, ...]) -> np.ndarray:
    """Return paired 24-conductor control and full current vectors."""

    shipped = np.asarray(profile.operator.external_current, dtype=float)
    if shipped.size != len(POLOIDAL_CONDUCTORS) + len(OMITTED_COILS):
        raise RuntimeError("profile does not carry the complete conductor response")
    control = shipped.copy()
    control[-len(OMITTED_COILS) :] = 0.0
    full = shipped.copy()
    full[-len(OMITTED_COILS) :] = np.asarray(recovered, dtype=float)
    return np.stack((control, full))


def diverted_cold_seeds(
    profile: ForwardProfile, label_seed: np.ndarray, arms: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct one production cold seed per current arm from topology geometry."""

    requested = TopologyClass.DIVERTED
    _masks, topology = profile.operator.read(jnp.asarray(label_seed))
    axis = np.asarray(topology.axis, dtype=float)
    saddle = np.asarray(topology.x_point, dtype=float)
    if not bool(topology.diverted) or not np.all(np.isfinite(axis + saddle)):
        raise RuntimeError("the selected label does not supply diverted seed geometry")
    cell_current = np.asarray(
        profile.operator.cell_current(label_seed, requested), dtype=float
    )
    plasma_current = float(np.sum(cell_current))
    if abs(plasma_current) <= np.finfo(float).tiny:
        raise RuntimeError("the selected label has zero source-driven plasma current")
    coordinates = np.asarray(profile.lattice.coordinate, dtype=float)
    centroid = np.sum(cell_current[:, None] * coordinates, axis=0) / plasma_current
    geometry = SaddleSeedGeometry(tuple(axis), tuple(saddle))
    seeds = []
    receipts = []
    for current in arms:
        portfolio = profile.cold_seed_portfolio(
            plasma_current,
            centroid,
            current=current,
            diverted_geometry=geometry,
        )
        branch = jax.tree.map(
            lambda value: value[int(TopologyClass.DIVERTED)], portfolio.branches
        )
        seeds.append(np.asarray(branch.flux, dtype=float))
        receipts.append(
            {
                "plasma_current_a": float(branch.plasma_current),
                "current_centroid_rz_m": np.asarray(branch.centroid).tolist(),
                "declared_axis_rz_m": np.asarray(branch.declared_axis).tolist(),
                "declared_saddle_rz_m": np.asarray(branch.declared_boundary).tolist(),
                "stored_flux_samples_used": bool(branch.stored_flux_samples_used),
                "supported_cells": int(branch.supported_cells),
            }
        )
    return np.stack(seeds), {
        "arms": dict(zip(CURRENT_ARM_NAMES, receipts, strict=True))
    }


def label_distance(
    labelled: np.ndarray, solved: np.ndarray, interior: np.ndarray
) -> dict[str, float]:
    """Return the registered gauge-aligned diagnostic label distance."""

    selected = np.asarray(interior, dtype=bool) & np.isfinite(labelled + solved)
    gauge = float(np.mean(labelled[selected] - solved[selected]))
    difference = solved[selected] + gauge - labelled[selected]
    rms = float(np.sqrt(np.mean(difference**2)))
    span = float(np.ptp(labelled))
    return {
        "fractional_rms": rms / span,
        "rms_wb": rms,
        "label_range_wb": span,
        "additive_gauge_wb": gauge,
        "representability_ceiling": LABEL_REPRESENTABILITY_CEILING,
        "used_as_pass_criterion": False,
    }


def qualify_arm(
    *,
    residual: float,
    finite: bool,
    diverted: bool,
    iterations: int,
    x_point: np.ndarray,
    diagnostic: dict[str, float],
    trace: np.ndarray,
) -> dict[str, Any]:
    """Serialize independent fixed-point, topology, and label diagnostics."""

    fixed_point_converged = bool(
        finite and np.isfinite(residual) and residual <= FIXED_POINT_CRITERION
    )
    simultaneous = bool(fixed_point_converged and diverted)
    point = np.asarray(x_point, dtype=float)
    return {
        "fixed_point": {
            "relative_residual": float(residual),
            "criterion": FIXED_POINT_CRITERION,
            "iterations": int(iterations),
            "finite": bool(finite),
            "converged": fixed_point_converged,
            "residual_trajectory": [
                float(value) if np.isfinite(value) else None for value in trace
            ],
        },
        "topology": {
            "class": "diverted" if diverted else "limited",
            "diverted": bool(diverted),
            "x_point_rz_m": point.tolist() if np.all(np.isfinite(point)) else None,
        },
        "simultaneously_converged_and_diverted": simultaneous,
        "label_map_diagnostic": diagnostic,
    }


def solve_frame(
    row: dict[str, Any],
    frame_input: FrameInput,
    geometry: dict[str, tuple[tuple[np.ndarray, float], ...]],
) -> dict[str, Any]:
    """Solve both current arms through one batched production branch path."""

    profile, seed, label, _wall, _reliable, _statement = build_profile(
        row, frame_input.frame, 0.02
    )
    profile = append_recovered_conductors(profile, geometry)
    arms = current_arms(profile, frame_input.recovered_currents_a)
    seeds, seed_receipt = diverted_cold_seeds(profile, seed, arms)

    def solve_one(initial, current):
        return profile.solve_branch(
            initial,
            TopologyClass.DIVERTED,
            route="newton_krylov",
            current=current,
            tolerance=FIXED_POINT_CRITERION,
            gmres_iterations=REGISTERED_ACCELERATED_GMRES_ITERATIONS,
            warmup=REGISTERED_ACCELERATED_WARMUP,
            relaxation=REGISTERED_ACCELERATED_RELAXATION,
            step_cap=REGISTERED_ACCELERATED_STEP_CAP,
        )

    branches = jax.jit(jax.vmap(solve_one))(jnp.asarray(seeds), jnp.asarray(arms))
    radius = profile.lattice.radius
    height = profile.lattice.height
    interior = _plasma_mask(row, frame_input.frame, radius, height)
    records = {}
    for index, name in enumerate(CURRENT_ARM_NAMES):
        branch = jax.tree.map(lambda value: value[index], branches)
        equilibrium = branch.equilibrium
        _masks, topology = profile.operator.read(equilibrium.flux)
        solved = np.asarray(equilibrium.flux[: profile.lattice.node_count]).reshape(
            profile.lattice.shape
        )
        records[name] = qualify_arm(
            residual=float(branch.residual),
            finite=bool(equilibrium.finite.passed),
            diverted=bool(topology.diverted),
            iterations=int(branch.iterations),
            x_point=np.asarray(topology.x_point),
            diagnostic=label_distance(label, solved, interior),
            trace=np.asarray(equilibrium.fixed_point.trace, dtype=float),
        )
    return {
        "shot": frame_input.shot,
        "frame": frame_input.frame,
        "time_ms": float(row["efit_times"][frame_input.frame]),
        "screened_out_of_affected_polarity_population": True,
        "shipped_channel_count": 20,
        "poloidal_conductor_count": len(POLOIDAL_CONDUCTORS) + len(OMITTED_COILS),
        "recovered_currents_a": dict(
            zip(OMITTED_COILS, frame_input.recovered_currents_a, strict=True)
        ),
        "coefficients_fitted": 0,
        "current_adjustments": 0,
        "cold_seed_receipt": seed_receipt,
        "arms": records,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the paired cohort verdict without using label distance as a gate."""

    full = [item["arms"][CURRENT_ARM_NAMES[1]] for item in records]
    control = [item["arms"][CURRENT_ARM_NAMES[0]] for item in records]
    full_passes = sum(item["simultaneously_converged_and_diverted"] for item in full)
    return {
        "frame_count": len(records),
        "all_shots_screened_free_of_affected_population": all(
            item["screened_out_of_affected_polarity_population"] for item in records
        ),
        "full_current_converged_diverted_frames": int(full_passes),
        "control_converged_diverted_frames": int(
            sum(item["simultaneously_converged_and_diverted"] for item in control)
        ),
        "control_fixed_point_converged_frames": int(
            sum(item["fixed_point"]["converged"] for item in control)
        ),
        "full_current_residuals": [
            item["fixed_point"]["relative_residual"] for item in full
        ],
        "control_residuals": [
            item["fixed_point"]["relative_residual"] for item in control
        ],
        "full_current_label_fractional_rms": [
            item["label_map_diagnostic"]["fractional_rms"] for item in full
        ],
        "label_representability_ceiling": LABEL_REPRESENTABILITY_CEILING,
        "label_distance_is_diagnostic_only": True,
        "passed": bool(
            len(records) >= FRAME_COUNT
            and full_passes == len(records)
            and all(
                item["screened_out_of_affected_polarity_population"] for item in records
            )
        ),
        "frames": records,
    }


def _figure(summary: dict[str, Any], path: Path) -> None:
    """Plot paired residual/topology outcomes and diagnostic label distance."""

    frames = summary["frames"]
    labels = [f"{item['shot'][9:17]}:{item['frame']}" for item in frames]
    x = np.arange(len(frames))
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    for name, color, marker in (
        (CURRENT_ARM_NAMES[0], "#4477aa", "o"),
        (CURRENT_ARM_NAMES[1], "#cc6677", "s"),
    ):
        values = [
            item["arms"][name]["fixed_point"]["relative_residual"] for item in frames
        ]
        axes[0].semilogy(x, values, marker=marker, color=color, label=name)
    axes[0].axhline(FIXED_POINT_CRITERION, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylabel("Relative fixed-point residual")
    axes[0].set_title("Root residual by unchanged current arm")
    axes[0].legend(frameon=False, fontsize=8)
    values = summary["full_current_label_fractional_rms"]
    axes[1].bar(x, values, color="#cc6677")
    axes[1].axhline(
        LABEL_REPRESENTABILITY_CEILING,
        color="black",
        linestyle="--",
        label="0.0429 representability ceiling",
    )
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylabel("Label-map fractional RMS (diagnostic)")
    axes[1].set_title("Not a root pass criterion")
    axes[1].legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(data: Path, output: Path) -> dict[str, Any]:
    """Run the fixed cohort and write checkpoint, receipt, and figure."""

    configure_dtypes()
    declaration = write_preregistration(output)
    recovery_path = RECOVERY_OUTPUT / RECOVERY_RECEIPT_NAME
    recovery = json.loads(recovery_path.read_text())
    polarity = json.loads(POLARITY_RECEIPT.read_text())["full_corpus_census"]
    affected = set(polarity["affected_shots"])
    if len(affected) != POLARITY_AFFECTED_SHOT_COUNT:
        raise RuntimeError("polarity authority is not the landed 603-shot population")
    selected = selected_inputs(recovery, affected)
    geometry = _omitted_vertices()
    columns = tuple(
        dict.fromkeys((*_LABEL_COLUMNS, *_CURRENT_COLUMNS, *_GEOMETRY_COLUMNS))
    )
    checkpoint = output / CHECKPOINT_NAME
    checkpoint.write_text("")
    records = []
    for number, frame_input in enumerate(selected, start=1):
        path = data / frame_input.shot
        row = _read(path, columns)
        row["_source_path"] = str(path)
        record = solve_frame(row, frame_input, geometry)
        records.append(record)
        with checkpoint.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        full = record["arms"][CURRENT_ARM_NAMES[1]]
        control = record["arms"][CURRENT_ARM_NAMES[0]]
        print(
            f"SOLVED {number}/{len(selected)} {frame_input.shot}:{frame_input.frame} "
            f"full={full['fixed_point']['relative_residual']:.6e}/"
            f"{full['topology']['class']} control="
            f"{control['fixed_point']['relative_residual']:.6e}/"
            f"{control['topology']['class']}",
            flush=True,
        )
    result = summarize(records)
    receipt = {
        "preregistration": preregistration(),
        "preregistration_path": str(declaration),
        "preregistration_sha256": _sha256(declaration),
        "authorities": {
            "recovery_receipt": str(recovery_path),
            "recovery_receipt_sha256": _sha256(recovery_path),
            "polarity_receipt": str(POLARITY_RECEIPT),
            "polarity_receipt_sha256": _sha256(POLARITY_RECEIPT),
            "affected_shot_count": len(affected),
            "omitted_geometry_entry": str(NETCDF_ENTRY),
            "omitted_geometry_dd_version": NETCDF_DD_VERSION,
        },
        "result": result,
    }
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _figure(result, output / FIGURE_NAME)
    if not result["passed"]:
        raise RuntimeError(
            "fewer than five full-current frames converged below 1e-6 as diverted"
        )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preregister-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.preregister_only:
        print(f"PREREGISTERED {write_preregistration(arguments.output)}")
        return
    receipt = run(arguments.data, arguments.output)
    headline = dict(receipt["result"])
    headline.pop("frames", None)
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
