"""Measure diverted-root movement as omitted solenoid halves are included.

The five omitted conductor geometries and signed turns come from the DIII-D
netCDF machine description.  Their currents do not: each is the released
ECOILA current from the same frame multiplied by a fixed, independently
measured common-drive scale.  The experiment therefore changes no current,
fits no coefficient, and never reads a label-derived current recovery.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import diiid_current_pinned_forward as pinned
from benchmarks.diiid_diverted_root_full_currents import (
    POLARITY_AFFECTED_SHOT_COUNT,
    _omitted_vertices,
    append_recovered_conductors as append_missing_conductor_responses,
)
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _plasma_mask,
    _read,
    _separatrix,
    build_profile,
    canonical_axes,
    gauge_metrics,
)
from benchmarks.diiid_state_of_play_figures import (
    _boundary_separation,
    boundary_gradient_minimum,
)
from nova.imas.diiid_description import POLOIDAL_CONDUCTORS
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path("docs/figures/current-constrained-forward-solve/inclusion-ladder")
PREREGISTRATION_NAME = "solenoid_inclusion_preregistration.json"
CHECKPOINT_NAME = "solenoid_inclusion_frames.jsonl"
RECEIPT_NAME = "solenoid_inclusion_receipt.json"
FIGURE_NAME = "solenoid_inclusion_ladder.png"
LABEL_REPRESENTABILITY_CEILING = 0.0429
LANDED_X_POINT_SEPARATION_M = 0.4552
LANDED_LCFS_SEPARATION_M = 0.0537
RELATIVE_RESIDUAL_CRITERION = pinned.RELATIVE_RESIDUAL_CRITERION
PSEUDO_WALL_EXPANSION = pinned.PSEUDO_WALL_EXPANSION
ECOILA_INDEX = POLOIDAL_CONDUCTORS.index("ECOILA")
MISSING_CONDUCTOR_ORDER = ("ECOILB", "E567UP", "E567DN", "E89UP", "E89DN")
CURRENT_SCALE = {
    "ECOILB": 1.0172,
    "E567UP": 0.9929,
    "E567DN": 0.9823,
    "E89UP": 0.9806,
    "E89DN": 1.0165,
}
CONDUCTOR_TURNS = {
    "ECOILB": 48,
    "E567UP": 6,
    "E567DN": 6,
    "E89UP": 7,
    "E89DN": 7,
}


@dataclass(frozen=True)
class Inclusion:
    """One cumulative set of omitted conductor responses."""

    name: str
    added: tuple[str, ...]


INCLUSIONS = (
    Inclusion("shipped_20", ()),
    Inclusion("plus_ecoilb", ("ECOILB",)),
    Inclusion("plus_e567_pair", ("ECOILB", "E567UP", "E567DN")),
    Inclusion("complete_24_poloidal", MISSING_CONDUCTOR_ORDER),
)
COHORT = tuple(
    {"shot": str(item["shot"]), "frame": int(item["frame"])}
    for item in pinned.REPRESENTATIVE_COHORT
)


def ampere_turn_fractions() -> list[dict[str, Any]]:
    """Return cumulative fractions of the prescribed missing ampere-turns."""

    weights = {
        name: abs(CURRENT_SCALE[name]) * CONDUCTOR_TURNS[name]
        for name in MISSING_CONDUCTOR_ORDER
    }
    total = sum(weights.values())
    return [
        {
            "name": inclusion.name,
            "added_conductors": list(inclusion.added),
            "cumulative_missing_ampere_turn_fraction": (
                sum(weights[name] for name in inclusion.added) / total
            ),
            "incremental_missing_ampere_turn_fraction": (
                sum(weights[name] for name in inclusion.added)
                - (
                    sum(weights[name] for name in INCLUSIONS[index - 1].added)
                    if index
                    else 0.0
                )
            )
            / total,
        }
        for index, inclusion in enumerate(INCLUSIONS)
    ]


def preregistration() -> dict[str, Any]:
    """Return the complete declaration fixed before any frame is scored."""

    return {
        "measurement": "omitted-solenoid cumulative inclusion ladder",
        "cohort": {
            "frames": list(COHORT),
            "frame_count": len(COHORT),
            "distinct_shots_required": True,
            "required_named_frame": {
                "shot": "d3d_shot_00000c4a7b.parquet",
                "frame": 102,
            },
            "polarity_screen": (
                "every shot must be absent from the landed 603-shot population"
            ),
        },
        "current_authority": {
            "source": "the shipped ECOILA channel on the same frame",
            "operation": (
                "multiply ECOILA by the fixed per-coil scale; no recovered "
                "current value is read"
            ),
            "scales": CURRENT_SCALE,
            "nothing_fitted": True,
            "no_current_adjusted": True,
        },
        "inclusions": ampere_turn_fractions(),
        "solver": {
            "map": "closed-form plasma-current amplitude elimination",
            "relative_residual_criterion": RELATIVE_RESIDUAL_CRITERION,
            "maximum_outer_iterations": pinned.HOST_OUTER_ITERATIONS,
            "maximum_inner_iterations": pinned.HOST_INNER_ITERATIONS,
            "terminal_topology_reported": True,
        },
        "metrics": {
            "x_point": "Euclidean R-Z separation from derived labelled X point",
            "lcfs": "symmetric mean closest-polyline separation",
            "flux": "gauge-free interior fractional RMS",
            "landed_x_point_separation_m": LANDED_X_POINT_SEPARATION_M,
            "landed_lcfs_separation_m": LANDED_LCFS_SEPARATION_M,
            "label_representability_ceiling": LABEL_REPRESENTABILITY_CEILING,
        },
        "monotonic_rule": (
            "X-point separation from the labelled X point must be non-increasing "
            "at every cumulative inclusion within 1e-9 m"
        ),
        "nova_equilibrium_modified": False,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_preregistration(output: Path) -> Path:
    """Persist the declaration before scoring and refuse policy drift."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / PREREGISTRATION_NAME
    encoded = json.dumps(preregistration(), indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("on-disk solenoid inclusion declaration differs")
    path.write_text(encoded)
    return path


def inclusion_currents(
    shipped_current: np.ndarray, ecoila_current_a: float
) -> list[np.ndarray]:
    """Build the four immutable current vectors from shipped ECOILA alone."""

    shipped = np.asarray(shipped_current, dtype=float)
    if shipped.size != len(POLOIDAL_CONDUCTORS) + len(MISSING_CONDUCTOR_ORDER):
        raise ValueError("conductor response vector is not the expected 24 columns")
    if not np.isclose(shipped[ECOILA_INDEX], ecoila_current_a, rtol=0.0, atol=1e-9):
        raise ValueError(
            "supplied ECOILA value differs from the shipped current vector"
        )
    if np.any(shipped[-len(MISSING_CONDUCTOR_ORDER) :] != 0.0):
        raise ValueError("omitted conductor current slots must start at zero")

    result = []
    for inclusion in INCLUSIONS:
        current = shipped.copy()
        for name in inclusion.added:
            offset = MISSING_CONDUCTOR_ORDER.index(name)
            current[-len(MISSING_CONDUCTOR_ORDER) + offset] = (
                CURRENT_SCALE[name] * ecoila_current_a
            )
        result.append(current)
    return result


def monotonic_toward_label(separations_m: list[float], tolerance: float = 1e-9) -> bool:
    """Return whether every cumulative inclusion approaches the labelled X point."""

    values = np.asarray(separations_m, dtype=float)
    return bool(np.all(np.isfinite(values)) and np.all(np.diff(values) <= tolerance))


def _label_topology(row: dict[str, Any], frame: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(row["efit_lcfs_n"][frame])
    boundary = np.column_stack(
        (
            np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
            np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
        )
    )
    radius, height = canonical_axes(row)
    field = np.asarray(row["efit_psirz"][frame], dtype=float)
    return boundary, boundary_gradient_minimum(radius, height, field, boundary)


def _serialise_solve(
    result: dict[str, Any],
    profile: Any,
    labelled_flux: np.ndarray,
    labelled_boundary: np.ndarray,
    labelled_x_point: np.ndarray,
    interior: np.ndarray,
) -> dict[str, Any]:
    """Attach the three label-distance diagnostics to one terminal solve."""

    state = np.asarray(result["state"], dtype=float)
    predicted = state[: profile.lattice.node_count].reshape(profile.lattice.shape)
    _r_squared, fractional_rms, gauge, aligned = gauge_metrics(
        labelled_flux, predicted, interior
    )
    _masks, topology = profile.operator.read(jnp.asarray(state))
    x_point = np.asarray(topology.x_point, dtype=float)
    boundary = _separatrix(
        np.asarray(profile.lattice.radius, dtype=float),
        np.asarray(profile.lattice.height, dtype=float),
        predicted,
        float(topology.axis_flux),
        float(topology.boundary_flux),
    )
    finite_x = bool(np.all(np.isfinite(x_point)))
    finite_boundary = len(boundary) >= 4
    return {
        "relative_residual": float(result["relative_residual"]),
        "iterations": int(result["iterations"]),
        "terminal_topology": str(result["topology"]),
        "converged_at_1e-6": bool(
            result["relative_residual"] <= RELATIVE_RESIDUAL_CRITERION
        ),
        "profile_amplitude": float(result["amplitude"]),
        "x_point_rz_m": x_point.tolist() if finite_x else None,
        "x_point_separation_m": (
            float(np.linalg.norm(x_point - labelled_x_point)) if finite_x else None
        ),
        "lcfs_symmetric_mean_separation_m": (
            _boundary_separation(boundary, labelled_boundary)
            if finite_boundary
            else None
        ),
        "gauge_free_fractional_rms": float(fractional_rms),
        "label_representability_ceiling": LABEL_REPRESENTABILITY_CEILING,
        "additive_gauge_total_flux_wb": float(gauge),
        "current_relative_error": float(result["current_relative_error"]),
        "lambda_guard_triggered": bool(result["lambda_guard_triggered"]),
        "termination": str(result["termination"]),
        "_aligned_flux": aligned,
    }


def solve_frame(
    row: dict[str, Any], frame: int, geometry: dict[str, Any]
) -> dict[str, Any]:
    """Solve all cumulative inclusions from the same labelled branch seed."""

    profile, seed, labelled, _wall, reliable, control_surface = build_profile(
        row, frame, PSEUDO_WALL_EXPANSION
    )
    profile = append_missing_conductor_responses(profile, geometry)
    shipped = np.asarray(profile.operator.external_current, dtype=float)
    ecoila = float(shipped[ECOILA_INDEX])
    currents = inclusion_currents(shipped, ecoila)
    time_ms = float(row["efit_times"][frame])
    target_current = pinned._target_current(row, time_ms)
    labelled_boundary, labelled_x_point = _label_topology(row, frame)
    interior = _plasma_mask(
        row,
        frame,
        np.asarray(profile.lattice.radius, dtype=float),
        np.asarray(profile.lattice.height, dtype=float),
    )

    inclusions = []
    for number, (definition, current) in enumerate(
        zip(INCLUSIONS, currents, strict=True)
    ):
        result = pinned.solve_eliminated(profile, seed, current, target_current)
        metrics = _serialise_solve(
            result,
            profile,
            labelled,
            labelled_boundary,
            labelled_x_point,
            interior,
        )
        metrics.pop("_aligned_flux")
        metrics.update(
            {
                "number": number,
                "name": definition.name,
                "added_conductors": list(definition.added),
                "poloidal_conductor_count": len(POLOIDAL_CONDUCTORS)
                + len(definition.added),
                "added_currents_a": {
                    name: float(CURRENT_SCALE[name] * ecoila)
                    for name in definition.added
                },
            }
        )
        inclusions.append(metrics)
        print(
            f"INCLUSION {number} {definition.name} "
            f"residual={metrics['relative_residual']:.12e} "
            f"topology={metrics['terminal_topology']} "
            f"x_distance={metrics['x_point_separation_m']}",
            flush=True,
        )

    x_distances = [item["x_point_separation_m"] for item in inclusions]
    monotonic = bool(
        all(value is not None for value in x_distances)
        and monotonic_toward_label([float(value) for value in x_distances])
    )
    return {
        "time_ms": time_ms,
        "target_plasma_current_a": target_current,
        "shipped_ecoila_current_a": ecoila,
        "reliable_extracted_flux_function_surfaces": reliable,
        "control_surface": control_surface,
        "labelled_x_point_rz_m": labelled_x_point.tolist(),
        "screened_out_of_affected_polarity_population": True,
        "same_label_branch_seed_all_inclusions": True,
        "nothing_fitted": True,
        "no_current_adjusted": True,
        "x_point_migration_monotonic_toward_label": monotonic,
        "x_point_migration_verdict": (
            "monotonic toward the labelled divertor leg"
            if monotonic
            else "non-monotonic; the omitted conductor set is not the whole deficit"
        ),
        "inclusions": inclusions,
    }


def summarize(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the cohort result without hiding a failed solve or trajectory."""

    per_inclusion = []
    fractions = ampere_turn_fractions()
    for number, definition in enumerate(INCLUSIONS):
        values = [frame["inclusions"][number] for frame in frames]
        per_inclusion.append(
            {
                **fractions[number],
                "median_relative_residual": float(
                    np.median([item["relative_residual"] for item in values])
                ),
                "converged_frames": int(
                    sum(item["converged_at_1e-6"] for item in values)
                ),
                "diverted_frames": int(
                    sum(item["terminal_topology"] == "diverted" for item in values)
                ),
                "median_x_point_separation_m": float(
                    np.median(
                        [
                            item["x_point_separation_m"]
                            for item in values
                            if item["x_point_separation_m"] is not None
                        ]
                    )
                ),
                "median_lcfs_symmetric_mean_separation_m": float(
                    np.median(
                        [
                            item["lcfs_symmetric_mean_separation_m"]
                            for item in values
                            if item["lcfs_symmetric_mean_separation_m"] is not None
                        ]
                    )
                ),
                "median_gauge_free_fractional_rms": float(
                    np.median([item["gauge_free_fractional_rms"] for item in values])
                ),
            }
        )
    monotonic_count = sum(
        frame["x_point_migration_monotonic_toward_label"] for frame in frames
    )
    return {
        "frame_count": len(frames),
        "distinct_shots": len({frame["shot"] for frame in frames}),
        "all_shots_screened_free_of_affected_population": all(
            frame["screened_out_of_affected_polarity_population"] for frame in frames
        ),
        "all_inclusions_derived_from_shipped_ecoila": True,
        "label_recovered_current_values_used": 0,
        "coefficients_fitted": 0,
        "currents_adjusted": 0,
        "monotonic_frames": int(monotonic_count),
        "non_monotonic_frames": int(len(frames) - monotonic_count),
        "pooled_x_point_migration_verdict": (
            "monotonic toward the labelled divertor leg on every frame"
            if monotonic_count == len(frames)
            else (
                "non-monotonic on at least one frame; the omitted conductor set "
                "is not the whole deficit"
            )
        ),
        "landed_comparators": {
            "x_point_separation_m": LANDED_X_POINT_SEPARATION_M,
            "lcfs_symmetric_mean_separation_m": LANDED_LCFS_SEPARATION_M,
            "label_representability_ceiling_fractional_rms": (
                LABEL_REPRESENTABILITY_CEILING
            ),
        },
        "per_inclusion": per_inclusion,
        "frames": frames,
    }


def render_figure(summary: dict[str, Any], path: Path) -> None:
    """Plot each physical X-point trajectory and the cohort distance ladder."""

    frames = summary["frames"]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(frames)))
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    x = np.arange(len(INCLUSIONS))
    for color, frame in zip(colors, frames, strict=True):
        label = f"{frame['shot'][9:17]}:{frame['frame']}"
        points = np.asarray(
            [item["x_point_rz_m"] for item in frame["inclusions"]], dtype=float
        )
        labelled = np.asarray(frame["labelled_x_point_rz_m"], dtype=float)
        axes[0].plot(points[:, 0], points[:, 1], "o-", color=color, label=label)
        axes[0].plot(*labelled, marker="x", color=color, markersize=8)
        axes[1].plot(
            x,
            [item["x_point_separation_m"] for item in frame["inclusions"]],
            "o-",
            color=color,
            label=label,
        )
    axes[0].set_xlabel("R [m]")
    axes[0].set_ylabel("Z [m]")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_title("Nova X-point path; × is the labelled X point")
    axes[1].axhline(
        LANDED_X_POINT_SEPARATION_M,
        color="black",
        linestyle="--",
        label="landed 0.4552 m baseline",
    )
    axes[1].set_xticks(x, [item.name for item in INCLUSIONS], rotation=25, ha="right")
    axes[1].set_ylabel("X-point separation from label [m]")
    axes[1].set_title("Cumulative missing-conductor inclusion")
    axes[1].legend(frameon=False, fontsize=7)
    figure.suptitle(
        "DIII-D current-pinned solenoid inclusion ladder: "
        "all added currents from ECOILA"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(data: Path, output: Path) -> dict[str, Any]:
    """Score the fixed cohort and write incremental, receipt, and figure artifacts."""

    configure_dtypes()
    declaration_path = write_preregistration(output)
    polarity = json.loads(pinned.POLARITY_RECEIPT.read_text())["full_corpus_census"]
    affected = set(polarity["affected_shots"])
    if len(affected) != POLARITY_AFFECTED_SHOT_COUNT:
        raise RuntimeError("polarity authority is not the landed 603-shot population")
    if len(COHORT) < 5 or len({item["shot"] for item in COHORT}) != len(COHORT):
        raise RuntimeError("cohort must contain five distinct shots")
    if any(item["shot"] in affected for item in COHORT):
        raise RuntimeError("cohort includes a polarity-affected shot")
    if {"shot": "d3d_shot_00000c4a7b.parquet", "frame": 102} not in COHORT:
        raise RuntimeError("required named frame is absent")

    columns = tuple(
        dict.fromkeys(
            (
                *_LABEL_COLUMNS,
                *_CURRENT_COLUMNS,
                *_GEOMETRY_COLUMNS,
                *pinned.PLASMA_CURRENT_COLUMNS,
            )
        )
    )
    geometry = _omitted_vertices()
    checkpoint = output / CHECKPOINT_NAME
    checkpoint.write_text("")
    frames = []
    for number, selected in enumerate(COHORT, start=1):
        source = data / selected["shot"]
        row = _read(source, columns)
        row["_source_path"] = str(source)
        frame = solve_frame(row, int(selected["frame"]), geometry)
        frame.update(
            {
                "shot": selected["shot"],
                "frame": int(selected["frame"]),
                "source_sha256": _sha256(source),
            }
        )
        frames.append(frame)
        with checkpoint.open("a") as stream:
            stream.write(json.dumps(frame, sort_keys=True, allow_nan=False) + "\n")
        print(
            f"FRAME {number}/{len(COHORT)} {selected['shot']}:{selected['frame']} "
            f"verdict={frame['x_point_migration_verdict']}",
            flush=True,
        )

    result = summarize(frames)
    figure_path = output / FIGURE_NAME
    render_figure(result, figure_path)
    receipt = {
        "preregistration": preregistration(),
        "preregistration_path": str(declaration_path),
        "preregistration_sha256": _sha256(declaration_path),
        "current_provenance": (
            "Every omitted-conductor current is the same-frame shipped ECOILA "
            "value times its fixed scale. The netCDF contributes geometry and "
            "turns only; no label-recovered current is read."
        ),
        "result": result,
        "artifacts": {
            "checkpoint": str(checkpoint),
            "figure": str(figure_path),
            "receipt": str(output / RECEIPT_NAME),
        },
        "nova_equilibrium_modified": False,
    }
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = run(arguments.data, arguments.output)
    print(json.dumps(result["result"], indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
