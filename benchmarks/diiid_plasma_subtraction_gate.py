"""Score DIII-D coils against labels after represented-plasma subtraction.

The labelled flux crosses the measured corpus-convention adapter once.  Its
flux functions are separated by ``extract_flux_functions`` and evaluated back
onto the shipped grid as a toroidal-current distribution.  Cell currents are
then projected through Nova's analytic filament Green function.  Subtracting
that projected plasma flux leaves the vacuum remainder scored against the
registered conductor model on every grid node.  No response coefficient is
estimated; only the physically irrelevant additive flux gauge is removed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as PolygonPath

from benchmarks.diiid_corpus_conventions import (
    CORPUS_COCOS,
    corpus_flux_to_nova_total,
    nova_total_flux_to_corpus,
)
from benchmarks.diiid_vacuum_quiescent_gate import (
    DERIVATIVE_THRESHOLDS,
    SELECTED_THRESHOLD,
    SMOOTHING_WINDOW_MS,
    ShotCensus,
    _read,
    census,
    normalised_flux,
    select_populations,
)
from nova.biot.greens import greens_psi
from nova.equilibrium.convention import toroidal_current_density
from nova.equilibrium.map_extraction import extract_flux_functions
from nova.imas.diiid_description import (
    DiiidDescriptionRegistry,
    vacuum_psi,
    vacuum_response,
)

DEFAULT_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
DEFAULT_OUTPUT = Path("/work/projects/imas_gpu/sophelio/vacuum-gate")
DEFAULT_FIGURES = Path("docs/figures/diiid-forward-onboarding/differencing")
PREREGISTRATION_NAME = "diiid_plasma_subtraction_preregistration.json"
RESULT_NAME = "diiid_plasma_subtraction_gate.json"
PASSIVE_CORPUS_NAME = "diiid_transient_plasma_subtracted_residuals.jsonl"
RECORDED_COIL_ONLY_EXTERIOR_R2 = 0.49526707859970365
RECORDED_CONSTRUCTION_EXACT_R2 = 0.9967218639797544
REGISTERED_WHOLE_GRID_R2_BAR = RECORDED_COIL_ONLY_EXTERIOR_R2
RESPONSE_CHUNK = 128


@dataclass(frozen=True)
class FrameProjection:
    """Plasma projection and its qualification for one labelled frame."""

    plasma_flux_per_radian: np.ndarray
    normalised_flux: np.ndarray
    current_a: float
    reliable_surfaces: int
    surface_projection_rms_a_per_m: float
    current_projection_fractional_rms: float


def preregistration() -> dict[str, Any]:
    """Return the immutable selection, scoring, and pass declaration."""

    return {
        "measurement": "whole-grid coil flux versus plasma-subtracted label",
        "corpus_cocos": CORPUS_COCOS,
        "selection": {
            "derivative": (
                f"centred {SMOOTHING_WINDOW_MS:g} ms native-rate smoothed derivative"
            ),
            "thresholds_ka_turn_per_s": list(DERIVATIVE_THRESHOLDS),
            "selected_threshold_ka_turn_per_s": SELECTED_THRESHOLD,
            "frames_per_population_per_shot": 6,
            "shots": 20,
        },
        "source_model": (
            "p-prime and FF-prime extracted from each label, evaluated on the "
            "label core, one cell filament per valid grid node"
        ),
        "score_region": "all finite nodes on the shipped 65 by 65 grid",
        "gauge": "one additive constant per frame",
        "coefficients_fitted": 0,
        "registered_pooled_quiescent_r2_bar": REGISTERED_WHOLE_GRID_R2_BAR,
        "reference_scores": {
            "coil_only_exterior_r2": RECORDED_COIL_ONLY_EXTERIOR_R2,
            "construction_exact_exterior_r2": RECORDED_CONSTRUCTION_EXACT_R2,
        },
    }


def write_preregistration(output: Path) -> Path:
    """Write the declaration that must precede every corpus scoring run."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / PREREGISTRATION_NAME
    path.write_text(json.dumps(preregistration(), indent=2, sort_keys=True) + "\n")
    return path


def _canonical_axes(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    radius = np.asarray(row["efit_grid_R"], dtype=float)
    height = np.asarray(row["efit_grid_Z"], dtype=float)
    return (
        np.linspace(radius[0], radius[-1], radius.size),
        np.linspace(height[0], height[-1], height.size),
    )


def _plasma_mask(
    row: dict[str, Any], frame: int, radius: np.ndarray, height: np.ndarray
) -> np.ndarray:
    count = int(row["efit_lcfs_n"][frame])
    contour = np.column_stack(
        [
            np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
            np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
        ]
    )
    radius_map, height_map = np.meshgrid(radius, height, indexing="ij")
    points = np.column_stack([radius_map.ravel(), height_map.ravel()])
    return PolygonPath(contour).contains_points(points).reshape(radius_map.shape)


def filament_response(radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Return full-grid corpus-flux response to every grid-node filament."""

    target_r, target_z = np.meshgrid(radius, height)
    source_r, source_z = np.meshgrid(radius, height)
    flat_target_r = target_r.ravel()
    flat_target_z = target_z.ravel()
    flat_source_r = source_r.ravel()
    flat_source_z = source_z.ravel()
    response = np.empty((flat_target_r.size, flat_source_r.size), dtype=float)
    for start in range(0, flat_target_r.size, RESPONSE_CHUNK):
        stop = min(start + RESPONSE_CHUNK, flat_target_r.size)
        block = greens_psi(
            flat_target_r[start:stop, None],
            flat_target_z[start:stop, None],
            flat_source_r[None, :],
            flat_source_z[None, :],
        )
        response[start:stop] = nova_total_flux_to_corpus(block)
    response[~np.isfinite(response)] = 0.0
    return response


def project_label_plasma(
    row: dict[str, Any],
    frame: int,
    radius: np.ndarray,
    height: np.ndarray,
    response: np.ndarray,
) -> FrameProjection:
    """Extract label profiles and represent their current with cell filaments."""

    label_per_radian = np.asarray(row["efit_psirz"][frame], dtype=float)
    label_total = corpus_flux_to_nova_total(label_per_radian).T
    psi_norm_zr = normalised_flux(row, frame)
    psi_norm = psi_norm_zr.T
    core = _plasma_mask(row, frame, radius, height)
    extraction = extract_flux_functions(
        radius,
        height,
        label_total,
        psi_norm,
        plasma_mask=core,
        min_samples=6,
    )
    reliable = (
        extraction.reliable
        & np.isfinite(extraction.p_prime)
        & np.isfinite(extraction.ff_prime)
    )
    if np.count_nonzero(reliable) < 2:
        raise ValueError("fewer than two reliable flux-function surfaces")
    surface = extraction.psi_norm[reliable]
    p_prime = np.interp(psi_norm, surface, extraction.p_prime[reliable])
    ff_prime = np.interp(psi_norm, surface, extraction.ff_prime[reliable])
    radius_map = np.broadcast_to(radius[:, None], psi_norm.shape)
    reconstructed = toroidal_current_density(radius_map, p_prime, ff_prime)
    active = core & extraction.current.valid & np.isfinite(reconstructed)
    active &= np.isfinite(psi_norm) & (psi_norm >= 0.0) & (psi_norm <= 1.0)
    if not np.any(active):
        raise ValueError("profile projection contains no active grid nodes")
    direct = extraction.current.toroidal_current_density
    difference = reconstructed[active] - direct[active]
    reference = np.sqrt(np.mean(direct[active] ** 2))
    fractional_rms = float(np.sqrt(np.mean(difference**2)) / reference)
    cell_area = float(np.diff(radius).mean() * np.diff(height).mean())
    source_current = np.zeros(label_per_radian.size, dtype=float)
    source_current[active.T.ravel()] = reconstructed.T[active.T] * cell_area
    projected = (response @ source_current).reshape(label_per_radian.shape)
    return FrameProjection(
        plasma_flux_per_radian=projected,
        normalised_flux=psi_norm_zr,
        current_a=float(np.sum(source_current)),
        reliable_surfaces=int(np.count_nonzero(reliable)),
        surface_projection_rms_a_per_m=float(
            np.sqrt(np.mean(extraction.projection_rms[reliable] ** 2))
        ),
        current_projection_fractional_rms=fractional_rms,
    )


def gauge_r_squared(
    actual: np.ndarray, predicted: np.ndarray
) -> tuple[float, float, float, float, np.ndarray]:
    """Score finite full-grid values after one additive-gauge alignment."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(actual + predicted)
    actual = actual[finite]
    predicted = predicted[finite]
    gauge = float(np.mean(actual - predicted))
    residual = actual - (predicted + gauge)
    squared_error = float(np.sum(residual**2))
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return 1.0 - squared_error / total, squared_error, total, gauge, residual


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _population(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    chosen = [record for record in records if record["population"] == name]
    by_shot: dict[str, list[dict[str, Any]]] = {}
    for record in chosen:
        by_shot.setdefault(record["shot"], []).append(record)
    shot_scores = []
    for shot_records in by_shot.values():
        error = sum(item["squared_error"] for item in shot_records)
        total = sum(item["total_sum_squares"] for item in shot_records)
        shot_scores.append(1.0 - error / total)
    error = sum(item["squared_error"] for item in chosen)
    total = sum(item["total_sum_squares"] for item in chosen)
    return {
        "frames": len(chosen),
        "shots": len(by_shot),
        "pooled_r2": float(1.0 - error / total),
        "frame_r2": _distribution([item["r2"] for item in chosen]),
        "shot_r2": _distribution(shot_scores),
        "current_projection_fractional_rms": _distribution(
            [item["current_projection_fractional_rms"] for item in chosen]
        ),
        "reliable_surfaces": _distribution(
            [item["reliable_surfaces"] for item in chosen]
        ),
    }


def score(
    selected: list[ShotCensus],
    frame_sets: dict[Path, dict[str, np.ndarray]],
    output: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Score the preselected matched populations and bank transient residuals."""

    first = _read(selected[0].path)
    radius, height = _canonical_axes(first)
    projection_response = filament_response(radius, height)
    registry = DiiidDescriptionRegistry()
    response_cache: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    residual_records: list[dict[str, Any]] = []
    figure_fields: dict[str, np.ndarray] = {}
    geometry_digests: set[str] = set()

    for shot_number, shot in enumerate(selected, start=1):
        row = first if shot_number == 1 else _read(shot.path)
        description = registry.ingest(row, source_row=shot.path.name)
        geometry_digests.add(description.physical_digest)
        coil_response = response_cache.get(description.physical_digest)
        if coil_response is None:
            coil_response = vacuum_response(
                description, row["efit_grid_R"], row["efit_grid_Z"]
            )
            response_cache[description.physical_digest] = coil_response
        coil_flux = nova_total_flux_to_corpus(
            vacuum_psi(row, description, coil_response)
        )
        for population in ("quiescent", "transient"):
            for frame_value in frame_sets[shot.path][population]:
                frame = int(frame_value)
                label = np.asarray(row["efit_psirz"][frame], dtype=float)
                plasma = project_label_plasma(
                    row, frame, radius, height, projection_response
                )
                remainder = label - plasma.plasma_flux_per_radian
                prediction = coil_flux[frame]
                r2, squared_error, total, gauge, residual = gauge_r_squared(
                    remainder, prediction
                )
                record = {
                    "shot": shot.path.name,
                    "frame": frame,
                    "time_ms": float(shot.times[frame]),
                    "population": population,
                    "maximum_derivative": float(shot.maximum_derivative[frame]),
                    "whole_grid_points": int(np.count_nonzero(np.isfinite(remainder))),
                    "r2": r2,
                    "squared_error": squared_error,
                    "total_sum_squares": total,
                    "additive_gauge_wb_per_rad": gauge,
                    "represented_plasma_current_a": plasma.current_a,
                    "reliable_surfaces": plasma.reliable_surfaces,
                    "surface_projection_rms_a_per_m": (
                        plasma.surface_projection_rms_a_per_m
                    ),
                    "current_projection_fractional_rms": (
                        plasma.current_projection_fractional_rms
                    ),
                }
                records.append(record)
                if not figure_fields and population == "quiescent":
                    figure_fields = {
                        "label": label,
                        "plasma": plasma.plasma_flux_per_radian,
                        "remainder": remainder,
                        "coil": prediction + gauge,
                        "residual": residual.reshape(label.shape),
                    }
                    figure_fields["metadata"] = np.asarray(
                        [shot_number, frame, shot.maximum_derivative[frame]]
                    )
                if population == "transient":
                    residual_records.append(
                        {
                            "shot": shot.path.name,
                            "frame": frame,
                            "time_ms": float(shot.times[frame]),
                            "maximum_derivative": float(shot.maximum_derivative[frame]),
                            "shape": list(label.shape),
                            "residual_wb_per_rad": residual.tolist(),
                        }
                    )
        print(f"SCORED {shot_number}/{len(selected)}", flush=True)

    passive_path = output / PASSIVE_CORPUS_NAME
    passive_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in residual_records)
    )
    quiet = _population(records, "quiescent")
    transient = _population(records, "transient")
    return (
        {
            "geometry_digests": sorted(geometry_digests),
            "geometry_provenance_complete": all(
                item.provenance_complete for item in registry.configurations.values()
            ),
            "source_projection": (
                "label p-prime and FF-prime evaluated on core grid nodes and "
                "projected through cell-filament Green functions"
            ),
            "score_region": "whole shipped grid",
            "coefficients_fitted": 0,
            "quiescent": quiet,
            "transient": transient,
            "registered_bar": REGISTERED_WHOLE_GRID_R2_BAR,
            "passed": bool(quiet["pooled_r2"] >= REGISTERED_WHOLE_GRID_R2_BAR),
            "references": {
                "coil_only_exterior_r2": RECORDED_COIL_ONLY_EXTERIOR_R2,
                "construction_exact_exterior_r2": RECORDED_CONSTRUCTION_EXACT_R2,
            },
            "transient_residual_corpus": str(passive_path),
            "transient_residual_frames": len(residual_records),
            "frame_records": records,
        },
        figure_fields,
    )


def selection_figure(
    selected: list[ShotCensus],
    frame_sets: dict[Path, dict[str, np.ndarray]],
    path: Path,
) -> None:
    """Plot the fixed population selection against the derivative predicate."""

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, shot in zip(axes.ravel(), selected[:4], strict=True):
        axis.plot(shot.times, shot.maximum_derivative, color="0.5", linewidth=0.8)
        for population, marker, colour in (
            ("quiescent", "o", "tab:blue"),
            ("transient", "x", "tab:red"),
        ):
            frames = frame_sets[shot.path][population]
            axis.scatter(
                shot.times[frames],
                shot.maximum_derivative[frames],
                marker=marker,
                color=colour,
                label=population,
            )
        for threshold in DERIVATIVE_THRESHOLDS:
            axis.axhline(threshold, color="0.75", linewidth=0.6)
        axis.set_yscale("log")
        axis.set_title(shot.path.stem)
        axis.set_xlabel("label time [ms]")
        axis.set_ylabel("maximum smoothed |dI/dt| [kA turn/s]")
    axes[0, 0].legend()
    figure.suptitle("Preregistered quiescent and matched-transient populations")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def differencing_figure(fields: dict[str, np.ndarray], path: Path) -> None:
    """Plot one complete label-to-vacuum differencing receipt."""

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    names = ("label", "plasma", "remainder", "coil", "residual")
    titles = (
        "Label",
        "Projected plasma",
        "Vacuum remainder",
        "Coil model + gauge",
        "Remainder - coil",
    )
    for axis, name, title in zip(axes.ravel(), names, titles, strict=False):
        image = axis.imshow(fields[name], origin="lower", aspect="auto")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label="Wb/rad")
    axes.ravel()[-1].axis("off")
    shot_number, frame, derivative = fields["metadata"]
    figure.suptitle(
        f"Whole-grid plasma subtraction: selected shot {int(shot_number)}, "
        f"frame {int(frame)}, max |dI/dt|={derivative:.2f} kA turn/s"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--census-limit", type=int)
    parser.add_argument("--preregister-only", action="store_true")
    args = parser.parse_args()
    preregistration_path = write_preregistration(args.output)
    print(f"PREREGISTERED {preregistration_path}", flush=True)
    if args.preregister_only:
        return
    paths = sorted(args.data.glob("*.parquet"))
    if args.census_limit is not None:
        paths = paths[: args.census_limit]
    shots, census_result = census(paths)
    selected, frame_sets = select_populations(shots, shot_count=args.shots)
    score_result, fields = score(selected, frame_sets, args.output)
    result = {
        "preregistration": preregistration(),
        "preregistration_path": str(preregistration_path),
        "census": census_result,
        "score": score_result,
    }
    result_path = args.output / RESULT_NAME
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.figures.mkdir(parents=True, exist_ok=True)
    selection_figure(selected, frame_sets, args.figures / "quiescent_selection.png")
    differencing_figure(fields, args.figures / "plasma_differencing.png")
    headline = dict(result)
    headline["score"] = {
        key: value for key, value in score_result.items() if key != "frame_records"
    }
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
