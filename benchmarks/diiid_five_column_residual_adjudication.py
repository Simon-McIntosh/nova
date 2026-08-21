"""Adjudicate DIII-D five-column recovery with interior current closure.

The fixed cohort, exact clipped-cell plasma field, released-conductor registry,
netCDF-only conductor geometry, polarity census, and current solver are all
composed from their landed implementations.  Every selected map is scored at
three subtraction states with one centred-stencil Delta-star operator.  Grid
cells without a complete stencil are reported separately and never contribute
to the interior current comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from benchmarks import diiid_boundary_current_recovery as recovery
from benchmarks import diiid_exact_clipped_tare as tare
from benchmarks import diiid_negative_tail_attribution as attribution
from benchmarks.diiid_corpus_conventions import (
    nova_total_flux_to_corpus,
)
from benchmarks.diiid_forward_gs_match import (
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
)
from nova.biot.polygon import polygon_greens
from nova.equilibrium.map_extraction import apply_delta_star
from nova.imas.diiid_description import DiiidDescriptionRegistry, vacuum_response
from nova.jax.config import configure_dtypes


DEFAULT_DATA = tare.DEFAULT_DATA
DEFAULT_OUTPUT = Path("docs/figures/coil-circuit-discovery")
SOURCE_RECEIPT = tare.DEFAULT_OUTPUT / tare.RECEIPT_NAME
OHMIC_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/ohmic-circuit/"
    "ohmic_circuit_retest_receipt.json"
)
REUSE_MAP = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/"
    "coil-circuit-discovery/reuse-map.md"
)
DIAGNOSTIC_MANIFEST = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20260821T204446383949-ohmic-circuit-driven-vacuum-field/manifest.md"
)
RECEIPT_NAME = "five_column_residual_adjudication_receipt.json"
COHORT_FIGURE_NAME = "residual_stage_comparison.png"
SPATIAL_FIGURE_NAME = "residual_spatial_classification.png"
EXACT_TARE_FLOOR_PERCENT = 0.4841505
STAGE_NAMES = (
    "after_exact_clipped_cell_tare",
    "after_released_conductor_removal",
    "after_five_column_recovery",
)
READ_COLUMNS = tuple(
    dict.fromkeys(
        (*tare.READ_COLUMNS, *_LABEL_COLUMNS, *_CURRENT_COLUMNS, *_GEOMETRY_COLUMNS)
    )
)


@dataclass(frozen=True)
class BankedFrame:
    """One exact-tare cohort member read from the committed source receipt."""

    path: Path
    frame: int
    time_ms: float
    expected_fraction: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def banked_cohort(
    data: Path, source_receipt: Path
) -> tuple[list[BankedFrame], dict[str, Any]]:
    """Load the exact sixty-frame authority without reselecting its members."""

    source = json.loads(source_receipt.read_text(encoding="utf-8"))
    records = source["records"]
    if source["selection"]["frames"] != 60 or source["selection"]["shots"] != 20:
        raise RuntimeError("the exact-tare authority is not the banked 60-frame cohort")
    if not source["selection"]["all_selected_absent_from_polarity_population"]:
        raise RuntimeError("the exact-tare authority is not fully polarity screened")
    cohort = [
        BankedFrame(
            path=data / item["shot"],
            frame=int(item["frame"]),
            time_ms=float(item["time_ms"]),
            expected_fraction=float(
                item["exact_clipped_moments"][
                    "absolute_signed_fraction_of_extracted_current"
                ]
            ),
        )
        for item in records
    ]
    if len(cohort) != 60 or len({item.path.name for item in cohort}) != 20:
        raise RuntimeError("the exact-tare frame list is incomplete")
    missing = [str(item.path) for item in cohort if not item.path.is_file()]
    if missing:
        raise FileNotFoundError(f"banked cohort files are missing: {missing[:3]}")
    return cohort, source


def _omitted_full_response(
    radius: np.ndarray, height: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return exact polygon flux maps for the five netCDF-only conductors."""

    import imas

    target_r, target_z = np.meshgrid(radius, height)
    maps = []
    records = []
    with imas.DBEntry(
        recovery.NETCDF_ENTRY,
        "r",
        dd_version=recovery.NETCDF_DD_VERSION,
    ) as entry:
        active = entry.get("pf_active", autoconvert=False)
        written_dd = str(active.ids_properties.version_put.data_dictionary)
        if written_dd != recovery.NETCDF_DD_VERSION:
            raise RuntimeError(
                f"expected DD {recovery.NETCDF_DD_VERSION}, read {written_dd}"
            )
        coils = {str(coil.name): coil for coil in active.coil}
        for name in recovery.OMITTED_COILS:
            flux = np.zeros(target_r.shape, dtype=float)
            turn_sum = 0.0
            for element in coils[name].element:
                geometry = element.geometry
                geometry_type = int(geometry.geometry_type)
                if geometry_type == 1:
                    vertices = np.c_[
                        np.asarray(geometry.outline.r, dtype=float),
                        np.asarray(geometry.outline.z, dtype=float),
                    ]
                elif geometry_type == 2:
                    vertices = recovery._rectangle_vertices(geometry)
                else:
                    raise ValueError(
                        f"unsupported geometry type {geometry_type} for {name}"
                    )
                turns = float(element.turns_with_sign)
                turn_sum += turns
                flux += turns * polygon_greens(
                    target_r.ravel(), target_z.ravel(), vertices
                )[0].reshape(target_r.shape)
            maps.append(flux)
            records.append(
                {
                    "coil": name,
                    "elements": len(coils[name].element),
                    "signed_turn_sum": turn_sum,
                }
            )
    return np.stack(maps), {
        "entry": str(recovery.NETCDF_ENTRY),
        "dd_version": recovery.NETCDF_DD_VERSION,
        "coils": records,
        "grid_points": int(target_r.size),
        "kernel": "nova.biot.polygon.polygon_greens",
    }


def current_residual_metrics(
    radius: np.ndarray,
    height: np.ndarray,
    residual_total_zr: np.ndarray,
    plasma_mask_rz: np.ndarray,
    extracted_current_a: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Measure current closure on complete stencils and qualify excluded edges."""

    operator = apply_delta_star(radius, height, np.asarray(residual_total_zr).T)
    density = np.asarray(operator.toroidal_current_density, dtype=float)
    valid = np.asarray(operator.valid, dtype=bool) & np.isfinite(density)
    selected = valid & np.asarray(plasma_mask_rz, dtype=bool)
    area = float(np.mean(np.diff(radius)) * np.mean(np.diff(height)))
    signed_current = float(np.sum(density[selected]) * area)
    absolute_current = float(np.sum(np.abs(density[selected])) * area)
    scale = max(abs(float(extracted_current_a)), np.finfo(float).tiny)
    flux_rz = np.asarray(residual_total_zr, dtype=float).T
    edge = ~np.asarray(operator.valid, dtype=bool)
    centred_flux = flux_rz - float(np.mean(flux_rz[valid]))
    metrics = {
        "signed_residual_current_a": signed_current,
        "absolute_residual_current_a": absolute_current,
        "absolute_signed_fraction_of_extracted_current": abs(signed_current) / scale,
        "absolute_signed_percent_of_extracted_current": 100.0
        * abs(signed_current)
        / scale,
        "l1_fraction_of_extracted_current": absolute_current / scale,
        "interior_valid_nodes": int(np.count_nonzero(selected)),
        "complete_stencil_nodes": int(np.count_nonzero(valid)),
        "edge_stencil": {
            "nodes": int(np.count_nonzero(edge)),
            "included_in_interior_current_comparison": False,
            "delta_star_value": None,
            "reason": (
                "a centred Delta-star stencil is incomplete in the outer grid bands"
            ),
            "gauge_centred_flux_rms_wb": float(
                np.sqrt(np.mean(np.square(centred_flux[edge])))
            ),
            "gauge_centred_flux_maximum_absolute_wb": float(
                np.max(np.abs(centred_flux[edge]))
            ),
        },
    }
    return metrics, density, valid


def classify_spatial_residual(
    density_rz: np.ndarray,
    valid_rz: np.ndarray,
    residual_total_zr: np.ndarray,
    omitted_total_response: np.ndarray,
) -> dict[str, Any]:
    """Classify surviving structure by edge, smooth, and conductor projections."""

    density = np.where(valid_rz, np.asarray(density_rz, dtype=float), 0.0)
    energy = float(np.sum(np.square(density)))
    radial, vertical = np.indices(density.shape)
    distance = np.minimum.reduce(
        [
            radial,
            vertical,
            density.shape[0] - 1 - radial,
            density.shape[1] - 1 - vertical,
        ]
    )
    edge_band = valid_rz & (distance <= 3)
    edge_fraction = (
        float(np.sum(np.square(density[edge_band])) / energy) if energy > 0.0 else 0.0
    )
    low_pass = gaussian_filter(density, sigma=2.0, mode="nearest")
    smooth_fraction = (
        float(np.sum(np.square(low_pass[valid_rz])) / energy) if energy > 0.0 else 0.0
    )
    flux = np.asarray(residual_total_zr, dtype=float).ravel()
    design = np.asarray(omitted_total_response, dtype=float).reshape(5, -1).T
    finite = np.isfinite(flux) & np.all(np.isfinite(design), axis=1)
    matrix = design[finite]
    values = flux[finite]
    matrix = matrix - np.mean(matrix, axis=0)
    values = values - np.mean(values)
    coefficient, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    fitted = matrix @ coefficient
    flux_energy = float(values @ values)
    conductor_fraction = (
        float(np.clip((fitted @ fitted) / flux_energy, 0.0, 1.0))
        if flux_energy > 0.0
        else 0.0
    )
    scores = {
        "edge_concentrated": edge_fraction,
        "smooth": smooth_fraction,
        "conductor_like": conductor_fraction,
    }
    classification = max(scores, key=scores.__getitem__)
    return {
        "classification": classification,
        "scores": scores,
        "definitions": {
            "edge_concentrated": (
                "fraction of Delta-star energy in the outer three "
                "complete-stencil rings"
            ),
            "smooth": (
                "fraction of Delta-star energy retained by a two-cell Gaussian low pass"
            ),
            "conductor_like": (
                "fraction of gauge-centred flux energy projected onto the five "
                "exact polygon maps"
            ),
        },
        "edge_band_nodes": int(np.count_nonzero(edge_band)),
        "projection_currents_a": {
            name: float(value)
            for name, value in zip(recovery.OMITTED_COILS, coefficient, strict=True)
        },
    }


def _manifest_context(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "terminal": False, "status": "missing"}
    text = path.read_text(encoding="utf-8")
    status = "unknown"
    for line in text.splitlines():
        if line.lower().startswith("status:"):
            status = line.split(":", 1)[1].strip().lower()
            break
    terminal = status in {"complete", "blocked", "failed"}
    result = {
        "path": str(path),
        "sha256": _sha256(path),
        "terminal": terminal,
        "status": status,
    }
    if terminal:
        result["content"] = text
    return result


def _render_cohort(records: list[dict[str, Any]], output: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    positions = np.arange(len(records))
    colors = ("#4477aa", "#ee7733", "#228833")
    labels = ("exact tare", "released removed", "five columns removed")
    for name, color, label in zip(STAGE_NAMES, colors, labels, strict=True):
        values = [
            record["residuals"][name]["absolute_signed_percent_of_extracted_current"]
            for record in records
        ]
        axes[0].plot(positions, values, color=color, lw=1.0, alpha=0.85, label=label)
    axes[0].axhline(EXACT_TARE_FLOOR_PERCENT, color="black", ls="--", lw=1.0)
    axes[0].set_xlabel("Banked frame")
    axes[0].set_ylabel("|signed interior residual| / |extracted current| [%]")
    axes[0].legend(frameon=False, fontsize=8)
    before = np.asarray(
        [
            record["residuals"][STAGE_NAMES[0]][
                "absolute_signed_percent_of_extracted_current"
            ]
            for record in records
        ]
    )
    after = np.asarray(
        [
            record["residuals"][STAGE_NAMES[2]][
                "absolute_signed_percent_of_extracted_current"
            ]
            for record in records
        ]
    )
    axes[1].scatter(before, after, s=22, color="#228833", alpha=0.8)
    limit = float(max(np.max(before), np.max(after), EXACT_TARE_FLOOR_PERCENT))
    axes[1].plot([0.0, limit], [0.0, limit], color="black", ls=":", lw=1.0)
    axes[1].axhline(EXACT_TARE_FLOOR_PERCENT, color="black", ls="--", lw=1.0)
    axes[1].set_xlabel("After exact tare [%]")
    axes[1].set_ylabel("After five-column recovery [%]")
    path = output / COHORT_FIGURE_NAME
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _render_spatial(
    record: dict[str, Any], fields: dict[str, np.ndarray], output: Path
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    densities = [fields[name] for name in STAGE_NAMES]
    finite = np.concatenate([value[np.isfinite(value)] for value in densities])
    limit = float(np.quantile(np.abs(finite), 0.99))
    image = None
    titles = ("Exact tare", "Released conductors removed", "Five columns removed")
    for axis, density, title in zip(axes, densities, titles, strict=True):
        image = axis.imshow(
            np.ma.masked_invalid(density).T / 1.0e6,
            origin="lower",
            cmap="coolwarm",
            vmin=-limit / 1.0e6,
            vmax=limit / 1.0e6,
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel("radial index")
    axes[0].set_ylabel("vertical index")
    assert image is not None
    figure.colorbar(
        image, ax=axes, shrink=0.82, label="residual current density [MA m⁻²]"
    )
    spatial = record["spatial_classification"]
    classification = spatial["classification"].replace("_", "-")
    figure.suptitle(
        f"{record['shot']} frame {record['frame']}: {classification} residual"
    )
    path = output / SPATIAL_FIGURE_NAME
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def run(
    data: Path = DEFAULT_DATA,
    output: Path = DEFAULT_OUTPUT,
    source_receipt: Path = SOURCE_RECEIPT,
) -> dict[str, Any]:
    """Run the complete sixty-frame residual adjudication."""

    configure_dtypes()
    output.mkdir(parents=True, exist_ok=True)
    cohort, source = banked_cohort(data, source_receipt)
    affected = tare.polarity_population()
    if any(item.path.name in affected for item in cohort):
        raise RuntimeError("a polarity-affected shot survived the banked cohort")
    rows = {
        name: tare._read(data / name, READ_COLUMNS)
        for name in sorted({item.path.name for item in cohort})
    }
    first = rows[cohort[0].path.name]
    radius, height = tare.canonical_axes(first)
    mesh, geometry, width, vertical_extent = tare.rectangular_geometry(radius, height)
    prepared = [
        tare.prepare_frame(item, rows[item.path.name], radius, height)
        for item in cohort
    ]
    source_mask = np.any(
        np.stack([frame.participation_zr.reshape(-1) for frame in prepared]), axis=0
    )
    source_indices = np.flatnonzero(source_mask & np.asarray(mesh.interior()))
    blocks = tare.response_blocks(mesh, source_indices, width, vertical_extent, 1)
    integrate = tare.moment_integrator(mesh, geometry)
    description = DiiidDescriptionRegistry().ingest(
        first, source_row=cohort[0].path.name
    )
    released_names, released_response = vacuum_response(description, radius, height)
    omitted_border, border_geometry = recovery.omitted_response(radius, height)
    omitted_total, full_geometry = _omitted_full_response(radius, height)
    border_mask = recovery.polarity._boundary_mask(radius, height)
    if not np.allclose(
        nova_total_flux_to_corpus(omitted_total[:, border_mask].T),
        omitted_border,
        rtol=2.0e-12,
        atol=2.0e-15,
    ):
        raise RuntimeError("full-grid and reusable border responses disagree")

    records: list[dict[str, Any]] = []
    representative_fields: dict[str, np.ndarray] = {}
    for prepared_frame, banked in zip(prepared, cohort, strict=True):
        vectors = integrate(
            prepared_frame.psi_norm_zr,
            prepared_frame.participation_zr,
            prepared_frame.profile_surface,
            prepared_frame.p_prime,
            prepared_frame.ff_prime,
        )
        exact_current, exact_radial, exact_vertical, _boundary = (
            np.asarray(value) for value in jax.block_until_ready(vectors)
        )
        exact_plasma_zr = (
            blocks[0] @ exact_current[source_indices]
            + blocks[1] @ exact_radial[source_indices]
            + blocks[2] @ exact_vertical[source_indices]
        ).reshape(prepared_frame.label_total_zr.shape)
        extracted_current_a = float(np.sum(exact_current))
        exact_tared_zr = prepared_frame.label_total_zr - exact_plasma_zr
        row = rows[banked.path.name]
        described = DiiidDescriptionRegistry().ingest(row, source_row=banked.path.name)
        if described.physical_digest != description.physical_digest:
            raise RuntimeError(
                "the banked cohort contains multiple released geometries"
            )
        released_currents = attribution._current_vector(
            row, described, released_names, banked.time_ms
        )
        released_flux_zr = np.einsum(
            "c,czr->zr", released_currents, released_response, optimize=True
        )
        released_removed_zr = exact_tared_zr - released_flux_zr
        solved = recovery.recover_currents(
            omitted_border,
            nova_total_flux_to_corpus(released_removed_zr)[border_mask],
        )
        recovered_currents = np.asarray(solved["currents_a"], dtype=float)
        recovered_flux_zr = np.einsum(
            "c,czr->zr", recovered_currents, omitted_total, optimize=True
        )
        recovered_removed_zr = released_removed_zr - recovered_flux_zr
        maps = {
            STAGE_NAMES[0]: exact_tared_zr,
            STAGE_NAMES[1]: released_removed_zr,
            STAGE_NAMES[2]: recovered_removed_zr,
        }
        residuals = {}
        densities = {}
        valid = None
        for name, residual_map in maps.items():
            metrics, density, current_valid = current_residual_metrics(
                radius,
                height,
                residual_map,
                prepared_frame.core_rz,
                extracted_current_a,
            )
            residuals[name] = metrics
            densities[name] = density
            valid = current_valid
        reproduced = residuals[STAGE_NAMES[0]][
            "absolute_signed_fraction_of_extracted_current"
        ]
        if not np.isclose(
            reproduced, banked.expected_fraction, rtol=2.0e-9, atol=2.0e-12
        ):
            raise RuntimeError(
                f"{banked.path.name} frame {banked.frame}: "
                "exact-tare receipt did not reproduce"
            )
        assert valid is not None
        spatial = classify_spatial_residual(
            densities[STAGE_NAMES[2]],
            valid,
            recovered_removed_zr,
            omitted_total,
        )
        records.append(
            {
                "shot": banked.path.name,
                "frame": banked.frame,
                "time_ms": banked.time_ms,
                "absent_from_landed_603_shot_polarity_population": True,
                "extracted_plasma_current_a": extracted_current_a,
                "recorded_plasma_current_a": prepared_frame.plasma_current_a,
                "normalisation_below_50ka_qualified": abs(
                    prepared_frame.plasma_current_a
                )
                < 50_000.0,
                "design_condition_number": float(solved["design_condition_number"]),
                "design_rank": int(solved["design_rank"]),
                "boundary_relative_residual": float(solved["relative_residual"]),
                "boundary_residual_rms_wb_per_radian": float(
                    solved["residual_rms_wb_per_radian"]
                ),
                "additive_gauge_wb_per_radian": float(solved["gauge_wb_per_radian"]),
                "recovered_currents_a": {
                    name: float(value)
                    for name, value in zip(
                        recovery.OMITTED_COILS, recovered_currents, strict=True
                    )
                },
                "released_currents_a_or_ampere_turn": {
                    name: float(value)
                    for name, value in zip(
                        released_names, released_currents, strict=True
                    )
                },
                "residuals": residuals,
                "spatial_classification": spatial,
            }
        )
        if not representative_fields:
            representative_fields = {
                name: np.where(valid, density, np.nan)
                for name, density in densities.items()
            }

    stage_summaries = {}
    for name in STAGE_NAMES:
        fractions = [
            item["residuals"][name]["absolute_signed_fraction_of_extracted_current"]
            for item in records
        ]
        percents = [100.0 * value for value in fractions]
        stage_summaries[name] = {
            "fraction_of_extracted_current": _distribution(fractions),
            "percent_of_extracted_current": _distribution(percents),
            "frames_at_or_below_exact_tare_floor": int(
                np.count_nonzero(np.asarray(percents) <= EXACT_TARE_FLOOR_PERCENT)
            ),
            "frames": len(records),
        }
    final_median = stage_summaries[STAGE_NAMES[2]]["percent_of_extracted_current"][
        "median"
    ]
    final_ratio = final_median / EXACT_TARE_FLOOR_PERCENT
    released_median = stage_summaries[STAGE_NAMES[1]]["percent_of_extracted_current"][
        "median"
    ]
    final_frames_at_floor = stage_summaries[STAGE_NAMES[2]][
        "frames_at_or_below_exact_tare_floor"
    ]
    final_frames_above_floor = len(records) - final_frames_at_floor
    structure_counts = {
        name: sum(
            item["spatial_classification"]["classification"] == name for item in records
        )
        for name in ("edge_concentrated", "smooth", "conductor_like")
    }
    closes_to_floor = final_frames_above_floor == 0
    if closes_to_floor:
        authority = "label_flux_recovery"
        statement = (
            "The median post-recovery interior residual reaches the exact-tare floor; "
            "under the declared adjudication rule the label-flux recovery is believed "
            "and the independently measured ohmic relation requires re-examination."
        )
    else:
        authority = "ohmic_circuit_relation"
        statement = (
            "Structured interior residual remains above the exact-tare floor after "
            "five-column recovery; the recovered coefficients are not physical "
            "currents, so the independently measured ohmic circuit relation is "
            "believed."
        )
    diagnostic = _manifest_context(DIAGNOSTIC_MANIFEST)
    receipt = {
        "measurement": "DIII-D five-column interior residual adjudication",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection": {
            "shots": len({item["shot"] for item in records}),
            "frames": len(records),
            "source": str(source_receipt),
            "source_sha256": _sha256(source_receipt),
            "all_frames_reproduced_from_banked_exact_tare_cohort": True,
            "landed_polarity_population_count": len(affected),
            "all_frames_absent_from_polarity_population": True,
        },
        "comparison": {
            "exact_tare_floor_percent": EXACT_TARE_FLOOR_PERCENT,
            "metric": (
                "absolute signed interior Delta-star residual as percent of exact "
                "extracted plasma current"
            ),
            "interior": "labelled LCFS nodes with complete centred Delta-star stencils",
            "edge_stencil_treatment": (
                "outer grid bands reported by flux statistics only and excluded "
                "from every current comparison"
            ),
            "states": list(STAGE_NAMES),
        },
        "stage_summaries": stage_summaries,
        "spatial_summary": {
            "classification_counts": structure_counts,
            "representative_frame": {
                "shot": records[0]["shot"],
                "frame": records[0]["frame"],
                **records[0]["spatial_classification"],
            },
        },
        "conditioning": {
            "design_rank": sorted({item["design_rank"] for item in records}),
            "design_condition_number": _distribution(
                [item["design_condition_number"] for item in records]
            ),
            "boundary_relative_residual": _distribution(
                [item["boundary_relative_residual"] for item in records]
            ),
        },
        "current_disagreement": {
            "label_flux_recovered_spread": 2.20,
            "independent_ohmic_scale_spread": 1.04,
            "post_recovery_median_to_exact_tare_floor_ratio": final_ratio,
            "post_recovery_frames_at_or_below_exact_tare_floor": (
                final_frames_at_floor
            ),
            "post_recovery_frames_above_exact_tare_floor": (final_frames_above_floor),
            "five_column_median_change_from_released_removed_percentage_points": (
                final_median - released_median
            ),
            "declared_closure_rule": (
                "every banked frame must reach the exact-tare floor; any surviving "
                "classified structure rejects the recovered coefficients as currents"
            ),
            "believed_authority": authority,
            "verdict": statement,
            "ohmic_receipt": str(OHMIC_RECEIPT),
            "ohmic_receipt_sha256": _sha256(OHMIC_RECEIPT),
            "ohmic_qualification": (
                "two of five traces pass the one-percent maximum-residual "
                "determinism bound"
            ),
            "inference_qualification": (
                "the ohmic relation is inference-admissible but transfers scales "
                "from a different netCDF pulse; label recovery matches the labelled "
                "vacuum better but consumes the labelled equilibrium"
            ),
            "corroborating_diagnostic_conflict": (
                "the terminal circuit diagnostic reports circuit median vacuum RMS "
                "10.587909 times the label-recovered control; this does not override "
                "the per-frame interior and spatial rejection"
            ),
        },
        "reuse_authority": {
            "map": str(REUSE_MAP),
            "sha256": _sha256(REUSE_MAP),
            "composed_capabilities": [
                "exact clipped-cell plasma tare",
                "centred-stencil Delta-star current density",
                "released registry polygon response",
                "netCDF-only exact polygon response",
                "gauge-free five-column current recovery",
                "independent ohmic relation",
                "complete corpus polarity screen",
            ],
            "diagnostic_manifest": diagnostic,
        },
        "geometry": {
            "grid_shape_zr": [int(height.size), int(radius.size)],
            "released_conductors": list(released_names),
            "omitted_border_response": border_geometry,
            "omitted_full_response": full_geometry,
            "exact_tare_geometry": source["geometry"],
        },
        "records": records,
    }
    cohort_figure = _render_cohort(records, output)
    spatial_figure = _render_spatial(records[0], representative_fields, output)
    receipt["artifacts"] = {
        "receipt": str(output / RECEIPT_NAME),
        "cohort_figure": str(cohort_figure),
        "spatial_figure": str(spatial_figure),
    }
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if len(records) != 60 or len({item["shot"] for item in records}) != 20:
        raise RuntimeError("the adjudication did not retain the complete banked cohort")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-receipt", type=Path, default=SOURCE_RECEIPT)
    args = parser.parse_args()
    result = run(args.data, args.output, args.source_receipt)
    print(
        json.dumps(
            {
                "frames": result["selection"]["frames"],
                "shots": result["selection"]["shots"],
                "post_recovery_median_percent": result["stage_summaries"][
                    STAGE_NAMES[2]
                ]["percent_of_extracted_current"]["median"],
                "exact_tare_floor_percent": EXACT_TARE_FLOOR_PERCENT,
                "believed_authority": result["current_disagreement"][
                    "believed_authority"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
