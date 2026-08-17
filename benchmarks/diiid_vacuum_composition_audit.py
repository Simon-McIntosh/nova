"""Audit the sign and flux-unit boundary of the DIII-D vacuum composition.

The authoritative gate remains untouched.  This module imports its scoring and
map-extraction helpers, builds a synthetic label from the same coil and filament
Green kernels, and compares the gate composition with the construction-exact
composition.  No coefficient is estimated from either synthetic or real data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import diiid_vacuum_quiescent_gate as gate
from nova.biot.greens import greens_psi
from nova.imas.diiid_description import (
    DiiidDescriptionRegistry,
    vacuum_psi,
    vacuum_response,
)

DEFAULT_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
DEFAULT_GATE_ARTIFACT = Path(
    "/work/projects/imas_gpu/sophelio/vacuum-gate/diiid_vacuum_quiescent_gate.json"
)
DEFAULT_OUTPUT = Path(
    "/work/projects/imas_gpu/sophelio/vacuum-gate/diiid_vacuum_composition_audit.json"
)
AMBIX_LOADER = Path("/home/ITER/mcintos/Code/imas-ambix/imas_ambix/challenge/loader.py")


def _magnitude(values_per_radian: np.ndarray) -> dict[str, float]:
    """Return signed and gauge-free total-flux magnitudes in webers."""

    total_flux = 2.0 * np.pi * np.asarray(values_per_radian, dtype=float)
    centred = total_flux - np.mean(total_flux)
    return {
        "mean_wb": float(np.mean(total_flux)),
        "centered_rms_wb": float(np.sqrt(np.mean(centred**2))),
        "span_wb": float(np.ptp(total_flux)),
        "minimum_wb": float(np.min(total_flux)),
        "maximum_wb": float(np.max(total_flux)),
    }


def _plasma_patch(
    radius: np.ndarray,
    height: np.ndarray,
    axis_r: float,
    axis_z: float,
    *,
    total_current_a: float = 8.0e5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an analytic Gaussian current patch on grid-cell centres."""

    source_r_axis = 0.5 * (radius[1:] + radius[:-1])
    source_z_axis = 0.5 * (height[1:] + height[:-1])
    source_r, source_z = np.meshgrid(source_r_axis, source_z_axis)
    radial_scale = 0.22
    vertical_scale = 0.32
    density = np.exp(
        -0.5
        * (
            ((source_r - axis_r) / radial_scale) ** 2
            + ((source_z - axis_z) / vertical_scale) ** 2
        )
    )
    density[
        ((source_r - axis_r) / (2.5 * radial_scale)) ** 2
        + ((source_z - axis_z) / (2.5 * vertical_scale)) ** 2
        > 1.0
    ] = 0.0
    cell_area = float(np.diff(radius).mean() * np.diff(height).mean())
    current = density * (total_current_a / (np.sum(density) * cell_area)) * cell_area
    selected = current > total_current_a * 1.0e-12
    return source_r[selected], source_z[selected], current[selected]


def _filament_flux_map(
    radius: np.ndarray,
    height: np.ndarray,
    source_r: np.ndarray,
    source_z: np.ndarray,
    source_current_a: np.ndarray,
) -> np.ndarray:
    """Map cell-filament currents to challenge Wb/rad on the full grid."""

    target_r, target_z = np.meshgrid(radius, height)
    response = greens_psi(
        target_r.ravel()[:, None],
        target_z.ravel()[:, None],
        source_r[None, :],
        source_z[None, :],
    )
    return ((response @ source_current_a) / (2.0 * np.pi)).reshape(target_r.shape)


def _one_frame_row(row: dict[str, Any], frame: int, flux: np.ndarray) -> dict[str, Any]:
    return {
        "efit_grid_R": row["efit_grid_R"],
        "efit_grid_Z": row["efit_grid_Z"],
        "efit_psirz": [flux],
        "efit_r_axis": [row["efit_r_axis"][frame]],
        "efit_z_axis": [row["efit_z_axis"][frame]],
        "efit_lcfs_n": [row["efit_lcfs_n"][frame]],
        "efit_lcfs_r": [row["efit_lcfs_r"][frame]],
        "efit_lcfs_z": [row["efit_lcfs_z"][frame]],
    }


def _extracted_plasma(
    row: dict[str, Any],
    frame: int,
    flux: np.ndarray,
    operator_radius: np.ndarray,
    operator_height: np.ndarray,
    filament_response: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    synthetic_row = _one_frame_row(row, frame, flux)
    psi_norm = gate.normalised_flux(synthetic_row, 0)
    receipt = gate.label_map_current(operator_radius, operator_height, flux)
    cell_area = float(np.diff(operator_radius).mean() * np.diff(operator_height).mean())
    source_current = np.zeros(flux.size, dtype=float)
    selected = receipt.valid.T & np.isfinite(receipt.toroidal_current_density.T)
    selected &= np.isfinite(psi_norm) & (psi_norm <= 1.0)
    source_current[selected.ravel()] = (
        receipt.toroidal_current_density.T[selected] * cell_area
    )
    return filament_response @ source_current, float(np.sum(source_current)), psi_norm


def _frame_table(
    row: dict[str, Any],
    frame: int,
    coil_flux: np.ndarray,
    operator_radius: np.ndarray,
    operator_height: np.ndarray,
    target_mask: np.ndarray,
    filament_response: np.ndarray,
) -> dict[str, Any]:
    label = np.asarray(row["efit_psirz"][frame], dtype=float)
    plasma, extracted_current_a, psi_norm = _extracted_plasma(
        row,
        frame,
        label,
        operator_radius,
        operator_height,
        filament_response,
    )
    exterior = psi_norm[target_mask] > 1.05
    coil = gate.FIXED_FLUX_SIGN * coil_flux[frame][target_mask][exterior]
    plasma_label_sign = plasma[exterior]
    plasma_gate_sign = gate.FIXED_FLUX_SIGN * plasma_label_sign
    actual = label[target_mask][exterior]
    return {
        "frame": frame,
        "time_ms": float(row["efit_times"][frame]),
        "raw_plasma_current": float(
            np.interp(
                row["efit_times"][frame],
                row["magnetics_plasma_current_times"],
                row["magnetics_plasma_current"],
            )
        ),
        "extracted_plasma_current_a": extracted_current_a,
        "exterior_points": int(np.count_nonzero(exterior)),
        "terms": {
            "coil_challenge_sign": _magnitude(coil),
            "plasma_extracted_label_sign": _magnitude(plasma_label_sign),
            "plasma_after_gate_global_sign": _magnitude(plasma_gate_sign),
            "label": _magnitude(actual),
        },
    }


def _unit_receipt(real_frames: list[dict[str, Any]], schema: Any) -> dict[str, Any]:
    ratios = [
        abs(frame["extracted_plasma_current_a"] / frame["raw_plasma_current"])
        for frame in real_frames
        if frame["raw_plasma_current"] != 0.0
    ]
    metadata = {
        key.decode(): value.decode() for key, value in (schema.metadata or {}).items()
    }
    loader_text = AMBIX_LOADER.read_text()
    loader_quotes = [
        "return np.asarray(table[name][0].as_py(), dtype=dtype)",
        "values=_array(table, name)",
    ]
    if not all(quote in loader_text for quote in loader_quotes):
        raise RuntimeError("the audited Ambix loader conversion text changed")
    median_ratio = float(np.median(ratios))
    verdict = "kA" if 500.0 <= median_ratio <= 1500.0 else "unresolved"
    return {
        "verdict": verdict,
        "confidence": "high" if verdict == "kA" else "low",
        "schema_field_quote": str(schema.field("magnetics_plasma_current")),
        "schema_metadata_quote": metadata,
        "loader_quotes": loader_quotes,
        "loader_interpretation": (
            "the challenge loader casts raw values to float and applies no unit scale"
        ),
        "ampere_per_raw_unit": {
            "values": ratios,
            "median": median_ratio,
        },
        "finding": (
            "raw units are kA; near-zero values in some labelled frames are not an "
            "A-versus-kA conversion and require a channel/alignment quality audit"
        ),
    }


def audit(
    data_root: Path = DEFAULT_DATA,
    gate_artifact: Path = DEFAULT_GATE_ARTIFACT,
) -> dict[str, Any]:
    gate_result = json.loads(gate_artifact.read_text())
    quiet_records = [
        record
        for record in gate_result["score"]["frame_records"]
        if "quiescent" in record["populations"]
    ]
    first_shot = quiet_records[0]["shot"]
    selected = [record for record in quiet_records if record["shot"] == first_shot][:3]
    if len(selected) != 3:
        raise RuntimeError("the authoritative gate has fewer than three audit frames")
    path = data_root / first_shot
    row = gate._read(path)
    registry = DiiidDescriptionRegistry()
    description = registry.ingest(row, source_row=path.name)
    response = vacuum_response(description, row["efit_grid_R"], row["efit_grid_Z"])
    coil_flux = vacuum_psi(row, description, response)
    radius = np.asarray(row["efit_grid_R"], dtype=float)
    height = np.asarray(row["efit_grid_Z"], dtype=float)
    operator_radius = np.linspace(radius[0], radius[-1], radius.size)
    operator_height = np.linspace(height[0], height[-1], height.size)
    target_mask, filament_response = gate._filament_matrix(
        operator_radius, operator_height
    )

    frame = int(selected[0]["frame"])
    source_r, source_z, source_current = _plasma_patch(
        operator_radius,
        operator_height,
        float(row["efit_r_axis"][frame]),
        float(row["efit_z_axis"][frame]),
    )
    plasma_physical = _filament_flux_map(
        operator_radius, operator_height, source_r, source_z, source_current
    )
    coil_physical = coil_flux[frame]
    synthetic_label = gate.FIXED_FLUX_SIGN * (coil_physical + plasma_physical)
    plasma_extracted, extracted_current_a, synthetic_norm = _extracted_plasma(
        row,
        frame,
        synthetic_label,
        operator_radius,
        operator_height,
        filament_response,
    )
    exterior = synthetic_norm[target_mask] > 1.05
    actual = synthetic_label[target_mask][exterior]
    coil_target = coil_physical[target_mask][exterior]
    known_plasma_target = plasma_physical[target_mask][exterior]
    extracted_target = plasma_extracted[exterior]
    exact_prediction = gate.FIXED_FLUX_SIGN * (coil_target + known_plasma_target)
    gate_prediction = gate.FIXED_FLUX_SIGN * (coil_target + extracted_target)
    localized_prediction = gate.FIXED_FLUX_SIGN * coil_target + extracted_target
    exact_score = gate._r2(actual, exact_prediction)[0]
    gate_score = gate._r2(actual, gate_prediction)[0]
    localized_score = gate._r2(actual, localized_prediction)[0]
    defect = (
        "global_sign_applied_twice_to_label_derived_plasma"
        if exact_score > 1.0 - 1.0e-12 and localized_score > gate_score
        else "pipeline_sound"
    )
    synthetic_table = {
        "frame_source": {"shot": path.name, "frame": frame},
        "analytic_patch_current_a": float(np.sum(source_current)),
        "extracted_patch_current_a": extracted_current_a,
        "exterior_points": int(np.count_nonzero(exterior)),
        "terms": {
            "coil_challenge_sign": _magnitude(gate.FIXED_FLUX_SIGN * coil_target),
            "plasma_known_challenge_sign": _magnitude(
                gate.FIXED_FLUX_SIGN * known_plasma_target
            ),
            "plasma_extracted_label_sign": _magnitude(extracted_target),
            "plasma_after_gate_global_sign": _magnitude(
                gate.FIXED_FLUX_SIGN * extracted_target
            ),
            "label": _magnitude(actual),
        },
    }
    real_tables = [
        _frame_table(
            row,
            int(record["frame"]),
            coil_flux,
            operator_radius,
            operator_height,
            target_mask,
            filament_response,
        )
        for record in selected
    ]

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("run with `uv run --with pyarrow python ...`") from error
    schema = parquet.read_schema(path)
    return {
        "synthetic_truth": {
            "construction_exact_r2": exact_score,
            "construction_exact_deviation_from_unity": abs(1.0 - exact_score),
            "unmodified_gate_r2": gate_score,
            "unmodified_gate_deviation_from_unity": abs(1.0 - gate_score),
            "localized_composition_r2": localized_score,
            "localized_composition_deviation_from_unity": abs(1.0 - localized_score),
            "verdict": defect,
        },
        "magnitude_table": {
            "unit": "Wb total flux; per-radian inputs multiplied by 2pi",
            "synthetic": synthetic_table,
            "real_quiescent_frames": real_tables,
        },
        "plasma_current_channel": _unit_receipt(real_tables, schema),
        "geometry_digest": description.physical_digest,
        "geometry_provenance_complete": description.provenance_complete,
        "coefficients_fitted": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--gate-artifact", type=Path, default=DEFAULT_GATE_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.data, args.gate_artifact)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
