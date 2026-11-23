"""Decompose the five DIII-D forward-label residuals without fitting a field.

The EFIT label is tared with the exact clipped-cell plasma moments.  The
vacuum contribution is then built from the persisted machine-description IDS,
including its wired ohmic circuit.  After removing one additive flux gauge,
the residual is split hierarchically: an orthogonal first-order R/Z component,
then the orthogonal remainder is allocated between vessel-shaped and
conductor-shaped current patches using the landed geometry discriminator.
No diagnostic component is added to the prediction and no coefficient is fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from benchmarks import diiid_exact_clipped_tare as exact_tare
from benchmarks import diiid_unclaimed_current_origin as origin
from benchmarks import diiid_unclaimed_current_patches as patches
from benchmarks import diiid_vacuum_against_exact_tare as vacuum_tare
from benchmarks.diiid_corpus_conventions import (
    corpus_flux_to_nova_total,
    nova_total_flux_to_corpus,
)
from benchmarks.diiid_forward_gs_match import (
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    build_profile,
)
from benchmarks.diiid_vacuum_from_description import (
    ALL_CONDUCTOR_NAMES,
    PERSISTED_DD_VERSION,
    PERSISTED_ENTRY,
    _persisted_response,
)
from nova.equilibrium.map_extraction import apply_delta_star
from nova.imas.diiid_description import PF_ACTIVE_CIRCUIT, POLOIDAL_CONDUCTORS
from nova.jax.config import configure_dtypes

DEFAULT_DATA = exact_tare.DEFAULT_DATA
DEFAULT_OUTPUT = Path("docs/figures/diiid-vertical-force-balance")
FRAME_SOURCE = Path("docs/figures/gs-absolute-accuracy/efit-reproduction.json")
DISPLACEMENT_SOURCE = Path(
    "docs/figures/diiid-forward-onboarding/forward-gs/forward_gs_receipt.json"
)
RECEIPT_NAME = "residual-decomposition.json"
LABEL_INCONSISTENCY_FRACTION = 0.95


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_selection() -> list[dict[str, Any]]:
    authority = json.loads(FRAME_SOURCE.read_text())["data"]
    selected = [
        {
            "shot": row["frame_identity"]["shot"],
            "frame": int(row["frame_identity"]["frame"]),
            "time_ms": float(row["frame_identity"]["time_ms"]),
            "label_gs_inconsistency": row["label_gs_inconsistency"],
        }
        for row in authority["rows"]
        if row["machine"] == "DIII-D"
    ]
    if len(selected) != 5:
        raise RuntimeError(
            f"expected five DIII-D authority rows, found {len(selected)}"
        )
    displacement = json.loads(DISPLACEMENT_SOURCE.read_text())["result"][
        "frame_records"
    ]
    by_key = {
        (row["shot"], int(row["frame"])): float(
            row["metrics"]["magnetic_axis_displacement_mm"]
        )
        / 1000.0
        for row in displacement
    }
    for item in selected:
        item["observed_axis_displacement_m"] = by_key[(item["shot"], item["frame"])]
    return selected


def _read_row(path: Path) -> dict[str, Any]:
    columns = tuple(
        dict.fromkeys(
            (
                *exact_tare.READ_COLUMNS,
                *_LABEL_COLUMNS,
                *_CURRENT_COLUMNS,
                *_GEOMETRY_COLUMNS,
            )
        )
    )
    return exact_tare._read(path, columns)


def _persisted_geometry() -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    import imas

    coils: list[np.ndarray] = []
    loops: list[np.ndarray] = []
    with imas.DBEntry(
        PERSISTED_ENTRY.resolve(), "r", dd_version=PERSISTED_DD_VERSION
    ) as entry:
        active = entry.get("pf_active", autoconvert=False)
        passive = entry.get("pf_passive", autoconvert=False)
        for coil in active.coil:
            for element in coil.element:
                coils.append(
                    np.column_stack(
                        (
                            np.asarray(element.geometry.outline.r, dtype=float),
                            np.asarray(element.geometry.outline.z, dtype=float),
                        )
                    )
                )
        for loop in passive.loop:
            for element in loop.element:
                loops.append(
                    np.column_stack(
                        (
                            np.asarray(element.geometry.outline.r, dtype=float),
                            np.asarray(element.geometry.outline.z, dtype=float),
                        )
                    )
                )
        circuit = np.asarray(active.circuit[0].connections, dtype=int)
        meta = {
            "coil_count": len(active.coil),
            "passive_loop_count": len(passive.loop),
            "circuit_count": len(active.circuit),
            "supply_count": len(active.supply),
            "ohmic_connections_shape": list(circuit.shape),
            "ohmic_connections_sha256": hashlib.sha256(circuit.tobytes()).hexdigest(),
        }
    return coils, loops, meta


def _currents(row: dict[str, Any], frame: int) -> tuple[np.ndarray, dict[str, Any]]:
    profile, *_unused = build_profile(row, frame, None)
    shipped = np.asarray(profile.operator.external_current, dtype=float)
    if shipped.size != len(POLOIDAL_CONDUCTORS):
        raise RuntimeError("forward profile does not carry the expected poloidal set")
    ecoila = float(shipped[POLOIDAL_CONDUCTORS.index("ECOILA")])
    derived = PF_ACTIVE_CIRCUIT.currents(ecoila)
    vector = np.r_[
        shipped, [derived[name] for name in PF_ACTIVE_CIRCUIT.component_order[2:]]
    ]
    if vector.size != len(ALL_CONDUCTOR_NAMES):
        raise RuntimeError("persisted-response current vector is incomplete")
    return vector, {
        "order": list(ALL_CONDUCTOR_NAMES),
        "currents_a_turn": vector.tolist(),
        "ohmic_source": "recorded same-frame ECOILA channel",
        "ohmic_effective_gains": {
            drive.conductor: drive.gain for drive in PF_ACTIVE_CIRCUIT.drives
        },
        "coefficients_fitted_in_this_measurement": 0,
    }


def _axis_field_conversion(
    residual_decomposition: dict[str, float],
    radius: np.ndarray,
    height: np.ndarray,
    label_total_zr: np.ndarray,
    axis_r: float,
    axis_z: float,
) -> dict[str, Any]:
    sigma_r = float(np.std(radius))
    sigma_z = float(np.std(height))
    radial_coefficient = residual_decomposition["radial_coefficient_wb_per_radian"]
    vertical_coefficient = residual_decomposition["vertical_coefficient_wb_per_radian"]
    vertical_field = abs(float(corpus_flux_to_nova_total(radial_coefficient))) / (
        axis_r * sigma_r
    )
    radial_field = abs(float(corpus_flux_to_nova_total(vertical_coefficient))) / (
        axis_r * sigma_z
    )

    radial_map, height_map = np.meshgrid(radius, height)
    total = np.asarray(label_total_zr, dtype=float)
    dpsi_dz = np.gradient(total, height, axis=0, edge_order=2)
    label_br = -dpsi_dz / radial_map
    dbr_dz = np.gradient(label_br, height, axis=0, edge_order=2)
    stiffness = float(
        RegularGridInterpolator((height, radius), dbr_dz)([[axis_z, axis_r]])[0]
    )
    implied = radial_field / abs(stiffness) if stiffness != 0.0 else None
    return {
        "vertical_field_from_radial_dipole_t": vertical_field,
        "radial_field_from_vertical_dipole_t": radial_field,
        "vertical_axis_stiffness_t_per_m": stiffness,
        "implied_vertical_axis_displacement_m": implied,
        "assumptions": [
            "axisymmetric field uses B_R = -(1/R) dpsi_total/dZ and "
            "B_Z = (1/R) dpsi_total/dR",
            "the normalised dipole basis has scales std(R) and std(Z) on "
            "the complete 65 by 65 grid",
            "the elongated plasma translates rigidly and locally, with "
            "linear vertical force balance",
            "the local EFIT-label dB_R/dZ at its magnetic axis is the "
            "restoring stiffness",
            "R-Z coupling, nonlinear shape response, active feedback and "
            "vessel-current feedback are omitted",
        ],
    }


def _non_gs_accounting(
    row: dict[str, Any],
    frame: exact_tare.PreparedFrame,
    radius: np.ndarray,
    height: np.ndarray,
    plasma_tared_zr: np.ndarray,
    wall: np.ndarray,
    reference_current_a: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    delta = apply_delta_star(radius, height, plasma_tared_zr.T)
    density = np.asarray(delta.toroidal_current_density, dtype=float)
    exterior = ~frame.core_rz & delta.valid & np.isfinite(density)
    found, metrics, _masks = patches.locate_patches(
        density,
        exterior,
        frame.core_rz,
        radius,
        height,
        wall,
        reference_current_a,
    )
    profile_source, reliable, label_total_rz = origin._profile_source(
        row, frame.selected.frame, radius, height
    )
    fixed = origin._operator(radius, height).solve(profile_source, label_total_rz)
    strict_non_gs = apply_delta_star(radius, height, label_total_rz - fixed)
    non_gs_density = np.asarray(strict_non_gs.toroidal_current_density, dtype=float)
    non_gs_exterior = ~frame.core_rz & strict_non_gs.valid & np.isfinite(non_gs_density)
    area = float(np.mean(np.diff(radius)) * np.mean(np.diff(height)))
    non_gs_l1 = float(np.sum(np.abs(non_gs_density[non_gs_exterior])) * area)
    native_l1 = metrics["total_unclaimed_ampere_turns_l1"]
    detectable = [item for item in found if item["detectable_above_tare_floor"]]
    vessel_l1 = sum(
        item["absolute_cell_current_a"]
        for item in found
        if item["classification"] == "vessel-shaped"
    )
    conductor_l1 = sum(
        item["absolute_cell_current_a"]
        for item in found
        if item["classification"] == "conductor-shaped"
    )
    return (
        {
            "reliable_profile_surfaces": reliable,
            "native_unclaimed_ampere_turns_l1": native_l1,
            "non_gs_apparent_ampere_turns_l1": non_gs_l1,
            "non_gs_to_native_unclaimed_ratio": non_gs_l1 / native_l1,
            "detectable_patch_count": len(detectable),
            "vessel_patch_current_l1_a_turn": vessel_l1,
            "conductor_patch_current_l1_a_turn": conductor_l1,
        },
        detectable,
    )


def _fractions(smooth_fraction: float, accounting: dict[str, Any]) -> dict[str, float]:
    remainder = max(0.0, 1.0 - smooth_fraction)
    vessel = accounting["vessel_patch_current_l1_a_turn"]
    conductor = accounting["conductor_patch_current_l1_a_turn"]
    source_total = vessel + conductor
    if source_total == 0.0:
        vessel_share = 0.0
    else:
        vessel_share = vessel / source_total
    return {
        "smooth_first_order_r_z": smooth_fraction,
        "vessel_following": remainder * vessel_share,
        "conductor_localised": remainder * (1.0 - vessel_share),
    }


def _figure(
    record: dict[str, Any],
    residual: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    coils: list[np.ndarray],
    loops: list[np.ndarray],
    output: Path,
) -> Path:
    figure, (axis, text_axis) = plt.subplots(
        1, 2, figsize=(11.5, 5.5), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    limit = float(np.nanmax(np.abs(residual)))
    image = axis.pcolormesh(
        radius,
        height,
        residual,
        shading="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    for polygon in loops:
        axis.plot(*polygon.T, color="0.35", linewidth=0.35, alpha=0.65)
    for polygon in coils:
        closed = np.vstack((polygon, polygon[0]))
        axis.plot(*closed.T, color="black", linewidth=0.65)
    axis.set_aspect("equal")
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_title("Gauge-free tared residual [Wb/rad]")
    figure.colorbar(image, ax=axis, shrink=0.82)

    fractions = record["decomposition_fractions"]
    conversion = record["dipole_field_and_axis_response"]
    lines = [
        f"{record['shot']} : frame {record['frame']}",
        f"residual RMS  {record['gauge_free_residual_rms_wb_per_radian']:.4g} Wb/rad",
        "",
        f"smooth R/Z    {fractions['smooth_first_order_r_z']:.1%}",
        f"conductor     {fractions['conductor_localised']:.1%}",
        f"vessel        {fractions['vessel_following']:.1%}",
        "",
        f"|Bz dipole|   {conversion['vertical_field_from_radial_dipole_t']:.4g} T",
        f"|Br dipole|   {conversion['radial_field_from_vertical_dipole_t']:.4g} T",
        f"implied |dZ|  {conversion['implied_vertical_axis_displacement_m']:.3g} m",
        f"observed |dX| {record['observed_axis_displacement_m']:.3g} m",
        "",
        record["verdict"]["headline"],
    ]
    text_axis.axis("off")
    text_axis.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace")
    figure.tight_layout()
    path = output / f"{Path(record['shot']).stem}-frame-{record['frame']}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def run(data: Path, output: Path, *, workers: int = 1) -> dict[str, Any]:
    configure_dtypes()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    selected_rows = _frame_selection()
    rows = {item["shot"]: _read_row(data / item["shot"]) for item in selected_rows}
    first = rows[selected_rows[0]["shot"]]
    radius, height = exact_tare.canonical_axes(first)
    mesh, geometry, width, vertical_extent = exact_tare.rectangular_geometry(
        radius, height
    )
    selected = [
        exact_tare.SelectedFrame(
            path=data / item["shot"],
            frame=item["frame"],
            time_ms=item["time_ms"],
        )
        for item in selected_rows
    ]
    prepared = [
        exact_tare.prepare_frame(item, rows[item.path.name], radius, height)
        for item in selected
    ]
    source_mask = np.any(
        np.stack([item.participation_zr.reshape(-1) for item in prepared]), axis=0
    )
    source_indices = np.flatnonzero(source_mask & np.asarray(mesh.interior()))
    blocks = exact_tare.response_blocks(
        mesh, source_indices, width, vertical_extent, max(1, workers)
    )
    integrate = exact_tare.moment_integrator(mesh, geometry)
    target_r, target_z = np.meshgrid(radius, height)
    persisted_psi, _persisted_br, _persisted_bz, response_meta = _persisted_response(
        PERSISTED_ENTRY,
        PERSISTED_DD_VERSION,
        ALL_CONDUCTOR_NAMES,
        target_r,
        target_z,
    )
    coil_geometry, loop_geometry, ids_meta = _persisted_geometry()
    with np.load(patches.VESSEL_ARTIFACT) as vessel:
        wall = np.asarray(vessel["limiter_contour_rz_m"], dtype=float)

    records: list[dict[str, Any]] = []
    figures: list[str] = []
    for authority, frame in zip(selected_rows, prepared, strict=True):
        exact_vectors = integrate(
            frame.psi_norm_zr,
            frame.participation_zr,
            frame.profile_surface,
            frame.p_prime,
            frame.ff_prime,
        )
        exact_current, exact_radial, exact_vertical, _boundary = (
            np.asarray(value) for value in jax.block_until_ready(exact_vectors)
        )
        plasma_flux_zr = (
            blocks[0] @ exact_current[source_indices]
            + blocks[1] @ exact_radial[source_indices]
            + blocks[2] @ exact_vertical[source_indices]
        ).reshape(frame.label_total_zr.shape)
        plasma_tared_zr = frame.label_total_zr - plasma_flux_zr
        current, current_meta = _currents(
            rows[frame.selected.path.name], frame.selected.frame
        )
        coil_flux_zr = np.einsum("c,czr->zr", current, persisted_psi)
        actual = nova_total_flux_to_corpus(plasma_tared_zr)
        prediction = nova_total_flux_to_corpus(coil_flux_zr)
        metrics, residual = vacuum_tare.comparison_metrics(actual, prediction)
        decomposition, _smooth, _remainder = vacuum_tare.first_order_decomposition(
            residual, radius, height
        )
        accounting, detectable = _non_gs_accounting(
            rows[frame.selected.path.name],
            frame,
            radius,
            height,
            plasma_tared_zr,
            wall,
            float(np.sum(exact_current)),
        )
        fractions = _fractions(
            decomposition["lowest_order_energy_fraction"], accounting
        )
        axis_r = float(
            rows[frame.selected.path.name]["efit_r_axis"][frame.selected.frame]
        )
        axis_z = float(
            rows[frame.selected.path.name]["efit_z_axis"][frame.selected.frame]
        )
        conversion = _axis_field_conversion(
            decomposition,
            radius,
            height,
            frame.label_total_zr,
            axis_r,
            axis_z,
        )
        inside_label = accounting["non_gs_to_native_unclaimed_ratio"] >= (
            LABEL_INCONSISTENCY_FRACTION
        )
        verdict = {
            "classification": (
                "inside-label-gs-inconsistency"
                if inside_label
                else "candidate-physical-source"
            ),
            "headline": (
                "Residual source is inside label GS inconsistency"
                if inside_label
                else (
                    "Conductor-localised source candidate"
                    if fractions["conductor_localised"] >= fractions["vessel_following"]
                    else "Vessel-following source candidate"
                )
            ),
            "reason": (
                "non-GS accounting carries "
                f"{accounting['non_gs_to_native_unclaimed_ratio']:.1%} of the "
                "exterior apparent current; all 24 known poloidal coils are "
                "already present"
            ),
        }
        record = {
            **authority,
            "exact_clipped_plasma_current_a": float(np.sum(exact_current)),
            "gauge_removed_wb_per_radian": metrics["additive_gauge_wb_per_radian"],
            "gauge_free_residual_rms_wb_per_radian": metrics["with_additive_gauge"][
                "residual_rms_wb_per_radian"
            ],
            "gauge_free_fractional_rms": metrics["with_additive_gauge"][
                "fractional_rms"
            ],
            "spatial_decomposition": decomposition,
            "decomposition_fractions": fractions,
            "decomposition_fraction_sum": float(sum(fractions.values())),
            "fraction_definition": (
                "smooth is its orthogonal flux-energy fraction; the orthogonal "
                "remainder is allocated between the landed vessel/conductor patch "
                "classes in proportion to their absolute patch current"
            ),
            "exterior_current_accounting": accounting,
            "detectable_patches": detectable,
            "dipole_field_and_axis_response": conversion,
            "current_drive": current_meta,
            "verdict": verdict,
        }
        path = _figure(
            record,
            residual,
            radius,
            height,
            coil_geometry,
            loop_geometry,
            output,
        )
        record["figure"] = str(path)
        figures.append(str(path))
        records.append(record)
        print(
            f"DECOMPOSED {len(records)}/5 {record['shot']}:{record['frame']}",
            flush=True,
        )

    receipt = {
        "measurement": (
            "exact clipped-cell plasma tare plus persisted-description coil tare, "
            "gauge removal and zero-fit source decomposition"
        ),
        "selection_authority": {
            "path": str(FRAME_SOURCE),
            "sha256": _sha256(FRAME_SOURCE),
            "frame_count": len(records),
        },
        "displacement_authority": {
            "path": str(DISPLACEMENT_SOURCE),
            "sha256": _sha256(DISPLACEMENT_SOURCE),
        },
        "persisted_description": {
            "path": str(PERSISTED_ENTRY),
            "sha256": _sha256(PERSISTED_ENTRY),
            "dd_version": PERSISTED_DD_VERSION,
            **ids_meta,
            "response": response_meta,
        },
        "physics": {
            "plasma_tare": (
                "exact clipped-cell current zeroth, radial and vertical moments"
            ),
            "coil_tare": (
                "all 24 persisted pf_active polygons driven through the "
                "recorded circuit relation"
            ),
            "additive_gauges_removed": len(records),
            "coefficients_fitted": 0,
            "passive_currents_assumed_a": 0.0,
            "passive_reason": (
                "the persisted 47-loop vessel supplies geometry and resistance "
                "but no measured loop-current state"
            ),
            "decomposition_is_diagnostic_only": True,
        },
        "verdict_counts": {
            name: sum(item["verdict"]["classification"] == name for item in records)
            for name in ("inside-label-gs-inconsistency", "candidate-physical-source")
        },
        "records": records,
        "cost": {"device": "cpu", "wall_seconds": time.perf_counter() - started},
        "artifacts": {"receipt": str(output / RECEIPT_NAME), "figures": figures},
    }
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    receipt = run(arguments.data, arguments.output, workers=arguments.workers)
    print(
        json.dumps(
            {
                "frames": len(receipt["records"]),
                "verdict_counts": receipt["verdict_counts"],
                "receipt": receipt["artifacts"]["receipt"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
