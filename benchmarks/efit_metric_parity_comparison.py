"""Compare EFM metrics with Nova geometry derived from the same flux map.

This is an unbounded measurement: it records signed differences and analytic
tolerance exceedances but emits no pass/fail verdict and registers no
cross-code tolerance.  Source COCOS 3 flux is converted from Wb/rad to total
Wb, and source safety factor is multiplied by minus one into COCOS 17.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.flux_surface_geometry import FluxSurfaceGeometry
from nova.equilibrium.wall_mask import inside_polygon
from nova.imas.mast_vacuum_cohort import SHOT_STORE

FROZEN_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)
COMMON_SLICE_INDEX = 46
PROFILE_NAMES = ("psi_norm", "areap_c", "fpsi_c", "qpsi_c", "volp_c")
PAIRED_NAMES = (
    *PROFILE_NAMES,
    "plasma_area",
    "plasma_volume",
    "q_axis",
    "q_90",
    "q_95",
    "q_100",
    "bvac_val",
    "bvac_r",
    "psi_axis",
    "psi_boundary",
    "magnetic_axis_r",
    "magnetic_axis_z",
)
SURFACE_AVERAGES_WITHOUT_REFERENCE = (
    "inverse_square_radius",
    "gradient_rho",
    "gradient_rho_squared",
    "gradient_rho_squared_over_radius_squared",
)
ANALYTIC_RELATIVE_TOLERANCE = {
    "psi_norm": 1.0e-6,
    "areap_c": 5.0e-6,
    "fpsi_c": 1.0e-8,
    "qpsi_c": 2.0e-7,
    "volp_c": 5.0e-6,
    "plasma_area": 5.0e-6,
    "plasma_volume": 5.0e-6,
    "q_axis": 2.0e-7,
    "q_90": 2.0e-7,
    "q_95": 2.0e-7,
    "q_100": 2.0e-7,
    "magnetic_axis_r": 1.0e-5,
}


def _uniform_axis(values: np.ndarray, name: str) -> np.ndarray:
    """Verify stored-dtype uniformity, then preserve endpoints in float64."""
    stored = np.asarray(values)
    if (
        stored.ndim != 1
        or stored.size < 2
        or not np.issubdtype(stored.dtype, np.floating)
    ):
        raise ValueError(f"{name} must be a one-dimensional floating coordinate")
    expected = np.linspace(stored[0], stored[-1], stored.size, dtype=stored.dtype)
    tolerance = 16.0 * np.finfo(stored.dtype).eps * max(1.0, abs(float(stored[-1])))
    deviation = float(np.max(np.abs(stored - expected)))
    if deviation > tolerance:
        raise ValueError(
            f"stored {name} is genuinely non-uniform at {stored.dtype} precision: "
            f"maximum deviation {deviation:.9g} exceeds {tolerance:.9g}"
        )
    return np.linspace(float(stored[0]), float(stored[-1]), stored.size)


def _finite_flux_map(group: zarr.Group, index: int, radius: np.ndarray) -> np.ndarray:
    """Return the finite live raster in radius-major order."""
    raw = np.asarray(group["psirz"][index], dtype=np.float64)
    columns = np.flatnonzero(np.all(np.isfinite(raw), axis=0))
    if columns.size != radius.size:
        raise ValueError(
            f"flux map has {columns.size} finite columns for {radius.size} radii"
        )
    profile_radius = np.asarray(group["profile_r"][:], dtype=np.float64)
    if not np.allclose(profile_radius[columns], radius, rtol=2.0e-7, atol=1.0e-8):
        raise ValueError("finite flux columns do not match efm/gridr")
    return raw[:, columns].T


def _lcfs(group: zarr.Group, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite stored boundary polygon."""
    count_value = float(group["lcfsn_c"][index])
    if not np.isfinite(count_value) or count_value != int(count_value):
        raise ValueError(f"invalid LCFS count {count_value}")
    count = int(count_value)
    radius = np.asarray(group["lcfs_r"][index, :count], dtype=np.float64)
    height = np.asarray(group["lcfs_z"][index, :count], dtype=np.float64)
    if count < 3 or not np.all(np.isfinite(radius)) or not np.all(np.isfinite(height)):
        raise ValueError("stored LCFS is not a finite polygon")
    return radius, height


def _distribution(values: np.ndarray) -> dict[str, float]:
    """Return median and interquartile range of one finite profile."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("difference profile must be non-empty and finite")
    q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
    return {
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "interquartile_range": float(q75 - q25),
    }


def _difference(
    name: str,
    position: np.ndarray,
    nova: np.ndarray,
    reference: np.ndarray,
    *,
    transform: str,
    classification: str,
    coincident_domain: bool = True,
    analytic_absolute_tolerance: float | None = None,
) -> dict[str, Any]:
    """Return an unbounded signed relative-difference record."""
    position = np.atleast_1d(np.asarray(position, dtype=np.float64))
    nova = np.atleast_1d(np.asarray(nova, dtype=np.float64))
    reference = np.atleast_1d(np.asarray(reference, dtype=np.float64))
    if nova.shape != reference.shape or nova.shape != position.shape:
        raise ValueError(f"{name} comparison shapes disagree")
    scale = max(float(np.max(np.abs(reference))), np.finfo(np.float64).tiny)
    denominator_floor = scale * 1.0e-12
    denominator = np.maximum(np.abs(reference), denominator_floor)
    difference = nova - reference
    signed = difference / denominator
    tolerance = ANALYTIC_RELATIVE_TOLERANCE.get(name)
    if analytic_absolute_tolerance is not None:
        exceed = np.flatnonzero(np.abs(difference) > analytic_absolute_tolerance)
        tolerance_class = {
            "kind": "absolute",
            "value": analytic_absolute_tolerance,
        }
    elif tolerance is not None:
        exceed = np.flatnonzero(np.abs(signed) > tolerance)
        tolerance_class = {"kind": "relative", "value": tolerance}
    else:
        exceed = np.array([], dtype=int)
        tolerance_class = None
    return {
        "store_path": f"efm/{name}",
        "surface_position_psi_norm": position.tolist(),
        "nova_value": nova.tolist(),
        "transformed_reference_value": reference.tolist(),
        "signed_relative_difference_profile": signed.tolist(),
        "distribution_across_surfaces": _distribution(signed),
        "relative_denominator_floor": denominator_floor,
        "declared_transform": transform,
        "difference_classification": classification,
        "domains_coincident": coincident_domain,
        "analytic_tolerance_class": tolerance_class,
        "analytic_tolerance_exceedance_positions": position[exceed].tolist(),
        "analytic_tolerance_exceedance_count": len(exceed),
        "verdict": "UNBOUNDED_OBSERVATION",
    }


def _sign_receipt(path: Path, shot: int, index: int) -> dict[str, Any]:
    """Return one per-shot current-sign result from the banked census."""
    report = json.loads(path.read_text())
    rows = [row for row in report["per_shot"] if row["shot"] == shot]
    if len(rows) != 1 or rows[0]["slice_index"] != index:
        raise ValueError(f"current-sign census has no unique shot {shot} row {index}")
    row = rows[0]
    if not row["all_four_signs_agree"]:
        raise ValueError(f"shot {shot} current signs disagree: {row['signs']}")
    signs = set(row["signs"].values())
    if len(signs) != 1:
        raise ValueError(f"shot {shot} current sign receipt is internally inconsistent")
    return {
        "sign": int(next(iter(signs))),
        "signs": row["signs"],
        "all_four_signs_agree": True,
        "signed_relative_current_closure": row[
            "signed_relative_closure_against_plasma_current_c"
        ],
    }


def compare_shot(
    store: Path,
    sign_report: Path,
    shot: int,
    index: int = COMMON_SLICE_INDEX,
) -> dict[str, Any]:
    """Build and compare one same-map geometry record."""
    group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
    time = np.asarray(group["time"][:], dtype=np.float64)
    if index >= time.size or not np.isfinite(time[index]):
        raise ValueError(f"shot {shot} row {index} is not a finite stored slice")
    sign = _sign_receipt(sign_report, shot, index)

    stored_radius = np.asarray(group["gridr"][:])
    stored_height = np.asarray(group["gridz"][:])
    radius = _uniform_axis(stored_radius, "gridr")
    height = _uniform_axis(stored_height, "gridz")
    flux_per_radian = _finite_flux_map(group, index, stored_radius)
    lattice = FluxLattice(radius, height)
    total_flux = TOTAL_FLUX_FACTOR * flux_per_radian.reshape(-1)
    stored_psi_norm = np.asarray(group["psi_norm"][:], dtype=np.float64)
    stored_field = np.asarray(group["fpsi_c"][index], dtype=np.float64)

    def field_function(psi_norm):
        return np.interp(psi_norm, stored_psi_norm, stored_field)

    axis_seed = (
        float(group["magnetic_axis_r"][index]),
        float(group["magnetic_axis_z"][index]),
    )
    boundary_flux = TOTAL_FLUX_FACTOR * float(group["psi_boundary"][index])
    record = FluxSurfaceGeometry.from_flux_map(
        lattice,
        total_flux,
        field_function,
        axis=axis_seed,
        boundary_flux=boundary_flux,
        reference_radius=float(group["bvac_r"][index]),
        surfaces=65,
        angles=128,
        edge_psi_norm=0.995,
    )

    lcfs_radius, lcfs_height = _lcfs(group, index)
    axis_inside = bool(
        inside_polygon(
            np.array([record.magnetic_axis[0]]),
            np.array([record.magnetic_axis[1]]),
            lcfs_radius,
            lcfs_height,
        )[0]
    )
    if not axis_inside:
        raise ValueError(f"shot {shot} refined magnetic axis lies outside stored LCFS")

    usable = stored_psi_norm <= record.psi_norm[-1]
    profile_position = stored_psi_norm[usable]
    profile_map = {
        "psi_norm": (record.psi_norm, stored_psi_norm),
        "areap_c": (record.area, np.asarray(group["areap_c"][index], dtype=float)),
        "fpsi_c": (record.field_function, stored_field),
        "qpsi_c": (
            record.safety_factor,
            -np.asarray(group["qpsi_c"][index], dtype=float),
        ),
        "volp_c": (record.volume, np.asarray(group["volp_c"][index], dtype=float)),
    }
    comparisons = {}
    for name, (nova_profile, reference_profile) in profile_map.items():
        resampled = np.interp(profile_position, record.psi_norm, nova_profile)
        transform = (
            "source COCOS 3 q multiplied by -1 into COCOS 17; monotone linear "
            "interpolation of Nova on psi_norm; no extrapolation"
            if name == "qpsi_c"
            else "monotone linear interpolation of Nova on psi_norm; no extrapolation"
        )
        comparisons[name] = _difference(
            name,
            profile_position,
            resampled,
            reference_profile[usable],
            transform=transform,
            classification=(
                "CONVENTION_ARTEFACT_RESOLVED"
                if name == "qpsi_c"
                else "GENUINE_NUMERICAL_DISAGREEMENT"
            ),
        )

    def scalar(
        name: str,
        nova: float,
        reference: float,
        position: float,
        transform: str = "none",
        convention: bool = False,
        coincident: bool = True,
        analytic_absolute_tolerance: float | None = None,
    ) -> None:
        comparisons[name] = _difference(
            name,
            np.array([position]),
            np.array([nova]),
            np.array([reference]),
            transform=transform,
            classification=(
                "CONVENTION_ARTEFACT_RESOLVED"
                if convention
                else "GENUINE_NUMERICAL_DISAGREEMENT"
            ),
            coincident_domain=coincident,
            analytic_absolute_tolerance=analytic_absolute_tolerance,
        )

    scalar(
        "plasma_area",
        record.area[-1],
        group["plasma_area"][index],
        record.psi_norm[-1],
        coincident=False,
    )
    scalar(
        "plasma_volume",
        record.volume[-1],
        group["plasma_volume"][index],
        record.psi_norm[-1],
        coincident=False,
    )
    for name, position in (("q_axis", 0.0), ("q_90", 0.90), ("q_95", 0.95)):
        scalar(
            name,
            float(np.interp(position, record.psi_norm, record.safety_factor)),
            -float(group[name][index]),
            position,
            "source COCOS 3 q multiplied by -1 into COCOS 17",
            True,
        )
    scalar(
        "q_100",
        record.safety_factor[-1],
        -float(group["q_100"][index]),
        record.psi_norm[-1],
        "source COCOS 3 q multiplied by -1 into COCOS 17; EFIT separatrix "
        "value compared at Nova outermost resolvable surface without extrapolation",
        True,
        False,
    )
    scalar("bvac_val", record.vacuum_field, group["bvac_val"][index], 1.0)
    scalar("bvac_r", record.reference_radius, group["bvac_r"][index], 1.0)
    scalar(
        "psi_axis",
        record.axis_flux,
        TOTAL_FLUX_FACTOR * float(group["psi_axis"][index]),
        0.0,
        "multiply source Wb/rad by 2*pi into total Wb",
        True,
    )
    scalar(
        "psi_boundary",
        record.boundary_flux,
        TOTAL_FLUX_FACTOR * float(group["psi_boundary"][index]),
        1.0,
        "multiply source Wb/rad by 2*pi into total Wb",
        True,
    )
    scalar(
        "magnetic_axis_r", record.magnetic_axis[0], group["magnetic_axis_r"][index], 0.0
    )
    scalar(
        "magnetic_axis_z",
        record.magnetic_axis[1],
        group["magnetic_axis_z"][index],
        0.0,
        analytic_absolute_tolerance=1.0e-5
        * abs(float(group["magnetic_axis_r"][index])),
    )
    if tuple(comparisons) != PAIRED_NAMES:
        raise ValueError("comparison output does not account for all 17 census pairs")

    return {
        "shot": shot,
        "slice_index": index,
        "time_s": float(time[index]),
        "plasma_current_sign_receipt": sign,
        "stored_lcfs": {
            "vertices": int(lcfs_radius.size),
            "refined_axis_inside": axis_inside,
            "surface_family_rule": (
                "first outward flux crossing from the stored-axis seed, bounded by "
                "the stored LCFS flux; outermost psi_norm=0.995"
            ),
        },
        "comparisons": comparisons,
    }


def build_report(store: Path, sign_report: Path) -> dict[str, Any]:
    """Return all six same-map, unbounded comparison records."""
    shots = [compare_shot(store, sign_report, shot) for shot in FROZEN_SHOTS]
    exceedances = []
    for row in shots:
        for name, comparison in row["comparisons"].items():
            if comparison["analytic_tolerance_exceedance_count"]:
                exceedances.append(
                    {
                        "shot": row["shot"],
                        "quantity": name,
                        "analytic_tolerance_class": comparison[
                            "analytic_tolerance_class"
                        ],
                        "surface_positions_psi_norm": comparison[
                            "analytic_tolerance_exceedance_positions"
                        ],
                    }
                )
    return {
        "policy": {
            "cross_code_tolerance": None,
            "verdict": "UNBOUNDED_FIRST_PASS",
            "pass_fail_applied": False,
            "tolerance_registered": False,
        },
        "store": str(store),
        "shots": list(FROZEN_SHOTS),
        "common_slice_index": COMMON_SLICE_INDEX,
        "current_sign_report": {
            "path": str(sign_report),
            "sha256": hashlib.sha256(sign_report.read_bytes()).hexdigest(),
        },
        "transforms": {
            "poloidal_flux": "source Wb/rad multiplied by 2*pi into total Wb",
            "safety_factor": "source COCOS 3 q multiplied by -1 into COCOS 17",
            "profile_resampling": (
                "monotone linear interpolation of Nova against "
                "FluxSurfaceGeometry.psi_norm onto efm/psi_norm; no extrapolation"
            ),
        },
        "analytic_tolerance_reference": {
            "source_commit": "b0af2d23291ab51d4d6e665150f67dbe6163ef49",
            "use": (
                "reported exceedance observations only; these analytic classes "
                "are not cross-code pass/fail thresholds"
            ),
        },
        "paired_quantity_count": len(PAIRED_NAMES),
        "per_shot": shots,
        "analytic_tolerance_exceedance_observations": exceedances,
        "flux_surface_averages_without_reference": [
            {"nova_column": name, "status": "NO_REFERENCE_COUNTERPART"}
            for name in SURFACE_AVERAGES_WITHOUT_REFERENCE
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--current-sign-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.store, args.current_sign_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    medians = {
        name: [
            row["comparisons"][name]["distribution_across_surfaces"]["median"]
            for row in report["per_shot"]
        ]
        for name in PAIRED_NAMES
    }
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shots": len(report["per_shot"]),
                "paired_quantities_per_shot": len(PAIRED_NAMES),
                "analytic_tolerance_exceedance_observations": len(
                    report["analytic_tolerance_exceedance_observations"]
                ),
                "median_ranges_by_quantity": {
                    name: [min(values), max(values)] for name, values in medians.items()
                },
                "verdict": "UNBOUNDED_FIRST_PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
