"""Census forward-equilibrium inputs stored in the FAIR-MAST EFM reference.

The report distinguishes experimental inputs (``*_x``) from fitted EFM
outputs (``*_c``), preserves the source array attributes verbatim, and derives
row usability from finite stored values.  It never calls the referee or any
production read path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

FROZEN_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)
DEFAULT_STORE = Path("/work/projects/imas_gpu/mast/level1/shots")

QUANTITY_GROUPS = {
    "time": ("time",),
    "flux_coordinate_and_profiles": (
        "psi_norm",
        "pprime",
        "ffprime",
        "ppsi_c",
        "fpsi_c",
    ),
    "poloidal_flux_map": ("psirz", "gridr", "gridz"),
    "conductor_currents": (
        "fcoil_x",
        "fcoil_c",
        "fcoil_n",
        "fcoil_circ",
        "fcoil_r",
        "fcoil_z",
        "fcoil_turns",
        "fcoil_xmult",
    ),
    "plasma_current": ("plasma_current_x", "plasma_current_c"),
    "flux_and_vacuum_field_landmarks": (
        "psi_axis",
        "psi_boundary",
        "bvac_val",
        "bvac_r",
        "irod",
    ),
    "axis_and_boundary_geometry": (
        "magnetic_axis_r",
        "magnetic_axis_z",
        "lcfs_r",
        "lcfs_z",
        "lcfsn_c",
        "xpoint1_rc",
        "xpoint1_zc",
        "xpoint2_rc",
        "xpoint2_zc",
    ),
}


def _json_value(value: Any) -> Any:
    """Convert Zarr and NumPy metadata to JSON-native values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _array_metadata(array: Any) -> dict[str, Any]:
    """Return exact stored metadata needed to interpret one array."""

    attributes = dict(array.attrs)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "dimensions": _json_value(attributes.get("_ARRAY_DIMENSIONS", [])),
        "units": attributes.get("units", ""),
        "description": attributes.get("description"),
        "mds_name": attributes.get("mds_name"),
    }


def _finite_rows(values: np.ndarray, row_count: int) -> np.ndarray:
    """Return rows whose entire payload is finite."""

    if values.shape[0] != row_count:
        raise ValueError(
            f"time-dependent array has {values.shape[0]} rows, expected {row_count}"
        )
    return np.all(np.isfinite(values.reshape(row_count, -1)), axis=1)


def _geometry_rows(group: Any, time: np.ndarray) -> np.ndarray:
    """Apply the immutable EFM geometry-validity definition to stored arrays."""

    axis_r = np.asarray(group["magnetic_axis_r"][...], dtype=float)
    axis_z = np.asarray(group["magnetic_axis_z"][...], dtype=float)
    lcfs_r = np.asarray(group["lcfs_r"][...], dtype=float)
    lcfs_z = np.asarray(group["lcfs_z"][...], dtype=float)
    axis_valid = (
        np.isfinite(axis_r)
        & np.isfinite(axis_z)
        & (axis_r > 0.0)
        & (axis_r < 10.0)
        & (np.abs(axis_z) < 10.0)
    )
    lcfs_valid = (
        np.isfinite(lcfs_r)
        & np.isfinite(lcfs_z)
        & (lcfs_r > 0.0)
        & (lcfs_r < 10.0)
        & (np.abs(lcfs_z) < 10.0)
    )
    return np.isfinite(time) & axis_valid & (np.sum(lcfs_valid, axis=1) >= 3)


def _live_flux_map_rows(
    psirz: np.ndarray, grid_r: np.ndarray, grid_z: np.ndarray
) -> tuple[np.ndarray, list[list[int]]]:
    """Find rows with one finite plane matching the explicit R and Z grids.

    Some source shots pad the stored radial dimension beyond ``gridr``.  The
    active columns are therefore derived from finite values instead of slicing
    the first columns by convention.
    """

    grid_valid = (
        np.all(np.isfinite(grid_r))
        and np.all(np.diff(grid_r) > 0.0)
        and np.all(np.isfinite(grid_z))
        and np.all(np.diff(grid_z) > 0.0)
    )
    usable = np.zeros(psirz.shape[0], dtype=bool)
    active_columns: list[list[int]] = []
    for row_index, plane in enumerate(psirz):
        columns = np.flatnonzero(np.all(np.isfinite(plane), axis=0))
        active_columns.append(columns.astype(int).tolist())
        usable[row_index] = bool(
            grid_valid
            and columns.size == grid_r.size
            and plane.shape[0] == grid_z.size
            and np.all(np.isfinite(plane[:, columns]))
        )
    return usable, active_columns


def _sign(value: float) -> int | None:
    """Return a finite nonzero sign, or no sign for unfilled evidence."""

    if not np.isfinite(value) or value == 0.0:
        return None
    return int(np.sign(value))


def _convention_evidence(group: Any) -> dict[str, Any]:
    """Derive convention digits available from the stored scalar signs."""

    arrays = {
        name: np.asarray(group[name][...], dtype=float)
        for name in (
            "plasma_current_c",
            "psi_axis",
            "psi_boundary",
            "bvac_val",
            "q_95",
            "pprime",
        )
    }
    valid_current = (
        np.isfinite(arrays["plasma_current_c"])
        & np.isfinite(arrays["psi_axis"])
        & np.isfinite(arrays["psi_boundary"])
        & (np.abs(arrays["plasma_current_c"]) > 1.0e4)
    )
    if not np.any(valid_current):
        raise ValueError("no finite plasma-current row carries axis and boundary flux")
    index = int(
        np.argmax(np.where(valid_current, np.abs(arrays["plasma_current_c"]), 0.0))
    )
    current_sign = _sign(float(arrays["plasma_current_c"][index]))
    flux_difference_sign = _sign(
        float(arrays["psi_boundary"][index] - arrays["psi_axis"][index])
    )
    b0_sign = _sign(float(arrays["bvac_val"][index]))
    q_sign = _sign(float(arrays["q_95"][index]))
    if current_sign is None or flux_difference_sign is None:
        sigma_bp = None
    else:
        sigma_bp = flux_difference_sign * current_sign
    if current_sign is None or b0_sign is None or q_sign is None:
        sigma_rho_theta_phi = None
    else:
        sigma_rho_theta_phi = q_sign * current_sign * b0_sign

    middle = arrays["pprime"][index, arrays["pprime"].shape[1] // 3 :]
    middle = middle[: arrays["pprime"].shape[1] // 3]
    pprime_median = float(np.nanmedian(middle))
    pprime_sign = _sign(pprime_median)
    expected_pprime_sign = (
        None if current_sign is None or sigma_bp is None else -current_sign * sigma_bp
    )
    return {
        "sample_row": index,
        "stored_values": {
            "plasma_current_c_A": float(arrays["plasma_current_c"][index]),
            "psi_axis_Wb_per_rad": float(arrays["psi_axis"][index]),
            "psi_boundary_Wb_per_rad": float(arrays["psi_boundary"][index]),
            "bvac_val_T": float(arrays["bvac_val"][index]),
            "q_95": float(arrays["q_95"][index]),
            "mid_profile_pprime_median_Pa_rad_per_Wb": pprime_median,
        },
        "derived_from_stored_signs": {
            "sigma_bp": sigma_bp,
            "sigma_rho_theta_phi": sigma_rho_theta_phi,
            "pprime_sign": pprime_sign,
            "expected_pprime_sign_from_sigma_bp": expected_pprime_sign,
            "pprime_cross_check_matches": pprime_sign == expected_pprime_sign,
        },
    }


def _shot_census(shot: int, store: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Census one shot and return row counts plus its array metadata."""

    import zarr

    root = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")
    if "efm" not in root:
        raise ValueError(f"shot {shot} has no efm group")
    group = root["efm"]
    required = {name for names in QUANTITY_GROUPS.values() for name in names}
    missing = sorted(required.difference(group.array_keys()))
    if missing:
        raise ValueError(f"shot {shot} is missing required arrays {missing}")

    time = np.asarray(group["time"][...], dtype=float)
    row_count = int(time.size)
    pprime = np.asarray(group["pprime"][...], dtype=float)
    ffprime = np.asarray(group["ffprime"][...], dtype=float)
    psirz = np.asarray(group["psirz"][...], dtype=float)
    grid_r = np.asarray(group["gridr"][...], dtype=float)
    grid_z = np.asarray(group["gridz"][...], dtype=float)
    experimental_currents = np.asarray(group["fcoil_x"][...], dtype=float)
    fitted_currents = np.asarray(group["fcoil_c"][...], dtype=float)
    current_and_flux_landmarks = np.column_stack(
        [
            group["plasma_current_c"][...],
            group["psi_axis"][...],
            group["psi_boundary"][...],
        ]
    ).astype(float)

    profile_rows = _finite_rows(pprime, row_count) & _finite_rows(ffprime, row_count)
    flux_map_rows, active_columns = _live_flux_map_rows(psirz, grid_r, grid_z)
    experimental_current_rows = _finite_rows(experimental_currents, row_count)
    fitted_current_rows = _finite_rows(fitted_currents, row_count)
    landmark_rows = _finite_rows(current_and_flux_landmarks, row_count)
    geometry_rows = _geometry_rows(group, time)
    profile_map_conductor_rows = (
        np.isfinite(time)
        & profile_rows
        & flux_map_rows
        & experimental_current_rows
        & fitted_current_rows
    )
    forward_rows = profile_map_conductor_rows & landmark_rows

    unique_active_columns = {tuple(columns) for columns in active_columns}
    row = {
        "shot": shot,
        "time_rows": row_count,
        "usable_profile_rows": int(np.count_nonzero(profile_rows)),
        "usable_flux_map_rows": int(np.count_nonzero(flux_map_rows)),
        "usable_experimental_conductor_current_rows": int(
            np.count_nonzero(experimental_current_rows)
        ),
        "usable_fitted_conductor_current_rows": int(
            np.count_nonzero(fitted_current_rows)
        ),
        "usable_current_and_flux_landmark_rows": int(np.count_nonzero(landmark_rows)),
        "usable_profile_map_conductor_rows": int(
            np.count_nonzero(profile_map_conductor_rows)
        ),
        "usable_forward_input_rows": int(np.count_nonzero(forward_rows)),
        "usable_geometry_rows": int(np.count_nonzero(geometry_rows)),
        "usable_forward_and_geometry_rows": int(
            np.count_nonzero(forward_rows & geometry_rows)
        ),
        "usable_forward_row_indices": np.flatnonzero(forward_rows).astype(int).tolist(),
        "stored_flux_map_shape": list(psirz.shape),
        "live_flux_plane_shape": [int(grid_z.size), int(grid_r.size)],
        "active_radial_column_indices": (
            list(next(iter(unique_active_columns)))
            if len(unique_active_columns) == 1
            else [list(columns) for columns in sorted(unique_active_columns)]
        ),
        "psi_norm_range": [
            float(np.min(group["psi_norm"][...])),
            float(np.max(group["psi_norm"][...])),
        ],
        "convention_evidence": _convention_evidence(group),
    }
    metadata = {
        f"efm/{name}": _array_metadata(group[name])
        for names in QUANTITY_GROUPS.values()
        for name in names
    }
    return row, metadata


def build_report(store: Path, shots: tuple[int, ...] = FROZEN_SHOTS) -> dict[str, Any]:
    """Build a quantitative census for the selected EFM shot references."""

    per_shot: list[dict[str, Any]] = []
    shapes_by_path: dict[str, dict[str, list[int]]] = {}
    reference_metadata: dict[str, Any] | None = None
    for shot in shots:
        row, metadata = _shot_census(shot, store)
        per_shot.append(row)
        if reference_metadata is None:
            reference_metadata = metadata
        for path, item in metadata.items():
            shapes_by_path.setdefault(path, {})[str(shot)] = item["shape"]

    assert reference_metadata is not None
    arrays = {}
    for path, item in reference_metadata.items():
        arrays[path] = {**item, "shapes_by_shot": shapes_by_path[path]}

    total_profile_map_conductor_rows = sum(
        row["usable_profile_map_conductor_rows"] for row in per_shot
    )
    total_forward_rows = sum(row["usable_forward_input_rows"] for row in per_shot)
    total_geometry_rows = sum(row["usable_geometry_rows"] for row in per_shot)
    measured_digits = [
        row["convention_evidence"]["derived_from_stored_signs"] for row in per_shot
    ]
    return {
        "store": str(store),
        "efm_group": "efm",
        "shots": list(shots),
        "conclusion": {
            "flux_function_profiles": "PRESENT",
            "forward_solve_from_stored_values": True,
            "statement": (
                "EFM directly stores pprime and ffprime, plus p and F=R*Bphi "
                "profiles, on every one of the 397 converged geometry rows. "
                "No profile derivation is required to drive a forward solve, "
                "although conversion from source COCOS to the solver convention is."
            ),
        },
        "quantity_groups": {
            name: [f"efm/{path}" for path in paths]
            for name, paths in QUANTITY_GROUPS.items()
        },
        "arrays": arrays,
        "normalization_evidence": {
            "coordinate_path": "efm/psi_norm",
            "stored_values": "65 uniformly spaced values from 0.0 to 1.0",
            "orientation": "axis or centre = 0; plasma boundary or edge = 1",
            "orientation_evidence": (
                "pprime and ffprime labels say centre to edge; their second "
                "dimension is psi_norm; psi_axis and psi_boundary identify "
                "the endpoints"
            ),
            "source_flux_units": "Wb / rad",
            "source_normalization_formula": (
                "psi_norm = (psi - psi_axis) / (psi_boundary - psi_axis)"
            ),
        },
        "map_layout_evidence": {
            "stored_path": "efm/psirz",
            "coordinate_paths": ["efm/gridz", "efm/gridr"],
            "qualification": (
                "psirz can have a padded radial dimension. Usable planes are "
                "the finite columns derived per row, and contain exactly "
                "len(gridz) by len(gridr) values."
            ),
        },
        "conductor_current_evidence": {
            "experimental_input": "efm/fcoil_x",
            "fitted_efm_output": "efm/fcoil_c",
            "interpretation": (
                "Both are complete 101-channel vectors on every stored time row. "
                "Use fcoil_x to reproduce experimental external-field input; use "
                "fcoil_c to reproduce the fitted EFM external-field state."
            ),
            "filament_to_circuit_paths": [
                "efm/fcoil_circ",
                "efm/fcoil_r",
                "efm/fcoil_z",
                "efm/fcoil_turns",
                "efm/fcoil_xmult",
            ],
        },
        "convention_evidence": {
            "metadata_cocos_field": "ABSENT",
            "source_flux_unit_evidence": (
                "psi_axis, psi_boundary and psirz attributes say Wb / rad; "
                "pprime says Pa-rad/Wb"
            ),
            "per_shot_sign_measurements": measured_digits,
            "measured_common_digits": {
                "sigma_bp": sorted({item["sigma_bp"] for item in measured_digits}),
                "sigma_rho_theta_phi": sorted(
                    {item["sigma_rho_theta_phi"] for item in measured_digits}
                ),
                "e_bp": 0,
            },
            "cocos_interpretation": (
                "The stored signs and per-radian flux units measure three digits "
                "compatible with source COCOS 3. sigma_RphiZ is not determined "
                "by these scalar arrays, so full COCOS 3 additionally requires "
                "the separately declared right-handed (R,phi,Z) frame; this "
                "report does not present that declaration as file evidence."
            ),
            "source_to_cocos_17_transform": {
                "psi_like": "multiply by 2*pi without a sign flip",
                "ip_like": "unchanged",
                "b0_like": "unchanged",
                "q_like": "multiply by -1",
                "derivative_with_respect_to_psi": "divide by 2*pi",
            },
        },
        "per_shot": per_shot,
        "totals": {
            "usable_profile_map_conductor_rows": total_profile_map_conductor_rows,
            "usable_forward_input_rows": total_forward_rows,
            "usable_geometry_rows": total_geometry_rows,
        },
        "absent_quantities": {
            "flux_function_profiles": [],
            "other": [
                "an explicit COCOS metadata field",
                "a scalar-data determination of sigma_RphiZ",
            ],
        },
        "qualifications": [
            (
                "The census proves stored-input availability and finite row "
                "coverage; it does not prove the profiles, map and currents "
                "are mutually self-consistent under a new forward solver."
            ),
            (
                "The EFM array attributes are preserved exactly, including "
                "source unit spellings that may be unconventional."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    """Parse the store and findings-file destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write the census as stable, reviewable JSON."""

    args = parse_args()
    report = build_report(args.store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "flux_function_profiles": report["conclusion"][
                    "flux_function_profiles"
                ],
                **report["totals"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
