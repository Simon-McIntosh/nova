"""Census plasma-current sign closure from stored MAST EFM flux maps.

For one common usable slice per frozen shot, this command reports both stored
plasma currents beside the current obtained by applying Nova's conservation
operator to the stored poloidal-flux map and integrating only inside the stored
LCFS.  Source flux is per toroidal radian, so it is converted to total flux
before applying ``mu0*jphi = -delta_star(Phi)/(2*pi*R)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import zarr
from scipy.constants import mu_0

jax.config.update("jax_enable_x64", True)

from nova.equilibrium.conservation import FluxLattice, delta_star  # noqa: E402
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR  # noqa: E402
from nova.equilibrium.wall_mask import inside_polygon  # noqa: E402
from nova.imas.mast_vacuum_cohort import SHOT_STORE  # noqa: E402

FROZEN_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)
COMMON_SLICE_INDEX = 46
CONTROL_SHOT = 21978
CONTROL_DERIVED_CURRENT_A = 925091.932
CONTROL_STORED_CURRENT_A = 925345.938
CONTROL_SIGNED_RELATIVE_CLOSURE = -2.744983e-4
CONTROL_CURRENT_ABSOLUTE_TOLERANCE_A = 0.01
CONTROL_RELATIVE_TOLERANCE = 1.0e-10


def _uniform_axis(values: np.ndarray, name: str) -> np.ndarray:
    """Return a float64 uniform axis while preserving its stored endpoints."""

    stored = np.asarray(values)
    resolution = np.finfo(stored.dtype).eps
    values = np.asarray(stored, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional coordinate axis")
    expected = np.linspace(values[0], values[-1], values.size)
    tolerance = 16.0 * resolution * max(1.0, abs(values[-1]))
    if not np.allclose(values, expected, rtol=0.0, atol=tolerance):
        deviation = float(np.max(np.abs(values - expected)))
        raise ValueError(f"{name} is not uniform; maximum deviation is {deviation:.6g}")
    return expected


def _finite_flux_map(
    group: zarr.Group, slice_index: int, stored_radius: np.ndarray
) -> np.ndarray:
    """Return the finite live flux plane from a possibly padded raster."""

    raw = np.asarray(group["psirz"][slice_index], dtype=np.float64)
    finite_columns = np.flatnonzero(np.all(np.isfinite(raw), axis=0))
    if finite_columns.size != stored_radius.size:
        raise ValueError(
            "the selected flux map does not carry exactly one finite column "
            f"per radial coordinate ({finite_columns.size} != {stored_radius.size})"
        )
    if "profile_r" in group:
        padded_radius = np.asarray(group["profile_r"][:], dtype=np.float64)
        if not np.allclose(
            padded_radius[finite_columns], stored_radius, rtol=2.0e-7, atol=1.0e-8
        ):
            raise ValueError("finite flux-map columns do not match the radial grid")
    return raw[:, finite_columns]


def _stored_lcfs(group: zarr.Group, slice_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite stored LCFS polygon for one time row."""

    count_value = float(group["lcfsn_c"][slice_index])
    if not np.isfinite(count_value) or count_value != int(count_value):
        raise ValueError(f"stored LCFS point count is invalid: {count_value}")
    count = int(count_value)
    radius = np.asarray(group["lcfs_r"][slice_index], dtype=np.float64)
    height = np.asarray(group["lcfs_z"][slice_index], dtype=np.float64)
    if count < 3 or count > radius.size or count > height.size:
        raise ValueError(f"stored LCFS point count {count} does not fit its arrays")
    radius = radius[:count]
    height = height[:count]
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(height)):
        raise ValueError("stored LCFS polygon contains non-finite vertices")
    return radius, height


def _sign(value: float) -> int:
    """Return the sign of a finite, nonzero current."""

    if not np.isfinite(value) or value == 0.0:
        raise ValueError(f"current must be finite and nonzero, received {value}")
    return int(np.sign(value))


def census_shot(
    store: Path,
    shot: int,
    slice_index: int = COMMON_SLICE_INDEX,
) -> dict[str, Any]:
    """Measure four-way sign agreement and signed current closure for one shot."""

    shot_path = store / f"{shot}.zarr"
    root = zarr.open_group(shot_path.as_posix(), mode="r")
    if "efm" not in root:
        raise ValueError(f"shot {shot} has no efm group")
    group = root["efm"]
    required = {
        "time",
        "gridr",
        "gridz",
        "psirz",
        "lcfs_r",
        "lcfs_z",
        "lcfsn_c",
        "plasma_current_x",
        "plasma_current_c",
    }
    missing = sorted(required.difference(group.array_keys()))
    if missing:
        raise ValueError(f"shot {shot} is missing EFM arrays {missing}")

    time = np.asarray(group["time"][:], dtype=np.float64)
    if slice_index < 0 or slice_index >= time.size:
        raise IndexError(
            f"slice index {slice_index} is outside shot {shot} rows 0..{time.size - 1}"
        )
    if not np.isfinite(time[slice_index]):
        raise ValueError(f"shot {shot} slice {slice_index} has no finite time")

    stored_radius = np.asarray(group["gridr"][:])
    stored_height = np.asarray(group["gridz"][:])
    radius = _uniform_axis(stored_radius, "gridr")
    height = _uniform_axis(stored_height, "gridz")
    flux_per_radian = _finite_flux_map(group, slice_index, stored_radius)
    if flux_per_radian.shape != (height.size, radius.size):
        raise ValueError(
            "live flux-map shape does not match the coordinate grid: "
            f"{flux_per_radian.shape} != {(height.size, radius.size)}"
        )

    lattice = FluxLattice(radius, height)
    total_flux = TOTAL_FLUX_FACTOR * flux_per_radian.T.reshape(-1)
    elliptic = np.asarray(delta_star(lattice, total_flux), dtype=np.float64)
    current_density = -elliptic / (TOTAL_FLUX_FACTOR * mu_0 * lattice.node_radius)
    lcfs_radius, lcfs_height = _stored_lcfs(group, slice_index)
    stored_lcfs_interior = inside_polygon(
        lattice.coordinate[:, 0],
        lattice.coordinate[:, 1],
        lcfs_radius,
        lcfs_height,
    )
    integration_mask = (
        np.asarray(lattice.interior(), dtype=bool)
        & stored_lcfs_interior
        & np.isfinite(current_density)
    )
    derived_current = float(
        np.sum(current_density[integration_mask] * lattice.cell_area[integration_mask])
    )
    experimental_current = float(group["plasma_current_x"][slice_index])
    fitted_current = float(group["plasma_current_c"][slice_index])
    signed_relative_closure = (derived_current - fitted_current) / abs(fitted_current)

    declared_cocos = 3
    declared_expected_sign = 1
    signs = {
        "plasma_current_x": _sign(experimental_current),
        "plasma_current_c": _sign(fitted_current),
        "derived_masked_integral": _sign(derived_current),
        "declared_cocos_3_expected": declared_expected_sign,
    }
    sign_agreement = len(set(signs.values())) == 1
    return {
        "shot": shot,
        "slice_index": slice_index,
        "time_s": float(time[slice_index]),
        "declared_cocos": declared_cocos,
        "declared_cocos_3_expected_current_sign": declared_expected_sign,
        "plasma_current_x_A": experimental_current,
        "plasma_current_c_A": fitted_current,
        "derived_masked_integral_A": derived_current,
        "signed_relative_closure_against_plasma_current_c": (signed_relative_closure),
        "signs": signs,
        "all_four_signs_agree": sign_agreement,
        "stored_lcfs_vertices": int(lcfs_radius.size),
        "masked_operator_cells": int(np.count_nonzero(integration_mask)),
        "live_flux_map_shape": list(flux_per_radian.shape),
        "source_paths": {
            "experimental_current": "efm/plasma_current_x",
            "fitted_current": "efm/plasma_current_c",
            "flux_map": "efm/psirz",
            "grid": ["efm/gridz", "efm/gridr"],
            "lcfs": ["efm/lcfs_r", "efm/lcfs_z", "efm/lcfsn_c"],
        },
    }


def build_report(
    store: Path = SHOT_STORE,
    shots: tuple[int, ...] = FROZEN_SHOTS,
    slice_index: int = COMMON_SLICE_INDEX,
) -> dict[str, Any]:
    """Build the frozen-shot current-sign census and verify its control datum."""

    per_shot = [census_shot(store, shot, slice_index) for shot in shots]
    disagreements = [
        {"shot": row["shot"], "slice_index": row["slice_index"]}
        for row in per_shot
        if not row["all_four_signs_agree"]
    ]
    control = next(row for row in per_shot if row["shot"] == CONTROL_SHOT)
    control_checks = {
        "derived_current_reproduced": bool(
            np.isclose(
                control["derived_masked_integral_A"],
                CONTROL_DERIVED_CURRENT_A,
                rtol=0.0,
                atol=CONTROL_CURRENT_ABSOLUTE_TOLERANCE_A,
            )
        ),
        "stored_current_reproduced": bool(
            np.isclose(
                control["plasma_current_c_A"],
                CONTROL_STORED_CURRENT_A,
                rtol=0.0,
                atol=CONTROL_CURRENT_ABSOLUTE_TOLERANCE_A,
            )
        ),
        "signed_relative_closure_reproduced": bool(
            np.isclose(
                control["signed_relative_closure_against_plasma_current_c"],
                CONTROL_SIGNED_RELATIVE_CLOSURE,
                rtol=0.0,
                atol=CONTROL_RELATIVE_TOLERANCE,
            )
        ),
    }
    if not all(control_checks.values()):
        raise ValueError(f"shot {CONTROL_SHOT} control datum changed: {control_checks}")

    return {
        "store": str(store),
        "shots": list(shots),
        "slice_selection": {
            "rule": "common stored row",
            "slice_index": slice_index,
            "reason": (
                "row 46 is finite and carries a valid stored LCFS on every frozen "
                "shot, and preserves the predeclared shot-21978 control"
            ),
        },
        "operator": {
            "implementation": "nova.equilibrium.conservation.delta_star",
            "source_flux_units": "Wb / rad",
            "operator_flux_units": "Wb",
            "conversion": "Phi = 2*pi*psi",
            "current_relation": "mu0*jphi = -delta_star(Phi)/(2*pi*R)",
            "integration_domain": (
                "stored LCFS interior intersected with operator interior"
            ),
        },
        "declared_frame": {
            "cocos": 3,
            "expected_plasma_current_sign": 1,
            "basis": (
                "the declared MAST COCOS 3 forward-current frame; source-to-target "
                "conversion leaves Ip unchanged"
            ),
        },
        "per_shot": per_shot,
        "disagreements": disagreements,
        "summary": {
            "shots_checked": len(per_shot),
            "shots_with_four_way_sign_agreement": len(per_shot) - len(disagreements),
            "shots_with_sign_disagreement": len(disagreements),
            "maximum_absolute_signed_relative_closure": float(
                max(
                    abs(row["signed_relative_closure_against_plasma_current_c"])
                    for row in per_shot
                )
            ),
        },
        "control": {
            "shot": CONTROL_SHOT,
            "slice_index": slice_index,
            "expected_derived_current_A": CONTROL_DERIVED_CURRENT_A,
            "expected_stored_current_A": CONTROL_STORED_CURRENT_A,
            "expected_signed_relative_closure": CONTROL_SIGNED_RELATIVE_CLOSURE,
            "checks": control_checks,
            "passed": all(control_checks.values()),
        },
        "qualification": (
            "This census establishes current-sign consistency and masked integral "
            "closure on one usable slice per shot; it does not establish temporal "
            "sign stability over every stored row."
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse source-store and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slice-index", type=int, default=COMMON_SLICE_INDEX)
    return parser.parse_args()


def main() -> None:
    """Write a stable JSON findings file and print its headline metrics."""

    args = parse_args()
    report = build_report(args.store, slice_index=args.slice_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **report["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
