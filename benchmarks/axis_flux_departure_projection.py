"""Project axis-flux and internal-current departures onto the span drift.

The diverted census has 43,017 finite rows whose Nova-to-EFIT span ratio has a
solenoid-current slope.  The preceding attribution measured profile amplitude,
boundary selection and external conductor source, and those three arms leave
the slope unexplained.  This benchmark measures the two remaining solve-side
quantities with the same paired projection:

* Nova minus EFIT magnetic-axis flux, divided by the EFIT reference span;
* the shape difference between Nova's enclosed-current profile and an EFIT
  cumulative current profile reconstructed from its stored current-density
  map and flux map.

Each arm has an exposure fit against solenoid current and an outcome fit with
solenoid current held fixed.  Their product is the portion of the ratio slope
attributed to that candidate.  No equilibrium is re-solved and no repair is
attempted.

The measurement is intended for one ``all_debug`` CPU allocation.  In a
worktree, the login-node invocation is::

    UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" \
        uv run --no-sync python benchmarks/axis_flux_departure_projection.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import jax
import netCDF4
import numpy as np
import zarr

from benchmarks.diverted_drift_regression import ols_fit
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "docs/figures/limited-boundary-census/limited-class-census.npz"
SESSION_ROOT = Path("/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29")
STORE_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")
OUTPUT = ROOT / "docs/figures/playable-forward-solve/span-ratio"
EXPECTED_ROWS = 43_017
EXPECTED_SLOPE = -0.013821226857584001
PROFILE_LEVELS = np.linspace(0.0, 0.995, 26)


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _strict(value: Any) -> Any:
    """Convert NumPy values and non-finite values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fit(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Return the common compact regression receipt."""
    result = ols_fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return {
        "slope": float(result["slope"]),
        "slope_se": float(result["slope_se"]),
        "r_squared": float(result["r_squared"]),
        "t_statistic": float(result["t_statistic"]),
        "n": int(result["n"]),
    }


def _controlled_fit(
    candidate: np.ndarray, ratio: np.ndarray, solenoid: np.ndarray
) -> dict[str, float | int]:
    """Fit ratio to candidate with solenoid current as a control."""
    design = np.column_stack((np.ones(candidate.size), solenoid, candidate))
    coefficients, *_ = np.linalg.lstsq(design, ratio, rcond=None)
    residual = ratio - design @ coefficients
    total_ss = float(np.sum((ratio - ratio.mean()) ** 2))
    reduced = np.column_stack((np.ones(candidate.size), solenoid))
    reduced_residual = ratio - reduced @ np.linalg.lstsq(reduced, ratio, rcond=None)[0]
    reduced_ss = float(np.sum(reduced_residual**2))
    full_ss = float(np.sum(residual**2))
    return {
        "candidate_coefficient_per_unit": float(coefficients[2]),
        "solenoid_coefficient_per_kA": float(coefficients[1]),
        "r_squared": 1.0 - full_ss / total_ss,
        "partial_r_squared_given_solenoid": (reduced_ss - full_ss) / reduced_ss,
        "n": int(candidate.size),
    }


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return nearest source indices for target times."""
    insertion = np.searchsorted(source, target)
    right = np.clip(insertion, 0, source.size - 1)
    left = np.clip(insertion - 1, 0, source.size - 1)
    choose_left = np.abs(target - source[left]) <= np.abs(source[right] - target)
    return np.where(choose_left, left, right).astype(int)


def _session(shot: int, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read selected persisted session arrays."""
    with netCDF4.Dataset(str(SESSION_ROOT / f"{shot}.nc"), "r") as store:
        group = next(iter(store.groups.values()))
        return {name: np.asarray(group[name][:], dtype=np.float64) for name in names}


def _population() -> dict[str, np.ndarray]:
    """Build the exact finite/profile-qualified population used by landed arms."""
    census = np.load(CENSUS, allow_pickle=True)
    shot = np.asarray(census["shot"], dtype=int)
    time = np.asarray(census["time_s"], dtype=np.float64)
    ratio = np.asarray(census["span_ratio_nova_over_efit"], dtype=np.float64)
    positions: list[np.ndarray] = []
    efit_rows: list[np.ndarray] = []
    members: list[np.ndarray] = []
    for shot_number in np.unique(shot):
        selected = np.flatnonzero(
            (shot == shot_number)
            & np.asarray(census["diverted"], dtype=bool)
            & np.isfinite(time)
            & np.isfinite(ratio)
        )
        group = zarr.open_group(str(STORE_ROOT / f"{shot_number}.zarr"), mode="r")[
            "efm"
        ]
        efm_time = np.asarray(group["time"], dtype=np.float64)
        rows = _nearest_indices(efm_time, time[selected])
        axis_r = np.asarray(group["magnetic_axis_r"], dtype=np.float64)[rows]
        axis_z = np.asarray(group["magnetic_axis_z"], dtype=np.float64)[rows]
        xpoints = np.column_stack(
            (
                np.asarray(group["xpoint1_rc"], dtype=np.float64)[rows],
                np.asarray(group["xpoint1_zc"], dtype=np.float64)[rows],
                np.asarray(group["xpoint2_rc"], dtype=np.float64)[rows],
                np.asarray(group["xpoint2_zc"], dtype=np.float64)[rows],
            )
        )
        has_xpoint = np.isfinite(xpoints).reshape(rows.size, 2, 2).any(axis=(1, 2))
        pressure = np.asarray(group["pprime"], dtype=np.float64)[rows]
        pressure_count = np.isfinite(pressure).sum(axis=1)
        pressure_mean = np.divide(
            np.nansum(pressure, axis=1),
            pressure_count,
            out=np.full(rows.size, np.nan),
            where=pressure_count > 0,
        )
        reference_span = TOTAL_FLUX_FACTOR * np.abs(
            np.asarray(group["psi_boundary"], dtype=np.float64)[rows]
            - np.asarray(group["psi_axis"], dtype=np.float64)[rows]
        )
        admitted = (
            np.isfinite(axis_r)
            & np.isfinite(axis_z)
            & has_xpoint
            & np.isfinite(reference_span)
            & (reference_span > 0.0)
            & (pressure_count > 5)
            & (pressure_mean > 0.0)
        )
        positions.append(selected[admitted])
        efit_rows.append(rows[admitted])
        members.append(np.full(int(admitted.sum()), int(shot_number), dtype=int))
    position = np.concatenate(positions)
    row = np.concatenate(efit_rows)
    member = np.concatenate(members)
    order = np.argsort(position)
    position, row, member = position[order], row[order], member[order]
    if position.size != EXPECTED_ROWS:
        raise RuntimeError(
            f"population has {position.size} rows, expected {EXPECTED_ROWS}"
        )
    solenoid = np.empty(position.size, dtype=np.float64)
    for shot_number in np.unique(member):
        mask = member == shot_number
        group = zarr.open_group(str(STORE_ROOT / f"{shot_number}.zarr"), mode="r")[
            "efm"
        ]
        solenoid[mask] = (
            np.asarray(group["fcoil_c"], dtype=np.float64)[row[mask], 0] / 1.0e3
        )
    return {
        "census": census,
        "position": position,
        "row": row,
        "member": member,
        "solenoid": solenoid,
        "ratio": np.asarray(census["span_ratio_nova_over_efit"], dtype=np.float64)[
            position
        ],
        "time": np.asarray(census["time_s"], dtype=np.float64)[position],
    }


def _axis_candidate(
    population: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the normalized Nova-minus-EFIT axis flux departure."""
    candidate = np.empty(EXPECTED_ROWS, dtype=np.float64)
    axis_delta = np.empty(EXPECTED_ROWS, dtype=np.float64)
    reference_span = np.empty(EXPECTED_ROWS, dtype=np.float64)
    for shot_number in np.unique(population["member"]):
        mask = population["member"] == shot_number
        group = zarr.open_group(str(STORE_ROOT / f"{shot_number}.zarr"), mode="r")[
            "efm"
        ]
        rows = population["row"][mask]
        session = _session(int(shot_number), ("time", "flux_surface_psi"))
        frames = _nearest_indices(session["time"], population["time"][mask])
        nova_axis = session["flux_surface_psi"][0, frames]
        efit_axis = (
            TOTAL_FLUX_FACTOR * np.asarray(group["psi_axis"], dtype=np.float64)[rows]
        )
        efit_boundary = (
            TOTAL_FLUX_FACTOR
            * np.asarray(group["psi_boundary"], dtype=np.float64)[rows]
        )
        span = np.abs(efit_boundary - efit_axis)
        axis_delta[mask] = nova_axis - efit_axis
        reference_span[mask] = span
        candidate[mask] = axis_delta[mask] / span
    if not np.all(np.isfinite(candidate)):
        raise RuntimeError("axis-flux candidate contains non-finite rows")
    return candidate, {
        "axis_flux_delta_on_solenoid": _fit(population["solenoid"], axis_delta),
        "reference_span_on_solenoid": _fit(population["solenoid"], reference_span),
        "candidate_distribution": {
            "median": float(np.median(candidate)),
            "p10": float(np.quantile(candidate, 0.1)),
            "p90": float(np.quantile(candidate, 0.9)),
        },
    }


def _efit_current_profile(group: zarr.Group, row: int) -> np.ndarray:
    """Reconstruct EFIT enclosed current on the Nova profile levels."""
    density = np.asarray(group["plasma_current_rz"], dtype=np.float64)[row]
    flux = np.asarray(group["psirz"], dtype=np.float64)[row][:, ::2]
    axis = float(np.asarray(group["psi_axis"], dtype=np.float64)[row])
    boundary = float(np.asarray(group["psi_boundary"], dtype=np.float64)[row])
    normalized = (flux - axis) / (boundary - axis)
    radial = np.asarray(group["gridr"], dtype=np.float64)
    vertical = np.asarray(group["gridz"], dtype=np.float64)
    weights = (
        density
        * float(np.median(np.diff(radial)))
        * float(np.median(np.diff(vertical)))
    )
    valid = np.isfinite(normalized) & np.isfinite(weights) & (normalized >= 0.0)
    total = np.nansum(
        np.where(valid & (normalized <= PROFILE_LEVELS[-1]), weights, 0.0)
    )
    if not np.isfinite(total) or abs(total) <= 0.0:
        raise RuntimeError(
            f"EFIT current profile has no finite enclosed total at row {row}"
        )
    values = np.asarray(
        [
            np.nansum(np.where(valid & (normalized <= level), weights, 0.0))
            for level in PROFILE_LEVELS
        ],
        dtype=np.float64,
    )
    return values / values[-1]


def _current_candidate(
    population: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return normalized cumulative-current shape mismatch per row."""
    candidate = np.empty(EXPECTED_ROWS, dtype=np.float64)
    mismatch_count = 0
    for shot_number in np.unique(population["member"]):
        mask = population["member"] == shot_number
        group = zarr.open_group(str(STORE_ROOT / f"{shot_number}.zarr"), mode="r")[
            "efm"
        ]
        rows = population["row"][mask]
        session = _session(int(shot_number), ("time", "Ip_profile"))
        frames = _nearest_indices(session["time"], population["time"][mask])
        nova = session["Ip_profile"][:, frames]
        nova_total = nova[-1]
        nova_shape = nova / nova_total[None, :]
        for source_row in np.unique(rows):
            row_mask = rows == source_row
            efit_shape = _efit_current_profile(group, int(source_row))
            differences = nova_shape[:, row_mask].T - efit_shape[None, :]
            values = np.sqrt(np.mean(differences[:, 1:] ** 2, axis=1))
            candidate[mask][row_mask] = values
            mismatch_count += int(np.sum(~np.isfinite(values)))
    if mismatch_count or not np.all(np.isfinite(candidate)):
        raise RuntimeError(
            f"current-distribution candidate has {mismatch_count} non-finite rows"
        )
    return candidate, {
        "profile_levels": PROFILE_LEVELS.tolist(),
        "comparison": (
            "RMS difference of normalized enclosed-current profiles, excluding "
            "the axis point"
        ),
        "candidate_distribution": {
            "median": float(np.median(candidate)),
            "p10": float(np.quantile(candidate, 0.1)),
            "p90": float(np.quantile(candidate, 0.9)),
        },
    }


def _arm(
    name: str,
    candidate: np.ndarray,
    population: dict[str, np.ndarray],
    definition: str,
    units: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Build one durable paired-projection receipt."""
    total = _fit(population["solenoid"], population["ratio"])
    if not np.isclose(total["slope"], EXPECTED_SLOPE, rtol=0.0, atol=2.0e-12):
        raise RuntimeError(f"census slope changed: {total['slope']}")
    exposure = _fit(population["solenoid"], candidate)
    outcome = _controlled_fit(candidate, population["ratio"], population["solenoid"])
    projected = float(exposure["slope"]) * float(
        outcome["candidate_coefficient_per_unit"]
    )
    return {
        "candidate": name,
        "source_revision": _revision(),
        "x64_enabled": bool(jax.config.x64_enabled),
        "population": {
            "rows": EXPECTED_ROWS,
            "definition": (
                "the same diverted finite/profile-qualified census rows as the "
                "landed solenoid-span attribution"
            ),
        },
        "candidate_definition": definition,
        "candidate_units": units,
        "paired_projection": {
            "exposure_arm_candidate_on_solenoid": exposure,
            "outcome_arm_ratio_on_candidate_with_solenoid_fixed": outcome,
            "marginal_ratio_on_candidate": _fit(candidate, population["ratio"]),
            "projected_slope_per_kA": projected,
            "fraction_of_fitted_slope": projected / float(total["slope"]),
            "total_ratio_slope": total,
        },
        "diagnostics": diagnostics,
    }


def main(argv: list[str] | None = None) -> None:
    """Run both arms and persist each arm before the combined receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    configure_dtypes()
    if not jax.config.x64_enabled:
        raise RuntimeError("configure_dtypes did not enable extended precision")
    population = _population()
    axis, axis_diagnostics = _axis_candidate(population)
    axis_receipt = _arm(
        "nova-minus-EFIT axis flux departure",
        axis,
        population,
        (
            "Nova flux_surface_psi axis value minus 2pi-scaled EFIT psi_axis, "
            "divided by the EFIT axis-to-boundary flux span"
        ),
        "fraction of reference flux span",
        axis_diagnostics,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / "axis-flux-arm.json", axis_receipt)
    current, current_diagnostics = _current_candidate(population)
    current_receipt = _arm(
        "internal current distribution shape mismatch",
        current,
        population,
        (
            "RMS difference between Nova's normalized enclosed-current profile and "
            "EFIT's normalized cumulative plasma_current_rz profile on the same "
            "26 flux levels"
        ),
        "normalized profile RMS",
        current_diagnostics,
    )
    _write(args.output / "current-distribution-arm.json", current_receipt)
    fractions = {
        "axis_flux": float(
            axis_receipt["paired_projection"]["fraction_of_fitted_slope"]
        ),
        "internal_current_distribution": float(
            current_receipt["paired_projection"]["fraction_of_fitted_slope"]
        ),
    }
    combined = {
        "receipt": "paired axis-flux and internal-current projection",
        "source_revision": _revision(),
        "x64_enabled": bool(jax.config.x64_enabled),
        "population_rows": EXPECTED_ROWS,
        "total_ratio_slope": axis_receipt["paired_projection"]["total_ratio_slope"],
        "fraction_of_fitted_slope": fractions,
        "fraction_sum": float(sum(fractions.values())),
        "remainder_fraction": float(1.0 - sum(fractions.values())),
        "arms": {
            "axis_flux": "axis-flux-arm.json",
            "internal_current_distribution": "current-distribution-arm.json",
        },
        "axis_flux_diagnosis": {
            "most_of_slope": bool(abs(fractions["axis_flux"]) > 0.5),
            "interpretation": (
                "Axis-flux departure is the dominant measured carrier; distinguish "
                "normalization, seed and current-distribution causes in the "
                "forward solve."
                if abs(fractions["axis_flux"]) > 0.5
                else (
                    "Axis-flux departure does not account for most of the fitted slope."
                )
            ),
            "unqualified_axis_normalisation_mechanism": (
                "not established by this projection; the axis read and "
                "normalization guard require a separate identity check"
            ),
        },
    }
    _write(args.output / "axis-flux-projection.json", combined)
    print(json.dumps({"rows": EXPECTED_ROWS, "fractions": fractions}, allow_nan=False))


if __name__ == "__main__":
    main()
