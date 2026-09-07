"""Diagnose the diverted class's monotonic span-ratio drift against EFIT.

The committed limited-class census measured, per guarded frame, the ratio of
nova's stored axis-to-boundary flux span to EFIT's on the nearest efm slice.
Banded in time the diverted median ratio rises monotonically from about 0.70
in the earliest band through 0.83 to 1.15 late, crossing unity around 300 to
400 ms.  A trend that crosses unity is not a scale error or a missing filter:
it looks like a quantity that varies through a discharge.  This benchmark
breaks the confound between time and everything that varies with time by
regressing the diverted ratio against per-slice quantities -- plasma current,
solenoid current, X-point distance from the magnetic axis, X-point distance
from the solve grid (a resolution proxy), boundary elongation and fitted
pressure peaking -- and reports which, if any, explain the drift better than
time does, or states plainly that none do.

The per-slice quantities are read on the same nearest-efm-slice time base the
census ratio itself uses, so a candidate and the ratio are measured at the
same instant.

Diagnosis only: no repair is proposed or implemented here.

Usage:
    UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync \\
        python benchmarks/diverted_drift_regression.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import scipy.stats
import zarr

CENSUS = Path("docs/figures/limited-boundary-census/limited-class-census.npz")
SESSION_ROOT = Path("/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29")
STORE_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")
OUTPUT_DIR = Path("docs/figures/playable-forward-solve/diverted-drift")
BANDS = (
    (0.0, 0.05, "0-50 ms"),
    (0.05, 0.10, "50-100 ms"),
    (0.10, 0.15, "100-150 ms"),
    (0.15, 0.20, "150-200 ms"),
    (0.20, 0.30, "200-300 ms"),
    (0.30, 0.40, "300-400 ms"),
    (0.40, 1.0, "400-1000 ms"),
)

CANDIDATES = (
    ("ip", "plasma current [kA]"),
    ("isol", "solenoid current [kA]"),
    ("d_xp_axis", "X-point to axis [m]"),
    ("d_xp_grid", "X-point to nearest grid node [m]"),
    ("kappa", "boundary elongation"),
    ("peaking", "pressure peaking p(0)/<p>"),
)


def ols_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least-squares y = b0 + b1 x with a t-statistic on the slope."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack((np.ones_like(x), x))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    freedom = max(int(y.size - design.shape[1]), 1)
    sigma = float(np.sum(residual**2) / freedom)
    covariance = sigma * np.linalg.inv(design.T @ design)
    standard_error = float(np.sqrt(covariance[1, 1]))
    slope = float(coefficients[1])
    t_statistic = slope / standard_error if standard_error > 0.0 else np.nan
    p_value = (
        2.0 * (1.0 - scipy.stats.t.cdf(abs(t_statistic), freedom))
        if np.isfinite(t_statistic)
        else np.nan
    )
    ss_total = float(np.sum((y - y.mean()) ** 2))
    r_squared = (
        1.0 - float(np.sum(residual**2)) / ss_total if ss_total > 0.0 else np.nan
    )
    log_likelihood = (
        -0.5 * y.size * (np.log(2.0 * np.pi * sigma) + 1.0)
        if np.isfinite(sigma) and sigma > 0.0
        else np.nan
    )
    return {
        "intercept": float(coefficients[0]),
        "slope": slope,
        "slope_se": standard_error,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "r_squared": r_squared,
        "aic": 2.0 * design.shape[1] - 2.0 * log_likelihood,
        "n": int(y.size),
    }


def ols_fit_multi(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least-squares y = b0 + b1 x1 + b2 x2; report the x2 (last) term."""
    design = np.column_stack((np.ones(x.shape[0]), *np.moveaxis(x, -1, 0)))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    freedom = max(int(y.size - design.shape[1]), 1)
    sigma = float(np.sum(residual**2) / freedom)
    covariance = sigma * np.linalg.inv(design.T @ design)
    standard_error = float(np.sqrt(covariance[-1, -1]))
    slope = float(coefficients[-1])
    t_statistic = slope / standard_error if standard_error > 0.0 else np.nan
    p_value = (
        2.0 * (1.0 - scipy.stats.t.cdf(abs(t_statistic), freedom))
        if np.isfinite(t_statistic)
        else np.nan
    )
    ss_total = float(np.sum((y - y.mean()) ** 2))
    r_squared = (
        1.0 - float(np.sum(residual**2)) / ss_total if ss_total > 0.0 else np.nan
    )
    log_likelihood = (
        -0.5 * y.size * (np.log(2.0 * np.pi * sigma) + 1.0)
        if np.isfinite(sigma) and sigma > 0.0
        else np.nan
    )
    return {
        "slope": slope,
        "slope_se": standard_error,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "r_squared": r_squared,
        "aic": 2.0 * design.shape[1] - 2.0 * log_likelihood,
        "n": int(y.size),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return the rank correlation and its population p-value."""
    statistic, p_value = scipy.stats.spearmanr(x, y)
    return float(statistic), float(p_value)


def _load_efm(group: zarr.Group, slice_index: int) -> dict[str, Any]:
    """Return the per-slice reference quantities for one efm slice."""
    return {
        "axis_r": float(np.asarray(group["magnetic_axis_r"], dtype=float)[slice_index]),
        "axis_z": float(np.asarray(group["magnetic_axis_z"], dtype=float)[slice_index]),
        "xpoints": (
            np.asarray(group["xpoint1_rc"], dtype=float)[slice_index],
            np.asarray(group["xpoint1_zc"], dtype=float)[slice_index],
            np.asarray(group["xpoint2_rc"], dtype=float)[slice_index],
            np.asarray(group["xpoint2_zc"], dtype=float)[slice_index],
        ),
        "ip": np.asarray(group["plasma_current_c"], dtype=float),
        "isol": np.asarray(group["fcoil_c"], dtype=float)[:, 0],
        "kappa": np.asarray(group["elongation"], dtype=float),
        "ppsi_c": np.asarray(group["ppsi_c"], dtype=float),
        "grid": np.column_stack(
            (
                np.asarray(group["gridr"], dtype=float)[::2],
                np.asarray(group["gridz"], dtype=float)[::2],
            )
        ),
    }


def _session_arrays(shot: int) -> dict[str, np.ndarray]:
    """Read the session's needed channels directly from the netCDF group."""
    path = SESSION_ROOT / f"{shot}.nc"
    result: dict[str, np.ndarray] = {}
    with netCDF4.Dataset(str(path), "r") as store:
        group_name = next(iter(store.groups))
        group = store[group_name]
        for name in (
            "time",
            "x_point_r",
            "x_point_z",
            "magnetic_axis_r",
            "magnetic_axis_z",
            "elongation",
        ):
            result[name] = np.asarray(group[name][:])
    return result


def per_frame_row(
    efm: dict[str, Any], session: dict[str, np.ndarray], frame: int, slice_index: int
) -> dict[str, float]:
    """Return this frame's per-slice candidate quantities on the efm slice."""
    x1_r, x1_z, x2_r, x2_z = efm["xpoints"]
    axis_r, axis_z = efm["axis_r"], efm["axis_z"]
    distances = [
        float(np.hypot(radius - axis_r, height - axis_z))
        for radius, height in ((x1_r, x1_z), (x2_r, x2_z))
        if np.isfinite(radius) and np.isfinite(height)
    ]
    d_xp_axis = min(distances) if distances else np.nan

    grid_distances = [
        float(np.min(np.hypot(efm["grid"][:, 0] - radius, efm["grid"][:, 1] - height)))
        for radius, height in ((x1_r, x1_z), (x2_r, x2_z))
        if np.isfinite(radius) and np.isfinite(height)
    ]
    d_xp_grid = min(grid_distances) if grid_distances else np.nan

    pressure = efm["ppsi_c"][slice_index]
    finite = pressure[np.isfinite(pressure)]
    peaking = (
        float(finite[0] / np.mean(finite))
        if (finite.size > 5 and np.mean(finite) > 0.0)
        else np.nan
    )

    nova_axis_r = float(session["magnetic_axis_r"][frame])
    nova_axis_z = float(session["magnetic_axis_z"][frame])
    if np.isfinite(nova_axis_r):
        nova_distances = np.hypot(
            session["x_point_r"][:, frame] - nova_axis_r,
            session["x_point_z"][:, frame] - nova_axis_z,
        )
        d_xp_axis_nova = float(np.nanmin(nova_distances))
    else:
        d_xp_axis_nova = np.nan

    return {
        "ip": float(efm["ip"][slice_index]) / 1.0e3,
        "isol": float(efm["isol"][slice_index]) / 1.0e3,
        "d_xp_axis": d_xp_axis,
        "d_xp_axis_nova": d_xp_axis_nova,
        "d_xp_grid": d_xp_grid,
        "kappa": float(efm["kappa"][slice_index]),
        "boundary_elongation_nova": float(session["elongation"][-1, frame]),
        "peaking": peaking,
    }


def _grid_spacing_details(group: zarr.Group) -> dict[str, float]:
    """State the solve grid's uniformity so the resolution proxy is read aright."""
    radius = np.asarray(group["gridr"], dtype=float)[::2]
    height = np.asarray(group["gridz"], dtype=float)[::2]
    return {
        "radial_step_m": float(np.median(np.diff(radius))),
        "vertical_step_m": float(np.median(np.diff(height))),
        "node_count": int(radius.size * height.size),
        "half_cell_diagonal_max_m": float(
            np.hypot(np.median(np.diff(radius)) / 2.0, np.median(np.diff(height)) / 2.0)
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Run the diverted-drift regressions and write the receipt and figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, default=CENSUS)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args(argv)

    census = np.load(arguments.census, allow_pickle=True)
    shot = census["shot"]
    time_s = census["time_s"]
    diverted = census["diverted"]
    cold = census["cold_start"]
    ratio = census["span_ratio_nova_over_efit"]

    select = diverted & np.isfinite(time_s) & np.isfinite(ratio)
    rows: list[dict[str, float]] = []
    grid_report: dict[str, float] | None = None
    for member in np.unique(shot[select]):
        group = zarr.open_group(str(STORE_ROOT / f"{member}.zarr"), mode="r")["efm"]
        efm_time = np.asarray(group["time"], dtype=float)
        session = _session_arrays(int(member))
        session_time = session["time"]
        for position in np.flatnonzero((shot == member) & select):
            frame = int(np.argmin(np.abs(session_time - time_s[position])))
            slice_index = int(np.argmin(np.abs(efm_time - time_s[position])))
            values = per_frame_row(
                _load_efm(group, slice_index), session, frame, slice_index
            )
            values["time_s"] = float(time_s[position])
            values["cold"] = bool(cold[position])
            values["ratio"] = float(ratio[position])
            rows.append(values)
        if grid_report is None:
            grid_report = _grid_spacing_details(group)

    n_all = len(rows)
    rows = [
        row
        for row in rows
        if np.isfinite(row["peaking"]) and np.isfinite(row["d_xp_axis"])
    ]
    n_finite = len(rows)

    table = np.asarray(
        [
            (
                row["time_s"],
                row["ratio"],
                row["ip"],
                row["isol"],
                row["d_xp_axis"],
                row["d_xp_grid"],
                row["kappa"],
                row["peaking"],
                row["cold"],
            )
            for row in rows
        ],
        dtype=float,
    )
    time_axis, ratio_axis = table[:, 0], table[:, 1]
    candidates = {
        "ip": table[:, 2],
        "isol": table[:, 3],
        "d_xp_axis": table[:, 4],
        "d_xp_grid": table[:, 5],
        "kappa": table[:, 6],
        "peaking": table[:, 7],
    }
    cold_axis = table[:, 8] > 0.5

    results: dict[str, Any] = {}
    results["n_diverted_rows"] = n_all
    results["n_finite_rows"] = n_finite
    results["grid_uniformity"] = grid_report
    results["primary"] = _regress_population(
        "primary", time_axis, ratio_axis, candidates
    )
    results["warm"] = _regress_population(
        "warm",
        time_axis[~cold_axis],
        ratio_axis[~cold_axis],
        {key: values[~cold_axis] for key, values in candidates.items()},
    )
    results["banded"] = _band_table(time_axis, ratio_axis, candidates, cold_axis)
    results["earliest_band"] = _earliest_band(time_axis, ratio_axis, cold_axis)
    results["candidate_statistics"] = _candidate_statistics(candidates)

    arguments.output.mkdir(parents=True, exist_ok=True)
    receipt_path = arguments.output / "diverted-drift-receipt.json"
    receipt_path.write_text(json.dumps(results, indent=1) + "\n")
    _draw_figures(
        arguments.output, results, time_axis, ratio_axis, candidates, cold_axis
    )

    for population in ("primary", "warm"):
        summary = results[population]["summary"]
        print(
            f"{population}: n={results[population]['n']} "
            f"any_beats_time={summary['any_beats_time']} best={summary['best_vs_time']}"
        )
    print(f"wrote {receipt_path}")


def _regress_population(
    name: str, t: np.ndarray, r: np.ndarray, candidates: dict[str, np.ndarray]
) -> dict[str, object]:
    """Return the candidate fits, the time baseline, and the beyond-time test."""
    time_fit = ols_fit(t, r)
    time_spearman = spearman(t, r)
    candidate_fits = {}
    for key, label in CANDIDATES:
        values = candidates[key]
        fit = ols_fit(values, r)
        rho, rho_p = spearman(values, r)
        combined = ols_fit_multi(np.column_stack((t, values)), r)
        candidate_fits[key] = {
            "label": label,
            "fit": fit,
            "spearman": {"rho": rho, "p_value": rho_p},
            "after_time": {
                "slope": combined["slope"],
                "p_value": combined["p_value"],
                "r_squared": combined["r_squared"],
                "aic": combined["aic"],
            },
        }
    best = min(
        candidate_fits,
        key=lambda key: abs(
            abs(candidate_fits[key]["spearman"]["rho"]) - abs(time_spearman[0])
        ),
    )
    return {
        "name": name,
        "n": int(t.size),
        "time_baseline": {
            "fit": time_fit,
            "spearman": {"rho": time_spearman[0], "p_value": time_spearman[1]},
        },
        "candidates": candidate_fits,
        "summary": {
            "best_vs_time": best,
            "any_beats_time": any(
                abs(candidate_fits[key]["spearman"]["rho"]) > abs(time_spearman[0])
                and candidate_fits[key]["fit"]["r_squared"] > time_fit["r_squared"]
                for key, _ in CANDIDATES
            ),
        },
    }


def _band_table(
    t: np.ndarray, r: np.ndarray, candidates: dict[str, np.ndarray], cold: np.ndarray
) -> list[dict[str, object]]:
    bands = []
    for low, high, label in BANDS:
        mask = (t >= low) & (t < high)
        band_rows = [
            {
                "candidate": key,
                "median": float(np.nanmedian(values[mask])),
                "p10": float(np.nanquantile(values[mask], 0.1)),
                "p90": float(np.nanquantile(values[mask], 0.9)),
            }
            for key, _ in CANDIDATES
            for values in (candidates[key],)
        ]
        bands.append(
            {
                "band": label,
                "n": int(mask.sum()),
                "ratio_median": float(np.median(r[mask])),
                "ratio_p10": float(np.quantile(r[mask], 0.1)),
                "ratio_p90": float(np.quantile(r[mask], 0.9)),
                "cold_fraction": float(np.mean(cold[mask])),
                "candidates": band_rows,
            }
        )
    return bands


def _earliest_band(t: np.ndarray, r: np.ndarray, cold: np.ndarray) -> dict[str, object]:
    mask = (t >= 0.0) & (t < 0.05)
    return {
        "band": "0-50 ms",
        "n": int(mask.sum()),
        "n_warm": int((mask & ~cold).sum()),
        "cold_fraction": float(np.mean(cold[mask])),
        "ratio_median": float(np.median(r[mask])),
        "ratio_p10": float(np.quantile(r[mask], 0.1)),
        "ratio_p90": float(np.quantile(r[mask], 0.9)),
    }


def _candidate_statistics(candidates: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        key: {
            "median": float(np.nanmedian(values)),
            "p10": float(np.nanquantile(values, 0.1)),
            "p90": float(np.nanquantile(values, 0.9)),
            "finite_n": int(np.isfinite(values).sum()),
        }
        for key, values in candidates.items()
    }


def _draw_figures(
    output: Path,
    results: dict[str, object],
    t: np.ndarray,
    r: np.ndarray,
    candidates: dict[str, np.ndarray],
    cold: np.ndarray,
) -> None:
    """Draw the ratio-candidate scatter grid and the banded median curve."""
    warm = ~cold
    figure, axes = plt.subplots(2, 4, figsize=(16.0, 7.5))
    panels = (("time_s", "time [s]"),) + CANDIDATES
    for index, (key, label) in enumerate(panels):
        axis = axes.flat[index]
        values = t if key == "time_s" else candidates[key]
        axis.scatter(
            values[warm], r[warm], s=4, alpha=0.22, color="#3b6ea5", linewidths=0
        )
        axis.scatter(
            values[cold], r[cold], s=4, alpha=0.45, color="#a53b3b", linewidths=0
        )
        fit = ols_fit(values, r)
        order_axis = np.argsort(values)
        axis.plot(
            values[order_axis],
            fit["intercept"] + fit["slope"] * values[order_axis],
            "-",
            color="black",
            lw=1.0,
        )
        rho, _ = spearman(values, r)
        axis.set_title(
            f"{label}\nrho={rho:+.3f}  R2={fit['r_squared']:.3f}", fontsize=9
        )
        axis.set_xlabel(label, fontsize=8)
        axis.set_ylabel("span ratio nova/EFIT", fontsize=8)
        axis.tick_params(labelsize=7)
    axes.flat[7].axis("off")
    figure.suptitle(
        "Diverted span ratio vs per-slice quantities (red = cold start)", fontsize=11
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output / "diverted-drift-scatters.png", dpi=130)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.0, 4.6))
    banded = results["banded"]
    indices = np.arange(len(banded))
    axis.plot(
        indices, [b["ratio_median"] for b in banded], "-o", color="#3b6ea5", lw=1.6
    )
    axis.axhline(1.0, color="grey", lw=0.8, ls="--")
    axis.set_xticks(indices)
    axis.set_xticklabels([b["band"] for b in banded], fontsize=8)
    axis.set_ylabel("median diverted span ratio nova/EFIT", fontsize=9)
    axis.set_xlabel("time band", fontsize=9)
    for band_index, band in enumerate(banded):
        axis.annotate(
            f"n={band['n']}",
            (band_index, band["ratio_median"]),
            textcoords="offset points",
            xytext=(0, 6),
            fontsize=7,
            ha="center",
        )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "diverted-drift-band-medians.png", dpi=130)
    plt.close(figure)


if __name__ == "__main__":
    main()
