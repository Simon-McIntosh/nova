"""Attribute the diverted span-ratio drift against EFIT to the solenoid.

The diverted-drift regression found the solenoid current the sole per-slice
quantity that beats time in explaining the diverted span ratio nova/EFIT
(R2 0.497, Spearman -0.83, slope -0.01383 per kA over 43,011 finite rows).
This benchmark tests, in order, the three mechanisms by which the solenoid's
flux could enter the recorded boundary span:

(a) the direct external-field contribution of the solenoid at the X-point and
    the axis -- the solenoid's vacuum flux image per ampere, projected onto
    the span change and compared with the fitted slope;
(b) the saddle position relative to the solve grid as the solenoid ramps
    (does the X-point walk across a resolution boundary, driving a
    resolution-dependent read of the boundary flux);
(c) the p_prime and ff_prime interpolation from the reference 65-point
    psi_norm grid onto nova's fixed face grid, as the profile shape evolves
    with the ramp.

For each candidate the benchmark reports the fraction of the fitted slope it
accounts for on the same 43,011 census rows, and names the carrier or states
plainly that none of the three carries it.  Diagnosis only: no repair.

The only new solve-like computation is the solenoid flux image, which is the
vacuum response: a direct Biot-Savart loop-flux evaluation of the solenoid
winding per ampere, no plasma, no Newton.  Everything else reads recorded
census and per-slice quantities.

Usage (debug partition; per-job TMPDIR):
    UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync \\
        python benchmarks/solenoid_span_attribution.py \\
        --rows <cache.npz> --output <output_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import zarr

from benchmarks.diverted_drift_regression import _session_arrays, ols_fit
from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.imas.mast_vacuum_response import coil_sections, loop_response_matrix

CENSUS = Path("docs/figures/limited-boundary-census/limited-class-census.npz")
SESSION_ROOT = Path("/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29")
STORE_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")

#: The stored efm flux grid is a uniform 65 by 65; the forward solve lattice
#: strides it by this factor exactly as the parity benchmark and labeller do.
GRID_STRIDE = 2

ROW_KEYS = (
    "frame",
    "slice_index",
    "time_s",
    "ratio",
    "isol_kA",
    "ref_span",
    "nova_span",
    "axis_r",
    "axis_z",
    "xp_r",
    "xp_z",
    "nova_xp_r",
    "nova_xp_z",
    "d_xp_axis",
    "d_xp_grid",
    "peaking",
    "trunc_p",
    "trunc_ff",
    "shot",
)


def _ols(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least-squares slope, its standard error and R2 of y on x."""
    fit = ols_fit(x, y)
    return {
        "slope": float(fit["slope"]),
        "slope_se": float(fit["slope_se"]),
        "r_squared": float(fit["r_squared"]),
    }


def _outcome_path_slope(x, y, isol) -> float:
    """Return the x coefficient in y ~ b0 + b1 x + b2 isol (the path arm).

    The mediation of isol's effect on y through x splits into the marginal
    regression of x on isol (the exposure arm) and this outcome arm: y on x
    with isol held fixed.  Controlling the exposure arm would be wrong -- the
    outcome is not a confounder of the mediator.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    isol = np.asarray(isol, dtype=float)
    design = np.column_stack((np.ones_like(x), x, isol))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[1])


def _nearest_xpoint(radius, height, axis_r, axis_z) -> tuple[float, float]:
    """Return the X-point nearer the axis, mirroring the drift regression."""
    candidates = [
        (r, z) for r, z in zip(radius, height) if np.isfinite(r) and np.isfinite(z)
    ]
    if not candidates:
        return float("nan"), float("nan")
    return min(candidates, key=lambda p: float(np.hypot(p[0] - axis_r, p[1] - axis_z)))


def _grid_geometry(group: zarr.Group) -> dict[str, Any]:
    """State the solve grid's uniformity and resolution from the efm axes."""
    stored_r = np.asarray(group["gridr"], dtype=float)
    stored_z = np.asarray(group["gridz"], dtype=float)
    radius = stored_r[0::GRID_STRIDE]
    height = stored_z[0::GRID_STRIDE]
    radial_step = float(np.median(np.diff(radius)))
    vertical_step = float(np.median(np.diff(height)))
    return {
        "stored_axis_points": int(stored_r.size),
        "mode": "stored_axis_stride_intervention",
        "stored_axis_stride": GRID_STRIDE,
        "solve_node_count": int(radius.size * height.size),
        "solve_axis_counts": [int(radius.size), int(height.size)],
        "radial_step_m": radial_step,
        "vertical_step_m": vertical_step,
        "uniform": bool(
            np.allclose(np.diff(radius), np.diff(radius)[0])
            and np.allclose(np.diff(height), np.diff(height)[0])
        ),
    }


def _profile_truncation(group: zarr.Group, row: int) -> dict[str, float]:
    """Estimate the reference-grid linear-interpolation truncation error.

    The solved profile is the piecewise-linear interpolation of the stored
    65-point samples on the reference psi_norm grid, so its truncation
    relative to the underlying smooth profile is bounded by the second
    difference of those samples (h^2 / 8 times the local curvature).  The
    returned numbers are the bound relative to the profile amplitude.
    """
    entries: dict[str, float] = {}
    for name, values, sign in (
        ("p_prime", np.asarray(group["pprime"][row], dtype=float), -1.0),
        ("ff_prime", np.asarray(group["ffprime"][row], dtype=float), -1.0),
    ):
        scaled = sign * values / TOTAL_FLUX_FACTOR
        amplitude = float(np.max(np.abs(scaled)))
        if amplitude <= 0.0 or scaled.size < 4:
            entries[name] = float("nan")
            continue
        second = scaled[2:] - 2.0 * scaled[1:-1] + scaled[:-2]
        error_si = float(np.max(np.abs(second)) / 8.0)
        entries[name] = error_si / amplitude
    return entries


def _frame_rows(
    shot: int, session, group: zarr.Group, time_s, ratio, select
) -> list[dict]:
    """Return one row per diverted finite frame of one shot."""
    efm_time = np.asarray(group["time"], dtype=float)
    session_time = session["time"]
    axis_r = np.asarray(group["magnetic_axis_r"], dtype=float)
    axis_z = np.asarray(group["magnetic_axis_z"], dtype=float)
    x1r = np.asarray(group["xpoint1_rc"], dtype=float)
    x1z = np.asarray(group["xpoint1_zc"], dtype=float)
    x2r = np.asarray(group["xpoint2_rc"], dtype=float)
    x2z = np.asarray(group["xpoint2_zc"], dtype=float)
    isol = np.asarray(group["fcoil_c"], dtype=float)[:, 0]
    psi_axis = np.asarray(group["psi_axis"], dtype=float)
    psi_boundary = np.asarray(group["psi_boundary"], dtype=float)
    pprime = np.asarray(group["pprime"], dtype=float)
    grid_r = np.asarray(group["gridr"], dtype=float)[::GRID_STRIDE]
    grid_z = np.asarray(group["gridz"], dtype=float)[::GRID_STRIDE]

    rows: list[dict] = []
    for frame in range(time_s.size):
        if not select[frame]:
            continue
        slice_index = int(np.argmin(np.abs(efm_time - time_s[frame])))
        axis = (float(axis_r[slice_index]), float(axis_z[slice_index]))
        if not np.isfinite(axis[0]) or not np.isfinite(axis[1]):
            continue
        xp = _nearest_xpoint(
            (x1r[slice_index], x2r[slice_index]),
            (x1z[slice_index], x2z[slice_index]),
            axis[0],
            axis[1],
        )
        if not np.isfinite(xp[0]) or not np.isfinite(xp[1]):
            continue
        grid_distances = np.hypot(grid_r - xp[0], grid_z - xp[1])
        d_xp_grid = float(np.min(grid_distances))
        d_xp_axis = float(np.hypot(xp[0] - axis[0], xp[1] - axis[1]))
        pressure = pprime[slice_index]
        finite = pressure[np.isfinite(pressure)]
        peaking = (
            float(abs(finite[0] / np.mean(finite)))
            if (finite.size > 5 and np.mean(finite) > 0.0)
            else float("nan")
        )
        ref_span = abs(
            TOTAL_FLUX_FACTOR * (psi_axis[slice_index] - psi_boundary[slice_index])
        )
        if not np.isfinite(ref_span) or ref_span <= 0.0:
            continue
        trunc = _profile_truncation(group, slice_index)

        nova_frame = int(np.argmin(np.abs(session_time - time_s[frame])))
        nova_axis_r = float(session["magnetic_axis_r"][nova_frame])
        nova_axis_z = float(session["magnetic_axis_z"][nova_frame])
        nova_xr = np.asarray(session["x_point_r"][:, nova_frame], dtype=float)
        nova_xz = np.asarray(session["x_point_z"][:, nova_frame], dtype=float)
        nova_dists = np.hypot(nova_xr - nova_axis_r, nova_xz - nova_axis_z)
        if np.isfinite(nova_axis_r) and np.isfinite(nova_dists).any():
            index = int(np.nanargmin(nova_dists))
            nova_xp = (float(nova_xr[index]), float(nova_xz[index]))
        else:
            nova_xp = (float("nan"), float("nan"))

        rows.append(
            {
                "frame": frame,
                "slice_index": slice_index,
                "time_s": float(time_s[frame]),
                "ratio": float(ratio[frame]),
                "isol_kA": float(isol[slice_index]) / 1.0e3,
                "ref_span": ref_span,
                "nova_span": float(ratio[frame]) * ref_span,
                "axis_r": axis[0],
                "axis_z": axis[1],
                "xp_r": xp[0],
                "xp_z": xp[1],
                "nova_xp_r": nova_xp[0],
                "nova_xp_z": nova_xp[1],
                "d_xp_axis": d_xp_axis,
                "d_xp_grid": d_xp_grid,
                "peaking": peaking,
                "trunc_p": trunc["p_prime"],
                "trunc_ff": trunc["ff_prime"],
                "shot": int(shot),
            }
        )
    return rows


def _load_rows(arguments: argparse.Namespace) -> list[dict]:
    """Build or load the census row table."""
    cache = Path(arguments.rows) if arguments.rows else None
    if cache is not None and cache.exists():
        stored = np.load(cache, allow_pickle=True)
        records = np.asarray(stored["rows"])
        integral = ("frame", "slice_index", "shot")
        rows = [
            {
                key: (int(value) if key in integral else float(value))
                for key, value in zip(ROW_KEYS, record)
            }
            for record in records
        ]
        return rows

    census = np.load(CENSUS, allow_pickle=True)
    shot_all = census["shot"]
    time_s_all = census["time_s"]
    diverted = census["diverted"]
    ratio_all = census["span_ratio_nova_over_efit"]
    select_all = diverted & np.isfinite(time_s_all) & np.isfinite(ratio_all)

    rows: list[dict] = []
    for member in np.unique(shot_all[select_all]):
        group = zarr.open_group(str(STORE_ROOT / f"{member}.zarr"), mode="r")["efm"]
        session = _session_arrays(int(member))
        member_select = (shot_all == member) & select_all
        time_s = time_s_all[member_select]
        ratio = np.asarray(ratio_all[member_select], dtype=float)
        rows.extend(
            _frame_rows(
                int(member),
                session,
                group,
                time_s,
                ratio,
                np.ones(time_s.size, dtype=bool),
            )
        )
    rows = [
        row
        for row in rows
        if np.isfinite(row["peaking"]) and np.isfinite(row["d_xp_axis"])
    ]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        table = np.asarray(
            [[row[key] for key in ROW_KEYS] for row in rows], dtype=float
        )
        np.savez(cache, rows=table, keys=ROW_KEYS)
    return rows


def _solenoid_response(
    rows: list[dict], registry: MachineGeometryRegistry
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-row solenoid flux beta at axis, efm X-point and nova X-point.

    Units are Wb per raw ampere of the solenoid circuit (the loop flux per
    ampere-turn times the winding turn-multiplier sum, 328 turns).
    """
    shots = np.asarray([int(row["shot"]) for row in rows])
    shot_to_digest: dict[int, str] = {}
    for shot in sorted(set(int(s) for s in shots)):
        shot_to_digest[shot] = registry.select(shot).configuration.physical_digest
    digests = sorted(set(shot_to_digest.values()))
    geometries: dict[str, tuple[dict[str, Any], float, zarr.Group]] = {}
    for digest in digests:
        shot = next(s for s, d in shot_to_digest.items() if d == digest)
        selection = registry.select(shot)
        geometry = selection.configuration.geometry
        order = tuple(sorted(coil_sections(geometry)))
        if "sol" not in order:
            raise RuntimeError(f"geometry {digest} carries no solenoid family")
        group = zarr.open_group(str(STORE_ROOT / f"{shot}.zarr"), mode="r")["efm"]
        turns = np.asarray(group["fcoil_turns"], dtype=float)
        multiplier = np.asarray(group["fcoil_xmult"], dtype=float)
        circuit = np.asarray(group["fcoil_circ"], dtype=int)
        scale = float(np.sum(turns[circuit == 1] * multiplier[circuit == 1]))
        geometries[digest] = (geometry, scale, group)

    points_all = np.concatenate(
        [
            np.asarray(
                [
                    [row["axis_r"], row["axis_z"]],
                    [row["xp_r"], row["xp_z"]],
                    [row["nova_xp_r"], row["nova_xp_z"]],
                ],
                dtype=float,
            )
            for row in rows
        ]
    )
    beta_axis = np.full(len(rows), np.nan)
    beta_xp = np.full(len(rows), np.nan)
    beta_xp_nova = np.full(len(rows), np.nan)
    for digest in digests:
        member = np.array([shot_to_digest[s] == digest for s in shots])
        geometry, scale, _ = geometries[digest]
        points = points_all[np.repeat(member, 3)]
        # the solenoid family alone: the other active components contribute
        # nothing to this attribution and cost an order of magnitude more
        matrix = loop_response_matrix(geometry, points, families=("sol",))
        column = matrix[:, 0] * scale
        beta_axis[member] = column[0::3]
        beta_xp[member] = column[1::3]
        beta_xp_nova[member] = column[2::3]
    return beta_axis, beta_xp, beta_xp_nova


def _attribution(
    rows: list[dict],
    beta_axis: np.ndarray,
    beta_xp: np.ndarray,
    beta_xp_nova: np.ndarray,
) -> dict[str, Any]:
    """Return the per-candidate accounted fractions on the census rows."""
    table = np.asarray(
        [
            (
                row["ratio"],
                row["isol_kA"],
                row["ref_span"],
                row["d_xp_grid"],
                row["trunc_p"],
                row["trunc_ff"],
            )
            for row in rows
        ],
        dtype=float,
    )
    ratio, isol = table[:, 0], table[:, 1]
    ref_span, d_xp_grid = table[:, 2], table[:, 3]
    trunc_p, trunc_ff = table[:, 4], table[:, 5]
    composite = np.sqrt(np.nanmean(np.stack((trunc_p, trunc_ff)) ** 2, axis=0))

    delta_beta_nova = beta_xp_nova - beta_axis  # nova's boundary read point
    delta_beta_efit = beta_xp - beta_axis  # efm's stored X-point

    fit = ols_fit(isol, ratio)
    fitted_slope_kA = float(fit["slope"])

    # candidate (a): the direct vacuum image of the solenoid
    per_row_full = delta_beta_nova / ref_span - ratio * delta_beta_efit / ref_span
    per_row_net = delta_beta_nova * (1.0 - ratio) / ref_span
    per_row_naive = delta_beta_nova / ref_span
    m_a = float(np.nanmean(per_row_full)) * 1.0e3
    m_a_net = float(np.nanmean(per_row_net)) * 1.0e3
    m_a_naive = float(np.nanmean(per_row_naive)) * 1.0e3

    # candidate (b): the saddle relative to the uniform solve grid
    gamma_b = _ols(isol, d_xp_grid)["slope"]  # m per kA
    beta_b = _outcome_path_slope(d_xp_grid, ratio, isol)  # ratio per m
    m_b = gamma_b * beta_b

    gamma_c = _ols(isol, composite)["slope"]  # per kA
    beta_c = _outcome_path_slope(composite, ratio, isol)  # ratio per unit
    m_c = gamma_c * beta_c

    fractions = {
        "a_direct_field_full": m_a / fitted_slope_kA,
        "a_direct_field_common_mode_net": m_a_net / fitted_slope_kA,
        "a_direct_field_nova_only": m_a_naive / fitted_slope_kA,
        "b_saddle_vs_grid": m_b / fitted_slope_kA,
        "c_profile_interpolation": m_c / fitted_slope_kA,
    }
    carrier = max(fractions, key=lambda key: abs(fractions[key]))
    return {
        "n_rows": int(len(rows)),
        "fitted_ratio_isol_per_kA": {
            "slope": fitted_slope_kA,
            "slope_se": float(fit["slope_se"]),
            "r_squared": float(fit["r_squared"]),
            "t_statistic": float(fit["t_statistic"]),
        },
        "solenoid_response": {
            "mean_beta_axis_wb_per_A": float(np.nanmean(beta_axis)),
            "mean_beta_xp_efit_wb_per_A": float(np.nanmean(beta_xp)),
            "mean_beta_xp_nova_wb_per_A": float(np.nanmean(beta_xp_nova)),
            "mean_delta_beta_nova_wb_per_A": float(np.nanmean(delta_beta_nova)),
            "mean_delta_beta_efit_wb_per_A": float(np.nanmean(delta_beta_efit)),
        },
        "candidate_a_direct_field": {
            "method": (
                "solenoid loop-flux per raw ampere at the boundary read point "
                "minus the axis, divided by the efm reference span; the full "
                "prediction removes the denominator's own walk with the same "
                "image at the efm X-point (common-mode cancellation)"
            ),
            "predicted_slope_per_kA_naive_nova_only": m_a_naive,
            "predicted_slope_per_kA_common_mode_net": m_a_net,
            "predicted_slope_per_kA_full": m_a,
        },
        "candidate_b_saddle_vs_grid": {
            "method": (
                "path mediation: gamma_b = d(d_xp_grid)/d I_sol (marginal) "
                "times beta_b = d(ratio)/d d_xp_grid with I_sol held fixed"
            ),
            "d_xp_grid_vs_isol_m_per_kA": gamma_b,
            "ratio_vs_d_xp_grid_per_m": beta_b,
            "predicted_slope_per_kA": m_b,
        },
        "candidate_c_profile_interpolation": {
            "method": (
                "path mediation of the composite relative truncation error "
                "of the piecewise-linear reference-grid interpolation: "
                "gamma_c = d(error)/d I_sol times beta_c = d(ratio)/d error "
                "with I_sol held fixed"
            ),
            "trunc_error_vs_isol_per_kA": gamma_c,
            "ratio_vs_trunc_per_unit": beta_c,
            "predicted_slope_per_kA": m_c,
        },
        "fraction_of_fitted_slope": {k: float(v) for k, v in fractions.items()},
        "carrier": carrier,
    }


def _draw_figures(
    output: Path,
    rows: list[dict],
    attribution: dict[str, Any],
    beta_axis: np.ndarray,
    beta_xp: np.ndarray,
) -> None:
    """Draw the per-candidate attribution panels."""
    table = np.asarray(
        [
            (
                row["ratio"],
                row["isol_kA"],
                row["ref_span"],
                row["d_xp_grid"],
                row["trunc_p"],
                row["trunc_ff"],
            )
            for row in rows
        ],
        dtype=float,
    )
    isol, ref_span = table[:, 1], table[:, 2]
    d_xp_grid, trunc_p, trunc_ff = table[:, 3], table[:, 4], table[:, 5]
    delta_beta = beta_xp - beta_axis
    predicted_a = delta_beta / ref_span * 1.0e3
    composite = np.sqrt(trunc_p**2 + trunc_ff**2)

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    panels = [
        (
            "candidate (a): direct solenoid vacuum image",
            isol,
            predicted_a,
            "predicted d(ratio)/dI_sol [per kA]",
        ),
        (
            "candidate (b): X-point to nearest grid node",
            isol,
            d_xp_grid,
            "d_xp_grid [m]",
        ),
        (
            "candidate (c): profile interpolation truncation",
            isol,
            composite,
            "composite relative truncation",
        ),
    ]
    for index, (title, x, y, ylabel) in enumerate(panels):
        axis = axes.flat[index]
        axis.scatter(x, y, s=3, alpha=0.12, color="#3b6ea5", linewidths=0)
        fit_xy = ols_fit(x, y)
        order = np.argsort(x)
        axis.plot(
            x[order],
            fit_xy["intercept"] + fit_xy["slope"] * x[order],
            "-",
            color="black",
            lw=1.0,
        )
        rho, _ = scipy.stats.spearmanr(x, y)
        axis.set_title(f"{title}\nrho={rho:+.3f}", fontsize=9)
        axis.set_xlabel("solenoid current [kA]", fontsize=8)
        axis.set_ylabel(ylabel, fontsize=8)
        axis.tick_params(labelsize=7)
        if index == 0:
            slope = attribution["fitted_ratio_isol_per_kA"]["slope"]
            axis.axhline(slope, color="red", lw=1.2, ls="--")
    figure.suptitle(
        "Solenoid span-ratio attribution -- three candidates against the fitted "
        f"slope ({attribution['fitted_ratio_isol_per_kA']['slope']:.4f}/kA, red)",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output / "solenoid-span-attribution.png", dpi=130)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    """Run the attribution and write the receipt, figure and summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/playable-forward-solve/solenoid-span"),
    )
    arguments = parser.parse_args(argv)

    rows = _load_rows(arguments)
    if not rows:
        raise RuntimeError("no census rows built")
    registry = MachineGeometryRegistry.default()

    group = zarr.open_group(str(STORE_ROOT / "15122.zarr"), mode="r")["efm"]
    grid = _grid_geometry(group)

    beta_axis, beta_xp, beta_xp_nova = _solenoid_response(rows, registry)
    isol = np.asarray([row["isol_kA"] for row in rows])
    ref_span = np.asarray([row["ref_span"] for row in rows])
    efit_slope_kA = _ols(isol, ref_span)["slope"]

    attribution = _attribution(rows, beta_axis, beta_xp, beta_xp_nova)
    attribution["grid"] = grid
    attribution["efit_span_isol_slope_wb_per_kA"] = efit_slope_kA

    arguments.output.mkdir(parents=True, exist_ok=True)
    receipt_path = arguments.output / "solenoid-span-receipt.json"
    receipt_path.write_text(json.dumps(attribution, indent=1) + "\n")
    _draw_figures(arguments.output, rows, attribution, beta_axis, beta_xp)

    fractions = attribution["fraction_of_fitted_slope"]
    candidate_a = attribution["candidate_a_direct_field"]
    candidate_b = attribution["candidate_b_saddle_vs_grid"]
    candidate_c = attribution["candidate_c_profile_interpolation"]
    print(
        f"rows={attribution['n_rows']} fitted_slope="
        f"{attribution['fitted_ratio_isol_per_kA']['slope']:.6f}/kA"
    )
    print("  efm span vs isol [Wb/kA]:", f"{efit_slope_kA:.6f}")
    print(
        "candidate a naive/net/full [per kA]:",
        f"{candidate_a['predicted_slope_per_kA_naive_nova_only']:.6f} / "
        f"{candidate_a['predicted_slope_per_kA_common_mode_net']:.6f} / "
        f"{candidate_a['predicted_slope_per_kA_full']:.6f}",
    )
    print("candidate b [per kA]:", f"{candidate_b['predicted_slope_per_kA']:.6f}")
    print("candidate c [per kA]:", f"{candidate_c['predicted_slope_per_kA']:.6f}")
    print("fractions_of_fitted_slope:")
    for key, value in fractions.items():
        print(f"  {key:42s} {value: .3f}")
    print("carrier:", attribution["carrier"])
    print("wrote", receipt_path)


if __name__ == "__main__":
    main()
