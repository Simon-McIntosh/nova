"""Settle the 4-6 cm axis disagreement with EFIT: definition or offset.

The consumer fidelity gate scores nova forward-labeller sessions against the
EFIT referee on 51 free converged flat-top slices and the axis misses a 2 cm
bound on 48 of them (median 4.56 cm, p90 5.75, max 6.24) while the boundary
radius RMS is inside the bound on 24.  This node settles which of three
outcomes that is:

* a definition difference — the referee publishes a different quantity than
  nova returns, so the gate compares the wrong pair;
* a real offset — nova's solved equilibrium genuinely places its magnetic axis
  away from EFIT's, and the solver owns it;
* a mixture — the split between the two, quantified.

The measurements, over exactly the same slices:

1. Signed R and Z components of nova_axis - EFIT_axis per slice and per shot.
   A definition difference shows as a consistent offset in one component
   across shots; a fidelity gap does not.
2. The same slices scored against the alternative referents the session
   carries — current_centroid_r/z and reference_centroid_z — and against the
   EFM store's own geometric axis and current centroid.  If the referee's
   ``magnetic_axis`` is secretly a centre-of-flux-surfaces or a current
   centroid, the nova quantity that tracks it is exposed.
3. The referee's own quantity, named from the store and the data-model
   documentation rather than inferred: the L2 equilibrium group reads
   ``EFM_MAGNETIC_AXIS_R/Z`` mapped to IMAS
   ``equilibrium.global_quantities.magnetic_axis``, documented as the
   "geometrical position of the magnetic axis (R, Z) ... the coordinate
   origin" of the flux surfaces.  The EFM store carries three distinct MAST
   definitions on one time base — magnetic_axis_r/z, geom_axis_rc/zc and
   current_centrd_r/z — so which one the referee serves is measured, not
   assumed.

The eligible slice set, the flat-top window and the referee read mirror the
landed consumer gate (``imas_ambix/worldmodel/physics_fidelity_gate.py``):
written rows whose free solve converged, not conditioned, inside the flat top
(interpolated |I_p_efm| >= 0.80 * peak) with a finite nova axis.  For this
corpus (the pre-correction six-carrier layout) no conditioned row carries a
zero-trip exception, so semantic convergence reduces to the row's own
``converged`` flag.  No solver module is edited; this is a read over sessions
and stores already written.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SESSION_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29"
)
DEFAULT_LEVEL1_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")
DEFAULT_LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")
FROZEN_CARRIER_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)
FLAT_TOP_CURRENT_FRACTION = 0.80


# ---------------------------------------------------------------------------
# Session / companion / manifest reads (mirrors the consumer gate)
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"{path} is not an atomically complete session")
    if not isinstance(manifest.get("slices"), list):
        raise ValueError(f"{path} has no slice-row list")
    return manifest


def _load_companion(
    path: Path, slices: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    written = [row for row in slices if bool(row.get("written", False))]
    with np.load(path, allow_pickle=False) as companion:
        required = {"row", "time", "conditioned"}
        missing = required - set(companion.files)
        if missing:
            raise ValueError(f"{path} companion lacks {sorted(missing)}")
        order = np.argsort(companion["row"], kind="stable")
        rows = np.asarray(companion["row"])[order]
        times = np.asarray(companion["time"], dtype=np.float64)[order]
        conditioned = np.asarray(companion["conditioned"], dtype=bool)[order]
        if rows.size != len(written):
            raise ValueError(
                f"{path} companion row count {rows.size} != written {len(written)}"
            )
    return times, conditioned, rows


def _load_efit_current(
    shot_id: int, level1_root: Path
) -> tuple[np.ndarray, np.ndarray]:
    import zarr  # noqa: PLC0415

    store = zarr.open_group(str(level1_root / f"{shot_id}.zarr"), mode="r")
    if "efm" not in set(store.group_keys()):
        raise KeyError(f"shot {shot_id}: no efm group at {level1_root}")
    group = store["efm"]
    if not {"time", "plasma_current_c"}.issubset(set(group.array_keys())):
        raise KeyError(f"shot {shot_id}: efm lacks time or plasma_current_c")
    return (
        np.asarray(group["time"], dtype=np.float64),
        np.asarray(group["plasma_current_c"], dtype=np.float64),
    )


def flat_top_mask(
    current_times: np.ndarray,
    plasma_current_a: np.ndarray,
    slice_times: np.ndarray,
) -> np.ndarray:
    """Mask where interpolated |I_p,efm| is >= 80% of the shot peak (gate rule)."""
    times = np.asarray(current_times, dtype=np.float64).reshape(-1)
    current = np.abs(np.asarray(plasma_current_a, dtype=np.float64).reshape(-1))
    query = np.asarray(slice_times, dtype=np.float64).reshape(-1)
    finite = np.isfinite(times) & np.isfinite(current)
    times, current = times[finite], current[finite]
    order = np.argsort(times, kind="stable")
    times, current = times[order], current[order]
    unique = np.concatenate(([True], np.diff(times) > 0.0))
    times, current = times[unique], current[unique]
    peak = float(np.max(current)) if current.size else 0.0
    if peak <= 0.0:
        raise ValueError("EFIT plasma-current trace has no non-zero sample")
    interpolated = np.interp(query, times, current, left=np.nan, right=np.nan)
    return np.isfinite(interpolated) & (
        interpolated >= FLAT_TOP_CURRENT_FRACTION * peak
    )


def _interp_at(t_source: np.ndarray, values: np.ndarray, t_query: float) -> float:
    """Linear interpolation of one scalar at an absolute time; NaN when the
    value is unmasked at no bracketing sample (the referee's semantics)."""
    order = np.argsort(t_source, kind="stable")
    ts = t_source[order]
    vs = values[order]
    keep = np.concatenate(([True], np.diff(ts) > 0.0)) & np.isfinite(vs)
    ts, vs = ts[keep], vs[keep]
    if ts.size < 2:
        return float("nan")
    return float(np.interp(t_query, ts, vs, left=np.nan, right=np.nan))


# ---------------------------------------------------------------------------
# Data accessors
# ---------------------------------------------------------------------------


def _session_fields(
    shot_id: int, session_root: Path
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import xarray as xr  # noqa: PLC0415

    path = session_root / f"{shot_id}.nc"
    with xr.open_dataset(path, group="steering", engine="h5netcdf") as session:
        times = np.asarray(session["time"], dtype=np.float64).reshape(-1)
        arrays = {
            name: np.asarray(session[name], dtype=np.float64).reshape(-1)
            for name in (
                "magnetic_axis_r",
                "magnetic_axis_z",
                "current_centroid_r",
                "current_centroid_z",
                "reference_centroid_z",
            )
        }
    return times, arrays


def _efm_scalars(shot_id: int, level1_root: Path) -> dict[str, np.ndarray]:
    import zarr  # noqa: PLC0415

    store = zarr.open_group(str(level1_root / f"{shot_id}.zarr"), mode="r")["efm"]
    out: dict[str, np.ndarray] = {"time": np.asarray(store["time"], dtype=np.float64)}
    for name in (
        "magnetic_axis_r",
        "magnetic_axis_z",
        "geom_axis_rc",
        "geom_axis_zc",
        "current_centrd_r",
        "current_centrd_z",
    ):
        if name in set(store.array_keys()):
            out[name] = np.asarray(store[name], dtype=np.float64)
    return out


def _l2_axis(shot_id: int, level2_root: Path) -> tuple[np.ndarray, np.ndarray]:
    import zarr  # noqa: PLC0415

    store = zarr.open_group(str(level2_root / f"{shot_id}.zarr"), mode="r")[
        "equilibrium"
    ]
    return (
        np.asarray(store["time"], dtype=np.float64),
        np.asarray(store["magnetic_axis_r"], dtype=np.float64),
        np.asarray(store["magnetic_axis_z"], dtype=np.float64),
    )


def _l2_psi_extremum(
    shot_id: int, level2_root: Path, t_query: float, axis_r: float, axis_z: float
) -> tuple[float, float, str]:
    """Offset of EFIT's own psi-grid extremum from the stored magnetic axis.

    The L2 equilibrium group stores the EFIT psi field as ``psi[Z, R, t]``
    (first array axis is Z, second is R — established by matching the local
    extremum R to the stored axis to sub-mm).  A local parabolic fit over the
    3x3 grid-cell neighbourhood of the stored axis gives the extremum vertex;
    its distance from the stored ``magnetic_axis`` is the definitional check
    "does EFIT's magnetic axis sit at the extremum of (this representation
    of) its own psi?".  When the fit is degenerate (flat psi, or the vertex
    stepping outside the neighbourhood) the status records why.
    """
    import zarr  # noqa: PLC0415

    store = zarr.open_group(str(level2_root / f"{shot_id}.zarr"), mode="r")[
        "equilibrium"
    ]
    t = np.asarray(store["time"], dtype=np.float64)
    r = np.asarray(store["major_radius"], dtype=np.float64)
    z = np.asarray(store["z"], dtype=np.float64)
    if t.size == 0:
        return float("nan"), float("nan"), "no_time"
    ti = int(np.argmin(np.abs(t - t_query)))
    if not ((t[ti] - t_query) ** 2 < (t[1] - t[0]) ** 2 * 2.0):
        return float("nan"), float("nan"), "outside_time"
    psi = np.asarray(store["psi"][:, :, ti])
    ir = int(np.argmin(np.abs(r - axis_r)))
    iz = int(np.argmin(np.abs(z - axis_z)))
    if ir < 1 or ir > r.size - 2 or iz < 1 or iz > z.size - 2:
        return float("nan"), float("nan"), "axis_cell_on_edge"
    # psi[Z, R]; axes: row=Z (dr/z pitch), col=R (r pitch)
    c = psi[iz - 1 : iz + 2, ir - 1 : ir + 2]
    if not np.isfinite(c).all():
        return float("nan"), float("nan"), "neighbourhood_non_finite"
    grid_x = np.tile([-1.0, 0.0, 1.0], 3)  # column (R) offsets
    grid_y = np.repeat([-1.0, 0.0, 1.0], 3)  # row (Z) offsets
    a_mat = np.c_[
        np.ones(9), grid_x, grid_y, grid_x * grid_x, grid_x * grid_y, grid_y * grid_y
    ]
    coef, *_ = np.linalg.lstsq(a_mat, c.ravel(), rcond=None)
    det = 4.0 * coef[3] * coef[5] - coef[4] * coef[4]
    if abs(det) < 1.0e-12:
        return float("nan"), float("nan"), "degenerate_fit"
    dr = -(2.0 * coef[5] * coef[1] - coef[4] * coef[2]) / det
    dz = -(2.0 * coef[3] * coef[2] - coef[4] * coef[1]) / det
    if abs(dr) > 1.5 or abs(dz) > 1.5:
        return float("nan"), float("nan"), "vertex_outside"
    r_pitch = r[1] - r[0] if r.size > 1 else 0.0
    z_pitch = z[1] - z[0] if z.size > 1 else 0.0
    dr_cm = 100.0 * (dr * r_pitch)
    dz_cm = 100.0 * (dz * z_pitch)
    return float(dr_cm), float(dz_cm), "ok"


# ---------------------------------------------------------------------------
# Per-slice row
# ---------------------------------------------------------------------------


@dataclass
class SliceRow:
    shot: int
    session_index: int
    time_s: float
    nova_r: float
    nova_z: float
    nova_centroid_r: float
    nova_centroid_z: float
    reference_centroid_z: float
    efit_r: float
    efit_z: float
    efit_geom_r: float
    efit_geom_z: float
    efit_centroid_r: float
    efit_centroid_z: float
    l2_r: float
    l2_z: float
    flat_top: bool
    d_r_cm: float = field(default=float("nan"))
    d_z_cm: float = field(default=float("nan"))
    l2_minus_efit_cm: float = field(default=float("nan"))
    psi_extremum_dr_cm: float = field(default=float("nan"))
    psi_extremum_dz_cm: float = field(default=float("nan"))
    psi_extremum_status: str = field(default="absent")


def collect_rows(
    session_root: Path,
    level1_root: Path,
    level2_root: Path,
    shots: tuple[int, ...],
) -> tuple[list[SliceRow], dict[str, Any]]:
    rows: list[SliceRow] = []
    per_shot: dict[str, Any] = {}
    for shot in shots:
        manifest = _load_manifest(session_root / f"{shot}.manifest.json")
        srows = manifest["slices"]
        session_path = session_root / f"{shot}.nc"
        if not session_path.is_file():
            raise FileNotFoundError(session_path)
        companion_times, companion_cond, companion_rows = _load_companion(
            session_root / f"{shot}.npz", srows
        )
        session_times, session_arrays = _session_fields(shot, session_root)
        if session_times.shape != companion_times.shape or not np.allclose(
            session_times, companion_times, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(f"{session_path} times do not align with its companion")

        # free converged rows: written, converged, not conditioned
        written_rows = [r for r in srows if bool(r.get("written", False))]
        free_index = [
            i
            for i, r in enumerate(written_rows)
            if bool(r.get("converged", False)) and not bool(companion_cond[i])
        ]
        row_to_session = {int(r): i for i, r in enumerate(companion_rows)}
        session_indices = [
            int(row_to_session[int(written_rows[i]["row"])]) for i in free_index
        ]
        slice_times = companion_times[session_indices]
        flat = flat_top_mask(*_load_efit_current(shot, level1_root), slice_times)

        efm = _efm_scalars(shot, level1_root)
        l2_t, l2_r, l2_z = _l2_axis(shot, level2_root)

        eligible = 0
        for k, idx in enumerate(session_indices):
            if not bool(flat[k]):
                continue
            nova_r = float(session_arrays["magnetic_axis_r"][idx])
            nova_z = float(session_arrays["magnetic_axis_z"][idx])
            if not (math.isfinite(nova_r) and math.isfinite(nova_z)):
                continue
            eligible += 1
            tt = float(slice_times[k])
            efit_r = _interp_at(efm["time"], efm["magnetic_axis_r"], tt)
            efit_z = _interp_at(efm["time"], efm["magnetic_axis_z"], tt)
            efit_geom_r = (
                _interp_at(efm["time"], efm["geom_axis_rc"], tt)
                if "geom_axis_rc" in efm
                else float("nan")
            )
            efit_geom_z = (
                _interp_at(efm["time"], efm["geom_axis_zc"], tt)
                if "geom_axis_zc" in efm
                else float("nan")
            )
            efit_centroid_r = (
                _interp_at(efm["time"], efm["current_centrd_r"], tt)
                if "current_centrd_r" in efm
                else float("nan")
            )
            efit_centroid_z = (
                _interp_at(efm["time"], efm["current_centrd_z"], tt)
                if "current_centrd_z" in efm
                else float("nan")
            )
            l2_e_r = _interp_at(l2_t, l2_r, tt)
            l2_e_z = _interp_at(l2_t, l2_z, tt)
            row = SliceRow(
                shot=int(shot),
                session_index=int(idx),
                time_s=tt,
                nova_r=nova_r,
                nova_z=nova_z,
                nova_centroid_r=float(session_arrays["current_centroid_r"][idx]),
                nova_centroid_z=float(session_arrays["current_centroid_z"][idx]),
                reference_centroid_z=float(session_arrays["reference_centroid_z"][idx]),
                efit_r=efit_r,
                efit_z=efit_z,
                efit_geom_r=efit_geom_r,
                efit_geom_z=efit_geom_z,
                efit_centroid_r=efit_centroid_r,
                efit_centroid_z=efit_centroid_z,
                l2_r=l2_e_r,
                l2_z=l2_e_z,
                flat_top=True,
            )
            row.d_r_cm = (
                100.0 * (nova_r - efit_r)
                if math.isfinite(nova_r) and math.isfinite(efit_r)
                else float("nan")
            )
            row.d_z_cm = (
                100.0 * (nova_z - efit_z)
                if math.isfinite(nova_z) and math.isfinite(efit_z)
                else float("nan")
            )
            row.l2_minus_efit_cm = (
                math.hypot(100.0 * (l2_e_r - efit_r), 100.0 * (l2_e_z - efit_z))
                if math.isfinite(l2_e_r) and math.isfinite(l2_e_z)
                else float("nan")
            )
            dr_fit, dz_fit, fit_status = _l2_psi_extremum(
                shot, level2_root, tt, l2_e_r, l2_e_z
            )
            row.psi_extremum_dr_cm = dr_fit
            row.psi_extremum_dz_cm = dz_fit
            row.psi_extremum_status = fit_status
            rows.append(row)
        per_shot[str(shot)] = {
            "session_path": str(session_path),
            "slices_total": len(srows),
            "written": len(written_rows),
            "free_converged": len(free_index),
            "free_converged_flat_top": eligible,
        }
    return rows, per_shot


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "max_abs": float(np.max(np.abs(arr))),
        "all_positive": int(np.all(arr > 0)),
        "all_negative": int(np.all(arr < 0)),
        "within_2cm": int(np.all(np.abs(arr) <= 2.0)),
    }


def _offset_stats(rows: list[SliceRow]) -> dict[str, float | int | None]:
    r = np.asarray([math.hypot(x.d_r_cm, x.d_z_cm) for x in rows], dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"n": 0}
    return {
        "n": int(r.size),
        "median": float(np.median(r)),
        "mean": float(np.mean(r)),
        "p90": float(np.quantile(r, 0.90)),
        "max": float(np.max(r)),
        "within_2cm": int(np.sum(r <= 2.0)),
    }


def _signed_space_stats(
    d_r: list[float],
) -> dict[str, Any]:
    """Per-component sign consistency, the definition-vs-fidelity tell."""
    arr = np.asarray([v for v in d_r if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "all_positive": int(np.all(arr > 0)),
        "all_negative": int(np.all(arr < 0)),
        "mixed": int(np.any(arr > 0) and np.any(arr < 0)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--level1-root", type=Path, default=DEFAULT_LEVEL1_ROOT)
    parser.add_argument("--level2-root", type=Path, default=DEFAULT_LEVEL2_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/playable-forward-solve/axis-definition"),
    )
    parser.add_argument(
        "--shots", type=int, nargs="+", default=list(FROZEN_CARRIER_SHOTS)
    )
    args = parser.parse_args(argv)

    rows, per_shot = collect_rows(
        args.session_root, args.level1_root, args.level2_root, tuple(args.shots)
    )

    eligible = [r for r in rows if r.flat_top]
    total = len(eligible)
    # L2/EFM referee consistency: the referee's own L2 store tracks its EFM
    # source (both are the IMAS magnetic axis; if they diverged the referee
    # read would be the wrong thing even internally).
    l2_consistency = _stats([r.l2_minus_efit_cm for r in eligible])

    per_shot_stats: dict[str, Any] = {}
    for shot in sorted({r.shot for r in eligible}):
        srows = [r for r in eligible if r.shot == shot]
        per_shot_stats[str(shot)] = {
            "n": len(srows),
            "d_r_cm": _stats([r.d_r_cm for r in srows]),
            "d_z_cm": _stats([r.d_z_cm for r in srows]),
        }

    efit_internal = {
        "efit_geom_axis_r - efit_magnetic_axis_r (cm)": _stats(
            [100.0 * (r.efit_geom_r - r.efit_r) for r in eligible]
        ),
        "efit_geom_axis_z - efit_magnetic_axis_z (cm)": _stats(
            [100.0 * (r.efit_geom_z - r.efit_z) for r in eligible]
        ),
        "efit_current_centroid_r - efit_magnetic_axis_r (cm)": _stats(
            [100.0 * (r.efit_centroid_r - r.efit_r) for r in eligible]
        ),
        "efit_current_centroid_z - efit_magnetic_axis_z (cm)": _stats(
            [100.0 * (r.efit_centroid_z - r.efit_z) for r in eligible]
        ),
    }

    referents = {
        "nova_magnetic_axis_vs_efit_axis": _offset_stats(eligible),
        "nova_current_centroid_vs_efit_axis": _offset_stats(
            [
                SliceRow(
                    shot=r.shot,
                    session_index=r.session_index,
                    time_s=r.time_s,
                    nova_r=r.nova_centroid_r,
                    nova_z=r.nova_centroid_z,
                    nova_centroid_r=float("nan"),
                    nova_centroid_z=float("nan"),
                    reference_centroid_z=float("nan"),
                    efit_r=r.efit_r,
                    efit_z=r.efit_z,
                    efit_geom_r=float("nan"),
                    efit_geom_z=float("nan"),
                    efit_centroid_r=float("nan"),
                    efit_centroid_z=float("nan"),
                    l2_r=float("nan"),
                    l2_z=float("nan"),
                    flat_top=True,
                    d_r_cm=100.0 * (r.nova_centroid_r - r.efit_r),
                    d_z_cm=100.0 * (r.nova_centroid_z - r.efit_z),
                )
                for r in eligible
            ]
        ),
        "efit_geometric_axis_vs_efit_axis": _offset_stats(
            [
                SliceRow(
                    shot=r.shot,
                    session_index=r.session_index,
                    time_s=r.time_s,
                    nova_r=r.efit_geom_r,
                    nova_z=r.efit_geom_z,
                    nova_centroid_r=float("nan"),
                    nova_centroid_z=float("nan"),
                    reference_centroid_z=float("nan"),
                    efit_r=r.efit_r,
                    efit_z=r.efit_z,
                    efit_geom_r=float("nan"),
                    efit_geom_z=float("nan"),
                    efit_centroid_r=float("nan"),
                    efit_centroid_z=float("nan"),
                    l2_r=float("nan"),
                    l2_z=float("nan"),
                    flat_top=True,
                    d_r_cm=100.0 * (r.efit_geom_r - r.efit_r),
                    d_z_cm=100.0 * (r.efit_geom_z - r.efit_z),
                )
                for r in eligible
            ]
        ),
        "efit_current_centroid_vs_efit_axis": _offset_stats(
            [
                SliceRow(
                    shot=r.shot,
                    session_index=r.session_index,
                    time_s=r.time_s,
                    nova_r=r.efit_centroid_r,
                    nova_z=r.efit_centroid_z,
                    nova_centroid_r=float("nan"),
                    nova_centroid_z=float("nan"),
                    reference_centroid_z=float("nan"),
                    efit_r=r.efit_r,
                    efit_z=r.efit_z,
                    efit_geom_r=float("nan"),
                    efit_geom_z=float("nan"),
                    efit_centroid_r=float("nan"),
                    efit_centroid_z=float("nan"),
                    l2_r=float("nan"),
                    l2_z=float("nan"),
                    flat_top=True,
                    d_r_cm=100.0 * (r.efit_centroid_r - r.efit_r),
                    d_z_cm=100.0 * (r.efit_centroid_z - r.efit_z),
                )
                for r in eligible
            ]
        ),
        "reference_centroid_z_vs_efit_axis_z": _stats(
            [100.0 * (r.reference_centroid_z - r.efit_z) for r in eligible]
        ),
    }

    # The definition tell: does each nova/EFM quantity sit consistently on one
    # side of the EFIT magnetic axis in R?
    d_r = [r.d_r_cm for r in eligible]
    d_z = [r.d_z_cm for r in eligible]
    sign_tell = {
        "nova_axis_dR": _signed_space_stats(d_r),
        "nova_axis_dZ": _signed_space_stats(d_z),
        "nova_centroid_dR": _signed_space_stats(
            [100.0 * (r.nova_centroid_r - r.efit_r) for r in eligible]
        ),
        "efit_geom_axis_dR": _signed_space_stats(
            [100.0 * (r.efit_geom_r - r.efit_r) for r in eligible]
        ),
        "efit_centroid_dR": _signed_space_stats(
            [100.0 * (r.efit_centroid_r - r.efit_r) for r in eligible]
        ),
    }

    summary = {
        "eligible_free_flat_top_slices": total,
        "published_fifty_one_match": total == 51,
        "axis_offset_cm": _offset_stats(eligible),
        "signed_d_r_cm": _stats(d_r),
        "signed_d_z_cm": _stats(d_z),
        "sign_tell": sign_tell,
        "l2_referee_matches_efm_source_cm": l2_consistency,
        "efit_own_psi_extremum_vs_stored_axis_cm": {
            "dr_cm": _stats([r.psi_extremum_dr_cm for r in eligible]),
            "dz_cm": _stats([r.psi_extremum_dz_cm for r in eligible]),
            "euclidean_cm": _offset_stats(
                [
                    SliceRow(
                        shot=r.shot,
                        session_index=r.session_index,
                        time_s=r.time_s,
                        nova_r=0.0,
                        nova_z=0.0,
                        nova_centroid_r=float("nan"),
                        nova_centroid_z=float("nan"),
                        reference_centroid_z=float("nan"),
                        efit_r=0.0,
                        efit_z=0.0,
                        efit_geom_r=float("nan"),
                        efit_geom_z=float("nan"),
                        efit_centroid_r=float("nan"),
                        efit_centroid_z=float("nan"),
                        l2_r=float("nan"),
                        l2_z=float("nan"),
                        flat_top=True,
                        d_r_cm=r.psi_extremum_dr_cm,
                        d_z_cm=r.psi_extremum_dz_cm,
                    )
                    for r in eligible
                ]
            ),
            "status_counts": {
                s: sum(1 for r in eligible if r.psi_extremum_status == s)
                for s in sorted({r.psi_extremum_status for r in eligible})
            },
        },
        "per_shot": per_shot_stats,
        "efit_definition_space_cm": efit_internal,
        "referent_offset_cm": referents,
    }

    receipt = {
        "processor": "axis_definition_referents",
        "session_root": str(args.session_root),
        "level1_root": str(args.level1_root),
        "level2_root": str(args.level2_root),
        "shots": list(args.shots),
        "flat_top_rule": "interpolated |I_p_efm| >= 0.80 * shot peak",
        "eligibility": (
            "written + free converged (recorded) + not conditioned "
            "+ flat top + finite nova axis"
        ),
        "summary": summary,
        "slices": [asdict(r) for r in eligible],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "axis-definition-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _write_figure(eligible, args.output / "axis-definition-signed.png")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _write_figure(rows: list[SliceRow], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shots = sorted({r.shot for r in rows})
    palette = plt.cm.tab10(np.linspace(0, 1, len(shots)))
    color = {s: c for s, c in zip(shots, palette, strict=True)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    ax = axes[0]
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(0.0, color="k", lw=0.8)
    for s in shots:
        srows = [r for r in rows if r.shot == s]
        ax.scatter(
            [r.d_r_cm for r in srows],
            [r.d_z_cm for r in srows],
            label=str(s),
            color=color[s],
            s=28,
        )
    ax.set_xlabel("dR = nova$R_{axis}$ - EFIT$R_{axis}$ (cm)")
    ax.set_ylabel("dZ = nova$Z_{axis}$ - EFIT$Z_{axis}$ (cm)")
    ax.set_title("Signed axis offset per slice")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    for s in shots:
        srows = [r for r in rows if r.shot == s]
        ax.scatter(
            [r.time_s for r in srows],
            [r.d_r_cm for r in srows],
            label=str(s),
            color=color[s],
            s=28,
        )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dR (cm)")
    ax.set_title("dR vs time")
    ax.grid(alpha=0.3)
    ax = axes[2]
    names = [
        "nova axis",
        "nova current\ncentroid",
        "EFIT geom\naxis",
        "EFIT current\ncentroid",
        "ref centroid Z\n(abs dZ)",
    ]
    values_series = {
        "nova axis": [math.hypot(r.d_r_cm, r.d_z_cm) for r in rows],
        "nova centroid": [
            math.hypot(
                100.0 * (r.nova_centroid_r - r.efit_r),
                100.0 * (r.nova_centroid_z - r.efit_z),
            )
            for r in rows
        ],
        "EFIT geom": [
            math.hypot(
                100.0 * (r.efit_geom_r - r.efit_r), 100.0 * (r.efit_geom_z - r.efit_z)
            )
            for r in rows
        ],
        "EFIT centroid": [
            math.hypot(
                100.0 * (r.efit_centroid_r - r.efit_r),
                100.0 * (r.efit_centroid_z - r.efit_z),
            )
            for r in rows
        ],
    }
    medians = [
        float(np.nanmedian(values_series[k]))
        for k in ("nova axis", "nova centroid", "EFIT geom", "EFIT centroid")
    ]
    medians.append(
        float(
            np.nanmedian(
                [abs(100.0 * (r.reference_centroid_z - r.efit_z)) for r in rows]
            )
        )
    )
    ax.bar(names, medians, color="#607D8B")
    ax.set_ylabel("median |offset from EFIT magnetic axis| (cm)")
    ax.set_title("Which referent tracks the EFIT magnetic axis")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
