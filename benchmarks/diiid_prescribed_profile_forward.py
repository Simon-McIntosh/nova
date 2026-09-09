"""Run the DIII-D prescribed-profile forward-solve evidence set.

Each source label crosses from its measured per-radian map representation to
Nova's total-flux convention exactly once.  The p-prime and FF-prime inputs
are then deterministically extracted from that converted map and prescribed to
the forward solve together with the recorded circuit-driven conductor currents.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jax
import numpy as np

from benchmarks.diiid_corpus_conventions import PSI_TO_NOVA, _axis_and_boundary
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    GATE_RESIDUAL_TOLERANCE,
    REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _PLASMA_CURRENT_COLUMNS,
    _label_state,
    _read,
    _solve_frame_retaining_failure,
    polarity_population,
    select_frames,
)
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

DEFAULT_OUTPUT = Path(
    "docs/figures/diiid-vertical-force-balance/prescribed-profile-prototype"
)
RECEIPT_NAME = "receipt.json"
GATE_FRAME_COUNT = 5


def _finite(value: Any) -> Any:
    """Return JSON-compatible finite numeric values, preserving missing evidence."""

    if isinstance(value, np.ndarray):
        return [_finite(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _finite(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_finite(item) for item in value]
    return value


def _interpolated_value(row: dict[str, Any], time_ms: float, column: str) -> float:
    values = np.asarray(row[column], dtype=float)
    times = np.asarray(row["magnetics_time"], dtype=float)
    valid = np.isfinite(times + values)
    if not np.any(valid):
        raise RuntimeError(f"no finite samples for {column}")
    return float(np.interp(time_ms, times[valid], values[valid]))


def _sign(value: float) -> int:
    if not np.isfinite(value) or value == 0.0:
        raise RuntimeError("a required COCOS sign is absent")
    return int(np.sign(value))


def _source_sign_tuple(row: dict[str, Any], frame: int) -> dict[str, Any]:
    """Measure the source signs without asserting a source COCOS index."""

    time_ms = float(row["efit_times"][frame])
    plasma_current_ka = float(
        np.interp(
            time_ms,
            np.asarray(row["magnetics_plasma_current_times"], dtype=float),
            np.asarray(row["magnetics_plasma_current"], dtype=float),
        )
    )
    bcoil = _interpolated_value(row, time_ms, "magnetics_bcoil")
    axis, boundary = _axis_and_boundary(row, frame)
    outward_flux = boundary - axis
    q95 = float(row["efit_q95"][frame])
    ip_sign = _sign(plasma_current_ka)
    b0_sign = _sign(bcoil)
    flux_sign = _sign(outward_flux)
    q95_sign = _sign(q95)
    return {
        "ip_ka": plasma_current_ka,
        "ip_sign": ip_sign,
        "bcoil": bcoil,
        "b0_sign": b0_sign,
        "psi_axis_wb_per_rad": axis,
        "psi_boundary_wb_per_rad": boundary,
        "psi_axis_to_boundary_wb_per_rad": outward_flux,
        "psi_axis_to_boundary_sign": flux_sign,
        "sigma_bp": ip_sign * flux_sign,
        "q95": q95,
        "q95_sign": q95_sign,
        "sigma_rho_theta_phi_from_reported_q95": q95_sign * ip_sign * b0_sign,
        "source_cocos_index": None,
        "source_cocos_index_status": "not uniquely derivable from this tuple",
    }


def _stored_profile_comparison(row: dict[str, Any]) -> dict[str, Any]:
    profile_columns = [
        name
        for name in row
        if "pprime" in name.lower()
        or "p_prime" in name.lower()
        or "ffprime" in name.lower()
        or "ff_prime" in name.lower()
    ]
    if profile_columns:
        raise RuntimeError(
            "a stored profile column appeared; its convention requires "
            "an explicit audit"
        )
    return {
        "status": "unavailable",
        "reason": (
            "the source Parquet carries no stored EFIT p-prime or FF-prime arrays"
        ),
        "source_profile_columns": profile_columns,
    }


def _forward_result(result: Any) -> dict[str, Any]:
    banked = result.banked_read
    return {
        "terminal_residual": _finite(result.fixed_point_relative_residual),
        "residual_tolerance": result.residual_tolerance,
        "converged": result.converged,
        "finite": result.finite,
        "solver_termination": result.solver_termination,
        "solve_exception_class": result.solve_exception_class,
        "achieved_topology_class": result.achieved_topology_class,
        "magnetic_axis_rz_m": _finite(banked.get("nova_axis_rz_m")),
        "x_point_rz_m": _finite(banked.get("nova_x_point_rz_m")),
        "target_current_a": _finite(result.target_current_a),
        "achieved_current_a": _finite(result.achieved_current_a),
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def run(data: Path = DEFAULT_DATA, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Solve the five declared frames and write one receipt with every input."""

    configure_dtypes()
    if not bool(jax.config.x64_enabled):
        raise RuntimeError("the prescribed-profile measurement requires JAX x64")
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())
    selected = select_frames(
        sorted(data.glob("*.parquet")), GATE_FRAME_COUNT, polarity_population()
    )
    rows: list[dict[str, Any]] = []
    for ordinal, selected_frame in enumerate(selected, start=1):
        row = _read(
            selected_frame.path,
            _LABEL_COLUMNS
            + _GEOMETRY_COLUMNS
            + _CURRENT_COLUMNS
            + _PLASMA_CURRENT_COLUMNS,
        )
        row["_source_path"] = str(selected_frame.path)
        sign_tuple = _source_sign_tuple(row, selected_frame.frame)
        psi_norm, p_prime, ff_prime = _label_state(row, selected_frame.frame)[1:]
        result, _fields = _solve_frame_retaining_failure(
            row,
            selected_frame.frame,
            REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
        )
        rows.append(
            {
                "ordinal": ordinal,
                "shot": selected_frame.path.name,
                "frame": selected_frame.frame,
                "time_ms": selected_frame.time_ms,
                "source_sign_tuple": sign_tuple,
                "translation": {
                    "map_factor_to_nova_total_flux": PSI_TO_NOVA,
                    "current_factor_to_nova": 1.0,
                    "map_applied_once_before_affine_extraction": True,
                    "current_prescription": (
                        "recorded corpus currents with fixed circuit wiring"
                    ),
                },
                "prescribed_profiles": {
                    "source": "affine extraction from converted source map",
                    "psi_norm": psi_norm,
                    "p_prime": p_prime,
                    "ff_prime": ff_prime,
                    "reliable_surface_count": int(len(psi_norm)),
                },
                "forward_result": _forward_result(result),
                "profile_comparison": _stored_profile_comparison(row),
            }
        )
        print(
            "SOLVED "
            f"{ordinal}/{GATE_FRAME_COUNT} {selected_frame.path.name}:"
            f"{selected_frame.frame} "
            f"residual={result.fixed_point_relative_residual:.6e} "
            f"converged={result.converged}",
            flush=True,
        )
    if len(rows) != GATE_FRAME_COUNT:
        raise RuntimeError("the receipt must contain exactly five gate-frame rows")
    receipt = _finite(
        {
            "measurement": "DIII-D prescribed-profile forward solves",
            "convention": {
                "target": "Nova COCOS 17 total flux",
                "source_cocos": (
                    "factor-equivalence class; no single source COCOS index is claimed"
                ),
                "map_factor_to_nova_total_flux": PSI_TO_NOVA,
                "current_factor_to_nova": 1.0,
                "profile_derivative_source": "affine extraction after map conversion",
            },
            "execution": {
                "jax_x64_enabled": bool(jax.config.x64_enabled),
                "gate_frame_count": GATE_FRAME_COUNT,
                "residual_tolerance": GATE_RESIDUAL_TOLERANCE,
                "git_head": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    check=True,
                    text=True,
                    capture_output=True,
                ).stdout.strip(),
            },
            "frames": rows,
        }
    )
    _atomic_write(output / RECEIPT_NAME, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = run(arguments.data, arguments.output)
    print(json.dumps(receipt["execution"], sort_keys=True))


if __name__ == "__main__":
    main()
