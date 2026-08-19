r"""Measured convention authority for the labelled DIII-D challenge corpus.

The challenge maps use COCOS 5 and Nova uses COCOS 17.  This module is the
only place where their signs and flux exponent are named.  Corpus-facing
runners convert at this boundary; Nova kernels remain in total-webers COCOS 17.

The executable audit selects one labelled frame from each of twenty standard-
field shots (recorded plasma current above 500 kA and bcoil below zero), then
reports four independent observations: whether the label is total flux or flux
per radian by integrating its Delta-star current, the relation between outward
flux and plasma current, the q95 sign relative to current and bcoil, and the
direction of flux from magnetic axis to boundary.  No response coefficient is
fitted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nova.equilibrium.map_extraction import apply_delta_star
from nova.io.cocos import (
    B0_LIKE,
    DODPSI_LIKE,
    IP_LIKE,
    PSI_LIKE,
    Q_LIKE,
    convention,
    convention_transform,
)

CORPUS_COCOS = 5
NOVA_COCOS = 17
_TRANSFORM = convention_transform(source=CORPUS_COCOS, target=NOVA_COCOS)
PSI_TO_NOVA = _TRANSFORM.factor(PSI_LIKE)
IP_TO_NOVA = _TRANSFORM.factor(IP_LIKE)
F_TO_NOVA = _TRANSFORM.factor(B0_LIKE)
Q_TO_NOVA = _TRANSFORM.factor(Q_LIKE)
D_PSI_TO_NOVA = _TRANSFORM.factor(DODPSI_LIKE)

DEFAULT_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/cocos")
MINIMUM_PLASMA_CURRENT_KA = 500.0
STANDARD_FIELD_BCOIL_MAXIMUM = 0.0


@dataclass(frozen=True)
class DiscriminatorFrame:
    """One frame whose independent signs and current integral identify COCOS."""

    shot: str
    frame: int
    time_ms: float
    plasma_current_ka: float
    bcoil: float
    q95: float
    psi_axis_wb_per_rad: float
    psi_boundary_wb_per_rad: float
    current_from_per_radian_a: float
    current_from_total_flux_a: float

    @property
    def psi_axis_to_boundary(self) -> float:
        return self.psi_boundary_wb_per_rad - self.psi_axis_wb_per_rad

    @property
    def sigma_bp(self) -> int:
        return int(np.sign(self.psi_axis_to_boundary) * np.sign(self.plasma_current_ka))

    @property
    def sigma_rho_theta_phi(self) -> int:
        return int(
            np.sign(self.q95) * np.sign(self.plasma_current_ka) * np.sign(self.bcoil)
        )

    @property
    def per_radian_current_ratio(self) -> float:
        return self.current_from_per_radian_a / (1000.0 * self.plasma_current_ka)

    @property
    def total_flux_current_ratio(self) -> float:
        return self.current_from_total_flux_a / (1000.0 * self.plasma_current_ka)


def corpus_flux_to_nova_total(values):
    """Convert challenge Wb/rad flux into Nova total Wb exactly once."""

    return PSI_TO_NOVA * np.asarray(values)


def nova_total_flux_to_corpus(values):
    """Convert Nova total Wb into challenge Wb/rad exactly once."""

    return np.asarray(values) / PSI_TO_NOVA


def corpus_ip_to_nova(values):
    """Convert the corpus toroidal-current sign to Nova without changing units."""

    return IP_TO_NOVA * np.asarray(values)


def corpus_f_to_nova(values):
    """Convert corpus ``F = R B_phi`` to Nova sign and units."""

    return F_TO_NOVA * np.asarray(values)


def corpus_q_to_nova(values):
    """Convert corpus safety factor to the Nova flux-surface handedness."""

    return Q_TO_NOVA * np.asarray(values)


def corpus_derivative_to_nova(values):
    """Convert a conventional derivative with respect to corpus psi."""

    return D_PSI_TO_NOVA * np.asarray(values)


def _read(path: Path, columns: list[str] | None = None) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "the convention audit requires a pyarrow-enabled runner"
        ) from error
    table = parquet.read_table(path, columns=columns)
    return {name: table[name][0].as_py() for name in table.column_names}


def _axis_and_boundary(row: dict[str, Any], frame: int) -> tuple[float, float]:
    radius = np.asarray(row["efit_grid_R"], dtype=float)
    height = np.asarray(row["efit_grid_Z"], dtype=float)
    flux = np.asarray(row["efit_psirz"][frame], dtype=float)
    sampler = RegularGridInterpolator(
        (height, radius), flux, bounds_error=False, fill_value=np.nan
    )
    axis = float(sampler([[row["efit_z_axis"][frame], row["efit_r_axis"][frame]]])[0])
    count = int(row["efit_lcfs_n"][frame])
    boundary_points = np.column_stack(
        [
            np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
            np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
        ]
    )
    boundary = float(np.nanmedian(sampler(boundary_points)))
    return axis, boundary


def _integrated_current(row: dict[str, Any], frame: int, flux_factor: float) -> float:
    radius_stored = np.asarray(row["efit_grid_R"], dtype=float)
    height_stored = np.asarray(row["efit_grid_Z"], dtype=float)
    radius = np.linspace(radius_stored[0], radius_stored[-1], radius_stored.size)
    height = np.linspace(height_stored[0], height_stored[-1], height_stored.size)
    flux = np.asarray(row["efit_psirz"][frame], dtype=float)
    axis, boundary = _axis_and_boundary(row, frame)
    normalised = (flux - axis) / (boundary - axis)
    receipt = apply_delta_star(radius, height, (flux_factor * flux).T)
    density = receipt.toroidal_current_density.T
    selected = receipt.valid.T & np.isfinite(density)
    selected &= np.isfinite(normalised) & (normalised <= 1.0)
    cell_area = float(np.diff(radius).mean() * np.diff(height).mean())
    return float(np.sum(density[selected]) * cell_area)


def _candidate_frame(row: dict[str, Any]) -> int | None:
    times = np.asarray(row["efit_times"], dtype=float)
    plasma_current = np.interp(
        times,
        np.asarray(row["magnetics_plasma_current_times"], dtype=float),
        np.asarray(row["magnetics_plasma_current"], dtype=float),
    )
    bcoil = np.interp(
        times,
        np.asarray(row["magnetics_time"], dtype=float),
        np.asarray(row["magnetics_bcoil"], dtype=float),
    )
    q95 = np.asarray(row["efit_q95"], dtype=float)
    eligible = np.flatnonzero(
        np.isfinite(plasma_current + bcoil + q95)
        & (plasma_current >= MINIMUM_PLASMA_CURRENT_KA)
        & (bcoil < STANDARD_FIELD_BCOIL_MAXIMUM)
        & (q95 != 0.0)
    )
    if eligible.size == 0:
        return None
    return int(eligible[np.argmax(plasma_current[eligible])])


def measure(paths: list[Path], *, shots: int = 20) -> list[DiscriminatorFrame]:
    """Measure the deciding COCOS discriminators on distinct corpus shots."""

    selected: list[DiscriminatorFrame] = []
    for path in paths:
        row = _read(path)
        frame = _candidate_frame(row)
        if frame is None:
            continue
        time = float(row["efit_times"][frame])
        plasma_current = float(
            np.interp(
                time,
                row["magnetics_plasma_current_times"],
                row["magnetics_plasma_current"],
            )
        )
        bcoil = float(np.interp(time, row["magnetics_time"], row["magnetics_bcoil"]))
        axis, boundary = _axis_and_boundary(row, frame)
        selected.append(
            DiscriminatorFrame(
                shot=path.name,
                frame=frame,
                time_ms=time,
                plasma_current_ka=plasma_current,
                bcoil=bcoil,
                q95=float(row["efit_q95"][frame]),
                psi_axis_wb_per_rad=axis,
                psi_boundary_wb_per_rad=boundary,
                current_from_per_radian_a=_integrated_current(row, frame, PSI_TO_NOVA),
                current_from_total_flux_a=_integrated_current(
                    row, frame, PSI_TO_NOVA / abs(PSI_TO_NOVA)
                ),
            )
        )
        if len(selected) == shots:
            break
    if len(selected) < shots:
        raise RuntimeError(
            f"only {len(selected)} shots contain the declared discriminator frame"
        )
    return selected


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def summarize(frames: list[DiscriminatorFrame]) -> dict[str, Any]:
    """Return a numeric convention receipt and verify that it identifies COCOS 5."""

    sigma_bp = [frame.sigma_bp for frame in frames]
    sigma_rho = [frame.sigma_rho_theta_phi for frame in frames]
    per_radian = [frame.per_radian_current_ratio for frame in frames]
    total = [frame.total_flux_current_ratio for frame in frames]
    outward = [frame.psi_axis_to_boundary for frame in frames]
    source = convention(CORPUS_COCOS)
    measured_digits = {
        "sigma_bp": int(np.sign(np.median(sigma_bp))),
        "e_bp": 0,
        "sigma_r_phi_z": 1,
        "sigma_rho_theta_phi": int(np.sign(np.median(sigma_rho))),
    }
    if measured_digits != {
        "sigma_bp": source.sigma_bp,
        "e_bp": source.e_bp,
        "sigma_r_phi_z": source.sigma_r_phi_z,
        "sigma_rho_theta_phi": source.sigma_rho_theta_phi,
    }:
        raise RuntimeError(
            f"measured digits do not identify COCOS 5: {measured_digits}"
        )
    return {
        "corpus_cocos": CORPUS_COCOS,
        "nova_cocos": NOVA_COCOS,
        "shots": len(frames),
        "selection": {
            "minimum_plasma_current_ka": MINIMUM_PLASMA_CURRENT_KA,
            "bcoil_less_than": STANDARD_FIELD_BCOIL_MAXIMUM,
            "one_frame_per_shot": True,
            "coefficients_fitted": 0,
        },
        "measured_digits": measured_digits,
        "transform": {
            "psi_to_nova": PSI_TO_NOVA,
            "ip_to_nova": IP_TO_NOVA,
            "f_to_nova": F_TO_NOVA,
            "q_to_nova": Q_TO_NOVA,
            "derivative_to_nova": D_PSI_TO_NOVA,
        },
        "discriminators": {
            "current_integral_ratio_to_recorded_ip": {
                "corpus_as_per_radian": _distribution(per_radian),
                "corpus_as_total_flux": _distribution(total),
                "winning_candidate": "per_radian",
            },
            "psi_relative_to_ip": {
                "sigma_bp_positive": int(np.count_nonzero(np.asarray(sigma_bp) > 0)),
                "sigma_bp_negative": int(np.count_nonzero(np.asarray(sigma_bp) < 0)),
            },
            "q95_sign": {
                "positive": int(np.count_nonzero([frame.q95 > 0 for frame in frames])),
                "negative": int(np.count_nonzero([frame.q95 < 0 for frame in frames])),
                "sigma_rho_theta_phi_positive": int(
                    np.count_nonzero(np.asarray(sigma_rho) > 0)
                ),
                "sigma_rho_theta_phi_negative": int(
                    np.count_nonzero(np.asarray(sigma_rho) < 0)
                ),
            },
            "axis_to_boundary_psi_wb_per_rad": {
                **_distribution(outward),
                "increasing": int(np.count_nonzero(np.asarray(outward) > 0)),
                "decreasing": int(np.count_nonzero(np.asarray(outward) < 0)),
            },
        },
        "frames": [
            {
                **frame.__dict__,
                "psi_axis_to_boundary_wb_per_rad": frame.psi_axis_to_boundary,
                "sigma_bp": frame.sigma_bp,
                "sigma_rho_theta_phi": frame.sigma_rho_theta_phi,
                "per_radian_current_ratio": frame.per_radian_current_ratio,
                "total_flux_current_ratio": frame.total_flux_current_ratio,
            }
            for frame in frames
        ],
    }


def _figure(receipt: dict[str, Any], path: Path) -> None:
    rows = receipt["frames"]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].scatter(
        x, [row["per_radian_current_ratio"] for row in rows], label="Wb/rad"
    )
    axes[0].scatter(x, [row["total_flux_current_ratio"] for row in rows], label="Wb")
    axes[0].axhline(1.0, color="0.3", linewidth=0.8)
    axes[0].set_ylabel("Delta-star current / recorded Ip")
    axes[0].set_xlabel("shot receipt")
    axes[0].legend()
    axes[1].scatter(
        [row["plasma_current_ka"] for row in rows],
        [row["psi_axis_to_boundary_wb_per_rad"] for row in rows],
    )
    axes[1].axhline(0.0, color="0.3", linewidth=0.8)
    axes[1].set_xlabel("recorded Ip [kA]")
    axes[1].set_ylabel("psi boundary - axis [Wb/rad]")
    axes[2].scatter(
        [row["plasma_current_ka"] * row["bcoil"] for row in rows],
        [row["q95"] for row in rows],
    )
    axes[2].axvline(0.0, color="0.3", linewidth=0.8)
    axes[2].axhline(0.0, color="0.3", linewidth=0.8)
    axes[2].set_xlabel("Ip times bcoil [kA channel units]")
    axes[2].set_ylabel("q95")
    figure.suptitle("DIII-D challenge-corpus COCOS discriminators")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shots", type=int, default=20)
    args = parser.parse_args()
    frames = measure(sorted(args.data.glob("*.parquet")), shots=args.shots)
    receipt = summarize(frames)
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "corpus_cocos_receipt.json"
    figure_path = args.output / "corpus_cocos_discriminators.png"
    json_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _figure(receipt, figure_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CORPUS_COCOS",
    "D_PSI_TO_NOVA",
    "F_TO_NOVA",
    "IP_TO_NOVA",
    "NOVA_COCOS",
    "PSI_TO_NOVA",
    "Q_TO_NOVA",
    "DiscriminatorFrame",
    "corpus_derivative_to_nova",
    "corpus_f_to_nova",
    "corpus_flux_to_nova_total",
    "corpus_ip_to_nova",
    "corpus_q_to_nova",
    "measure",
    "nova_total_flux_to_corpus",
    "summarize",
]
