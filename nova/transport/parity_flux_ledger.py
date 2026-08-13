"""Shot-level temporal consistency scoring for reconstruction parity.

Each adjacent pair of reconstructed equilibria supplies an observed poloidal-flux
swing.  :class:`~nova.transport.current_diffusion.CurrentDiffusion` advances the
first equilibrium over the same time interval and measured-current endpoints.
The registered metric is the RMS difference between predicted and observed flux
swings divided by the RMS observed swing, accumulated across every interval and
radial face in the shot.

The report retains the physical ledger rather than reducing it to one verdict:
observed and predicted surface-flux consumption, predicted resistive and
inductive contributions, and the independently visible ``li`` and ``beta_p``
traces.  An unavailable shot remains in the ordered result with its reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from nova.imas.parity_tolerances import (
    MetricTolerance,
    ScorecardField,
    registered_tolerances,
)
from nova.transport.current_diffusion import (
    CurrentDiffusion,
    EtaProfile,
    FluxSurfaceGeometry,
    poloidal_field_energy_li,
)


FROZEN_GATE_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)


def flux_ledger_tolerance() -> MetricTolerance:
    """Return the registered temporal tolerance used for every shot."""

    field = ScorecardField.CURRENT_DIFFUSION_FLUX_LEDGER_RMS_FRACTION
    return registered_tolerances()[field.value]


@dataclass(frozen=True)
class ReconstructedEquilibrium:
    """Transport-ready state of one reconstructed equilibrium slice."""

    time_s: float
    geometry: FluxSurfaceGeometry
    poloidal_beta: float


@dataclass(frozen=True)
class ShotFluxLedger:
    """Current-diffusion consistency and physical ledger for one shot."""

    shot: int
    bound: float
    rms_fraction: float | None
    passed: bool | None
    slice_rms_fractions: tuple[float, ...]
    observed_surface_flux_consumption_wb: float | None
    predicted_surface_flux_consumption_wb: float | None
    resistive_flux_consumption_wb: float | None
    inductive_flux_consumption_wb: float | None
    internal_inductance: tuple[float, ...]
    poloidal_beta: tuple[float, ...]
    li_minus_beta_p: tuple[float, ...]
    reason: str | None = None

    @property
    def available(self) -> bool:
        """Return whether the shot produced a numeric ledger and verdict."""

        return self.reason is None

    @classmethod
    def unavailable(cls, shot: int, reason: str) -> ShotFluxLedger:
        """Keep an unscored shot visible with an actionable reason."""

        return cls(
            shot=int(shot),
            bound=flux_ledger_tolerance().bound,
            rms_fraction=None,
            passed=None,
            slice_rms_fractions=(),
            observed_surface_flux_consumption_wb=None,
            predicted_surface_flux_consumption_wb=None,
            resistive_flux_consumption_wb=None,
            inductive_flux_consumption_wb=None,
            internal_inductance=(),
            poloidal_beta=(),
            li_minus_beta_p=(),
            reason=str(reason),
        )

    def as_dict(self) -> Mapping[str, object]:
        """Return a serialization-ready report with the verdict kept explicit."""

        return MappingProxyType(
            {
                "shot": self.shot,
                "bound": self.bound,
                "rms_fraction": self.rms_fraction,
                "passed": self.passed,
                "slice_rms_fractions": list(self.slice_rms_fractions),
                "observed_surface_flux_consumption_wb": (
                    self.observed_surface_flux_consumption_wb
                ),
                "predicted_surface_flux_consumption_wb": (
                    self.predicted_surface_flux_consumption_wb
                ),
                "resistive_flux_consumption_wb": self.resistive_flux_consumption_wb,
                "inductive_flux_consumption_wb": self.inductive_flux_consumption_wb,
                "internal_inductance": list(self.internal_inductance),
                "poloidal_beta": list(self.poloidal_beta),
                "li_minus_beta_p": list(self.li_minus_beta_p),
                "reason": self.reason,
            }
        )


def _validate_slices(slices: Sequence[ReconstructedEquilibrium]) -> None:
    if len(slices) < 2:
        raise ValueError("at least two reconstructed slices are required")
    times = np.asarray([row.time_s for row in slices], dtype=np.float64)
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0.0):
        raise ValueError(
            "reconstructed slice times must be finite and strictly increasing"
        )
    for index, row in enumerate(slices):
        geometry = row.geometry
        arrays = {
            "rho_face": geometry.rho_face,
            "psi_face": geometry.psi_face,
        }
        for name, values in arrays.items():
            values = np.asarray(values, dtype=np.float64)
            if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
                raise ValueError(f"slice {index} {name} must be a finite radial vector")
        if not np.all(np.diff(np.asarray(geometry.rho_face)) > 0.0):
            raise ValueError(f"slice {index} rho_face must be strictly increasing")
        if not np.isfinite(row.poloidal_beta):
            raise ValueError(f"slice {index} poloidal_beta is not finite")


def score_shot_flux_ledger(
    shot: int,
    slices: Sequence[ReconstructedEquilibrium],
    *,
    eta: EtaProfile = EtaProfile(),
) -> ShotFluxLedger:
    """Run the reconstructed shot through the current-diffusion flux ledger."""

    _validate_slices(slices)
    residual_rows: list[np.ndarray] = []
    observed_rows: list[np.ndarray] = []
    interval_fractions: list[float] = []
    predicted_surface = 0.0
    resistive = 0.0
    inductive = 0.0

    for start, end in zip(slices[:-1], slices[1:], strict=True):
        solver = CurrentDiffusion(start.geometry, eta)
        step = solver.evolve(
            np.asarray([start.time_s, end.time_s], dtype=np.float64),
            np.asarray(
                [start.geometry.ip_amperes, end.geometry.ip_amperes],
                dtype=np.float64,
            ),
        )
        rho = np.asarray(start.geometry.rho_face, dtype=np.float64)
        next_flux = np.interp(
            rho,
            np.asarray(end.geometry.rho_face, dtype=np.float64),
            np.asarray(end.geometry.psi_face, dtype=np.float64),
        )
        initial_flux = np.asarray(start.geometry.psi_face, dtype=np.float64)
        observed = next_flux - initial_flux
        predicted = np.asarray(step["psi_face"][-1], dtype=np.float64) - initial_flux
        residual = predicted - observed
        scale = float(np.sqrt(np.mean(np.square(observed))))
        if not np.isfinite(scale) or scale <= np.finfo(np.float64).tiny:
            raise ValueError(
                f"interval {start.time_s:g}-{end.time_s:g} s has zero "
                "observed flux swing"
            )
        interval_fractions.append(float(np.sqrt(np.mean(np.square(residual))) / scale))
        residual_rows.append(residual)
        observed_rows.append(observed)
        budget = solver.budget(step)
        predicted_surface += float(budget["d_psi_bdry"])
        resistive += float(budget["d_psi_axis"])
        inductive += float(budget["d_psi_internal"])

    residual = np.concatenate(residual_rows)
    observed = np.concatenate(observed_rows)
    observed_scale = float(np.sqrt(np.mean(np.square(observed))))
    rms_fraction = float(np.sqrt(np.mean(np.square(residual))) / observed_scale)
    tolerance = flux_ledger_tolerance()
    internal_inductance = tuple(
        float(poloidal_field_energy_li(row.geometry)) for row in slices
    )
    poloidal_beta = tuple(float(row.poloidal_beta) for row in slices)
    separation = tuple(
        li - beta for li, beta in zip(internal_inductance, poloidal_beta, strict=True)
    )
    observed_surface = float(
        slices[-1].geometry.psi_face[-1] - slices[0].geometry.psi_face[-1]
    )
    slice_fractions = tuple([interval_fractions[0], *interval_fractions])
    return ShotFluxLedger(
        shot=int(shot),
        bound=tolerance.bound,
        rms_fraction=rms_fraction,
        passed=tolerance.passes(rms_fraction),
        slice_rms_fractions=slice_fractions,
        observed_surface_flux_consumption_wb=observed_surface,
        predicted_surface_flux_consumption_wb=predicted_surface,
        resistive_flux_consumption_wb=resistive,
        inductive_flux_consumption_wb=inductive,
        internal_inductance=internal_inductance,
        poloidal_beta=poloidal_beta,
        li_minus_beta_p=separation,
    )


def score_gated_flux_ledgers(
    reconstructions: Mapping[int, Sequence[ReconstructedEquilibrium]],
    *,
    shots: Sequence[int] = FROZEN_GATE_SHOTS,
    eta: EtaProfile = EtaProfile(),
) -> tuple[ShotFluxLedger, ...]:
    """Score every requested shot without dropping absent or invalid ledgers."""

    reports = []
    for shot in shots:
        if shot not in reconstructions:
            reports.append(
                ShotFluxLedger.unavailable(
                    shot, "reconstructed equilibria are unavailable"
                )
            )
            continue
        try:
            reports.append(score_shot_flux_ledger(shot, reconstructions[shot], eta=eta))
        except Exception as error:
            reports.append(
                ShotFluxLedger.unavailable(shot, f"{type(error).__name__}: {error}")
            )
    return tuple(reports)


__all__ = [
    "FROZEN_GATE_SHOTS",
    "ReconstructedEquilibrium",
    "ShotFluxLedger",
    "flux_ledger_tolerance",
    "score_gated_flux_ledgers",
    "score_shot_flux_ledger",
]
