"""Pre-registered tolerances and held-out shots for MAST parity scoring.

The registry is deliberately importable without opening the shot store.  Bounds
come from committed benchmark artifacts and are fixed before a scorecard is
accepted.  The raw-magnetics metric has two explicit comparison classes because
changing the reconstruction substrate and changing the machine description have
different measured noise floors.
"""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping


class ScorecardField(enum.StrEnum):
    """Every numeric field emitted by the reconstruction parity scorecard."""

    MAGNETIC_AXIS_DISTANCE_M = "magnetic_axis_distance_m"
    LCFS_DISTANCE_M = "lcfs_distance_m"
    X_POINT_DISTANCE_M = "x_point_distance_m"
    TOPOLOGY_CLASS_AGREEMENT_FRACTION = "topology_class_agreement_fraction"
    PROFILE_RESIDUAL_RMS = "profile_residual_rms"
    FIXED_POINT_DEFECT = "fixed_point_defect"
    MAGNETICS_RESIDUAL_WHITENED_RMS = "magnetics_residual_whitened_rms"
    CONVERGED_FRACTION = "converged_fraction"
    CONFINED_FRACTION = "confined_fraction"
    ITERATION_COUNT = "iteration_count"
    THROUGHPUT_SLICES_PER_CORE_S = "throughput_slices_per_core_s"
    CURRENT_DIFFUSION_FLUX_LEDGER_RMS_FRACTION = (
        "current_diffusion_flux_ledger_rms_fraction"
    )


SCORECARD_FIELDS = frozenset(field.value for field in ScorecardField)


class BoundDirection(enum.StrEnum):
    """The side of a numeric tolerance that passes."""

    AT_MOST = "at-most"
    AT_LEAST = "at-least"


class MagneticsBudgetClass(enum.StrEnum):
    """The measured comparison class for the absolute magnetics residual."""

    SAME_SOURCE = "same-source"
    SOURCE_CUTOVER = "source-cutover"


@dataclass(frozen=True)
class MetricTolerance:
    """One numeric bound together with the measurement that licensed it."""

    field: ScorecardField | str
    bound: float
    direction: BoundDirection
    unit: str
    basis: str
    evidence: str

    def passes(self, value: float) -> bool:
        """Return whether ``value`` is finite and lies on the passing side."""
        if not math.isfinite(value):
            return False
        if self.direction is BoundDirection.AT_MOST:
            return value <= self.bound
        return value >= self.bound


GEOMETRY_REFERENCE_IDENTITY = (
    "sha256:8df7a0a6c3f6162dbe0f226660bc069f37de8eb69f0f7c80bbfedc2bd4be220c"
)
PRECISION_AUDIT = "docs/figures/jax-dissolution/fieldnull_production_route.json"
TRANSPORT_CROSS_CHECK = "CurrentDiffusion/TORAX flux-ledger cross-check"

MAGNETICS_REFERENCE = 0.7401030841611733
SAME_SOURCE_RESIDUAL_BUDGET = 0.01
CROSS_SOURCE_COLUMN_PROPAGATION = 0.0285
SOLVE_FEEDBACK_ALLOWANCE = 0.0337
STATED_WEIGHT_CALIBRATION_ALLOWANCE = 0.0171
SOURCE_CUTOVER_RESIDUAL_BUDGET = (
    CROSS_SOURCE_COLUMN_PROPAGATION
    + SOLVE_FEEDBACK_ALLOWANCE
    + STATED_WEIGHT_CALIBRATION_ALLOWANCE
)


def _common_tolerances() -> dict[str, MetricTolerance]:
    return {
        ScorecardField.MAGNETIC_AXIS_DISTANCE_M: MetricTolerance(
            field=ScorecardField.MAGNETIC_AXIS_DISTANCE_M,
            bound=0.00101344,
            direction=BoundDirection.AT_MOST,
            unit="m",
            basis=(
                "four times the 0.025336 cm committed-stamp axis reference; "
                "the deterministic reproduction spread was zero"
            ),
            evidence=GEOMETRY_REFERENCE_IDENTITY,
        ),
        ScorecardField.LCFS_DISTANCE_M: MetricTolerance(
            field=ScorecardField.LCFS_DISTANCE_M,
            bound=0.00033047,
            direction=BoundDirection.AT_MOST,
            unit="m",
            basis=(
                "four times the 0.008262 cm committed-stamp LCFS reference; "
                "the deterministic reproduction spread was zero"
            ),
            evidence=GEOMETRY_REFERENCE_IDENTITY,
        ),
        ScorecardField.X_POINT_DISTANCE_M: MetricTolerance(
            field=ScorecardField.X_POINT_DISTANCE_M,
            bound=0.014,
            direction=BoundDirection.AT_MOST,
            unit="m",
            basis=(
                "0.522 of the 26.64 mm MAST radial cell, the committed "
                "held-out noisy X-point localisation maximum, rounded upward"
            ),
            evidence=PRECISION_AUDIT,
        ),
        ScorecardField.TOPOLOGY_CLASS_AGREEMENT_FRACTION: MetricTolerance(
            field=ScorecardField.TOPOLOGY_CLASS_AGREEMENT_FRACTION,
            bound=1.0,
            direction=BoundDirection.AT_LEAST,
            unit="fraction",
            basis=(
                "categorical topology is quantized per slice; any disagreement "
                "is one whole observed class error, so no fractional margin exists"
            ),
            evidence=PRECISION_AUDIT,
        ),
        ScorecardField.PROFILE_RESIDUAL_RMS: MetricTolerance(
            field=ScorecardField.PROFILE_RESIDUAL_RMS,
            bound=0.075868,
            direction=BoundDirection.AT_MOST,
            unit="normalized RMS",
            basis=(
                "four times the 0.018967 committed-stamp profile reference; "
                "the deterministic reproduction spread was zero"
            ),
            evidence=GEOMETRY_REFERENCE_IDENTITY,
        ),
        ScorecardField.FIXED_POINT_DEFECT: MetricTolerance(
            field=ScorecardField.FIXED_POINT_DEFECT,
            bound=1.0e-8,
            direction=BoundDirection.AT_MOST,
            unit="relative sup norm",
            basis=(
                "the profile accelerator's existing strict convergence criterion, "
                "applied to max|g(x)-x|/max|g(x)| without substituting the "
                "cross-substrate current-reproduction bound"
            ),
            evidence="nova/equilibrium/fixed_point.py",
        ),
        ScorecardField.CONVERGED_FRACTION: MetricTolerance(
            field=ScorecardField.CONVERGED_FRACTION,
            bound=1.0,
            direction=BoundDirection.AT_LEAST,
            unit="fraction",
            basis=(
                "the frozen stamps score at most six slices per shot, so the "
                "smallest observed loss is at least 1/6 and exact 1.0 is the only gate"
            ),
            evidence=GEOMETRY_REFERENCE_IDENTITY,
        ),
        ScorecardField.CONFINED_FRACTION: MetricTolerance(
            field=ScorecardField.CONFINED_FRACTION,
            bound=1.0,
            direction=BoundDirection.AT_LEAST,
            unit="fraction",
            basis=(
                "the frozen stamps score at most six slices per shot, so the "
                "smallest observed loss is at least 1/6 and exact 1.0 is the only gate"
            ),
            evidence=GEOMETRY_REFERENCE_IDENTITY,
        ),
        ScorecardField.ITERATION_COUNT: MetricTolerance(
            field=ScorecardField.ITERATION_COUNT,
            bound=8.0,
            direction=BoundDirection.AT_MOST,
            unit="fixed sweeps per slice",
            basis=(
                "the committed frozen-shot runs use the fixed-shape eight-sweep "
                "profile path; exceeding its registered compute budget is a failure"
            ),
            evidence=GEOMETRY_REFERENCE_IDENTITY,
        ),
        ScorecardField.THROUGHPUT_SLICES_PER_CORE_S: MetricTolerance(
            field=ScorecardField.THROUGHPUT_SLICES_PER_CORE_S,
            bound=0.1886677065919248,
            direction=BoundDirection.AT_LEAST,
            unit="slices/(core s)",
            basis=(
                "25% below the 0.2515569421 before-path throughput; the allowance "
                "clears the measured 19.3% shared-node spread"
            ),
            evidence=GEOMETRY_REFERENCE_IDENTITY,
        ),
        ScorecardField.CURRENT_DIFFUSION_FLUX_LEDGER_RMS_FRACTION: MetricTolerance(
            field=ScorecardField.CURRENT_DIFFUSION_FLUX_LEDGER_RMS_FRACTION,
            bound=0.004,
            direction=BoundDirection.AT_MOST,
            unit="fractional RMS",
            basis=(
                "the independent transport cross-check measured 0.4% RMS flux-ledger "
                "disagreement, registered as the temporal consistency ceiling"
            ),
            evidence=TRANSPORT_CROSS_CHECK,
        ),
    }


def registered_tolerances(
    magnetics_budget: MagneticsBudgetClass = MagneticsBudgetClass.SAME_SOURCE,
) -> Mapping[str, MetricTolerance]:
    """Return the complete immutable registry for one comparison class."""
    tolerances = _common_tolerances()
    if magnetics_budget is MagneticsBudgetClass.SAME_SOURCE:
        residual_budget = SAME_SOURCE_RESIDUAL_BUDGET
        basis = (
            "1% above the committed absolute reference: above the 7.3e-4 "
            "cross-substrate floor and below the 1.65% price of a 10 mm sensor "
            "displacement"
        )
    else:
        residual_budget = SOURCE_CUTOVER_RESIDUAL_BUDGET
        basis = (
            "7.93% source-cutover class: 2.85% measured column propagation plus "
            "3.37% solve feedback and 1.71% stated-weight calibration asymmetry"
        )
    tolerances[ScorecardField.MAGNETICS_RESIDUAL_WHITENED_RMS] = MetricTolerance(
        field=ScorecardField.MAGNETICS_RESIDUAL_WHITENED_RMS,
        bound=MAGNETICS_REFERENCE * (1.0 + residual_budget),
        direction=BoundDirection.AT_MOST,
        unit="whitened RMS",
        basis=basis,
        evidence=(
            f"{GEOMETRY_REFERENCE_IDENTITY}; "
            "docs/research/mast-cutover-parity-evidence.html"
        ),
    )
    if set(tolerances) != SCORECARD_FIELDS:
        raise RuntimeError("the parity tolerance registry is incomplete")
    return MappingProxyType(tolerances)


PARITY_TOLERANCES = registered_tolerances()

# Stored-equilibrium comparisons use physical distances and flux-span fractions.
IMAS_COMPARISON_BANDS = MappingProxyType(
    {
        "magnetic_axis_distance_m": 0.20,
        "plasma_current_relative_error": 0.015,
        "axis_flux_span_error": 0.05,
        "boundary_flux_span_error": 0.05,
        "q95_magnitude_error": 2.0,
    }
)


def validate_scorecard_fields(fields: Iterable[str]) -> None:
    """Reject a scorecard unless its fields exactly match the registry."""
    observed = set(fields)
    missing = SCORECARD_FIELDS - observed
    unknown = observed - SCORECARD_FIELDS
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing registered fields: {sorted(missing)}")
        if unknown:
            parts.append(f"unregistered fields: {sorted(unknown)}")
        raise ValueError("; ".join(parts))


def scorecard_verdicts(
    scorecard: Mapping[str, float],
    magnetics_budget: MagneticsBudgetClass = MagneticsBudgetClass.SAME_SOURCE,
) -> Mapping[str, bool]:
    """Score every registered field after enforcing exact schema coverage."""
    validate_scorecard_fields(scorecard)
    tolerances = registered_tolerances(magnetics_budget)
    return MappingProxyType(
        {
            field: tolerance.passes(float(scorecard[field]))
            for field, tolerance in tolerances.items()
        }
    )


class FieldPolarity(enum.StrEnum):
    """The two coupled plasma-current and toroidal-field sign cohorts."""

    FORWARD = "Ip>0,Bt<0"
    REVERSED = "Ip<0,Bt>0"


@dataclass(frozen=True)
class HeldOutCandidate:
    """A store-verified shot in one campaign and field-polarity stratum."""

    shot_id: int
    campaign: str
    field_polarity: FieldPolarity


CAMPAIGN_EARLY_HIGH_FILAMENT = "mp78-fl46-fc1004-lim37-9425ae4a8bf3bc15"
CAMPAIGN_LATE_HIGH_FILAMENT = "mp78-fl46-fc1004-lim37-edd753d282903679"
CAMPAIGN_MAIN = "mp78-fl46-fc938-lim37-1cb6f2ee742c4ee4"

HELD_OUT_DRAW_SEED = 20260812
HELD_OUT_CANDIDATE_STRATA: tuple[tuple[HeldOutCandidate, ...], ...] = (
    tuple(
        HeldOutCandidate(shot, CAMPAIGN_EARLY_HIGH_FILAMENT, FieldPolarity.FORWARD)
        for shot in (11794, 11920, 12200, 12400)
    ),
    tuple(
        HeldOutCandidate(shot, CAMPAIGN_LATE_HIGH_FILAMENT, FieldPolarity.FORWARD)
        for shot in (12417, 12434, 12450, 12470)
    ),
    tuple(
        HeldOutCandidate(shot, CAMPAIGN_MAIN, FieldPolarity.FORWARD)
        for shot in (18502, 25000, 27000, 29879)
    ),
    tuple(
        HeldOutCandidate(shot, CAMPAIGN_MAIN, FieldPolarity.REVERSED)
        for shot in (13469, 13550, 22376, 22500)
    ),
)


def draw_held_out_extension(
    seed: int = HELD_OUT_DRAW_SEED,
) -> tuple[HeldOutCandidate, ...]:
    """Draw one shot from each recorded campaign/polarity stratum."""
    generator = random.Random(seed)
    return tuple(
        stratum[generator.randrange(len(stratum))]
        for stratum in HELD_OUT_CANDIDATE_STRATA
    )


HELD_OUT_EXTENSION = draw_held_out_extension()


STATE_RECEIPT = (
    "docs/figures/inverse-equilibrium-baseline/efit-on-nova-rows/receipt.json"
)

STATE_SCOREBOARD_FIELDS = (
    ScorecardField.MAGNETIC_AXIS_DISTANCE_M.value,
    "plasma_current_fraction",
    "axis_flux_fraction",
    "boundary_flux_fraction",
    "q95_difference",
)

REFERENCE_FREE_FIELDS = ("boundary_area_fraction",)


def _state_scoreboard_tolerances() -> dict[str, MetricTolerance]:
    """Return the bands one reconstructed state is judged against a reference by.

    The four bands are the ones recorded on the row-exchange case, read from the
    receipt committed beside its panels; the axis band is a metre distance and
    every flux band is a fraction of the reference's axis-to-boundary span.
    """

    return {
        ScorecardField.MAGNETIC_AXIS_DISTANCE_M.value: MetricTolerance(
            field=ScorecardField.MAGNETIC_AXIS_DISTANCE_M.value,
            bound=0.02,
            direction=BoundDirection.AT_MOST,
            unit="m",
            basis="the recorded row-exchange band on the magnetic axis position",
            evidence=STATE_RECEIPT,
        ),
        "plasma_current_fraction": MetricTolerance(
            field="plasma_current_fraction",
            bound=0.015,
            direction=BoundDirection.AT_MOST,
            unit="fraction of the reference current",
            basis="the recorded row-exchange band on the plasma current",
            evidence=STATE_RECEIPT,
        ),
        "axis_flux_fraction": MetricTolerance(
            field="axis_flux_fraction",
            bound=0.02,
            direction=BoundDirection.AT_MOST,
            unit="fraction of the reference flux span",
            basis="the recorded row-exchange band on the axis flux",
            evidence=STATE_RECEIPT,
        ),
        "boundary_flux_fraction": MetricTolerance(
            field="boundary_flux_fraction",
            bound=0.02,
            direction=BoundDirection.AT_MOST,
            unit="fraction of the reference flux span, after gauge alignment",
            basis="the recorded row-exchange band on the boundary flux",
            evidence=STATE_RECEIPT,
        ),
        "q95_difference": MetricTolerance(
            field="q95_difference",
            bound=0.1,
            direction=BoundDirection.AT_MOST,
            unit="magnitude",
            basis=(
                "the recorded row-exchange band on q95 read from both states' own "
                "profiles, magnitudes compared with each stored sign reported"
            ),
            evidence=STATE_RECEIPT,
        ),
    }


STATE_SCOREBOARD_TOLERANCES = MappingProxyType(_state_scoreboard_tolerances())

REFERENCE_FREE_TOLERANCES = MappingProxyType(
    {
        "boundary_area_fraction": MetricTolerance(
            field="boundary_area_fraction",
            bound=0.5,
            direction=BoundDirection.AT_LEAST,
            unit="fraction of the boundary's bounding box",
            basis=(
                "a closed outline that fails to cover half its own bounding box is "
                "not a plasma boundary"
            ),
            evidence="nova/imas/mast_efit_referee.py",
        )
    }
)


def state_scoreboard_verdicts(metrics: Mapping[str, float]) -> Mapping[str, bool]:
    """Return one verdict per registered state-scoreboard metric."""

    return MappingProxyType(
        {
            field: STATE_SCOREBOARD_TOLERANCES[field].passes(float(metrics[field]))
            for field in STATE_SCOREBOARD_FIELDS
        }
    )


def reference_free_verdicts(
    checks: Mapping[str, float | bool | None],
) -> Mapping[str, bool]:
    """Return one verdict per reference-free check, refusing an absent measure."""

    verdicts: dict[str, bool] = {}
    for field in REFERENCE_FREE_FIELDS:
        value = checks.get(field)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(f"reference-free check {field!r} carries no measure")
        verdicts[field] = REFERENCE_FREE_TOLERANCES[field].passes(float(value))
    for field in ("axis_inside_wall", "axis_inside_boundary"):
        value = checks.get(field)
        if value is None:
            raise ValueError(f"reference-free check {field!r} carries no measure")
        verdicts[field] = bool(value)
    return MappingProxyType(verdicts)
