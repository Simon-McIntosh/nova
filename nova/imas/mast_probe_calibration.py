"""Fit each probe's own gain and orientation, and apply the fixed discriminant.

:mod:`nova.imas.mast_probe_discriminant` fixes what separates a probe's
calibration error from a misdescribed field, and this module measures the five
statistics that criterion is a function of.  Nothing here may choose a threshold;
the thresholds arrive from there and are written beside the results so a reader
can check the run applied the criterion it cites.

The fit is per probe and per excitation family, which is what makes the
discriminant possible at all.  Holding the family fixed, the model is one number:

    y(t) = gain * sum_c G(p, c) w_c I_c(t)

with ``G`` the geometry-derived response, ``w`` the drive weights the description
already carries, and ``I`` the measured currents.  A probe-side error makes
``gain`` the same whichever family is driving.  A misrepresented current
arrangement inside a neighbouring pack does not, because how wrong a uniform
density is falls off with distance from the pack.

The orientation term is fitted against a *measurement* rather than against the
model.  Each outboard axial probe shares its position with an outboard radial
probe, so

    y_axial(t) = a * B_z_model(t) + b * y_radial(t)

recovers ``gain = hypot(a, b)`` and ``tilt = atan2(b, a)`` while reading the
radial component off the instrument that measures it.  That matters: the radial
field beside P5 is half again the axial field there, so a model that gets the
winding arrangement wrong gets the radial component wrong too, and a tilt fitted
against the model would absorb the very error it is meant to be distinguished
from.

One shot is one measurement.  A waveform's samples are not independent -- ten
thousand of them from one pulse carry roughly one pulse worth of information --
so every quantity is estimated per shot first and pooled across shots, and its
standard error is the scatter of the per-shot values.  Estimating it from the
sample count instead would shrink every error bar by two orders of magnitude and
make every probe look excitation-selective, which is to say it would make the
discriminant always return the same answer.

The near-field screen the turn fits rely on is deliberately *not* applied here.
That screen exists to stop a probe standing beside an excited pack from biasing
that pack's turn count, and the two probes under test stand about one pack width
from P5.  Excluding them would remove the only data that can answer the
question, so the standoff is recorded beside every per-family gain and becomes
the explanatory variable instead of a cut.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from nova.imas.mast_probe_discriminant import (
    CO_LOCATION_TOLERANCE,
    FAMILY_LEVERAGE,
    DiscriminantStatistics,
    PreRegistration,
    ProbeVerdict,
    adjudicate,
    promotable,
)
from nova.imas.mast_vacuum_cohort import (
    CohortError,
    ShotWaveforms,
    parse_probe_channel,
)
from nova.imas.mast_vacuum_response import (
    MINIMUM_STANDOFF,
    ResponseModel,
)

MINIMUM_SAMPLES = 200
"""Admitted samples a shot must give a probe for its gain to be estimated."""

MINIMUM_SHOTS = 3
"""Shots a probe-family gain needs before it carries a standard error."""

MAXIMUM_COLLINEARITY = 0.95
"""Correlation between the two rigid-fit columns that still separates them.

The same threshold the coil response uses to refuse a series-wired pair, and for
the same reason: two columns this nearly proportional are one column, and a solve
handed them returns a pair of numbers whose difference the data never
constrained.
"""

MAXIMUM_CONDITION = 1.0e4
"""Condition number past which a pooled scale-and-rotation solve is not read.

A well-separated pair sits near unity; a cohort that only ever drove one coil at
a probe pushes it to infinity.  Four orders is where the smaller singular value
stops carrying more signal than double precision noise on the larger one.
"""

MINIMUM_SHAPE_AGREEMENT = 0.5
"""Share of a probe's variance one coil's modelled shape must explain.

A gain is a scale, and a scale only means something if the thing being scaled has
the right shape.  Where the best scale still leaves more than half the probe's
variance, what the fit returned is a projection of one waveform onto a different
one -- reporting it as that coil's gain would put a shape error into a number
read as an amplitude.
"""


class CalibrationError(CohortError):
    """Raised when a probe's calibration cannot be estimated or adjudicated."""


@dataclass(frozen=True, order=True)
class OrthogonalPair:
    """Two probes at one point measuring different poloidal components."""

    channel: str
    partner: str
    separation: float
    r: float
    z: float

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channel": self.channel,
            "partner": self.partner,
            "r": self.r,
            "separation": self.separation,
            "z": self.z,
        }


def orthogonal_pairs(
    model: ResponseModel,
    *,
    tolerance: float = CO_LOCATION_TOLERANCE,
) -> tuple[OrthogonalPair, ...]:
    """Pair every probe with one at the same point on the other axis.

    Pairing is by position and by sensitive axis, never by channel number: the
    numbering happens to line the two outboard arrays up, and a pair built on
    that coincidence would silently survive a renumbering while measuring two
    different places.
    """

    targets = model.targets
    pairs = []
    for probe in targets:
        candidates = [
            (math.hypot(probe.r - other.r, probe.z - other.z), other)
            for other in targets
            if other.channel != probe.channel
            and abs(probe.radial_cosine - other.radial_cosine) > 0.5
        ]
        if not candidates:
            continue
        separation, closest = min(candidates, key=lambda row: row[0])
        if separation > tolerance:
            continue
        pairs.append(
            OrthogonalPair(
                channel=probe.channel,
                partner=closest.channel,
                separation=separation,
                r=probe.r,
                z=probe.z,
            )
        )
    return tuple(pairs)


@dataclass(frozen=True)
class ShotGain:
    """One probe's scale on one shot, and which family it answers for."""

    shot: int
    channel: str
    family: str
    gain: float
    leverage: float
    sample_count: int
    residual: float
    signal: float
    shape_agreement: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channel": self.channel,
            "family": self.family,
            "gain": self.gain,
            "leverage": self.leverage,
            "residual": self.residual,
            "sample_count": self.sample_count,
            "shape_agreement": self.shape_agreement,
            "shot": self.shot,
            "signal": self.signal,
        }


@dataclass(frozen=True)
class ShotRigidFit:
    """One probe's joint scale and rotation on one shot.

    ``collinearity`` is the correlation between the two columns the fit solves
    over -- the modelled axial field and the measured radial one.  On a shot that
    drove a single coil the two carry the same waveform, so the correlation goes
    to one and the pair is not separable however clean the data is: any scale can
    be traded against any angle for the same prediction.  The fit is still
    reported, because its residual is a valid statement about the shot, but a
    ``gain`` or ``tilt`` read off a collinear shot is meaningless and
    :func:`pool_rigid_systems` is what the two parameters must come from.
    """

    shot: int
    channel: str
    partner: str
    gain: float
    tilt: float
    residual: float
    signal: float
    sample_count: int
    tilt_residual: float
    collinearity: float = 0.0

    @property
    def separable(self) -> bool:
        """Return whether this shot alone can divide scale from rotation."""

        return abs(self.collinearity) < MAXIMUM_COLLINEARITY

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channel": self.channel,
            "collinearity": self.collinearity,
            "gain": self.gain,
            "partner": self.partner,
            "residual": self.residual,
            "sample_count": self.sample_count,
            "separable": self.separable,
            "shot": self.shot,
            "signal": self.signal,
            "tilt": self.tilt,
            "tilt_residual": self.tilt_residual,
        }


@dataclass(frozen=True)
class ShotRigidSystem:
    """One shot's contribution to a probe's pooled scale-and-rotation solve.

    Carried as normal equations rather than as a fitted pair because a single
    shot usually cannot separate the two parameters and a pooled solve can: each
    coil presents the probe with a different ratio of radial to axial field, so
    the columns that are collinear within one shot stop being collinear once
    shots driving different coils are added together.
    """

    shot: int
    channel: str
    partner: str
    gram: tuple[float, float, float]
    moment: tuple[float, float]
    observed_square: float
    sample_count: int

    @property
    def matrix(self) -> np.ndarray:
        """Return the symmetric two-by-two normal matrix."""

        first, cross, second = self.gram
        return np.asarray([[first, cross], [cross, second]], dtype=float)


@dataclass(frozen=True, order=True)
class PooledRigidFit:
    """A probe's scale and rotation solved over every shot at once.

    ``condition`` is the normal matrix's condition number, which is the honest
    statement of whether the cohort separated the two parameters.  ``error`` is a
    jackknife over shots: each shot is left out in turn and the spread of the
    resulting pairs is what a standard error means when the samples inside a shot
    are correlated.
    """

    channel: str
    partner: str
    shot_count: int
    gain: float
    tilt: float
    gain_error: float
    tilt_error: float
    condition: float
    residual: float
    signal: float
    tilt_variance_removed: float

    @property
    def identified(self) -> bool:
        """Return whether the pooled solve separated scale from rotation."""

        return (
            self.shot_count >= MINIMUM_SHOTS
            and math.isfinite(self.condition)
            and self.condition < MAXIMUM_CONDITION
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channel": self.channel,
            "condition": None if math.isinf(self.condition) else self.condition,
            "gain": self.gain,
            "gain_error": self.gain_error,
            "identified": self.identified,
            "partner": self.partner,
            "residual": self.residual,
            "shot_count": self.shot_count,
            "signal": self.signal,
            "tilt": self.tilt,
            "tilt_error": self.tilt_error,
            "tilt_variance_removed": self.tilt_variance_removed,
        }


def _solve_rigid(matrix: np.ndarray, moment: np.ndarray) -> tuple[float, float] | None:
    try:
        solution = np.linalg.solve(matrix, moment)
    except np.linalg.LinAlgError:
        return None
    return float(solution[0]), float(solution[1])


def pool_rigid_systems(
    systems: Iterable[ShotRigidSystem],
) -> tuple[PooledRigidFit, ...]:
    """Solve each probe's scale and rotation over every shot that constrains it."""

    grouped: dict[str, list[ShotRigidSystem]] = {}
    for row in systems:
        grouped.setdefault(row.channel, []).append(row)

    fits = []
    for channel, rows in sorted(grouped.items()):
        matrix = sum((row.matrix for row in rows), np.zeros((2, 2)))
        moment = np.asarray(
            [sum(row.moment[0] for row in rows), sum(row.moment[1] for row in rows)],
            dtype=float,
        )
        observed = float(sum(row.observed_square for row in rows))
        samples = int(sum(row.sample_count for row in rows))
        solution = _solve_rigid(matrix, moment)
        if solution is None or samples == 0:
            continue
        axial, cross = solution
        coefficients = np.asarray(solution, dtype=float)
        residual_square = max(
            0.0,
            observed
            - 2.0 * float(coefficients @ moment)
            + float(coefficients @ matrix @ coefficients),
        )
        cross_only = moment[1] / matrix[1, 1] if matrix[1, 1] > 0.0 else 0.0
        tilt_square = max(
            0.0,
            observed - 2.0 * cross_only * moment[1] + cross_only**2 * matrix[1, 1],
        )
        removed = (
            0.0
            if observed <= 0.0
            else float(max(0.0, 1.0 - residual_square / observed))
        )
        jackknife = []
        if len(rows) > 1:
            for index in range(len(rows)):
                kept = rows[:index] + rows[index + 1 :]
                partial = sum((row.matrix for row in kept), np.zeros((2, 2)))
                vector = np.asarray(
                    [
                        sum(row.moment[0] for row in kept),
                        sum(row.moment[1] for row in kept),
                    ],
                    dtype=float,
                )
                leave = _solve_rigid(partial, vector)
                if leave is not None:
                    jackknife.append(
                        (math.hypot(*leave), math.atan2(leave[1], leave[0]))
                    )
        gains = np.asarray([row[0] for row in jackknife], dtype=float)
        tilts = np.asarray([row[1] for row in jackknife], dtype=float)
        scale = math.sqrt(max(0, len(rows) - 1)) if len(rows) > 1 else 0.0
        fits.append(
            PooledRigidFit(
                channel=channel,
                partner=rows[0].partner,
                shot_count=len(rows),
                gain=float(math.hypot(axial, cross)),
                tilt=float(math.atan2(cross, axial)),
                gain_error=float(gains.std() * scale) if gains.size else math.inf,
                tilt_error=float(tilts.std() * scale) if tilts.size else math.inf,
                condition=float(np.linalg.cond(matrix)),
                residual=float(math.sqrt(residual_square / samples)),
                signal=float(math.sqrt(observed / samples)),
                tilt_variance_removed=float(
                    removed
                    if tilt_square <= 0.0 or observed <= 0.0
                    else max(0.0, 1.0 - residual_square / tilt_square)
                ),
            )
        )
    return tuple(fits)


def _centred(signal: np.ndarray, quiet: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(signal)
    if not (finite & quiet).any():
        return None
    return signal - float(np.mean(signal[finite & quiet]))


def shot_gains(
    model: ResponseModel,
    waveforms: ShotWaveforms,
    weights: Mapping[str, float],
    *,
    pairs: Mapping[str, str] | None = None,
    refused_channels: Iterable[str] = (),
    stride: int = 1,
    family_leverage: float = FAMILY_LEVERAGE,
) -> tuple[tuple[ShotGain, ...], tuple[ShotRigidFit, ...], tuple[ShotRigidSystem, ...]]:
    """Estimate one shot's per-family gains and its rigid-fit contribution.

    A family's gain is only recorded where that family produces at least
    ``family_leverage`` of the probe's predicted signal power on this shot AND
    where the best scale on that family's modelled shape explains at least
    :data:`MINIMUM_SHAPE_AGREEMENT` of the probe's variance -- a scale fitted to
    a waveform of the wrong shape is a projection, not a gain.

    The rigid fit is returned twice over.  The per-shot pair is reported with the
    collinearity that says whether this shot could separate its two parameters,
    and the normal equations are returned alongside so the pair a probe is
    actually judged on comes from every shot at once.
    """

    excluded = set(refused_channels)
    quiet = waveforms.baseline_mask
    samples = np.flatnonzero(waveforms.sample_mask)[::stride]
    if samples.size == 0:
        return (), (), ()

    drive = np.zeros((waveforms.time.size, len(model.families)), dtype=float)
    for column, family in enumerate(model.families):
        values = waveforms.drives.get(family)
        if values is not None:
            drive[:, column] = np.nan_to_num(values) * float(weights.get(family, 0.0))

    centred = {}
    for channel, signal in waveforms.probes.items():
        if channel in excluded or signal.shape != waveforms.time.shape:
            continue
        adjusted = _centred(signal, quiet)
        if adjusted is not None:
            centred[channel] = adjusted

    gains: list[ShotGain] = []
    rigid: list[ShotRigidFit] = []
    systems: list[ShotRigidSystem] = []
    for row, target in enumerate(model.targets):
        observed = centred.get(target.channel)
        if observed is None:
            continue
        columns = drive * model.response[row, :]
        prediction = columns.sum(axis=1)
        usable = np.isfinite(observed[samples]) & np.isfinite(prediction[samples])
        keep = samples[usable]
        if keep.size < MINIMUM_SAMPLES:
            continue
        y = observed[keep]
        total = float(np.dot(prediction[keep], prediction[keep]))
        if total <= 0.0:
            continue
        for column, family in enumerate(model.families):
            partial = columns[keep, column]
            power = float(np.dot(partial, partial))
            if power <= 0.0 or power / total < family_leverage:
                continue
            gain = float(np.dot(partial, y) / power)
            residual = y - gain * partial
            observed_square = float(np.dot(y, y))
            if observed_square <= 0.0:
                continue
            explained = 1.0 - float(np.dot(residual, residual)) / observed_square
            if explained < MINIMUM_SHAPE_AGREEMENT:
                continue
            gains.append(
                ShotGain(
                    shot=waveforms.shot,
                    channel=target.channel,
                    family=family,
                    gain=gain,
                    leverage=power / total,
                    sample_count=int(keep.size),
                    residual=float(np.sqrt(np.mean(residual**2))),
                    signal=float(np.sqrt(np.mean(y**2))),
                    shape_agreement=explained,
                )
            )
        partner = (pairs or {}).get(target.channel)
        measured = centred.get(partner) if partner else None
        if measured is None:
            continue
        design = np.column_stack([prediction[keep], measured[keep]])
        if not np.isfinite(design).all():
            continue
        try:
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        axial, cross = (float(coefficients[0]), float(coefficients[1]))
        joint = y - design @ coefficients
        offset = y - prediction[keep]
        cross_power = float(np.dot(measured[keep], measured[keep]))
        if cross_power <= 0.0:
            continue
        tilt_only = float(np.dot(measured[keep], offset) / cross_power)
        first, second = design[:, 0], design[:, 1]
        collinearity = 0.0
        if np.std(first) > 0.0 and np.std(second) > 0.0:
            collinearity = float(np.corrcoef(first, second)[0, 1])
        rigid.append(
            ShotRigidFit(
                shot=waveforms.shot,
                channel=target.channel,
                partner=partner,
                gain=float(math.hypot(axial, cross)),
                tilt=float(math.atan2(cross, axial)),
                residual=float(np.sqrt(np.mean(joint**2))),
                signal=float(np.sqrt(np.mean(y**2))),
                sample_count=int(keep.size),
                tilt_residual=float(
                    np.sqrt(np.mean((offset - tilt_only * measured[keep]) ** 2))
                ),
                collinearity=collinearity,
            )
        )
        systems.append(
            ShotRigidSystem(
                shot=waveforms.shot,
                channel=target.channel,
                partner=partner,
                gram=(
                    float(np.dot(first, first)),
                    float(np.dot(first, second)),
                    float(np.dot(second, second)),
                ),
                moment=(float(np.dot(first, y)), float(np.dot(second, y))),
                observed_square=float(np.dot(y, y)),
                sample_count=int(keep.size),
            )
        )
    return tuple(gains), tuple(rigid), tuple(systems)


def score_rigid_correction(
    model: ResponseModel,
    waveforms: ShotWaveforms,
    weights: Mapping[str, float],
    coefficients: Mapping[str, tuple[float, float]],
    *,
    pairs: Mapping[str, str] | None = None,
    refused_channels: Iterable[str] = (),
    stride: int = 1,
) -> dict[str, tuple[float, float]]:
    """Score a supplied correction on one shot the correction never saw.

    Returns, per probe channel, the residual the description gives as it stands
    and the residual the supplied scale and rotation give.  The coefficients are
    the ``(gain, tilt)`` pair a fit produced elsewhere, so the two numbers are a
    prediction and not a fit -- which is the only form in which a calibration can
    be challenged.
    """

    excluded = set(refused_channels)
    quiet = waveforms.baseline_mask
    samples = np.flatnonzero(waveforms.sample_mask)[::stride]
    if samples.size == 0:
        return {}
    drive = np.zeros((waveforms.time.size, len(model.families)), dtype=float)
    for column, family in enumerate(model.families):
        values = waveforms.drives.get(family)
        if values is not None:
            drive[:, column] = np.nan_to_num(values) * float(weights.get(family, 0.0))
    centred = {}
    for channel, signal in waveforms.probes.items():
        if channel in excluded or signal.shape != waveforms.time.shape:
            continue
        adjusted = _centred(signal, quiet)
        if adjusted is not None:
            centred[channel] = adjusted

    scores: dict[str, tuple[float, float]] = {}
    for row, target in enumerate(model.targets):
        pair = (coefficients.get(target.channel), (pairs or {}).get(target.channel))
        correction, partner = pair
        observed = centred.get(target.channel)
        measured = centred.get(partner) if partner else None
        if correction is None or observed is None or measured is None:
            continue
        prediction = (drive * model.response[row, :]).sum(axis=1)
        usable = (
            np.isfinite(observed[samples])
            & np.isfinite(prediction[samples])
            & np.isfinite(measured[samples])
        )
        keep = samples[usable]
        if keep.size < MINIMUM_SAMPLES:
            continue
        gain, tilt = correction
        reference = observed[keep] - prediction[keep]
        corrected = observed[keep] - (
            gain * math.cos(tilt) * prediction[keep]
            + gain * math.sin(tilt) * measured[keep]
        )
        scores[target.channel] = (
            float(np.sqrt(np.mean(corrected**2))),
            float(np.sqrt(np.mean(reference**2))),
        )
    return scores


def pooled_corrections(
    rigid: Iterable[PooledRigidFit],
) -> dict[str, tuple[float, float]]:
    """Return the scale and rotation each conditioned pooled solve produced.

    A probe whose solve was not conditioned is left out, so the held-out
    challenge is never asked to score a correction the cohort did not determine.
    """

    return {
        row.channel: (row.gain, row.tilt) for row in sorted(rigid) if row.identified
    }


def pooled_held_out(
    scores: Iterable[Mapping[str, tuple[float, float]]],
) -> dict[str, tuple[float, float]]:
    """Pool per-shot held-out scores into one pair per channel."""

    corrected: dict[str, list[float]] = {}
    reference: dict[str, list[float]] = {}
    for row in scores:
        for channel, (fitted, plain) in row.items():
            corrected.setdefault(channel, []).append(fitted)
            reference.setdefault(channel, []).append(plain)
    return {
        channel: (
            float(np.mean(corrected[channel])),
            float(np.mean(reference[channel])),
        )
        for channel in sorted(corrected)
    }


@dataclass(frozen=True, order=True)
class FamilyGain:
    """One probe's scale for one coil family, pooled over shots."""

    channel: str
    family: str
    shot_count: int
    gain: float
    standard_error: float
    standoff: float
    leverage: float

    @property
    def identified(self) -> bool:
        """Return whether enough shots back the scale to bound it."""

        return self.shot_count >= MINIMUM_SHOTS

    @property
    def near_field(self) -> bool:
        """Return whether the probe stands inside this coil's near field.

        The boundary is the one the turn fits were measured to need, so near and
        far here mean the same thing they mean there rather than being a fresh
        judgement.
        """

        return self.standoff < MINIMUM_STANDOFF

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channel": self.channel,
            "family": self.family,
            "gain": self.gain,
            "identified": self.identified,
            "leverage": self.leverage,
            "near_field": self.near_field,
            "shot_count": self.shot_count,
            "standard_error": self.standard_error,
            "standoff": self.standoff,
        }


def _pooled(values: Sequence[float]) -> tuple[float, float]:
    """Return the mean of per-shot values and the standard error of that mean."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return math.nan, math.inf
    if array.size == 1:
        return float(array[0]), math.inf
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(array.size))


def aggregate_family_gains(
    gains: Iterable[ShotGain],
    standoff: Mapping[tuple[str, str], float],
) -> tuple[FamilyGain, ...]:
    """Pool per-shot gains into one scale per probe and family."""

    grouped: dict[tuple[str, str], list[ShotGain]] = {}
    for row in gains:
        grouped.setdefault((row.channel, row.family), []).append(row)
    result = []
    for (channel, family), rows in sorted(grouped.items()):
        gain, error = _pooled([row.gain for row in rows])
        result.append(
            FamilyGain(
                channel=channel,
                family=family,
                shot_count=len(rows),
                gain=gain,
                standard_error=error,
                standoff=float(standoff.get((channel, family), math.nan)),
                leverage=float(np.mean([row.leverage for row in rows])),
            )
        )
    return tuple(result)


def build_statistics(
    channel: str,
    family_gains: Sequence[FamilyGain],
    rigid: PooledRigidFit | None,
    *,
    noise_floor: float,
    partner_excess_share: float = 0.0,
) -> DiscriminantStatistics:
    """Assemble one probe's five discriminant statistics.

    The near and distant gains are means over the families on each side of the
    standoff boundary, so a probe with no family on one side reports that side as
    the other -- which makes its ``near_field_contrast`` zero rather than
    inventing a difference from an absent measurement.

    The rotation comes from the pooled solve and only when that solve was
    conditioned.  An unconditioned one reports no angle at all rather than the
    number a degenerate system happened to return, which would otherwise arrive
    at the criterion looking like a measured tilt of a hundred degrees.
    """

    identified = [row for row in family_gains if row.identified]
    gains = [row.gain for row in identified]
    errors = [
        row.standard_error for row in identified if math.isfinite(row.standard_error)
    ]
    near = [row.gain for row in identified if row.near_field]
    distant = [row.gain for row in identified if not row.near_field]
    pooled_error = (
        float(math.sqrt(float(np.mean(np.square(errors))))) if errors else math.inf
    )
    overall = float(np.mean(gains)) if gains else math.nan
    usable = rigid is not None and rigid.identified
    tilt = rigid.tilt if usable else 0.0
    tilt_error = rigid.tilt_error if usable else 0.0
    removed = rigid.tilt_variance_removed if usable else 0.0
    rigid_residual = rigid.residual if rigid is not None else 0.0
    partner = rigid.partner if rigid is not None else ""
    return DiscriminantStatistics(
        channel=channel,
        family_count=len(identified),
        gain=overall if math.isfinite(overall) else 1.0,
        gain_standard_error=pooled_error if math.isfinite(pooled_error) else 0.0,
        gain_spread=float(max(gains) - min(gains)) if len(gains) > 1 else 0.0,
        near_coil_gain=float(np.mean(near)) if near else (overall if gains else 1.0),
        distant_coil_gain=(
            float(np.mean(distant)) if distant else (overall if gains else 1.0)
        ),
        tilt=tilt if math.isfinite(tilt) else 0.0,
        tilt_standard_error=tilt_error if math.isfinite(tilt_error) else 0.0,
        tilt_variance_removed=removed,
        partner_channel=partner,
        partner_excess_share=partner_excess_share,
        rigid_residual=rigid_residual,
        noise_floor=noise_floor,
    )


@dataclass(frozen=True, order=True)
class ProbeAdjudication:
    """One probe's verdict, the statistics behind it, and what it licenses."""

    channel: str
    verdict: ProbeVerdict
    statistics: DiscriminantStatistics
    family_gains: tuple[FamilyGain, ...]
    held_out_residual: float = math.nan
    held_out_reference: float = math.nan
    rigid: PooledRigidFit | None = None

    @property
    def promoted(self) -> bool:
        """Return whether this verdict writes a value into the description."""

        return promotable(self.verdict) and self.improves_held_out

    @property
    def improves_held_out(self) -> bool:
        """Return whether the correction predicts unseen shots better.

        A calibration that fits the shots it was measured on and does not
        improve the ones it never saw is a description of those shots, not of the
        probe.  A probe with no held-out coverage cannot claim the improvement
        and is not promoted.
        """

        if not math.isfinite(self.held_out_residual):
            return False
        if not math.isfinite(self.held_out_reference):
            return False
        return self.held_out_residual < self.held_out_reference

    @property
    def interval(self) -> tuple[float, float]:
        """Return the promoted quantity's interval at two standard errors."""

        if self.verdict is ProbeVerdict.CALIBRATION_TILT:
            centre, error = self.statistics.tilt, self.statistics.tilt_standard_error
        else:
            centre, error = self.statistics.gain, self.statistics.gain_standard_error
        if not math.isfinite(error):
            return (centre, centre)
        return (centre - 2.0 * error, centre + 2.0 * error)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        lower, upper = self.interval
        return {
            "channel": self.channel,
            "family_gains": [row.as_dict() for row in self.family_gains],
            "held_out_reference": (
                None
                if not math.isfinite(self.held_out_reference)
                else self.held_out_reference
            ),
            "held_out_residual": (
                None
                if not math.isfinite(self.held_out_residual)
                else self.held_out_residual
            ),
            "improves_held_out": self.improves_held_out,
            "interval": [lower, upper],
            "promoted": self.promoted,
            "rigid": None if self.rigid is None else self.rigid.as_dict(),
            "statistics": self.statistics.as_dict(),
            "verdict": str(self.verdict),
        }


def adjudicate_probes(
    family_gains: Iterable[FamilyGain],
    rigid: Iterable[PooledRigidFit],
    noise_floor: Mapping[str, float],
    *,
    excess_share: Mapping[str, float] | None = None,
    held_out: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[ProbeAdjudication, ...]:
    """Adjudicate every probe the fit produced statistics for."""

    by_channel: dict[str, list[FamilyGain]] = {}
    for row in family_gains:
        by_channel.setdefault(row.channel, []).append(row)
    fits = {row.channel: row for row in rigid}
    shares = dict(excess_share or {})
    scores = dict(held_out or {})

    result = []
    for channel in sorted(set(by_channel) | set(fits)):
        floor = noise_floor.get(channel)
        if floor is None or floor <= 0.0:
            continue
        fit = fits.get(channel)
        statistics = build_statistics(
            channel,
            by_channel.get(channel, ()),
            fit,
            noise_floor=float(floor),
            partner_excess_share=float(
                shares.get(fit.partner, 0.0) if fit is not None else 0.0
            ),
        )
        residual, reference = scores.get(channel, (math.nan, math.nan))
        result.append(
            ProbeAdjudication(
                channel=channel,
                verdict=adjudicate(statistics),
                statistics=statistics,
                family_gains=tuple(sorted(by_channel.get(channel, ()))),
                held_out_residual=residual,
                held_out_reference=reference,
                rigid=fit,
            )
        )
    return tuple(result)


def standoff_table(model: ResponseModel) -> dict[tuple[str, str], float]:
    """Return every probe's standoff from every coil, in that coil's pack widths."""

    return {
        (target.channel, family): float(model.standoff[row, column])
        for row, target in enumerate(model.targets)
        for column, family in enumerate(model.families)
    }


def verdict_counts(
    adjudications: Iterable[ProbeAdjudication],
) -> dict[str, int]:
    """Count probes by verdict, every verdict present as a key."""

    counts = {str(verdict): 0 for verdict in ProbeVerdict}
    for row in adjudications:
        counts[str(row.verdict)] += 1
    return counts


def calibration_record(
    adjudications: Sequence[ProbeAdjudication],
    pairs: Sequence[OrthogonalPair],
    *,
    training_shots: Sequence[int],
    held_out_shots: Sequence[int],
    refused_shots: Sequence[int],
) -> dict[str, Any]:
    """Assemble the run's record, criterion included.

    The pre-registration travels with the results so the criterion a run claims
    can be compared with the one it applied, rather than being taken on trust
    from a commit ordering.
    """

    unpaired = sorted(
        {row.channel for row in adjudications} - {pair.channel for pair in pairs}
    )
    return {
        "adjudications": [row.as_dict() for row in adjudications],
        "held_out_shots": list(held_out_shots),
        "orthogonal_pairs": [pair.as_dict() for pair in pairs],
        "pre_registration": PreRegistration().as_dict(),
        "promoted": sorted(row.channel for row in adjudications if row.promoted),
        "refused_shots": list(refused_shots),
        "training_shots": list(training_shots),
        "unpaired_channels": unpaired,
        "verdict_counts": verdict_counts(adjudications),
    }


def probe_family(channel: str) -> str:
    """Return the array a probe channel belongs to."""

    family, _ = parse_probe_channel(channel)
    return family
