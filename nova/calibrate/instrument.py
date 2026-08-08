"""Measure what a channel reads where the field is zero, and whether it comes back.

An instrument-quiet window has no field in it by construction, so whatever the
channel reports there is the instrument: a standing offset, and the walk of an
integrator whose zero moves.  Both are measurable on every pulse the machine ever
fired, which is what makes them worth taking from the archive rather than from the
few shots somebody designed for the purpose.

The walk is fitted as a quadratic, not a line.  An integrator's zero moves under a
slowly varying input offset, so its output carries a second-order term as well as a
first, and fitting only a line reports the average of the two over the window --
a number that changes when the window moves and describes neither term.  The
curvature also settles a question a line cannot: whether a channel that fails to
return to its extrapolated baseline is defective or merely curved.

Where the fit is taken from and where it is reported are deliberately different
instants.  A quadratic fitted about its own window centre is well conditioned; the
same quadratic expressed about an origin many window-widths away is the same
polynomial with coefficients that are large, opposite in sign, and cancelling.  The
fit therefore runs in the window's own centred coordinate and is shifted to the
caller's reference exactly, which costs nothing and keeps the conditioning of the
short window rather than of the long lever arm.  The reference matters because the
read path needs one: a drift rate is a slope, and a slope without the instant it is
taken at removes a different ramp than the one that was measured.

The closure test is what the two windows are for.  A channel whose pre-pulse walk,
extrapolated across the pulse, lands on its post-pulse walk shed everything it
accumulated; one that lands elsewhere did not, and the miss is the part of a flux
measurement that no per-window offset subtraction removes.  Two extrapolations are
carried rather than one, because a genuinely curved walk misses a linear
extrapolation by half its curvature times the gap squared -- a miss that grows with
the pulse length and looks exactly like accumulated integrator error to a consumer
who only has the linear number.

What the defect is scored against matters as much as the defect.  The two windows'
combined sample scatter is the obvious yardstick and it is not sufficient on its
own: a prediction carried across the pulse also inherits the leading window's rate
error multiplied by the gap, and a pre-pulse window of a few tens of milliseconds
extrapolated across more than a second misses its own target by far more than the
noise.  Measured on one archive pulse, scoring against scatter alone called
sixty-five of seventy-three channels non-closing on a machine that had done nothing
to them.  The two contributions are added in quadrature and reported separately, so
a channel that needs a longer window is distinguishable from one that needs an
explanation.  Neither is the standard error of a window mean: the samples inside a
window are correlated, so that error shrinks by two orders of magnitude and would
report every channel as defective.

The terms leave here as ``diagnostic-correction-schema`` records of the offset and
drift_rate kinds, which is where a consumer reads them from.  The schema carries no
kind for the curvature -- it declares the drift rate as a slope with an optional
curvature term and gives it no slot -- so the curvature is stated in the drift
record's notes, in its own units, rather than being folded into a slope that is not
it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from nova.calibrate.correction_model import (
    ChannelCorrection,
    CorrectionKind,
    CorrectionStatus,
    Provenance,
    Uncertainty,
    ValidityInterval,
)
from nova.calibrate.gain import MINIMUM_FITS
from nova.calibrate.windows import PulseWindow

MINIMUM_WINDOW_SAMPLES = 32
"""Samples a quiet window must give before its walk is fitted.

Three parameters over-determined by an order of magnitude.  A quadratic through
three samples passes through them exactly and reports a curvature that is the
arrangement of three noise draws, which is worse than no curvature at all because
it comes with the same dataclass as a measured one.
"""

MAXIMUM_CLOSURE_SIGNIFICANCE = 3.0
"""Multiples of the prediction's own uncertainty a defect may reach and still close.

Three of them is the usual bar for calling an excursion something other than what
produced the numbers either side of it.
"""


class InstrumentError(ValueError):
    """Raised when an instrument term cannot be measured from what was supplied."""


@dataclass(frozen=True)
class InstrumentTerms:
    """One channel's offset and integrator walk over one quiet window.

    The walk is ``offset + drift_rate * (t - reference_time) + drift_curvature *
    (t - reference_time)**2 / 2``, so ``drift_rate`` is the slope at the reference
    instant and ``drift_curvature`` is the second derivative, constant across the
    window.  ``scatter`` is what is left once the walk is removed, which is the
    channel's floor rather than its variability.

    ``rate_fit_error`` is the formal standard error of the rate from this window's
    own least squares, and it is not a reproducibility bound: the samples inside a
    window are correlated, so it understates how far the rate would move on the next
    pulse, and pooling across pulses is what measures that.  It is carried for the
    one job it is right for -- propagating the rate's uncertainty across a gap, where
    no across-pulse number exists because the gap belongs to this pulse alone.
    """

    channel: str
    offset: float
    drift_rate: float
    drift_curvature: float
    reference_time: float
    scatter: float
    start: float
    stop: float
    sample_count: int
    rate_fit_error: float = 0.0
    pulse: int | None = None

    @property
    def centre(self) -> float:
        """Return the instant the fit is best determined at."""

        return 0.5 * (self.start + self.stop)

    def level(self, time: float) -> float:
        """Return the walk's value at an instant."""

        elapsed = time - self.reference_time
        return (
            self.offset
            + self.drift_rate * elapsed
            + 0.5 * self.drift_curvature * elapsed**2
        )

    def slope(self, time: float) -> float:
        """Return the walk's rate at an instant."""

        return self.drift_rate + self.drift_curvature * (time - self.reference_time)


@dataclass(frozen=True)
class ClosureDefect:
    """Whether a channel's integrator returned to the baseline it left.

    ``defect`` extrapolates the leading window's walk across the pulse as a
    straight line and ``curved_defect`` carries its curvature along too.  Both are
    reported because the difference between them is the part of the miss the walk's
    own shape explains, and only the remainder is accumulated error.

    ``uncertainty`` is what the defect is scored against, and it is not the sample
    scatter alone.  A prediction carried across the pulse inherits the leading
    window's rate error multiplied by the gap, so a short window extrapolated a long
    way misses its own target by an amount that has nothing to do with the
    instrument.  Both parts are kept separate -- ``scatter`` for the two windows'
    combined floor, ``extrapolation_error`` for the lever arm -- because which of
    them dominates says whether a channel needs a longer window or a real
    explanation.
    """

    channel: str
    defect: float
    curved_defect: float
    gap: float
    scatter: float
    extrapolation_error: float
    uncertainty: float
    significance: float

    @property
    def closes(self) -> bool:
        """Return whether the channel came back inside what its fit can resolve."""

        return self.significance <= MAXIMUM_CLOSURE_SIGNIFICANCE


@dataclass(frozen=True)
class PooledInstrumentTerms:
    """One channel's instrument terms pooled over the windows that measured them.

    The errors are the scatter of the per-window values, because a term estimated
    from correlated samples inside one window has a formal error two orders of
    magnitude below its real reproducibility.
    """

    channel: str
    offset: float
    offset_error: float
    drift_rate: float
    drift_rate_error: float
    drift_curvature: float
    drift_curvature_error: float
    scatter: float
    pulses: tuple[int, ...]
    fit_count: int

    @property
    def identified(self) -> bool:
        """Return whether enough windows back the terms to bound them."""

        return self.fit_count >= MINIMUM_FITS


def fit_instrument_terms(
    time: np.ndarray | Sequence[float],
    signal: np.ndarray | Sequence[float],
    window: PulseWindow,
    *,
    channel: str = "",
    pulse: int | None = None,
    reference_time: float | None = None,
    minimum_samples: int = MINIMUM_WINDOW_SAMPLES,
) -> InstrumentTerms:
    """Fit a channel's offset and integrator walk over one instrument-quiet window.

    ``window`` names one interval, not a set of them: a walk fitted across the union
    of the windows either side of a pulse spans an interval the channel was never
    observed in, and what it would measure there is the closure defect rather than a
    drift rate.

    ``reference_time`` is the instant the offset and the rate are reported at, and
    defaults to the window's own start.  A consumer removing the terms from a whole
    record wants the record's origin instead, because that is the instant its own
    elapsed time is counted from.
    """

    axis = np.asarray(time, dtype=float)
    values = np.asarray(signal, dtype=float)
    if axis.shape != values.shape:
        raise InstrumentError(f"{axis.size} times against {values.size} samples")
    span = window.indices
    clock, level = axis[span], values[span]
    keep = np.isfinite(clock) & np.isfinite(level)
    admitted = int(keep.sum())
    if admitted < minimum_samples:
        raise InstrumentError(
            f"the window offers {admitted} finite samples against a floor of "
            f"{minimum_samples}, which is too few to separate a curvature from the "
            "arrangement of the noise that produced it"
        )
    clock, level = clock[keep], level[keep]
    start, stop = float(clock[0]), float(clock[-1])
    centre = 0.5 * (start + stop)
    local = clock - centre
    coefficients = np.polynomial.polynomial.polyfit(local, level, 2)
    residual = level - np.polynomial.polynomial.polyval(local, coefficients)
    reference = start if reference_time is None else float(reference_time)
    shift = centre - reference
    constant, linear, quadratic = (float(row) for row in coefficients)
    return InstrumentTerms(
        channel=channel,
        offset=constant - linear * shift + quadratic * shift**2,
        drift_rate=linear - 2.0 * quadratic * shift,
        drift_curvature=2.0 * quadratic,
        reference_time=reference,
        scatter=float(np.std(residual)),
        start=start,
        stop=stop,
        sample_count=admitted,
        rate_fit_error=_rate_fit_error(local, residual),
        pulse=pulse,
    )


def _rate_fit_error(local: np.ndarray, residual: np.ndarray) -> float:
    """Return the least-squares standard error of the rate about the window centre.

    Taken about the centre because that is where the fit was formed and where the
    rate is least entangled with the other two coefficients; a caller wanting the
    rate elsewhere carries the curvature's own uncertainty over the lever arm too,
    which is the extrapolation this number exists to price.
    """

    samples = local.size
    if samples <= 3:
        return math.inf
    design = np.column_stack([np.ones(samples), local, local**2])
    try:
        covariance = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return math.inf
    variance = float(residual @ residual) / (samples - 3)
    return float(math.sqrt(max(0.0, variance * float(covariance[1, 1]))))


def closure_defect(
    leading: InstrumentTerms,
    trailing: InstrumentTerms,
) -> ClosureDefect:
    """Ask whether a channel returned to the baseline its pre-pulse walk predicts.

    Both extrapolations start from the leading window's own centre, where its fit is
    best determined, rather than from the reference instant the terms are reported
    at.  The two are the same polynomial, but a straight line taken about a distant
    origin is not the same straight line, and the linear extrapolation is the number
    a consumer without the curvature would form.
    """

    if leading.reference_time != trailing.reference_time:
        raise InstrumentError(
            f"the windows are reported about reference instants "
            f"{leading.reference_time} and {trailing.reference_time}; the linear part "
            "of a quadratic depends on where it is taken, so differencing them would "
            "measure the origins as much as the instrument"
        )
    if leading.channel != trailing.channel:
        raise InstrumentError(
            f"{leading.channel!r} before the pulse against {trailing.channel!r} after "
            "it: a closure defect is one channel's own accumulated error, and the "
            "difference of two channels is a comparison of their calibrations"
        )
    origin, moment = leading.centre, trailing.centre
    gap = moment - origin
    straight = leading.level(origin) + leading.slope(origin) * gap
    curved = straight + 0.5 * leading.drift_curvature * gap**2
    observed = trailing.level(moment)
    defect = observed - straight
    scatter = float(math.hypot(leading.scatter, trailing.scatter))
    lever = abs(leading.rate_fit_error * gap)
    uncertainty = float(math.hypot(scatter, lever))
    if uncertainty <= 0.0:
        significance = 0.0 if defect == 0.0 else math.inf
    else:
        significance = abs(defect) / uncertainty
    return ClosureDefect(
        channel=leading.channel,
        defect=defect,
        curved_defect=observed - curved,
        gap=gap,
        scatter=scatter,
        extrapolation_error=lever,
        uncertainty=uncertainty,
        significance=significance,
    )


def _pool(values: Sequence[float]) -> tuple[float, float]:
    """Return the mean of per-window values and the standard error of that mean."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.inf
    if array.size == 1:
        return float(array[0]), math.inf
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(array.size))


def pool_instrument_terms(
    terms: Iterable[InstrumentTerms],
) -> tuple[PooledInstrumentTerms, ...]:
    """Pool each channel's terms over every quiet window that measured them.

    The mean is unweighted.  Weighting by a window's sample count would let the
    long post-pulse window decide a channel's offset while the short pre-pulse one,
    which is the window with no passive current in it at all, contributed a fraction
    of the answer.
    """

    grouped: dict[str, list[InstrumentTerms]] = {}
    for row in terms:
        grouped.setdefault(row.channel, []).append(row)

    pooled = []
    for channel, rows in sorted(grouped.items()):
        offset, offset_error = _pool([row.offset for row in rows])
        rate, rate_error = _pool([row.drift_rate for row in rows])
        curvature, curvature_error = _pool([row.drift_curvature for row in rows])
        scatters = np.asarray([row.scatter for row in rows], dtype=float)
        pooled.append(
            PooledInstrumentTerms(
                channel=channel,
                offset=offset,
                offset_error=offset_error,
                drift_rate=rate,
                drift_rate_error=rate_error,
                drift_curvature=curvature,
                drift_curvature_error=curvature_error,
                scatter=float(np.sqrt(np.mean(scatters**2))),
                pulses=tuple(
                    sorted({row.pulse for row in rows if row.pulse is not None})
                ),
                fit_count=len(rows),
            )
        )
    return tuple(pooled)


def _bounds(
    row: PooledInstrumentTerms,
    pulse_start: int | None,
    pulse_end: int | None,
) -> ValidityInterval:
    """Return the pulse span a channel's terms are recorded over."""

    first = pulse_start if pulse_start is not None else min(row.pulses, default=None)
    last = pulse_end if pulse_end is not None else max(row.pulses, default=None)
    if first is None and last is None and not row.pulses:
        raise InstrumentError(
            f"{row.channel} carries terms measured on no named pulse and no span was "
            "supplied, so the record would claim to hold everywhere on the strength "
            "of nothing"
        )
    inside = [
        pulse
        for pulse in row.pulses
        if (first is None or pulse >= first) and (last is None or pulse <= last)
    ]
    return ValidityInterval(
        pulse_start=first,
        pulse_end=last,
        measured_pulses=inside or None,
    )


def _uncertainty(value: float, error: float, unit: str) -> Uncertainty | None:
    """Return the interval a pooled value is supported to, where one exists."""

    if not math.isfinite(error):
        return None
    return Uncertainty(lower=value - error, upper=value + error, unit=unit)


def instrument_corrections(
    pooled: PooledInstrumentTerms | Iterable[PooledInstrumentTerms],
    *,
    provenance: Provenance,
    unit: str,
    status: CorrectionStatus = CorrectionStatus.recorded,
    pulse_start: int | None = None,
    pulse_end: int | None = None,
) -> tuple[ChannelCorrection, ...]:
    """Emit pooled instrument terms as correction records a read path can apply.

    Recorded rather than promoted by default.  A measured term and a term the read
    path divides data by are different claims, and the schema keeps them apart on
    purpose: promotion is a decision with its own gate, so it is made by a caller
    saying so rather than by this function's default.

    The curvature travels in the drift record's notes.  It is a real second-order
    term the fit measured, the schema declares no kind that carries it, and folding
    it into the slope would report a number that is not the slope at any instant.
    """

    rows = [pooled] if isinstance(pooled, PooledInstrumentTerms) else list(pooled)
    corrections: list[ChannelCorrection] = []
    for row in rows:
        interval = _bounds(row, pulse_start, pulse_end)
        corrections.append(
            ChannelCorrection(
                channel=row.channel,
                kind=CorrectionKind.offset,
                status=status,
                value=row.offset,
                unit=unit,
                uncertainty=_uncertainty(row.offset, row.offset_error, unit),
                validity=[interval],
                provenance=provenance,
                notes=(
                    f"scatter about the fitted walk {row.scatter:.6g} {unit}, "
                    f"pooled over {row.fit_count} quiet windows"
                ),
            )
        )
        corrections.append(
            ChannelCorrection(
                channel=row.channel,
                kind=CorrectionKind.drift_rate,
                status=status,
                value=row.drift_rate,
                unit=f"{unit}/s",
                uncertainty=_uncertainty(
                    row.drift_rate, row.drift_rate_error, f"{unit}/s"
                ),
                validity=[interval],
                provenance=provenance,
                notes=(
                    f"second time derivative of the integrator walk "
                    f"{row.drift_curvature:.6g} {unit}/s2, spread "
                    f"{row.drift_curvature_error:.6g} {unit}/s2"
                ),
            )
        )
    return tuple(corrections)
