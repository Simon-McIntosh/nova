"""Measure a channel's gain wherever recorded currents determine its field.

A vacuum-driven interval supplies an unusually strong calibration statement: no
plasma current contributes, so the described Green's columns contracted with the
recorded conductor currents are the channel's complete physical prediction. A
scale fitted there is one pulse's gain check. Repeating that check turns ordinary
pulses into a calibration history without teaching the numerical kernel anything
about a machine, archive, or store.

The fit remains through the origin. An offset or integrator walk is not a gain and
must be removed first, either by the adapter or through ``instrument_for``. Letting
an intercept absorb it would move an additive instrument term into a multiplier a
consumer later divides by.

Pickup state and gain are separated by shape, not amplitude. An adapter may supply
one response row per physically described pickup state. Each row is contracted
with the same recorded currents and fitted with its own scalar gain; the state whose
prediction leaves the smallest residual wins only when it beats the next candidate
by a stated share of the measured signal. Response states that differ only by an
overall multiplier are therefore refused as unidentifiable, because no scalar fit
can tell that multiplier from gain.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from nova.calibrate.gain import MINIMUM_SAMPLES, MINIMUM_SHAPE_AGREEMENT, ScalarFit
from nova.calibrate.gain import through_origin_fit
from nova.calibrate.windows import PulseWindow, WindowKind

MINIMUM_STATE_SEPARATION = 0.05
"""Required residual advantage over the next response state.

The advantage is normalised by the measured signal RMS. Five percent keeps a
state decision from being made on a numerical tie while leaving the actual noise
floor and the machine's response geometry to the fit result.
"""


class GainCheckError(ValueError):
    """Raised when a vacuum-driven gain check is not identifiable."""


@dataclass(frozen=True)
class CandidateGain:
    """The scalar fit obtained under one described response state."""

    state: str
    gain: float
    shape_agreement: float
    residual: float
    signal: float
    sample_count: int


@dataclass(frozen=True)
class PulseGainCheck:
    """One channel's gain and response state measured on one pulse."""

    channel: str
    gain: float
    response_state: str
    shape_agreement: float
    residual: float
    signal: float
    sample_count: int
    window_count: int
    state_separation: float
    candidates: tuple[CandidateGain, ...]
    pulse: int | None = None

    @property
    def relative_residual(self) -> float:
        """Return the fitted residual as a share of measured signal RMS."""

        return self.residual / self.signal if self.signal > 0.0 else math.inf


@dataclass(frozen=True)
class RejectedGainCheck:
    """A requested channel for which this pulse could not identify a gain."""

    channel: str
    reason: str
    pulse: int | None = None


@dataclass(frozen=True)
class PulseGainChecks:
    """Accepted and refused channel checks from one pulse."""

    checks: tuple[PulseGainCheck, ...]
    rejected: tuple[RejectedGainCheck, ...]
    pulse: int | None = None

    def for_channel(self, channel: str) -> PulseGainCheck:
        """Return one accepted channel or explain that it was not accepted."""

        for check in self.checks:
            if check.channel == channel:
                return check
        refused = next(
            (row for row in self.rejected if row.channel == channel),
            None,
        )
        if refused is not None:
            raise GainCheckError(refused.reason)
        raise GainCheckError(f"pulse carries no requested channel {channel!r}")


ResponseRows = np.ndarray | Sequence[float] | Mapping[str, np.ndarray | Sequence[float]]
"""One response row, or one row per candidate pickup state."""


def _driven_mask(samples: int, windows: Sequence[PulseWindow]) -> np.ndarray:
    """Return the union of supplied driven windows after validating their bounds."""

    mask = np.zeros(samples, dtype=bool)
    for window in windows:
        if window.kind is not WindowKind.driven:
            raise GainCheckError(
                f"a {window.kind.value} window cannot support a vacuum-driven gain fit"
            )
        if not (0 <= window.start_index < window.stop_index <= samples):
            raise GainCheckError(
                f"window indices [{window.start_index}, {window.stop_index}) lie "
                f"outside a record of {samples} samples"
            )
        if mask[window.indices].any():
            raise GainCheckError(
                "driven windows overlap, so samples would be counted twice"
            )
        mask[window.indices] = True
    if not mask.any():
        raise GainCheckError("the pulse carries no vacuum-driven window")
    return mask


def _candidate_rows(responses: ResponseRows, drives: int) -> dict[str, np.ndarray]:
    """Normalise one response row or a named set of rows."""

    rows = {"described": responses} if not isinstance(responses, Mapping) else responses
    if not rows:
        raise GainCheckError("at least one described response state is required")
    normalised: dict[str, np.ndarray] = {}
    for state, response in rows.items():
        name = str(state)
        if not name:
            raise GainCheckError("a described response state needs a non-empty name")
        row = np.asarray(response, dtype=float)
        if row.shape != (drives,):
            raise GainCheckError(
                f"response state {name!r} has shape {row.shape}, expected ({drives},)"
            )
        if not np.all(np.isfinite(row)):
            raise GainCheckError(f"response state {name!r} is not finite")
        normalised[name] = row
    return normalised


def _candidate(state: str, fit: ScalarFit) -> CandidateGain:
    return CandidateGain(
        state=state,
        gain=fit.slope,
        shape_agreement=fit.variance_explained,
        residual=fit.residual,
        signal=fit.signal,
        sample_count=fit.sample_count,
    )


def fit_gain_check(
    time: np.ndarray | Sequence[float],
    currents: np.ndarray,
    signal: np.ndarray | Sequence[float],
    responses: ResponseRows,
    windows: Sequence[PulseWindow],
    *,
    channel: str = "",
    pulse: int | None = None,
    instrument: np.ndarray | Sequence[float] | None = None,
    minimum_samples: int = MINIMUM_SAMPLES,
    minimum_shape_agreement: float = MINIMUM_SHAPE_AGREEMENT,
    minimum_state_separation: float = MINIMUM_STATE_SEPARATION,
) -> PulseGainCheck:
    """Fit one channel against its exact vacuum prediction on one pulse.

    ``currents`` has one recorded conductor-current column per entry of each
    response row. ``responses`` may be a single row or a mapping from candidate
    pickup-state names to rows. Only samples covered by ``windows`` enter the fit;
    every supplied window must be vacuum-driven.

    ``signal`` must already be free of offset and integrator walk unless
    ``instrument`` supplies those terms sample by sample. The latter is subtracted
    before fitting. No intercept is estimated here.
    """

    axis = np.asarray(time, dtype=float)
    drive = np.asarray(currents, dtype=float)
    observed = np.asarray(signal, dtype=float)
    if axis.ndim != 1:
        raise GainCheckError("the time base is not one-dimensional")
    if drive.ndim != 2 or drive.shape[0] != axis.size:
        raise GainCheckError(
            f"current array has shape {drive.shape}, expected ({axis.size}, drives)"
        )
    if observed.shape != axis.shape:
        raise GainCheckError(
            f"signal has shape {observed.shape}, expected {axis.shape}"
        )
    if not math.isfinite(minimum_shape_agreement) or not (
        0.0 <= minimum_shape_agreement <= 1.0
    ):
        raise GainCheckError("minimum shape agreement must lie between zero and one")
    if not math.isfinite(minimum_state_separation) or minimum_state_separation < 0.0:
        raise GainCheckError("minimum state separation must be finite and non-negative")
    if minimum_samples < 1:
        raise GainCheckError("minimum samples must be positive")

    corrected = observed.copy()
    if instrument is not None:
        nuisance = np.asarray(instrument, dtype=float)
        if nuisance.shape != axis.shape:
            raise GainCheckError(
                f"instrument terms have shape {nuisance.shape}, expected {axis.shape}"
            )
        corrected -= nuisance

    admitted = _driven_mask(axis.size, windows)
    rows = _candidate_rows(responses, drive.shape[1])
    candidates: list[CandidateGain] = []
    for state, row in rows.items():
        prediction = drive @ row
        finite = (
            admitted
            & np.isfinite(axis)
            & np.isfinite(prediction)
            & np.isfinite(corrected)
        )
        if int(finite.sum()) < minimum_samples:
            raise GainCheckError(
                f"channel {channel!r} offers {int(finite.sum())} finite driven samples "
                f"against a floor of {minimum_samples}"
            )
        try:
            fit = through_origin_fit(prediction, corrected, mask=finite)
        except ValueError as error:
            raise GainCheckError(str(error)) from error
        candidates.append(_candidate(state, fit))

    candidates.sort(key=lambda row: row.residual)
    best = candidates[0]
    if not math.isfinite(best.gain) or best.gain <= 0.0:
        raise GainCheckError(
            f"channel {channel!r} fits gain {best.gain:.6g}; a non-positive scale is "
            "a polarity or response fault, not a gain check"
        )
    if not math.isfinite(best.shape_agreement) or (
        best.shape_agreement < minimum_shape_agreement
    ):
        raise GainCheckError(
            f"channel {channel!r} best response state {best.state!r} explains "
            f"{best.shape_agreement:.3f} of its variance against a floor of "
            f"{minimum_shape_agreement:.3f}"
        )

    separation = math.inf
    if len(candidates) > 1:
        separation = (candidates[1].residual - best.residual) / best.signal
        if not math.isfinite(separation) or separation < minimum_state_separation:
            raise GainCheckError(
                f"channel {channel!r} response states are separated by "
                f"{separation:.3f} of signal against a floor of "
                f"{minimum_state_separation:.3f}; pair state and gain are confounded"
            )

    return PulseGainCheck(
        channel=channel,
        gain=best.gain,
        response_state=best.state,
        shape_agreement=best.shape_agreement,
        residual=best.residual,
        signal=best.signal,
        sample_count=best.sample_count,
        window_count=len(windows),
        state_separation=separation,
        candidates=tuple(candidates),
        pulse=pulse,
    )


def fit_pulse_gain_checks(
    time: np.ndarray | Sequence[float],
    currents: np.ndarray,
    windows: Sequence[PulseWindow],
    channels: Iterable[str],
    signal_for: Callable[[str], np.ndarray | Sequence[float]],
    response_for: Callable[[str], ResponseRows],
    *,
    instrument_for: Callable[[str, np.ndarray], np.ndarray | Sequence[float]]
    | None = None,
    pulse: int | None = None,
    minimum_samples: int = MINIMUM_SAMPLES,
    minimum_shape_agreement: float = MINIMUM_SHAPE_AGREEMENT,
    minimum_state_separation: float = MINIMUM_STATE_SEPARATION,
) -> PulseGainChecks:
    """Run one pulse's gain checks through machine-owned adapter callbacks.

    The callbacks translate channel names to arrays and described Green's rows.
    They are the only adapter surface: this function opens no store, assumes no
    channel convention, and gives every refused channel back with its reason so an
    archive sweep can distinguish missing evidence from a clean result.
    """

    axis = np.asarray(time, dtype=float)
    accepted: list[PulseGainCheck] = []
    rejected: list[RejectedGainCheck] = []
    for channel in channels:
        name = str(channel)
        try:
            instrument = None if instrument_for is None else instrument_for(name, axis)
            accepted.append(
                fit_gain_check(
                    axis,
                    currents,
                    signal_for(name),
                    response_for(name),
                    windows,
                    channel=name,
                    pulse=pulse,
                    instrument=instrument,
                    minimum_samples=minimum_samples,
                    minimum_shape_agreement=minimum_shape_agreement,
                    minimum_state_separation=minimum_state_separation,
                )
            )
        except (GainCheckError, KeyError) as error:
            rejected.append(RejectedGainCheck(name, str(error), pulse))
    return PulseGainChecks(tuple(accepted), tuple(rejected), pulse)
