"""Apply a channel's corrections to its samples, in the order the schema fixes.

Reading a correction document tells a consumer what a channel needs.  Turning that
into corrected samples is a second job with its own ways of going wrong, and this
module does it once so that no consumer has to.

Three of those ways are worth naming, because each is silent in the arithmetic.

The order is not the consumer's to choose.  Subtracting a baseline after dividing
by a gain removes a different number than subtracting it before, and dividing a
gain out before an acquisition rung attributes the rung to the sensor.  The order
is therefore taken from the schema's own
:class:`~nova.calibrate.correction_model.ApplicationStage` ranks by way of
:data:`~nova.calibrate.correction_set.APPLICATION_ORDER`, never restated here.

Two corrections can land on one stage even in a document the reader accepts.  The
reader's non-overlap rule is scoped within one channel, kind and status, so a
channel-scoped gain and a gain scoped to a group the channel belongs to can both
cover one pulse, and so can two corrections of different statuses once a consumer
widens what it draws from.  The schema declares no precedence between a channel and
its group, so choosing one would encode a rule the document does not state: the
chain refuses instead, and a caller that knows the precedence states it by
narrowing the scope it asks for.

A value can be absent because nothing single describes the interval.  A channel
flipping between two states pulse to pulse carries its states in
``candidate_values`` with no value, and the mean of them describes no pulse.  Such a
correction is refused rather than skipped -- skipping it would return samples that
look corrected -- unless the caller resolves it explicitly, which is the one form in
which a consumer may assert which state a pulse was in.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from nova.calibrate.correction_model import (
    ApplicationStage,
    ChannelCorrection,
    CorrectionKind,
    CorrectionSet,
    CorrectionStatus,
    QualityStatus,
    ValidityInterval,
)
from nova.calibrate.correction_set import (
    APPLICATION_ORDER,
    CorrectionSetError,
    stage,
    target,
)

APPLIED_STATUSES = frozenset({CorrectionStatus.promoted})
"""The statuses a chain draws from unless the caller widens it.

Promoted is the one status that means "the read path divides by this".  A caller
may widen the set -- to see what a recorded pair state would do, say -- and doing so
is what makes two corrections able to share a stage, which is why widening it is a
decision rather than a default.
"""

ADDITIVE_STAGES = frozenset({ApplicationStage.offset, ApplicationStage.drift})
"""Stages that subtract from a signal rather than dividing it.

A drift is additive in the same sense an offset is: its value is a slope, and what
comes out of the samples is that slope times elapsed time.
"""


class CorrectionApplicationError(CorrectionSetError):
    """Raised when a channel's corrections cannot be applied as they stand."""


@dataclass(frozen=True)
class CorrectionStep:
    """One correction a chain applies, and what selecting it rested on."""

    stage: ApplicationStage
    kind: CorrectionKind
    value: float
    scope: str
    status: CorrectionStatus
    measured: bool
    resolved: bool = False

    @property
    def additive(self) -> bool:
        """Return whether the step subtracts rather than divides."""

        return self.stage in ADDITIVE_STAGES


@dataclass(frozen=True)
class CorrectionChain:
    """What one channel needs at one point, ordered and ready to apply.

    ``steps`` is in application order.  ``exclusions`` and ``quality`` carry the
    kinds that change no sample: they describe the channel or instruct a consumer to
    drop it, and a consumer that ignores them gets arithmetically correct samples
    from a channel it was told not to use.
    """

    channel: str
    steps: tuple[CorrectionStep, ...] = ()
    exclusions: tuple[str, ...] = ()
    quality: tuple[tuple[QualityStatus, str], ...] = ()

    @property
    def excluded(self) -> bool:
        """Return whether a correction instructs a consumer to drop the channel."""

        return bool(self.exclusions)

    @property
    def multiplier(self) -> float:
        """Return the product of every multiplicative value in the chain.

        The number a consumer divides a flat, baseline-free signal by.  It does not
        describe a chain carrying an offset or a drift, because those depend on the
        samples and on time; :func:`apply_chain` is what such a chain goes through.
        """

        product = 1.0
        for step in self.steps:
            if not step.additive:
                product *= step.value
        return product

    @property
    def extrapolated(self) -> bool:
        """Return whether any step was measured somewhere other than this point.

        A block running over five thousand pulses may rest on thirty-six of them, so
        a read inside the span is covered by the correction without being measured by
        it.  Both are legitimate; only one is a measurement, and a consumer weighing
        the two needs to be able to tell.
        """

        return any(not step.measured for step in self.steps)


def _interval_axes(interval: ValidityInterval) -> frozenset[str]:
    """Return the axes an interval is bounded on, empty when it is unbounded."""

    axes = set()
    if interval.pulse_start is not None or interval.pulse_end is not None:
        axes.add("pulse")
    if interval.time_start is not None or interval.time_end is not None:
        axes.add("time")
    return frozenset(axes)


def _covers(
    interval: ValidityInterval, *, pulse: int | None, moment: float | None
) -> bool:
    """Return whether a point lies inside a bounded interval."""

    if pulse is not None:
        lower, upper = interval.pulse_start, interval.pulse_end
        if (lower is not None and pulse < lower) or (
            upper is not None and pulse > upper
        ):
            return False
    if moment is not None:
        lower, upper = interval.time_start, interval.time_end
        if (lower is not None and moment < lower) or (
            upper is not None and moment > upper
        ):
            return False
    return True


def select_interval(
    correction: ChannelCorrection,
    *,
    pulse: int | None = None,
    time: float | None = None,
) -> ValidityInterval | None:
    """Return the interval of a correction that holds at a point, or None.

    An interval bounded on an axis the caller did not supply is refused rather than
    matched or skipped.  Both alternatives are wrong in the same direction: the
    caller has not said where it is reading, so nothing here can tell whether the
    correction applies, and either answer would be a guess dressed as a selection.
    """

    for interval in correction.validity:
        axes = _interval_axes(interval)
        if "pulse" in axes and pulse is None:
            raise CorrectionApplicationError(
                f"{target(correction)} {correction.kind} holds over a pulse range and "
                "the read names no pulse, so whether it applies is not decidable here"
            )
        if "time" in axes and time is None:
            raise CorrectionApplicationError(
                f"{target(correction)} {correction.kind} holds over a time span and "
                "the read names no time, so whether it applies is not decidable here"
            )
        if _covers(interval, pulse=pulse, moment=time):
            return interval
    return None


def _measured_at(interval: ValidityInterval, pulse: int | None) -> bool:
    """Return whether the point is one the correction was measured on."""

    pulses = interval.measured_pulses
    if not pulses or pulse is None:
        return not pulses
    return pulse in set(pulses)


def _scopes(channel: str, groups: Mapping[str, Iterable[str]] | None) -> frozenset[str]:
    """Return every scope string a channel is addressed by."""

    names = {channel}
    for name, members in (groups or {}).items():
        if channel in set(members):
            names.add(f"group:{name}")
    return frozenset(names)


def _resolved_value(
    correction: ChannelCorrection,
    resolution: Mapping[tuple[str, CorrectionKind], float] | None,
    channel: str,
) -> tuple[float, bool]:
    """Return the value to apply and whether the caller supplied it."""

    kind = CorrectionKind(correction.kind)
    supplied = (resolution or {}).get((channel, kind))
    if supplied is None and resolution:
        supplied = resolution.get((channel, kind.value))  # type: ignore[call-overload]
    if supplied is not None:
        return float(supplied), True
    if correction.value is not None:
        return float(correction.value), False
    candidates = list(correction.candidate_values or ())
    detail = (
        f"observed values {candidates}"
        if candidates
        else "no value and no candidates were recorded"
    )
    raise CorrectionApplicationError(
        f"{channel} {kind.value} carries no single value that describes this read "
        f"({detail}); resolve it explicitly to state which one held, because an "
        "average of discrete states describes none of them"
    )


def build_chain(
    document: CorrectionSet,
    channel: str,
    *,
    pulse: int | None = None,
    time: float | None = None,
    groups: Mapping[str, Iterable[str]] | None = None,
    statuses: Iterable[CorrectionStatus] = APPLIED_STATUSES,
    resolution: Mapping[tuple[str, CorrectionKind], float] | None = None,
) -> CorrectionChain:
    """Return one channel's ordered chain at one point in pulse or time.

    ``groups`` maps a group name to its member channels, because which channels a
    named group holds is a property of the machine rather than of the correction set,
    and a document that carried it would have to be rewritten whenever an array was
    rewired.  A channel absent from every group simply matches no group correction.
    """

    wanted = frozenset(CorrectionStatus(value) for value in statuses)
    scopes = _scopes(channel, groups)
    steps: dict[ApplicationStage, CorrectionStep] = {}
    exclusions: list[str] = []
    quality: list[tuple[QualityStatus, str]] = []
    for correction in document.corrections:
        if target(correction) not in scopes:
            continue
        status = CorrectionStatus(correction.status)
        if status not in wanted:
            continue
        interval = select_interval(correction, pulse=pulse, time=time)
        if interval is None:
            continue
        kind = CorrectionKind(correction.kind)
        if kind is CorrectionKind.exclusion:
            exclusions.append(correction.cause or "no cause recorded")
            continue
        if kind is CorrectionKind.quality:
            quality.append(
                (
                    QualityStatus(correction.quality_status),
                    correction.cause or "no cause recorded",
                )
            )
            continue
        entry = stage(correction)
        if entry is None:
            raise CorrectionApplicationError(
                f"{channel} carries a {kind.value} correction that the schema gives "
                "no application stage, so where it enters the chain is undefined"
            )
        value, resolved = _resolved_value(correction, resolution, channel)
        step = CorrectionStep(
            stage=entry,
            kind=kind,
            value=value,
            scope=target(correction),
            status=status,
            measured=_measured_at(interval, pulse),
            resolved=resolved,
        )
        if entry in steps:
            first = steps[entry]
            raise CorrectionApplicationError(
                f"{channel} draws two corrections onto the {entry.value} stage at "
                f"this read -- {first.scope} {first.kind.value} ({first.status.value}) "
                f"and {step.scope} {step.kind.value} ({step.status.value}). The schema "
                "orders the stages and not the corrections within one, so applying "
                "either would assert a precedence the document does not carry"
            )
        _validate_step(channel, step)
        steps[entry] = step
    return CorrectionChain(
        channel=channel,
        steps=tuple(steps[entry] for entry in APPLICATION_ORDER if entry in steps),
        exclusions=tuple(exclusions),
        quality=tuple(quality),
    )


def _validate_step(channel: str, step: CorrectionStep) -> None:
    """Refuse a value that would erase the channel rather than correct it."""

    if not math.isfinite(step.value):
        raise CorrectionApplicationError(
            f"{channel} {step.kind.value} resolves to a value that is not finite"
        )
    if not step.additive and step.value == 0.0:
        raise CorrectionApplicationError(
            f"{channel} {step.kind.value} resolves to a multiplier of zero, which "
            "erases the channel rather than correcting it"
        )


def apply_chain(
    chain: CorrectionChain,
    samples: np.ndarray | Sequence[float],
    *,
    time: np.ndarray | Sequence[float] | None = None,
    reference_time: float | None = None,
    allow_excluded: bool = False,
) -> np.ndarray:
    """Return the samples with every step of the chain removed, in order.

    A drift is a slope, so removing it needs a time base; ``reference_time`` fixes
    the instant the ramp is zero at and defaults to the first sample.  A chain
    carrying a drift and no time base is refused rather than treated as an offset.

    An excluded channel refuses by default.  Returning its samples silently is the
    one outcome that makes the exclusion useless -- the arithmetic is correct and the
    consumer never learns it was told not to read this channel.
    """

    if chain.excluded and not allow_excluded:
        causes = "; ".join(chain.exclusions)
        raise CorrectionApplicationError(
            f"{chain.channel} is excluded ({causes}); correcting it returns numbers a "
            "consumer was instructed not to use, so the exclusion has to be overridden "
            "deliberately"
        )
    values = np.asarray(samples, dtype=float)
    for step in chain.steps:
        if step.stage is ApplicationStage.offset:
            values = values - step.value
        elif step.stage is ApplicationStage.drift:
            values = values - step.value * _elapsed(chain, values, time, reference_time)
        else:
            values = values / step.value
    return values


def _elapsed(
    chain: CorrectionChain,
    values: np.ndarray,
    time: np.ndarray | Sequence[float] | None,
    reference_time: float | None,
) -> np.ndarray:
    """Return time since the ramp's origin, one entry per sample."""

    if time is None:
        raise CorrectionApplicationError(
            f"{chain.channel} carries a drift rate, whose value is a slope, and the "
            "call supplies no time base to multiply it by"
        )
    axis = np.asarray(time, dtype=float)
    if axis.shape != values.shape:
        raise CorrectionApplicationError(
            f"{chain.channel} was given {axis.size} times for {values.size} samples"
        )
    origin = float(axis[0]) if reference_time is None else float(reference_time)
    return axis - origin


def apply_corrections(
    document: CorrectionSet,
    channel: str,
    samples: np.ndarray | Sequence[float],
    *,
    pulse: int | None = None,
    time: np.ndarray | Sequence[float] | None = None,
    reference_time: float | None = None,
    groups: Mapping[str, Iterable[str]] | None = None,
    statuses: Iterable[CorrectionStatus] = APPLIED_STATUSES,
    resolution: Mapping[tuple[str, CorrectionKind], float] | None = None,
    allow_excluded: bool = False,
) -> tuple[np.ndarray, CorrectionChain]:
    """Correct one channel's samples and return them beside the chain applied.

    The chain comes back with the samples because what was done to them is not
    recoverable from them.  A consumer that records only the corrected array has no
    way to say later which gain era it read, whether the read point was measured or
    merely covered, or whether a state it depended on was resolved by hand.

    A chain selected at a moment in time uses the first sample's time when no
    ``pulse`` is given, so a correction scoped in time is resolved at the start of
    the window it is applied over.
    """

    values = np.asarray(samples, dtype=float)
    axis = None if time is None else np.asarray(time, dtype=float)
    moment: float | None = None
    if pulse is None and axis is not None and axis.size:
        moment = float(axis[0])
    chain = build_chain(
        document,
        channel,
        pulse=pulse,
        time=moment,
        groups=groups,
        statuses=statuses,
        resolution=resolution,
    )
    corrected = apply_chain(
        chain,
        values,
        time=axis,
        reference_time=reference_time,
        allow_excluded=allow_excluded,
    )
    return corrected, chain
