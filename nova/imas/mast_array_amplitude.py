"""Measure one channel's amplitude against the rest of the array, model-free.

The acquisition setting of a probe channel is normally measured against the described
field: predict what the coils should produce, divide the recording by it, and the
quotient is what the acquisition did.  That route is the right one where it applies,
but it only applies where the description predicts the shot -- a plasma-free shot with
a sustained excitation.  Those shots arrive in clusters, so a switch that happened
between two clusters is bracketed by thousands of shots the route cannot read, and a
correction whose boundaries are that loose can be applied to the cohort and to little
else.

This module supplies a second observable that needs no description at all.  A range
setting is applied to one channel, so it changes that channel's amplitude relative to
every other channel reading the same shot.  Taking each channel's amplitude as a ratio
to the median amplitude of the array removes everything the channels have in common --
how hard the shot was driven, which coils it drove, how much current the vessel
carried, whether a plasma formed -- because all of that is in the numerator and the
denominator alike.  What survives is per-channel, which is what a range setting is.

Three properties make it usable on shots the fitted route refuses.

It is robust to the steppers themselves, to a bound worth stating.  The reference is a
median over sixty-odd channels, so the handful that stepped on any one boundary barely
moves it -- but "barely" is not "not at all": channels that double leave the lower half
of the ordering, which walks the median a little.  Measured over a seventy-six channel
array with a full decade of channel-to-channel spread, six channels doubling together
moves every other channel's ratio by under five percent and makes the doubled ones read
1.91 rather than 2.00.  The archive shows at most six moving at one boundary, and a step
has to reach 1.41 to be called one, so the bias sits an order inside the margin.  It
would not be safe for reading a value off, which is the second reason this route does
not set one.

It carries its own confound, and states it.  A channel's ratio to the array also moves
when the *field pattern* moves, because a different coil lights up a different part of
the array.  That is a real effect and it is not small between excitation classes, so
this observable is declared here as a boundary-placing instrument only: it says on
which shot a channel's amplitude jumped by a factor from the ladder, and it is not
permitted to set the value of a block.  The value keeps coming from the fitted route.

It is checkable against the route it extends.  On the plasma-free shots both routes
read, they must agree about which channels stepped and where.  :func:`agreement` is
that comparison, and it is the gate: an instrument that cannot reproduce the blocks
already measured has no business placing a boundary where nothing else can look.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from nova.imas.mast_acquisition_scale import STEP_RATIO

MINIMUM_CHANNELS = 20
"""Channels a shot must record before its median is a reference rather than a sample.

Below this the median is itself moved by the channels that stepped, and the whole
point of referring to the array is that the reference does not move.  Twenty leaves
the median untouched by the largest number seen stepping at one boundary.
"""

MINIMUM_AMPLITUDE = 1.0e-4
"""Tesla of root-mean-square signal below which a channel says nothing.

A channel that barely moved on a shot has a ratio dominated by its own noise, and
admitting it would put scatter into a comparison whose whole job is to see a factor of
two.  The value sits an order above the measured sensor floor.
"""


class ArrayAmplitudeError(ValueError):
    """Raised when a shot cannot yield an amplitude relative to its array."""


def channel_amplitudes(
    probes: Mapping[str, np.ndarray],
    *,
    baseline: np.ndarray | None = None,
    minimum_channels: int = MINIMUM_CHANNELS,
    minimum_amplitude: float = MINIMUM_AMPLITUDE,
) -> dict[str, float]:
    """Return each channel's amplitude on one shot, as a ratio to the array median.

    ``baseline`` marks the samples a channel's standing offset is measured in, so an
    offset is removed before the amplitude is taken rather than counted as signal.
    A channel too quiet to measure is left out rather than given a ratio, because its
    ratio would be a statement about its noise.
    """

    levels: dict[str, float] = {}
    for channel, values in probes.items():
        signal = np.asarray(values, dtype=float)
        finite = np.isfinite(signal)
        if baseline is not None and baseline.shape == signal.shape:
            window = finite & baseline
            if window.any():
                signal = signal - float(np.mean(signal[window]))
        if not finite.any():
            continue
        level = float(np.sqrt(np.mean(signal[finite] ** 2)))
        if math.isfinite(level) and level >= minimum_amplitude:
            levels[channel] = level
    if len(levels) < minimum_channels:
        return {}
    reference = float(np.median(list(levels.values())))
    if not math.isfinite(reference) or reference <= 0.0:
        return {}
    return {channel: level / reference for channel, level in sorted(levels.items())}


@dataclass(frozen=True, order=True)
class RouteAgreement:
    """Whether the model-free observable reproduces one channel's fitted blocks.

    ``fitted_steps`` and ``array_steps`` count the boundaries each route places over
    the same shots.  ``matched`` counts the fitted boundaries the array route places
    within :attr:`tolerance` shots of the same gap.  A channel the fitted route calls
    steady must come back steady, which is the harder half of the test: an instrument
    that invents boundaries is worse than one that misses them, because a boundary
    splits a block that a read would otherwise get right.
    """

    channel: str
    fitted_steps: int
    array_steps: int
    matched: int
    shared_shots: int

    @property
    def agrees(self) -> bool:
        """Return whether the two routes tell the same story about this channel."""

        return (
            self.matched == self.fitted_steps and self.array_steps == self.fitted_steps
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "agrees": self.agrees,
            "array_steps": self.array_steps,
            "channel": self.channel,
            "fitted_steps": self.fitted_steps,
            "matched": self.matched,
            "shared_shots": self.shared_shots,
        }


def _boundaries(
    rows: Sequence[tuple[int, float]], *, step_ratio: float = STEP_RATIO
) -> list[tuple[int, int]]:
    """Return the consecutive shot pairs a series steps across."""

    found = []
    for (before, first), (after, second) in zip(rows, rows[1:], strict=False):
        if first <= 0.0:
            continue
        ratio = second / first
        if ratio > step_ratio or ratio < 1.0 / step_ratio:
            found.append((before, after))
    return found


def agreement(
    fitted: Mapping[str, Mapping[int, Sequence[float]]],
    array: Mapping[str, Mapping[int, float]],
    *,
    step_ratio: float = STEP_RATIO,
) -> tuple[RouteAgreement, ...]:
    """Compare the two routes on every channel and shot they both read.

    Only the shots both routes read enter, so the comparison is about the observable
    and not about coverage.  A boundary matches when both routes place it across the
    same pair of consecutive shared shots, which is the strictest reading available:
    on the shared shots the two routes see the same gaps.
    """

    rows = []
    for channel, fitted_rows in sorted(fitted.items()):
        array_rows = array.get(channel, {})
        shared = sorted(set(fitted_rows) & set(array_rows))
        if len(shared) < 2:
            continue
        first = _boundaries(
            [(shot, float(np.median(fitted_rows[shot]))) for shot in shared],
            step_ratio=step_ratio,
        )
        second = _boundaries(
            [(shot, float(array_rows[shot])) for shot in shared], step_ratio=step_ratio
        )
        rows.append(
            RouteAgreement(
                channel=channel,
                fitted_steps=len(first),
                array_steps=len(second),
                matched=len(set(first) & set(second)),
                shared_shots=len(shared),
            )
        )
    return tuple(rows)


@dataclass(frozen=True, order=True)
class NarrowedBracket:
    """A fitted bracket after the model-free route said where inside it the step is.

    ``before_shot`` and ``after_shot`` are the consecutive shots the model-free
    amplitude crossed between, so they replace the fitted bracket's endpoints.  The
    fitted rungs are carried unchanged: this route moved the boundary and was never
    allowed to touch the value.

    ``crossing`` is the factor the amplitude moved by at that pair.  It is reported
    rather than tested against the fitted rung, because the reference walks a little
    when several channels step together -- so a crossing of 1.9 where the fitted rung
    is 2 is agreement, and demanding better would refuse a boundary the data placed.
    """

    channel: str
    before_shot: int
    after_shot: int
    crossing: float
    fitted_before: int
    fitted_after: int
    readings: int

    @property
    def width(self) -> int:
        """Return how many shot numbers the narrowed bracket still spans."""

        return int(self.after_shot - self.before_shot)

    @property
    def fitted_width(self) -> int:
        """Return the width the fitted route left."""

        return int(self.fitted_after - self.fitted_before)

    @property
    def narrowed(self) -> bool:
        """Return whether this route placed the boundary more tightly."""

        return self.width < self.fitted_width

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "after_shot": self.after_shot,
            "before_shot": self.before_shot,
            "channel": self.channel,
            "crossing": float(self.crossing),
            "fitted_after": self.fitted_after,
            "fitted_before": self.fitted_before,
            "fitted_width": self.fitted_width,
            "narrowed": self.narrowed,
            "readings": self.readings,
            "width": self.width,
        }


def narrow_bracket(
    channel: str,
    before_shot: int,
    after_shot: int,
    ratio: float,
    amplitudes: Mapping[int, float],
    *,
    step_ratio: float = STEP_RATIO,
) -> NarrowedBracket | None:
    """Place a fitted bracket's boundary on the shot pair the array route crossed at.

    Only readings inside the fitted bracket's own endpoints are used, so this cannot
    move a boundary the fitted route already placed -- it can only divide the gap the
    fitted route left open.  The crossing chosen is the largest step in the direction
    the fitted rungs went, because a bracket can contain scatter as well as the switch
    and the switch is the one moving the right way by the largest factor.

    Returns nothing when the readings do not contain a step in that direction: the
    honest outcome for a bracket the model-free route cannot resolve either, and one
    that must leave the fitted bracket exactly as wide as it was.
    """

    rows = sorted(
        (shot, float(value))
        for shot, value in amplitudes.items()
        if before_shot <= shot <= after_shot and value > 0.0
    )
    if len(rows) < 2:
        return None
    rising = ratio > 1.0
    best: tuple[float, int, int] | None = None
    for (first_shot, first), (second_shot, second) in zip(rows, rows[1:], strict=False):
        crossing = second / first
        if rising and crossing < step_ratio:
            continue
        if not rising and crossing > 1.0 / step_ratio:
            continue
        size = crossing if rising else 1.0 / crossing
        if best is None or size > best[0]:
            best = (crossing, first_shot, second_shot)
    if best is None:
        return None
    crossing, first_shot, second_shot = best
    return NarrowedBracket(
        channel=channel,
        before_shot=int(first_shot),
        after_shot=int(second_shot),
        crossing=float(crossing),
        fitted_before=int(before_shot),
        fitted_after=int(after_shot),
        readings=len(rows),
    )


def narrowing_summary(rows: Iterable[NarrowedBracket]) -> dict[str, Any]:
    """Report how much the model-free route tightened the switch boundaries."""

    rows = list(rows)
    narrowed = [row for row in rows if row.narrowed]
    return {
        "brackets": [row.as_dict() for row in rows],
        "median_fitted_width": (
            float(np.median([row.fitted_width for row in rows])) if rows else 0.0
        ),
        "median_width": float(np.median([row.width for row in rows])) if rows else 0.0,
        "narrowed": len(narrowed),
        "placed": len(rows),
        "widest_width": max((row.width for row in rows), default=0),
    }


def agreement_summary(rows: Iterable[RouteAgreement]) -> dict[str, Any]:
    """Report whether the model-free route has earned the right to place boundaries."""

    rows = list(rows)
    stepping = [row for row in rows if row.fitted_steps > 0]
    steady = [row for row in rows if row.fitted_steps == 0]
    return {
        "channels": len(rows),
        "invented_on_steady": sum(1 for row in steady if row.array_steps > 0),
        "matched_steps": sum(row.matched for row in stepping),
        "rows": [row.as_dict() for row in rows],
        "steady_channels": len(steady),
        "steady_reproduced": sum(1 for row in steady if row.array_steps == 0),
        "stepping_channels": len(stepping),
        "stepping_reproduced": sum(1 for row in stepping if row.agrees),
        "total_steps": sum(row.fitted_steps for row in stepping),
    }
