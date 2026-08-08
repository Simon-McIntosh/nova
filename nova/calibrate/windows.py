"""Split a pulse into the intervals whose field is known without solving anything.

A pulse is not one measurement.  Before anything is energised the sensors read the
instrument alone: whatever they report is offset and integrator walk, because the
field is zero by construction.  Once conductors carry current and no plasma exists,
the field follows from the recorded currents and the described geometry, so the
channel has a right answer that no reconstruction was needed to obtain.  Once plasma
forms, neither statement holds.  After termination both return.

Those intervals exist in every pulse a machine ever fired, which is what makes them
worth harvesting: a calibration that needs designed shots is limited to the few
hundred somebody thought to run, and one that needs only the archive is limited by
the archive.

The classification itself is three thresholds and a run-length encoding.  What is
not trivial is the four ways it goes quietly wrong, and each has a guard here.

A conductor sitting at its acquisition noise floor is not a drive.  Comparing a
channel against its own peak would classify the noise of a coil nobody energised as
an excitation and remove every instrument-quiet window from exactly the shots that
have the most of them, so the bar is an absolute current the caller states.

An interval that follows a disturbance is not the instrument alone.  Vessel and case
currents induced by the drives, and by the plasma's termination, decay on their own
time constants and go on producing field into the quiet that follows.  Every window
after a disturbance therefore starts late, by a settling time the caller builds from
a measured decay time and the number of time constants it wants gone.  A window with
no disturbance anywhere before it is not delayed: nothing has been induced yet, and
clipping there would cost real samples to guard against something that never
happened.

The guard is measured from the last interval observed to be a disturbance, not from
whatever immediately precedes the window.  The two differ whenever the record has a
gap in it, and both ways of confusing them are wrong.  A gap between a drive and the
quiet after it does not reset the decay -- the current has been falling since the
drive stopped, and losing sight of it for a moment does not change that.  A gap
before anything was ever driven is not evidence that something was: the interval
before a record is unobserved whether or not its first sample digitised, and reading
one unrecorded sample as a disturbance costs the pre-pulse window the whole guard.

A gap in the record is not a quiet interval.  Samples that are not finite say nothing
about what the machine was doing, so they break a window rather than joining it, and
what they broke is reported rather than silently merged.

A missing plasma channel is two opposite statements about the same absence.  On a
designed vacuum shot there was no plasma and an adapter says so by supplying nothing;
on a shot whose plasma signal was not recorded, nothing is known about plasma and the
adapter says *that* by supplying values that are not finite.  Reading the first as
the second loses windows; reading the second as the first admits plasma intervals to
the vacuum cohort.  Only the adapter can tell them apart, so only the adapter does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

_GAP = 0
_QUIET = 1
_DRIVEN = 2
_PLASMA = 3
"""Per-sample labels, ordered so that a higher label outranks a lower one.

Plasma outranks a drive because a conductor still carrying current while plasma
exists does not make the field computable, and a gap outranks nothing because a
sample that was not recorded supports no statement at all.
"""


class WindowKind(StrEnum):
    """What is known about the field over an interval."""

    quiet = "quiet"
    """Nothing energised and no plasma: the channel reads the instrument alone."""

    driven = "driven"
    """Conductors energised with no plasma: the field follows from the currents."""

    plasma = "plasma"
    """Plasma present: neither of the other two statements holds."""


_KIND = {
    _QUIET: WindowKind.quiet,
    _DRIVEN: WindowKind.driven,
    _PLASMA: WindowKind.plasma,
}


class WindowError(ValueError):
    """Raised when a pulse cannot be classified from what was supplied."""


@dataclass(frozen=True)
class PulseWindow:
    """One interval of a pulse over which one statement about the field holds."""

    kind: WindowKind
    start: float
    stop: float
    start_index: int
    stop_index: int
    guarded: bool = False
    """Whether a settling guard removed this window's leading samples.

    A window that follows a disturbance and was not guarded still carries the
    passive current that disturbance induced, so the flag is what separates an
    interval that is instrument-only from one that merely looks it.
    """

    @property
    def sample_count(self) -> int:
        """Return how many samples the window spans."""

        return self.stop_index - self.start_index

    @property
    def duration(self) -> float:
        """Return the seconds between the window's first and last sample."""

        return self.stop - self.start

    @property
    def indices(self) -> slice:
        """Return the window as a slice into the record."""

        return slice(self.start_index, self.stop_index)

    def mask(self, samples: int) -> np.ndarray:
        """Return a boolean selector for this window over a record of ``samples``."""

        selector = np.zeros(samples, dtype=bool)
        selector[self.indices] = True
        return selector


@dataclass(frozen=True)
class RejectedWindow:
    """An interval the classifier found and then refused, and why.

    Reported rather than dropped because an absent window and a refused one are
    the same silence to a consumer, and only one of them is a property of the
    pulse.
    """

    kind: WindowKind | None
    start: float
    stop: float
    reason: str


@dataclass(frozen=True)
class PulseTimeline:
    """Every interval of one pulse, and every interval that did not survive."""

    windows: tuple[PulseWindow, ...]
    rejected: tuple[RejectedWindow, ...]
    samples: int
    settling_time: float = 0.0

    def of_kind(self, kind: WindowKind) -> tuple[PulseWindow, ...]:
        """Return the windows carrying one statement about the field."""

        return tuple(row for row in self.windows if row.kind is kind)

    @property
    def quiet_windows(self) -> tuple[PulseWindow, ...]:
        """Return the intervals reading the instrument alone."""

        return self.of_kind(WindowKind.quiet)

    @property
    def driven_windows(self) -> tuple[PulseWindow, ...]:
        """Return the intervals whose field follows from the recorded currents."""

        return self.of_kind(WindowKind.driven)

    @property
    def plasma_windows(self) -> tuple[PulseWindow, ...]:
        """Return the intervals carrying plasma."""

        return self.of_kind(WindowKind.plasma)

    @property
    def leading_quiet(self) -> PulseWindow | None:
        """Return the quiet interval the pulse begins in, if it begins in one."""

        if self.windows and self.windows[0].kind is WindowKind.quiet:
            return self.windows[0]
        return None

    @property
    def trailing_quiet(self) -> PulseWindow | None:
        """Return the quiet interval the pulse ends in, if it ends in one."""

        if self.windows and self.windows[-1].kind is WindowKind.quiet:
            return self.windows[-1]
        return None

    def mask(self, kind: WindowKind) -> np.ndarray:
        """Return a boolean selector for every window of one kind at once."""

        selector = np.zeros(self.samples, dtype=bool)
        for window in self.of_kind(kind):
            selector[window.indices] = True
        return selector


def _columns(drive: np.ndarray | Sequence[float], samples: int) -> np.ndarray:
    """Return the drive as a two-dimensional array of one column per circuit."""

    values = np.asarray(drive, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise WindowError(
            f"a drive of {values.ndim} dimensions names neither one circuit nor a "
            "set of them"
        )
    if values.shape[0] != samples:
        raise WindowError(f"{values.shape[0]} drive samples against {samples} times")
    return values


def _bar(value: float | None, name: str) -> float:
    """Return a threshold, refusing one that admits everything."""

    if value is None or not math.isfinite(value) or value <= 0.0:
        raise WindowError(
            f"the {name} threshold is {value}, and a threshold at or below zero "
            "admits a channel's own noise as a signal"
        )
    return float(value)


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Collapse a per-sample label into contiguous runs of one label each."""

    if labels.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(labels) != 0)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks + 1, [labels.size]])
    return [
        (int(labels[start]), int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
    ]


def classify_pulse(
    time: np.ndarray | Sequence[float],
    drive: np.ndarray | Sequence[float],
    *,
    plasma: np.ndarray | Sequence[float] | None = None,
    drive_threshold: float,
    plasma_threshold: float | None = None,
    decay_time: float = 0.0,
    settling_periods: float = 0.0,
    minimum_samples: int = 1,
) -> PulseTimeline:
    """Split one pulse's timeline into what is known about the field over it.

    ``drive`` is one column per circuit in the units ``drive_threshold`` is stated
    in; a single circuit may be passed as a flat array.  ``plasma`` is the plasma
    current where it was recorded, and is left out where the pulse carried none.

    The settling guard is ``settling_periods`` times ``decay_time``, both supplied
    by the caller because the decay time is a measured property of the machine's
    vessel and coil cases and the number of time constants worth removing is a
    judgement about how much of the induced current is tolerable.  It advances the
    start of every window whose predecessor was driven, carried plasma, or was not
    recorded at all; a window it advances past its own end is refused rather than
    returned empty.

    ``minimum_samples`` refuses a window too short to fit anything, after the guard
    has run.  It defaults to admitting everything: the floor a consumer needs is a
    property of what it intends to fit, and the fits here impose their own.
    """

    axis = np.asarray(time, dtype=float)
    if axis.ndim != 1:
        raise WindowError("the time base is not one-dimensional")
    samples = int(axis.size)
    currents = _columns(drive, samples)
    drive_bar = _bar(drive_threshold, "drive")

    finite = np.isfinite(axis) & np.isfinite(currents).all(axis=1)
    driven = (np.abs(np.nan_to_num(currents)) >= drive_bar).any(axis=1)
    carries_plasma = np.zeros(samples, dtype=bool)
    if plasma is not None:
        current = np.asarray(plasma, dtype=float)
        if current.shape != axis.shape:
            raise WindowError(f"{current.size} plasma samples against {samples} times")
        plasma_bar = _bar(plasma_threshold, "plasma")
        finite &= np.isfinite(current)
        carries_plasma = np.abs(np.nan_to_num(current)) >= plasma_bar

    labels = np.where(
        ~finite,
        _GAP,
        np.where(carries_plasma, _PLASMA, np.where(driven, _DRIVEN, _QUIET)),
    )

    settling = float(settling_periods) * float(decay_time)
    if settling < 0.0 or not math.isfinite(settling):
        raise WindowError(f"the settling guard is {settling} seconds")

    windows: list[PulseWindow] = []
    rejected: list[RejectedWindow] = []
    disturbance_end: float | None = None
    for label, start, stop in _runs(labels):
        span = (float(axis[start]), float(axis[stop - 1]))
        if label == _GAP:
            rejected.append(
                RejectedWindow(
                    None, *span, "the record is not finite over this interval"
                )
            )
            continue
        kind = _KIND[label]
        origin = disturbance_end
        if label in (_DRIVEN, _PLASMA):
            disturbance_end = float(axis[stop]) if stop < samples else span[1]
        guarded = settling > 0.0 and origin is not None
        first = start
        if guarded:
            admitted = np.flatnonzero(axis[start:stop] >= origin + settling)
            if admitted.size == 0:
                rejected.append(
                    RejectedWindow(
                        kind,
                        *span,
                        f"passive current induced at {origin:.4g} s is still decaying "
                        f"{settling:.4g} s later, which is past the interval's end",
                    )
                )
                continue
            first = start + int(admitted[0])
        if stop - first < minimum_samples:
            rejected.append(
                RejectedWindow(
                    kind,
                    float(axis[first]),
                    span[1],
                    f"{stop - first} samples is under the floor of {minimum_samples}",
                )
            )
            continue
        windows.append(
            PulseWindow(
                kind=kind,
                start=float(axis[first]),
                stop=span[1],
                start_index=first,
                stop_index=stop,
                guarded=guarded,
            )
        )

    return PulseTimeline(
        windows=tuple(windows),
        rejected=tuple(rejected),
        samples=samples,
        settling_time=settling,
    )
