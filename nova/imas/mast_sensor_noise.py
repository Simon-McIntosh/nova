"""Measure what each magnetic sensor reads when nothing is driving it.

A residual is only meaningful against the precision of the instrument that
produced it, and that precision was never measured for this machine -- the
description was refined until the fit stopped improving, with no statement of
where it should have stopped.  The archive answers the question directly: nine
hundred and forty-nine plasma-free shots drove no poloidal coil at all, and five
hundred and seventy-seven of those held only the toroidal field, which a poloidal
probe is oriented to reject.  On those shots every reading is the instrument.

Two quantities come out and they are not the same thing.  A probe is an
integrator, so its output wanders: over a five-second record the wander is a
smooth ramp of a definite slope, and it is removed by the pre-excitation offset
subtraction that every fit here already does.  What survives that subtraction is
the scatter about the ramp, and that is the floor a model can be asked to reach.
Reporting the raw standard deviation instead would charge the model for drift the
fit has already removed and overstate the floor several-fold, so the two are
separated: :attr:`ChannelNoise.scatter` is the floor and
:attr:`ChannelNoise.drift_rate` is the artefact.

Repeat experiments give a third and larger number.  The calibration campaigns
fired the same coil at the same current twice, so the difference between two such
readings is everything that changes between two shots -- supply reproducibility,
thermal state, whatever the acquisition did differently -- and it bounds how well
any single-shot fit can be expected to transfer.  A model already inside the
scatter but outside the repeat spread is limited by the machine, not by itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from nova.imas.mast_vacuum_cohort import ShotWaveforms

MINIMUM_NOISE_SAMPLES = 200
"""Samples a channel must contribute before its noise is reported.

The store records five seconds at five kilohertz, so a channel present for the
whole shot offers tens of thousands of samples and this floor only refuses a
channel that dropped out.  Reporting a standard deviation from a handful of
samples would put a number with no precision beside numbers that have some.
"""


class NoiseError(ValueError):
    """Raised when a sensor noise envelope cannot be measured."""


def _detrend(time: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Return the scatter about a straight line and that line's slope.

    Fitting and removing one straight line is the whole of the separation.  An
    integrator's zero moves smoothly, so a ramp absorbs it; anything left is what
    the pre-excitation offset subtraction cannot remove and a model has to match.
    """

    finite = np.isfinite(time) & np.isfinite(values)
    count = int(finite.sum())
    if count < MINIMUM_NOISE_SAMPLES:
        return (float("nan"), float("nan"))
    clock = time[finite]
    signal = values[finite]
    span = float(clock.max() - clock.min())
    if not math.isfinite(span) or span <= 0.0:
        return (float(np.std(signal)), 0.0)
    slope, intercept = np.polyfit(clock, signal, 1)
    residual = signal - (slope * clock + intercept)
    return (float(np.std(residual)), float(slope))


@dataclass(frozen=True, order=True)
class ChannelNoise:
    """One sensor's measured floor, drift and the shots it was measured on."""

    channel: str
    scatter: float
    drift_rate: float
    shot_count: int
    sample_count: int
    scatter_spread: float = 0.0

    def validate(self) -> None:
        """Reject a floor that is not a usable positive amplitude."""

        if not math.isfinite(self.scatter) or self.scatter <= 0.0:
            raise NoiseError(f"channel {self.channel!r} has no measurable floor")
        if self.shot_count <= 0:
            raise NoiseError(f"channel {self.channel!r} was measured on no shot")

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channel": self.channel,
            "drift_rate": float(self.drift_rate),
            "sample_count": self.sample_count,
            "scatter": float(self.scatter),
            "scatter_spread": float(self.scatter_spread),
            "shot_count": self.shot_count,
        }


@dataclass(frozen=True)
class NoiseEnvelope:
    """Every sensor's floor, and the aggregate the ladder is measured against."""

    channels: tuple[ChannelNoise, ...]
    shots: tuple[int, ...]

    def validate(self) -> None:
        """Reject an envelope with no channel in it."""

        if not self.channels:
            raise NoiseError("a noise envelope must carry at least one channel")
        for row in self.channels:
            row.validate()

    @property
    def scatter(self) -> np.ndarray:
        """Return every channel's floor, in channel order."""

        return np.asarray([row.scatter for row in self.channels], dtype=float)

    @property
    def pooled_scatter(self) -> float:
        """Return the root-mean-square floor over the array.

        The quadratic mean rather than the arithmetic one, because a residual
        pooled over the array is itself a quadratic mean and the two have to be
        the same kind of average to be compared.
        """

        values = self.scatter
        return float(np.sqrt(np.mean(values**2)))

    def family_scatter(self) -> dict[str, float]:
        """Return the pooled floor per probe family."""

        grouped: dict[str, list[float]] = {}
        for row in self.channels:
            family = row.channel.rstrip("0123456789")
            grouped.setdefault(family, []).append(row.scatter)
        return {
            family: float(np.sqrt(np.mean(np.asarray(values) ** 2)))
            for family, values in sorted(grouped.items())
        }

    def channel(self, name: str) -> ChannelNoise:
        """Return one channel's measured floor."""

        for row in self.channels:
            if row.channel == name:
                return row
        raise KeyError(f"no noise measurement for channel {name!r}")

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channels": [row.as_dict() for row in self.channels],
            "family_scatter": self.family_scatter(),
            "pooled_scatter": self.pooled_scatter,
            "shot_count": len(self.shots),
            "shots": list(self.shots),
        }


def measure_noise_envelope(
    waveforms: Iterable[ShotWaveforms],
    *,
    channels: Sequence[str] | None = None,
) -> NoiseEnvelope:
    """Measure every sensor's floor across shots that drove no poloidal coil.

    Each shot contributes one scatter per channel and the shots are combined by
    their quadratic mean, so a channel that is quiet on most shots and noisy on
    one reports a floor that admits the noisy one.  The spread across shots is
    carried alongside, because a floor that varies between shots is a statement
    about the acquisition rather than about the sensor.
    """

    collected: dict[str, list[tuple[float, float, int]]] = {}
    shots: list[int] = []
    for waveform in waveforms:
        shots.append(waveform.shot)
        for name, signal in sorted(waveform.probes.items()):
            if channels is not None and name not in channels:
                continue
            scatter, drift = _detrend(waveform.time, signal)
            if not math.isfinite(scatter) or scatter <= 0.0:
                continue
            samples = int(np.count_nonzero(np.isfinite(signal)))
            collected.setdefault(name, []).append((scatter, drift, samples))
    if not collected:
        raise NoiseError("no shot contributed a measurable sensor floor")

    rows = []
    for name, values in sorted(collected.items()):
        scatters = np.asarray([row[0] for row in values], dtype=float)
        drifts = np.asarray([row[1] for row in values], dtype=float)
        rows.append(
            ChannelNoise(
                channel=name,
                scatter=float(np.sqrt(np.mean(scatters**2))),
                drift_rate=float(np.median(np.abs(drifts))),
                shot_count=len(values),
                sample_count=int(sum(row[2] for row in values)),
                scatter_spread=float(np.std(scatters)) if scatters.size > 1 else 0.0,
            )
        )
    envelope = NoiseEnvelope(channels=tuple(rows), shots=tuple(sorted(shots)))
    envelope.validate()
    return envelope


@dataclass(frozen=True)
class RepeatScatter:
    """How far two repetitions of one designed excitation disagree."""

    family: str
    shots: tuple[int, ...]
    peak_current: float
    relative_scatter: float
    absolute_scatter: float
    channel_count: int

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "absolute_scatter": float(self.absolute_scatter),
            "channel_count": self.channel_count,
            "family": self.family,
            "peak_current": float(self.peak_current),
            "relative_scatter": float(self.relative_scatter),
            "shots": list(self.shots),
        }


def repeat_groups(
    experiments: Iterable[Any],
    *,
    current_tolerance: float = 0.02,
) -> tuple[tuple[str, tuple[int, ...], float], ...]:
    """Group experiments that repeat one coil at one current, as designed.

    Two shots repeat an experiment when they hold the same single coil to within
    ``current_tolerance`` of the same peak.  The campaigns deliberately fired each
    coil twice at each of two levels, so the groups this finds are the archive's
    own repetitions rather than coincidences of operation.
    """

    singles: dict[str, list[tuple[float, int]]] = {}
    for row in experiments:
        if len(row.identifies) != 1:
            continue
        singles.setdefault(row.identifies[0], []).append((row.peak_current, row.shot))
    groups: list[tuple[str, tuple[int, ...], float]] = []
    for family, members in sorted(singles.items()):
        members.sort()
        current: list[tuple[float, int]] = []
        for peak, shot in members:
            if (
                current
                and abs(peak - current[0][0]) > current_tolerance * current[0][0]
            ):
                if len(current) > 1:
                    groups.append(
                        (
                            family,
                            tuple(sorted(row[1] for row in current)),
                            float(np.mean([row[0] for row in current])),
                        )
                    )
                current = []
            current.append((peak, shot))
        if len(current) > 1:
            groups.append(
                (
                    family,
                    tuple(sorted(row[1] for row in current)),
                    float(np.mean([row[0] for row in current])),
                )
            )
    return tuple(groups)


def measure_repeat_scatter(
    family: str,
    shots: Sequence[int],
    waveforms: Mapping[int, ShotWaveforms],
    *,
    peak_current: float = 0.0,
) -> RepeatScatter:
    """Compare repetitions of one excitation channel by channel.

    Each shot's probe reading is normalised by the current its own channel
    measured, so a supply that delivered one percent less current on the second
    shot does not count as sensor disagreement.  What is left is everything the
    excitation channel does not explain.
    """

    available = [shot for shot in shots if shot in waveforms]
    if len(available) < 2:
        raise NoiseError(f"repeat scatter for {family!r} needs two readable shots")
    normalised: dict[str, list[float]] = {}
    amplitudes: dict[str, list[float]] = {}
    for shot in available:
        waveform = waveforms[shot]
        drive = waveform.drives.get(family)
        if drive is None:
            continue
        mask = waveform.sample_mask & np.isfinite(drive)
        if not mask.any():
            continue
        index = int(np.argmax(np.abs(np.where(mask, np.nan_to_num(drive), 0.0))))
        current = float(drive[index])
        if abs(current) <= 0.0:
            continue
        quiet = waveform.baseline_mask
        for name, signal in waveform.probes.items():
            if not np.isfinite(signal[index]) or not np.isfinite(signal[quiet]).any():
                continue
            offset = float(np.mean(signal[quiet & np.isfinite(signal)]))
            normalised.setdefault(name, []).append(
                (float(signal[index]) - offset) / current
            )
            amplitudes.setdefault(name, []).append(abs(float(signal[index]) - offset))
    relative: list[float] = []
    absolute: list[float] = []
    for name, values in normalised.items():
        if len(values) < 2:
            continue
        centre = float(np.mean(np.abs(values)))
        spread = float(np.std(values))
        absolute.append(spread * float(np.mean(np.abs(values))) / max(centre, 1.0e-30))
        if centre > 0.0:
            relative.append(spread / centre)
    if not relative:
        raise NoiseError(f"no channel repeats on the {family!r} experiments")
    scale = float(
        np.median([np.mean(values) for values in amplitudes.values() if values])
    )
    return RepeatScatter(
        family=family,
        shots=tuple(sorted(available)),
        peak_current=float(peak_current),
        relative_scatter=float(np.median(relative)),
        absolute_scatter=float(np.median(relative) * scale),
        channel_count=len(relative),
    )
