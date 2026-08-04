"""Screen a shot's non-axisymmetric excitation out of the vacuum calibration.

A poloidal-field coil produces the same field at every toroidal angle, so the
axisymmetric forward model every fit here is built on cannot represent a coil
that does not.  Whatever such a coil put on a probe therefore lands in the
residual, and a fit that reads that shot attributes it to whichever
axisymmetric term happens to correlate.  This module decides, per shot and per
probe channel, whether that can have happened -- and it decides it by
measurement rather than by exclusion, because the answer turns out to be that
almost nowhere in this archive can it.

Three things had to be established before a threshold could mean anything.

**Which channels are the non-axisymmetric ones.**  The store names them two
ways across campaigns: ``error_field_02``/``error_field_05`` on the later shots
and ``error_field_a``/``error_field_b`` on the earlier ones.  The two pairs are
not merely renamed -- the earlier pair is acquired on its own clock, carried
beside the shared one in the same group -- so a reader that knows only the later
names silently sees the earlier campaigns as having no such coil, and the
earlier campaigns hold the strongest excitation in the archive.

**Which channels are not.**  ``efps_current`` sits beside them and reaches
twenty kiloamperes, more than either error-field channel, so it looks like the
strongest non-axisymmetric drive in the store.  It is not one: it tracks a P2
feed current with correlation above 0.99 and matching amplitude on
substantially every shot where it carries anything, so it is a supply monitor on
an axisymmetric circuit and already in the forward model through the coil it
feeds.  :func:`supply_monitor_correlation` is the test, kept here rather than
asserted, because treating it as an error-field channel disqualifies most of the
calibration cohort for a drive the model already has.

**How much probe signal a non-axisymmetric ampere is worth.**  Shots exist that
drive an error-field channel with every poloidal coil quiet, so the coupling is
measurable directly: regress each probe on the error-field waveform and the
slope is the channel's response per ampere.  A threshold derived from that slope
and the channel's own quiescent scatter is a statement about when the drive
becomes visible, which is the only thing a screen needs to know.

The measurement's own shape is what makes the screen cheap.  A field varies
smoothly across an array whose channels sit seventy-five millimetres apart, so a
response an order of magnitude larger than its immediate neighbours' is not a
field -- it is a conductor shared with the excitation.  Reporting the
neighbour ratio beside every coupling keeps that distinction visible instead of
letting a coupled channel set a threshold for the whole array.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from nova.imas.mast_block_scale import BlockScaleTable, promoted_block_scales
from nova.imas.mast_vacuum_cohort import (
    CURRENT_GROUP,
    FIELD_GROUP,
    KILO,
    SHOT_STORE,
    CohortError,
    _resample,
    parse_probe_channel,
)

ERROR_FIELD_CHANNELS = ("error_field_02", "error_field_05")
"""Store names of the non-axisymmetric coil currents on the later campaigns."""

ERROR_FIELD_ALIASES = ("error_field_a", "error_field_b")
"""The same two coils' channels on the earlier campaigns, on their own clock.

Kept as a separate tuple rather than folded into one list because the pairing
between the two naming schemes is not established: what is established is that
both name non-axisymmetric coil current, and that a screen reading only one
scheme leaves the other campaign unscreened.  The earlier pair reaches twelve
kiloamperes, half again the strongest the later names ever carry, so the
omission is not academic.
"""

SUPPLY_MONITOR_CHANNELS = ("efps_current",)
"""Channels that sit beside the error-field ones and are not error-field ones.

Excluded on the measurement in :func:`supply_monitor_correlation`, not on the
name.
"""

CLOCK_CHANNELS = ("time", "timesec")
"""Time bases a current group may carry, longest-standing name first."""

SUPPLY_MONITOR_CORRELATION = 0.9
"""Correlation with an axisymmetric coil channel that identifies a monitor.

A channel measuring its own coil correlates with another coil's waveform only
as far as the two supplies happen to be programmed alike, which across a mixed
cohort is nowhere near this.  A channel that is a copy of another channel
reaches it on every shot.
"""

QUIESCENT_CURRENT = 100.0
"""Amperes below which a non-axisymmetric channel is treated as not driven.

The digitiser's own floor on these channels sits an order below this, and the
offset window every fit measures its zero in has to be a window in which the
error-field coils were off, so the threshold is the one that defines that
window rather than the one that decides whether a shot is usable.
"""

DRIVEN_CURRENT = 1.0e3
"""Amperes at which a non-axisymmetric channel counts as deliberately driven.

Matches the excitation threshold the poloidal coils are judged by, so a shot
described as driving an error-field coil means the same thing as a shot
described as driving P4.
"""

MINIMUM_COUPLING_SHOTS = 3
"""Isolated shots a channel's coupling must be measured on to be believed.

Two give a slope and no scatter, so a channel whose coupling happens to fit one
shot's residual cannot be told from one that reproduces.
"""

NEIGHBOUR_INCOHERENCE = 5.0
"""Ratio to adjacent channels past which a response is not a field.

The outboard and centre-column arrays are wound at seventy-five millimetre
pitch.  Any field a coil produces varies over that distance by a few percent at
most at these radii, so a channel responding several times its neighbours is
reading a shared conductor and not the field at its position.  Five is well
above the few tens of percent real field structure allows and well below the
factor of fifty the coupled channel in this archive exhibits.
"""


class ErrorFieldError(CohortError):
    """Raised when the non-axisymmetric excitation cannot be read or screened."""


def _clock(group: Any, samples: int) -> np.ndarray | None:
    """Return the time base of the given length carried in a current group."""

    for name in CLOCK_CHANNELS:
        if name in group and group[name].shape[0] == samples:
            return np.asarray(group[name][...], dtype=float)
    return None


@dataclass(frozen=True)
class ErrorFieldDrive:
    """One shot's non-axisymmetric excitation, on the probe clock.

    ``waveforms`` are in amperes at the probe sample times, so a probe signal can
    be regressed on them without further resampling.  ``absent`` names the
    channels this campaign does not carry, which is how a shot screened under one
    naming scheme is distinguished from a shot that was never screened at all.
    """

    shot: int
    time: np.ndarray
    waveforms: Mapping[str, np.ndarray]
    absent: tuple[str, ...]
    quiescent_mask: np.ndarray

    @property
    def peaks(self) -> dict[str, float]:
        """Return each channel's largest magnitude in amperes."""

        result = {}
        for channel, values in sorted(self.waveforms.items()):
            finite = values[np.isfinite(values)]
            result[channel] = float(np.max(np.abs(finite))) if finite.size else 0.0
        return result

    @property
    def peak(self) -> float:
        """Return the largest non-axisymmetric current the shot drove."""

        peaks = self.peaks
        return max(peaks.values()) if peaks else 0.0

    @property
    def driven(self) -> bool:
        """Return whether a non-axisymmetric coil was deliberately driven."""

        return self.peak >= DRIVEN_CURRENT

    @property
    def unmeasured(self) -> bool:
        """Return whether this shot recorded no non-axisymmetric channel at all.

        The earliest campaigns predate the recording of these channels, so their
        shots carry none of them.  That is NOT the same as carrying them at zero:
        a coil nobody measured may have been driven, and a screen that reads an
        absent channel as quiescent passes exactly the shots it cannot vouch for.
        Such a shot is reported unmeasured and refused.
        """

        return not self.waveforms

    @property
    def strongest_channel(self) -> str | None:
        """Return the channel carrying the shot's largest excitation."""

        peaks = self.peaks
        if not peaks:
            return None
        return max(peaks, key=lambda channel: peaks[channel])

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "absent": list(self.absent),
            "driven": self.driven,
            "peak": self.peak,
            "peaks": self.peaks,
            "shot": self.shot,
            "strongest_channel": self.strongest_channel,
            "unmeasured": self.unmeasured,
        }


def read_error_field_drive(
    shot: int,
    *,
    store: Path | str = SHOT_STORE,
) -> ErrorFieldDrive:
    """Read one shot's non-axisymmetric excitation onto the probe time base.

    Both naming schemes are looked for, and each channel's own clock is resolved
    from its sample count rather than assumed to be the group's shared one --
    the earlier campaigns' pair is acquired at a different rate and reading it
    against the shared clock raises rather than silently misaligns, but only
    because the lengths differ; resolving the clock removes the coincidence.
    """

    import zarr

    root = Path(store)
    group = zarr.open_group(f"{root}/{shot}.zarr", mode="r")
    currents = group[CURRENT_GROUP]
    fields = group[FIELD_GROUP]
    time = np.asarray(fields["time"][...], dtype=float)

    waveforms: dict[str, np.ndarray] = {}
    absent: list[str] = []
    for channel in ERROR_FIELD_CHANNELS + ERROR_FIELD_ALIASES:
        if channel not in currents:
            absent.append(channel)
            continue
        raw = np.asarray(currents[channel][...], dtype=float)
        clock = _clock(currents, raw.shape[0])
        if clock is None:
            raise ErrorFieldError(
                f"shot {shot} channel {channel!r} has {raw.shape[0]} samples and "
                "no time base of that length"
            )
        waveforms[channel] = _resample(time, clock, raw * KILO)

    quiescent = np.ones(time.shape, dtype=bool)
    for values in waveforms.values():
        quiescent &= np.abs(np.nan_to_num(values)) < QUIESCENT_CURRENT
    return ErrorFieldDrive(
        shot=shot,
        time=time,
        waveforms=waveforms,
        absent=tuple(sorted(absent)),
        quiescent_mask=quiescent,
    )


@dataclass(frozen=True, order=True)
class SupplyMonitor:
    """Evidence that a channel beside the error-field ones is not one of them."""

    channel: str
    shot_count: int
    axisymmetric_share: float
    correlation: float
    amplitude_ratio: float
    best_channel: str

    @property
    def identified(self) -> bool:
        """Return whether the channel is a copy of an axisymmetric coil's."""

        return (
            self.shot_count >= MINIMUM_COUPLING_SHOTS
            and self.axisymmetric_share > 0.9
            and abs(self.correlation) >= SUPPLY_MONITOR_CORRELATION
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "amplitude_ratio": self.amplitude_ratio,
            "axisymmetric_share": self.axisymmetric_share,
            "best_channel": self.best_channel,
            "channel": self.channel,
            "correlation": self.correlation,
            "identified": self.identified,
            "shot_count": self.shot_count,
        }


def supply_monitor_correlation(
    shot: int,
    channel: str,
    *,
    store: Path | str = SHOT_STORE,
) -> tuple[float, str, float]:
    """Return one shot's best-correlating current channel, and by how much.

    The comparison is against every other channel in the current group on the
    shared clock, so the answer is not restricted to the coils somebody thought
    to check.  A channel that measures its own coil wins against itself and
    nothing else; a monitor on somebody else's circuit names that circuit.
    """

    import zarr

    root = Path(store)
    currents = zarr.open_group(f"{root}/{shot}.zarr", mode="r")[CURRENT_GROUP]
    if channel not in currents:
        raise ErrorFieldError(f"shot {shot} carries no channel {channel!r}")
    target = np.asarray(currents[channel][...], dtype=float)
    best = (0.0, "", 0.0)
    for name in sorted(currents.keys()):
        if name in CLOCK_CHANNELS or name == channel:
            continue
        values = np.asarray(currents[name][...], dtype=float)
        if values.shape != target.shape:
            continue
        keep = np.isfinite(values) & np.isfinite(target)
        if int(keep.sum()) < 100:
            continue
        if np.std(values[keep]) == 0.0 or np.std(target[keep]) == 0.0:
            continue
        correlation = float(np.corrcoef(values[keep], target[keep])[0, 1])
        if abs(correlation) > abs(best[0]):
            span = float(np.max(np.abs(values[keep])))
            ratio = (
                float(np.max(np.abs(target[keep]))) / span if span > 0.0 else math.inf
            )
            best = (correlation, name, ratio)
    return best


def measure_supply_monitor(
    shots: Iterable[int],
    channel: str,
    *,
    store: Path | str = SHOT_STORE,
    minimum_peak: float = QUIESCENT_CURRENT,
) -> SupplyMonitor:
    """Pool the identity test over the shots on which a channel carries current.

    A shot on which the channel is at its floor says nothing about what it
    measures, so only shots above ``minimum_peak`` are counted.  The reported
    share is the fraction of those whose best correlate is an axisymmetric coil
    channel rather than an error-field one.
    """

    import zarr

    root = Path(store)
    correlations: list[float] = []
    ratios: list[float] = []
    axisymmetric = 0
    names: list[str] = []
    for shot in shots:
        try:
            currents = zarr.open_group(f"{root}/{shot}.zarr", mode="r")[CURRENT_GROUP]
        except Exception:  # noqa: BLE001 - a shot the store cannot open is skipped
            continue
        if channel not in currents:
            continue
        raw = np.asarray(currents[channel][...], dtype=float)
        finite = raw[np.isfinite(raw)]
        if not finite.size or float(np.max(np.abs(finite))) * KILO < minimum_peak:
            continue
        correlation, name, ratio = supply_monitor_correlation(
            shot, channel, store=store
        )
        if not name:
            continue
        correlations.append(correlation)
        names.append(name)
        if not name.startswith("error_field"):
            axisymmetric += 1
            ratios.append(ratio)
    count = len(correlations)
    if count == 0:
        raise ErrorFieldError(f"channel {channel!r} carries current on no given shot")
    common = max(set(names), key=names.count)
    return SupplyMonitor(
        channel=channel,
        shot_count=count,
        axisymmetric_share=axisymmetric / count,
        correlation=float(np.median(correlations)),
        amplitude_ratio=float(np.median(ratios)) if ratios else math.nan,
        best_channel=common,
    )


@dataclass(frozen=True, order=True)
class ChannelCoupling:
    """How much probe signal one non-axisymmetric ampere is worth on one channel.

    ``response`` is tesla per ampere, measured on shots that drove the
    error-field channel with every poloidal coil quiet.  ``threshold`` inverts it
    against the channel's own quiescent scatter: the excitation at which this
    channel starts reporting the non-axisymmetric coil above its own noise.
    """

    channel: str
    driver: str
    shot_count: int
    response: float
    scatter: float
    noise_floor: float
    neighbour_response: float

    def validate(self) -> None:
        """Reject a coupling that cannot produce a threshold."""

        if self.noise_floor <= 0.0:
            raise ErrorFieldError(
                f"{self.channel!r} needs a measured noise floor to be screened"
            )
        if self.shot_count < 0 or not math.isfinite(self.response):
            raise ErrorFieldError(f"{self.channel!r} coupling is malformed")

    @property
    def measured(self) -> bool:
        """Return whether enough isolated shots back the slope."""

        return self.shot_count >= MINIMUM_COUPLING_SHOTS

    @property
    def threshold(self) -> float:
        """Return the excitation at which this channel stops being clean."""

        self.validate()
        if not self.measured or abs(self.response) <= 0.0:
            return math.inf
        return self.noise_floor / abs(self.response)

    @property
    def neighbour_ratio(self) -> float:
        """Return how far this channel's response stands from its neighbours'."""

        if self.neighbour_response <= 0.0:
            return math.inf if abs(self.response) > 0.0 else 0.0
        return abs(self.response) / self.neighbour_response

    @property
    def shares_a_conductor(self) -> bool:
        """Return whether the response is too local to be a field.

        A coupling this far above its neighbours' is a mutual with the
        excitation's own wiring.  It still has to be screened -- the channel
        reports it either way -- but it must not be read as evidence about where
        the array sits, because it carries no field pattern.
        """

        return self.measured and self.neighbour_ratio > NEIGHBOUR_INCOHERENCE

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        threshold = self.threshold
        return {
            "channel": self.channel,
            "driver": self.driver,
            "measured": self.measured,
            "neighbour_ratio": (
                None if math.isinf(self.neighbour_ratio) else self.neighbour_ratio
            ),
            "neighbour_response": self.neighbour_response,
            "noise_floor": self.noise_floor,
            "response": self.response,
            "scatter": self.scatter,
            "shares_a_conductor": self.shares_a_conductor,
            "shot_count": self.shot_count,
            "threshold": None if math.isinf(threshold) else threshold,
        }


def _neighbour_names(channel: str) -> tuple[str, ...]:
    """Return the two channels adjacent to this one in its own array."""

    family, number = parse_probe_channel(channel)
    return tuple(
        f"{family}{index:02d}" for index in (number - 1, number + 1) if index >= 1
    )


def probe_response_to_drive(
    drive: ErrorFieldDrive,
    probes: Mapping[str, np.ndarray],
    *,
    channel: str | None = None,
) -> dict[str, float]:
    """Regress every probe on one shot's error-field waveform.

    Each probe's zero is measured in the window where the error-field channels
    were below :data:`QUIESCENT_CURRENT`, so the slope answers for the change the
    excitation produced.  The returned slopes are tesla per ampere.
    """

    driver = channel or drive.strongest_channel
    if driver is None or driver not in drive.waveforms:
        raise ErrorFieldError(f"shot {drive.shot} drove no error-field channel")
    excitation = drive.waveforms[driver]
    quiet = drive.quiescent_mask
    slopes: dict[str, float] = {}
    for name, signal in sorted(probes.items()):
        if signal.shape != drive.time.shape:
            continue
        finite = np.isfinite(signal) & np.isfinite(excitation)
        if not (finite & quiet).any():
            continue
        centred = signal - float(np.mean(signal[finite & quiet]))
        power = float(np.dot(excitation[finite], excitation[finite]))
        if int(finite.sum()) < 200 or power <= 0.0:
            continue
        slopes[name] = float(np.dot(excitation[finite], centred[finite]) / power)
    return slopes


def measure_error_field_coupling(
    slopes_by_shot: Sequence[tuple[str, Mapping[str, float]]],
    noise_floor: Mapping[str, float],
) -> tuple[ChannelCoupling, ...]:
    """Pool per-shot slopes into one coupling per probe channel and driver.

    The median over shots is taken rather than the mean, because a channel that
    misbehaves on one shot of a dozen would otherwise set its own threshold from
    that shot.  The neighbour response is the mean of the adjacent channels'
    pooled magnitudes under the same driver, which is what makes a locally
    coupled channel distinguishable from a field.
    """

    pooled: dict[tuple[str, str], list[float]] = {}
    for driver, slopes in slopes_by_shot:
        for channel, slope in slopes.items():
            pooled.setdefault((driver, channel), []).append(slope)
    medians = {key: float(np.median(values)) for key, values in sorted(pooled.items())}
    couplings = []
    for (driver, channel), values in sorted(pooled.items()):
        if channel not in noise_floor:
            continue
        neighbours = [
            abs(medians[(driver, name)])
            for name in _neighbour_names(channel)
            if (driver, name) in medians
        ]
        couplings.append(
            ChannelCoupling(
                channel=channel,
                driver=driver,
                shot_count=len(values),
                response=medians[(driver, channel)],
                scatter=float(np.std(values)),
                noise_floor=float(noise_floor[channel]),
                neighbour_response=float(np.mean(neighbours)) if neighbours else 0.0,
            )
        )
    return tuple(couplings)


PAIR_CURRENT_TOLERANCE = 0.05
"""Fractional agreement two shots' poloidal peaks need to be one pair.

Five percent is what the archive's repeat pulses reproduce each other to, so a
pair matched this closely differs in the non-axisymmetric drive and in nothing a
probe can tell apart.
"""


@dataclass(frozen=True, order=True)
class MatchedPair:
    """Two shots alike in poloidal drive and unalike in the other kind.

    A pair isolates the non-axisymmetric excitation without needing a model of
    it: subtract the two shots' probe readings and every axisymmetric term
    cancels to the accuracy the currents match, leaving whatever the error-field
    coil put on the array.  ``agreement`` is the worst fractional match over the
    coils either shot drove, which is what bounds that cancellation.
    """

    driven_shot: int
    quiet_shot: int
    family: str
    driven_error_field: float
    quiet_error_field: float
    agreement: float

    @property
    def usable(self) -> bool:
        """Return whether the pair's poloidal drives match closely enough."""

        return self.agreement <= PAIR_CURRENT_TOLERANCE

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "agreement": self.agreement,
            "driven_error_field": self.driven_error_field,
            "driven_shot": self.driven_shot,
            "family": self.family,
            "quiet_error_field": self.quiet_error_field,
            "quiet_shot": self.quiet_shot,
            "usable": self.usable,
        }


def matched_pairs(
    peaks: Mapping[int, Mapping[str, float]],
    families: Mapping[int, str],
    error_field: Mapping[int, float],
    *,
    tolerance: float = PAIR_CURRENT_TOLERANCE,
) -> tuple[MatchedPair, ...]:
    """Pair each error-field shot with the closest shot that drove it quiet.

    Matching is on the poloidal coils' own peak currents rather than on the
    excitation family label, because two shots can share a label and differ by a
    factor in what they drove.  Each driven shot keeps only its best partner, so
    the set is one pair per experiment rather than every combination.
    """

    quiet = [
        shot
        for shot, value in sorted(error_field.items())
        if value < QUIESCENT_CURRENT and shot in peaks
    ]
    pairs = []
    for shot, value in sorted(error_field.items()):
        if value < DRIVEN_CURRENT or shot not in peaks:
            continue
        driven = {
            family: abs(float(current))
            for family, current in peaks[shot].items()
            if abs(float(current)) > 0.0
        }
        if not driven:
            continue
        best: tuple[float, int] | None = None
        for other in quiet:
            candidate = peaks[other]
            worst = 0.0
            for family, current in driven.items():
                partner = abs(float(candidate.get(family, 0.0)))
                worst = max(worst, abs(partner - current) / max(current, partner, 1.0))
            for family, current in candidate.items():
                if abs(float(current)) > 0.0 and family not in driven:
                    worst = 1.0
            if best is None or worst < best[0]:
                best = (worst, other)
        if best is None or best[0] > tolerance:
            continue
        pairs.append(
            MatchedPair(
                driven_shot=shot,
                quiet_shot=best[1],
                family=str(families.get(shot, "")),
                driven_error_field=float(value),
                quiet_error_field=float(error_field.get(best[1], 0.0)),
                agreement=float(best[0]),
            )
        )
    return tuple(pairs)


@dataclass(frozen=True)
class ErrorFieldScreen:
    """Per-channel thresholds, and the verdict they give a shot.

    The screen is deliberately per channel rather than per shot.  A shot whose
    error-field drive is visible on one coupled channel and forty decibels below
    the floor on every other is not a shot to discard; it is a shot with one
    channel to leave out, and discarding it instead costs the cohort coverage
    for nothing.
    """

    couplings: tuple[ChannelCoupling, ...]

    def threshold(self, channel: str) -> float:
        """Return the strictest excitation this channel tolerates."""

        thresholds = [row.threshold for row in self.couplings if row.channel == channel]
        return min(thresholds) if thresholds else math.inf

    @property
    def thresholds(self) -> dict[str, float]:
        """Return every measured channel's threshold."""

        return {
            channel: self.threshold(channel)
            for channel in sorted({row.channel for row in self.couplings})
        }

    @property
    def coupled_channels(self) -> tuple[str, ...]:
        """Return the channels sharing a conductor with the excitation."""

        return tuple(
            sorted({row.channel for row in self.couplings if row.shares_a_conductor})
        )

    def refused(self, drive: ErrorFieldDrive) -> tuple[str, ...]:
        """Return the probe channels this shot's excitation disqualifies.

        A shot that recorded none of these channels loses every channel the screen
        has a threshold for, because nothing about it can be vouched for.  A shot
        that recorded them and found them quiet loses nothing.
        """

        if drive.unmeasured:
            return tuple(sorted(self.thresholds))
        peak = drive.peak
        if peak <= 0.0:
            return ()
        return tuple(
            channel
            for channel, limit in sorted(self.thresholds.items())
            if peak >= limit
        )

    def passes(self, drive: ErrorFieldDrive) -> bool:
        """Return whether every probe channel survives this shot."""

        return not self.refused(drive)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "coupled_channels": list(self.coupled_channels),
            "couplings": [row.as_dict() for row in self.couplings],
            "driven_current": DRIVEN_CURRENT,
            "neighbour_incoherence": NEIGHBOUR_INCOHERENCE,
            "quiescent_current": QUIESCENT_CURRENT,
        }


@dataclass(frozen=True)
class ScreenOutcome:
    """What the screen did to one named set of shots.

    Three dispositions, deliberately kept apart.  ``driven_shots`` recorded the
    excitation and found it running; ``unmeasured_shots`` did not record it at all
    and are refused wholesale; ``unscreened_shots`` were never looked at, which is a
    gap in this run rather than a fact about the archive.
    """

    name: str
    shot_count: int
    driven_shots: tuple[int, ...]
    unscreened_shots: tuple[int, ...]
    refusals: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    unmeasured_shots: tuple[int, ...] = ()

    @property
    def clean(self) -> bool:
        """Return whether no shot in the set lost a channel."""

        return (
            not self.refusals
            and not self.unscreened_shots
            and not self.unmeasured_shots
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "clean": self.clean,
            "driven_shots": list(self.driven_shots),
            "name": self.name,
            "refusals": {
                str(shot): list(channels)
                for shot, channels in sorted(self.refusals.items())
            },
            "shot_count": self.shot_count,
            "unmeasured_shots": list(self.unmeasured_shots),
            "unscreened_shots": list(self.unscreened_shots),
        }


def screen_shot_set(
    name: str,
    shots: Sequence[int],
    screen: ErrorFieldScreen,
    drives: Mapping[int, ErrorFieldDrive],
) -> ScreenOutcome:
    """Apply the screen to one named set and report what it removed.

    A shot missing from ``drives`` is reported as unscreened rather than as
    passing: the whole reason the earlier campaigns needed looking at is that a
    shot nobody screened reads exactly like a shot that passed.
    """

    driven = []
    unscreened = []
    unmeasured = []
    refusals: dict[int, tuple[str, ...]] = {}
    for shot in shots:
        drive = drives.get(shot)
        if drive is None:
            unscreened.append(shot)
            continue
        if drive.unmeasured:
            unmeasured.append(shot)
        elif drive.driven:
            driven.append(shot)
        removed = screen.refused(drive)
        if removed:
            refusals[shot] = removed
    return ScreenOutcome(
        name=name,
        shot_count=len(shots),
        driven_shots=tuple(driven),
        unscreened_shots=tuple(unscreened),
        refusals=refusals,
        unmeasured_shots=tuple(unmeasured),
    )


def read_probe_signals(
    shot: int,
    *,
    store: Path | str = SHOT_STORE,
    block_scale: BlockScaleTable | None = None,
) -> dict[str, np.ndarray]:
    """Read one shot's poloidal field probe channels, range setting divided out.

    ``block_scale`` names the table and defaults to the promoted one, so a coupling
    slope measured here is a slope in field rather than in whatever range the
    acquisition happened to be on.  An empty table reads the archive as published.
    """

    import zarr

    fields = zarr.open_group(f"{Path(store)}/{shot}.zarr", mode="r")[FIELD_GROUP]
    signals = {}
    for channel in sorted(fields.keys()):
        try:
            parse_probe_channel(channel)
        except CohortError:
            continue
        signals[channel] = np.asarray(fields[channel][...], dtype=float)
    table = promoted_block_scales() if block_scale is None else block_scale
    return table.normalise(shot, signals)[0]
