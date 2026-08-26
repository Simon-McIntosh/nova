"""Select and pin the no-plasma shots that identify the vacuum response.

A vacuum shot drives known currents through the poloidal field coils with no
plasma, so every diagnostic reading is the machine's own response to a measured
excitation.  That is the only experiment that separates a coil's signed turn
count from its geometry, because geometry alone fixes the shape of a coil's
field and not its amplitude or its polarity.

Three things have to be pinned before a fit may read a sample.  The store,
group and channel a number came from, so a residual can be traced back to a
signal rather than to a table.  The time window it was taken from, because a
shot is contaminated in parts and clean in others and the boundary is a physical
statement about what else was carrying current.  And the excitation family it
belongs to, because a fit that trains and tests inside one family has only shown
that it can interpolate a single waveform.

Channel identity here never rests on the store's own position metadata.  The
signal descriptions carry radii and axial positions that disagree with the
measured catalog by up to a centimetre and, for the flux loops, repeat one
position across a whole family.  The registry geometry is the authority; the
store supplies the ordered channel name, and the two are joined by family and
channel number.

Probe amplitude is not taken as published either.  The acquisition applied a
per-channel range setting that the store never normalised out, so
:func:`read_shot_waveforms` divides it out where the channel is read and reports
what it divided by -- see :mod:`~nova.imas.mast_block_scale`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from nova.imas.mast_block_scale import (
    BlockScaleTable,
    ScaleCorrection,
    ScaleReader,
    promoted_block_scales,
)

SHOT_STORE = Path("/work/projects/imas_gpu/mast/level1/shots")
"""Level-1 MAST shot store holding one Zarr group per shot."""

CURRENT_GROUP = "amc"
"""Store group carrying plasma, poloidal-field and toroidal-field currents."""

FIELD_GROUP = "amb"
"""Store group carrying the magnetic field probes and flux loops."""

WALL_GROUP = "amm"
"""Store group carrying the reduced vessel model's induced currents."""

KILO = 1.0e3
"""Store currents are recorded in kiloamperes; the fit works in amperes."""


class CohortError(ValueError):
    """Raised when a shot cannot be admitted to the vacuum cohort."""


@dataclass(frozen=True, order=True)
class CoilDrive:
    """The store channel that measures one coil's excitation.

    ``turns_per_channel_ampere`` is what a response fit recovers when it scales
    this channel: a channel measuring the current in one conductor returns the
    coil's turn count, and a channel already reporting ampere-turns returns one.
    It is the quantity under test, so it is never asserted here -- the field
    records which of the two a fit is being asked for, and the fit reports the
    number it found.
    """

    family: str
    channel: str
    circuit: str
    reports_ampere_turns: bool
    turn_to_channel_current_ratio: float = 1.0

    def validate(self) -> None:
        """Reject a drive whose channel or circuit identity is incomplete."""

        for value, context in (
            (self.family, "drive family"),
            (self.channel, "drive channel"),
            (self.circuit, "drive circuit"),
        ):
            if not value or value.strip() != value:
                raise CohortError(f"{context} must be non-empty trimmed text")
        if not math.isfinite(self.turn_to_channel_current_ratio):
            raise CohortError(f"drive {self.family!r} ratio must be finite")
        if self.turn_to_channel_current_ratio <= 0.0:
            raise CohortError(f"drive {self.family!r} ratio must be positive")


COIL_DRIVES = (
    CoilDrive("sol", "sol_current", "P1", False, 0.5),
    CoilDrive("p2_inner_lower", "p2il_feed_current", "P2", False),
    CoilDrive("p2_inner_upper", "p2iu_feed_current", "P2", False),
    CoilDrive("p2_outer_lower", "p2ol_feed_current", "P2", False),
    CoilDrive("p2_outer_upper", "p2ou_feed_current", "P2", False),
    CoilDrive("p3_lower", "p3l_feed_current", "P3", False),
    CoilDrive("p3_upper", "p3u_feed_current", "P3", False),
    CoilDrive("p4_lower", "p4l_feed_current", "P4", False),
    CoilDrive("p4_upper", "p4u_feed_current", "P4", False),
    CoilDrive("p5_lower", "p5l_feed_current", "P5", False),
    CoilDrive("p5_upper", "p5u_feed_current", "P5", False),
    CoilDrive("p6_lower", "p6l_current", "P6", True),
    CoilDrive("p6_upper", "p6u_current", "P6", True),
)
"""One excitation channel per registry active component, in registry order."""

DERIVED_TURN_CHANNELS = {
    "p2_inner_lower": ("p2il_coil_current", "p2il_feed_current"),
    "p2_inner_upper": ("p2iu_coil_current", "p2iu_feed_current"),
    "p2_outer_lower": ("p2ol_coil_current", "p2ol_feed_current"),
    "p2_outer_upper": ("p2ou_coil_current", "p2ou_feed_current"),
    "p3_lower": ("p3l_coil_current", "p3l_feed_current"),
    "p3_upper": ("p3u_coil_current", "p3u_feed_current"),
    "p4_lower": ("p4l_coil_current", "p4l_feed_current"),
    "p4_upper": ("p4u_coil_current", "p4u_feed_current"),
    "p5_lower": ("p5l_coil_current", "p5l_feed_current"),
    "p5_upper": ("p5u_coil_current", "p5u_feed_current"),
}
"""Ampere-turn and conductor-current channel pairs whose ratio is a multiplier.

The store publishes, for ten of the thirteen coils, both the current measured in
one conductor and a derived ampere-turn channel the reconstruction consumes.
Their ratio is a constant the archive applied, so it is a statement about turn
count made by the archive rather than by a fit.  It is read as a cross-check on
the fitted amplitude and never as the fitted value.
"""

CASE_CURRENT_CHANNELS = {
    "p2_lower": "p2l_case_current",
    "p2_upper": "p2u_case_current",
    "p3_lower": "p3l_case_current",
    "p3_upper": "p3u_case_current",
    "p4_lower": "p4l_case_current",
    "p4_upper": "p4u_case_current",
    "p5_lower": "p5l_case_current",
    "p5_upper": "p5u_case_current",
}
"""Coil-case current channels, the only per-family passive current published."""

ERROR_FIELD_CHANNELS = ("error_field_02", "error_field_05", "efps_current")
"""Channels carrying the deliberately non-axisymmetric excitation.

A poloidal-field coil produces the same field at every toroidal angle, so no
poloidal vacuum shot can say where a probe sits toroidally.  These channels are
the only excitation in the store that can, which is why what they carry is
recorded per shot rather than left to be discovered later: a shot that drove them
is the discriminating experiment, and a store in which none did closes the search
by evidence instead of by silence.
"""

PROBE_FAMILIES = ("ccbv", "obr", "obv")
"""Poloidal field probe families, in the registry's block order."""

RADIAL_AXIS_FAMILIES = frozenset({"obr"})
"""Probe families whose sensitive axis lies along the major radius.

The registry stores one poloidal angle for all seventy-eight probes, so the
radial families are not distinguishable inside it.  Which families belong here
is therefore a hypothesis the vacuum response is asked to confirm, not an input
the fit may assume: :mod:`nova.imas.mast_vacuum_response` scores both
assignments and reports the margin between them.
"""

_CHANNEL_PATTERN = re.compile(r"^(ccbv|obr|obv)(\d{2})$")


def _series_partners() -> dict[str, str]:
    """Map each coil to the coil its documented circuit wires it in series with."""

    partners: dict[str, str] = {}
    for suffix, other in (("_upper", "_lower"), ("_lower", "_upper")):
        for drive in COIL_DRIVES:
            if drive.family.endswith(suffix):
                candidate = drive.family[: -len(suffix)] + other
                if any(row.family == candidate for row in COIL_DRIVES):
                    partners[drive.family] = candidate
    return partners


@dataclass(frozen=True, order=True)
class ProbeChannel:
    """One field probe, joined from the store's channel name to the registry."""

    family: str
    number: int
    channel: str
    registry_index: int


def probe_channels(
    probe_families: Sequence[Mapping[str, Any]],
) -> tuple[ProbeChannel, ...]:
    """Join ordered store channel names onto the registry's probe blocks.

    ``probe_families`` is the registry's poloidal-probe sequence.  Each family
    occupies one contiguous block whose order matches the store's channel
    numbering, so channel ``obr07`` is the seventh entry of the ``obr`` block.
    A number outside its block is rejected rather than clamped, because a
    silently shifted probe reports a field from the wrong place.
    """

    blocks: dict[str, list[int]] = {}
    for index, row in enumerate(probe_families):
        blocks.setdefault(str(row["family"]), []).append(index)
    channels = []
    for family in PROBE_FAMILIES:
        block = blocks.get(family)
        if not block:
            raise CohortError(f"registry carries no {family!r} probe block")
        for offset, registry_index in enumerate(block):
            channels.append(
                ProbeChannel(
                    family=family,
                    number=offset + 1,
                    channel=f"{family}{offset + 1:02d}",
                    registry_index=registry_index,
                )
            )
    return tuple(channels)


def parse_probe_channel(channel: str) -> tuple[str, int]:
    """Split a store field-probe channel into its family and channel number."""

    match = _CHANNEL_PATTERN.match(channel)
    if match is None:
        raise CohortError(f"unrecognised field probe channel {channel!r}")
    return match[1], int(match[2])


ENERGISED_CURRENT = 200.0
"""Amperes below which a coil carries too little current to move a probe.

This is the threshold for MODELLING a coil, so it sits at the pickup floor: a
channel reading above it is carrying real current and belongs in the prediction
whether or not anybody meant to drive it.  Leaving such a coil out of the model
puts its field into the residual; a separate and much higher threshold decides
what counts as a deliberate excitation.
"""

SUSTAINED_HOLD = 0.05
"""Seconds a coil must stay near its peak for its field to be read cleanly.

Set by the two kinds of experiment in the store, which are separated by two
orders of magnitude rather than by a judgement call: the sustained individual-coil
pulses hold for about 0.62 s, and the fast ones for about 0.006 s.  Anything in
between does not occur, so the boundary's exact position does not matter -- what
matters is that a few-millisecond pulse never enters a turn fit, because the coil
case is still carrying up to half the feed current when the probes are read.
"""

EXCITATION_CURRENT = 1.0e3
"""Amperes above which a supply was deliberately driving a coil.

This is the threshold for LABELLING a shot, and it has to sit well clear of the
standing current the vertical-control coils hold between pulses -- their peak is
around three hundred amperes on a median shot and reaches five hundred, so a
lower cut would label almost every shot in the store as a P6 experiment.  The
deliberate individual-coil pulses run to fifteen or twenty kiloamperes, more than
an order clear of this line.
"""

PLASMA_FREE_CURRENT = 5.0e3
"""Amperes of plasma current below which a shot is treated as plasma-free.

The plasma-current channel carries inductive pickup from the coil ramps, so a
genuine vacuum shot does not read zero.  The threshold sits above that pickup
and two orders below the four hundred kiloamperes of an ordinary MAST pulse, so
no plasma shot is admitted by it and no vacuum shot is rejected by its noise.
"""


@dataclass(frozen=True)
class SignalProvenance:
    """Where one number came from, precisely enough to fetch it again."""

    store: str
    shot: int
    group: str
    channel: str
    group_identity: str

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channel": self.channel,
            "group": self.group,
            "group_identity": self.group_identity,
            "shot": self.shot,
            "store": self.store,
        }


@dataclass(frozen=True)
class ShotSurvey:
    """The cheap per-shot summary that cohort selection reads."""

    shot: int
    plasma_current_peak: float
    toroidal_current_peak: float
    coil_peaks: Mapping[str, float]
    coil_hold_times: Mapping[str, float]
    turn_multipliers: Mapping[str, float]
    absent_groups: tuple[str, ...]
    absent_channels: tuple[str, ...]
    field_channels: tuple[str, ...]
    current_identity: str = ""
    field_identity: str = ""
    case_peaks: Mapping[str, float] = field(default_factory=dict)
    error_field_peaks: Mapping[str, float] = field(default_factory=dict)
    toroidal_hold_time: float = 0.0

    @property
    def error_field_driven(self) -> bool:
        """Return whether a non-axisymmetric coil was deliberately driven."""

        return any(
            peak >= EXCITATION_CURRENT for peak in self.error_field_peaks.values()
        )

    def sustained_coils(self, hold: float = SUSTAINED_HOLD) -> tuple[str, ...]:
        """Return the coils held near their peak for long enough to be read.

        The archive's individual-coil experiments come in two kinds and only one
        of them measures a turn count.  Some hold a coil at twenty kiloamperes for
        two thirds of a second, and by the time the probes are read the vessel and
        the coil case have given their induced currents back; the reading is the
        coil's own field.  Others fire a few-millisecond pulse, and every
        conducting structure nearby is still carrying a large opposing current
        when the probes are read -- the coil case alone reaches half the feed
        current.  Fitting a turn count to the second kind measures the coil and
        its case together and returns a number that is neither.
        """

        return tuple(
            sorted(
                family
                for family, duration in self.coil_hold_times.items()
                if duration >= hold
                and self.coil_peaks.get(family, 0.0) >= EXCITATION_CURRENT
            )
        )

    @property
    def readable(self) -> bool:
        """Return whether both the excitation and the response were readable."""

        return not self.absent_groups and bool(self.field_channels)

    @property
    def energised_families(self) -> tuple[str, ...]:
        """Return the coils carrying enough current to move a probe."""

        return tuple(
            sorted(
                family
                for family, peak in self.coil_peaks.items()
                if peak >= ENERGISED_CURRENT
            )
        )

    @property
    def excited_families(self) -> tuple[str, ...]:
        """Return the coils a supply deliberately drove on this shot."""

        return tuple(
            sorted(
                family
                for family, peak in self.coil_peaks.items()
                if peak >= EXCITATION_CURRENT
            )
        )

    @property
    def energised_circuits(self) -> tuple[str, ...]:
        """Return the circuits a supply deliberately drove on this shot."""

        circuit = {drive.family: drive.circuit for drive in COIL_DRIVES}
        return tuple(sorted({circuit[family] for family in self.excited_families}))

    def asymmetric_coils(self, contrast: float = 0.2) -> tuple[str, ...]:
        """Return coils driven while their series partner was not.

        A pair wired in series carries one waveform, so a shot that drives both
        cannot say how the pair's total divides between them.  A shot that drives
        one member alone can, and those shots are what make the individual turn
        counts identifiable at all rather than only their sums.
        """

        partner = _series_partners()
        alone = []
        for family, peak in self.coil_peaks.items():
            if peak < EXCITATION_CURRENT:
                continue
            other = partner.get(family)
            if other is None or self.coil_peaks.get(other, 0.0) < contrast * peak:
                alone.append(family)
        return tuple(sorted(alone))

    @property
    def excitation_family(self) -> str:
        """Return the label identifying which circuits this shot exercised."""

        circuits = self.energised_circuits
        return "+".join(circuits) if circuits else "none"

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "absent_channels": list(self.absent_channels),
            "absent_groups": list(self.absent_groups),
            "case_peaks": {k: float(v) for k, v in sorted(self.case_peaks.items())},
            "coil_hold_times": {
                k: float(v) for k, v in sorted(self.coil_hold_times.items())
            },
            "coil_peaks": {k: float(v) for k, v in sorted(self.coil_peaks.items())},
            "current_identity": self.current_identity,
            "error_field_peaks": {
                k: float(v) for k, v in sorted(self.error_field_peaks.items())
            },
            "excitation_family": self.excitation_family,
            "field_channels": list(self.field_channels),
            "field_identity": self.field_identity,
            "plasma_current_peak": float(self.plasma_current_peak),
            "shot": self.shot,
            "toroidal_current_peak": float(self.toroidal_current_peak),
            "toroidal_hold_time": float(self.toroidal_hold_time),
            "turn_multipliers": {
                k: float(v) for k, v in sorted(self.turn_multipliers.items())
            },
        }


def _peak(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else 0.0


def _hold_time(time: np.ndarray, values: np.ndarray) -> float:
    """Return how long a channel stayed within half of its own peak.

    Measured as sample count times the median sample interval rather than as the
    span between first and last crossing, so a channel that pulses twice reports
    the time it was actually driven and not the gap between the pulses.
    """

    finite = np.isfinite(values) & np.isfinite(time)
    if int(finite.sum()) < 2:
        return 0.0
    magnitude = np.abs(values[finite])
    peak = float(magnitude.max())
    if peak <= 0.0:
        return 0.0
    interval = float(np.median(np.diff(time[finite])))
    if not math.isfinite(interval) or interval <= 0.0:
        return 0.0
    return float(np.count_nonzero(magnitude > 0.5 * peak)) * interval


def survey_shot(shot: int, store: Path | str = SHOT_STORE) -> ShotSurvey:
    """Summarise one shot's excitation and response without loading waveforms.

    Every read is guarded: the store is missing whole groups on some shots and
    individual channels on many, and an absent channel is recorded rather than
    treated as a zero reading.
    """

    import zarr

    root = Path(store)
    absent_groups: list[str] = []
    absent_channels: list[str] = []
    coil_peaks: dict[str, float] = {}
    hold_times: dict[str, float] = {}
    multipliers: dict[str, float] = {}
    case_peaks: dict[str, float] = {}
    error_field_peaks: dict[str, float] = {}
    plasma_peak = 0.0
    toroidal_peak = 0.0
    toroidal_hold = 0.0
    current_identity = ""
    field_identity = ""
    field_channels: tuple[str, ...] = ()

    try:
        group = zarr.open_group(f"{root}/{shot}.zarr", mode="r")
    except Exception as error:  # noqa: BLE001 - a shot may be absent or corrupt
        raise CohortError(f"shot {shot} is unreadable in {root}") from error

    try:
        currents = group[CURRENT_GROUP]
    except Exception:  # noqa: BLE001 - group absence is data, not failure
        absent_groups.append(CURRENT_GROUP)
    else:
        keys = set(currents.keys())
        current_identity = str(dict(currents.attrs).get("uuid", ""))
        clock = (
            np.asarray(currents["time"][...], dtype=float)
            if "time" in keys
            else np.zeros(0)
        )
        if "plasma_current" in keys:
            plasma_peak = _peak(currents["plasma_current"][...]) * KILO
        else:
            absent_channels.append("plasma_current")
        if "tf_current" in keys:
            values = np.asarray(currents["tf_current"][...], dtype=float)
            toroidal_peak = _peak(values) * KILO
            toroidal_hold = _hold_time(clock, values)
        else:
            absent_channels.append("tf_current")
        for family, channel in sorted(CASE_CURRENT_CHANNELS.items()):
            if channel in keys:
                case_peaks[family] = _peak(currents[channel][...]) * KILO
            else:
                absent_channels.append(channel)
        for channel in ERROR_FIELD_CHANNELS:
            if channel in keys:
                error_field_peaks[channel] = _peak(currents[channel][...]) * KILO
            else:
                absent_channels.append(channel)
        for drive in COIL_DRIVES:
            if drive.channel in keys:
                values = np.asarray(currents[drive.channel][...], dtype=float)
                coil_peaks[drive.family] = _peak(values) * KILO
                hold_times[drive.family] = _hold_time(clock, values)
            else:
                absent_channels.append(drive.channel)
        for family, (turns, feed) in sorted(DERIVED_TURN_CHANNELS.items()):
            if turns in keys and feed in keys:
                ratio = _channel_ratio(currents[turns][...], currents[feed][...])
                if ratio is not None:
                    multipliers[family] = ratio

    try:
        waveforms = read_shot_waveforms(shot, store=root)
    except Exception:  # noqa: BLE001 - group absence is data, not failure
        absent_groups.append(FIELD_GROUP)
    else:
        field_identity = next(
            (
                row.group_identity
                for row in waveforms.provenance
                if row.group == FIELD_GROUP
            ),
            "",
        )
        field_channels = tuple(sorted(waveforms.probes))

    return ShotSurvey(
        shot=shot,
        plasma_current_peak=plasma_peak,
        toroidal_current_peak=toroidal_peak,
        coil_peaks=coil_peaks,
        coil_hold_times=hold_times,
        turn_multipliers=multipliers,
        absent_groups=tuple(absent_groups),
        absent_channels=tuple(sorted(absent_channels)),
        field_channels=field_channels,
        current_identity=current_identity,
        field_identity=field_identity,
        case_peaks=case_peaks,
        error_field_peaks=error_field_peaks,
        toroidal_hold_time=toroidal_hold,
    )


def _channel_ratio(turns: np.ndarray, feed: np.ndarray) -> float | None:
    """Return the constant ratio between two channels over the driven samples."""

    turns = np.asarray(turns, dtype=float)
    feed = np.asarray(feed, dtype=float)
    if turns.shape != feed.shape:
        return None
    scale = _peak(feed)
    if scale <= 0.0:
        return None
    mask = np.isfinite(turns) & np.isfinite(feed) & (np.abs(feed) > 0.3 * scale)
    if int(mask.sum()) < 8:
        return None
    return float(np.median(turns[mask] / feed[mask]))


def store_shots(store: Path | str = SHOT_STORE) -> tuple[int, ...]:
    """Return every shot the store holds, in ascending order."""

    root = Path(store)
    if not root.is_dir():
        raise CohortError(f"shot store {root} is not a directory")
    shots = []
    for entry in root.iterdir():
        if entry.suffix != ".zarr":
            continue
        try:
            shots.append(int(entry.stem))
        except ValueError:
            continue
    if not shots:
        raise CohortError(f"shot store {root} holds no shot")
    return tuple(sorted(shots))


def _survey_one(argument: tuple[int, str]) -> ShotSurvey | None:
    shot, store = argument
    try:
        return survey_shot(shot, store)
    except CohortError:
        return None


def survey_store(
    shots: Sequence[int] | None = None,
    *,
    store: Path | str = SHOT_STORE,
    processes: int = 1,
) -> tuple[ShotSurvey, ...]:
    """Survey every shot in the store, in parallel when asked.

    Unreadable shots are dropped rather than raised: the store carries shots
    with no current group at all, and a census that stops at the first one never
    reaches the cohort.  How many were dropped is recoverable by comparing the
    result against :func:`store_shots`.
    """

    root = str(Path(store))
    selection = tuple(store_shots(root)) if shots is None else tuple(shots)
    arguments = [(shot, root) for shot in selection]
    if processes > 1:
        import multiprocessing

        with multiprocessing.Pool(processes) as pool:
            results = pool.map(_survey_one, arguments, chunksize=16)
    else:
        results = [_survey_one(argument) for argument in arguments]
    return tuple(row for row in results if row is not None)


@dataclass(frozen=True)
class ShotExclusion:
    """One shot kept out of the cohort, and the reason it was kept out."""

    shot: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {"reason": self.reason, "shot": self.shot}


@dataclass(frozen=True)
class VacuumCohort:
    """The admitted shots, their split, and every shot that was refused."""

    training: tuple[int, ...]
    held_out: tuple[int, ...]
    families: Mapping[int, str]
    exclusions: tuple[ShotExclusion, ...] = ()
    held_out_families: tuple[str, ...] = ()

    def validate(self) -> None:
        """Reject a cohort whose split leaks a shot or a family."""

        overlap = set(self.training) & set(self.held_out)
        if overlap:
            raise CohortError(f"shots {sorted(overlap)} are in both cohort arms")
        for shot in (*self.training, *self.held_out):
            if shot not in self.families:
                raise CohortError(f"shot {shot} carries no excitation family")
        if not self.held_out:
            raise CohortError("a cohort must hold out at least one shot")
        held = {self.families[shot] for shot in self.held_out}
        trained = {self.families[shot] for shot in self.training}
        if not set(self.held_out_families) <= held:
            missing = sorted(set(self.held_out_families) - held)
            raise CohortError(f"held-out families {missing} have no held-out shot")
        if set(self.held_out_families) & trained:
            leaked = sorted(set(self.held_out_families) & trained)
            raise CohortError(f"held-out families {leaked} also appear in training")

    @property
    def shots(self) -> tuple[int, ...]:
        """Return every admitted shot in ascending order."""

        return tuple(sorted((*self.training, *self.held_out)))

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "exclusions": [row.as_dict() for row in self.exclusions],
            "families": {str(k): v for k, v in sorted(self.families.items())},
            "held_out": list(self.held_out),
            "held_out_families": list(self.held_out_families),
            "training": list(self.training),
        }


def _family_rank(family: str) -> tuple[int, str]:
    """Order families by how many circuits they exercise, then by name."""

    return (len(family.split("+")), family)


def select_vacuum_cohort(
    surveys: Iterable[ShotSurvey],
    *,
    held_out_families: Sequence[str] = (),
    held_out_fraction: float = 0.25,
    minimum_probes: int = 40,
) -> VacuumCohort:
    """Admit the plasma-free, excited, readable shots and split them.

    A shot is admitted when its plasma current stays below the plasma-free
    threshold, at least one coil is driven above the noise floor, and enough
    field probes are present to over-determine the coil amplitudes.  Every
    refusal is recorded with its reason so the cohort's boundary can be audited
    without re-reading the store.

    The split holds out whole shots, never time windows inside a shot, because
    two windows of one shot share the same coil calibration and the same
    acquisition state and would not challenge a fitted parameter.  Whole
    excitation families named in ``held_out_families`` are withheld from
    training so a fitted amplitude is tested against a coil combination it never
    saw.
    """

    admitted: list[ShotSurvey] = []
    exclusions: list[ShotExclusion] = []
    for survey in sorted(surveys, key=lambda row: row.shot):
        if survey.absent_groups:
            exclusions.append(
                ShotExclusion(
                    survey.shot,
                    f"store groups absent: {', '.join(survey.absent_groups)}",
                )
            )
            continue
        if survey.plasma_current_peak >= PLASMA_FREE_CURRENT:
            exclusions.append(
                ShotExclusion(
                    survey.shot,
                    f"plasma current peak {survey.plasma_current_peak:.0f} A "
                    f"reaches the plasma-free limit {PLASMA_FREE_CURRENT:.0f} A",
                )
            )
            continue
        if not survey.excited_families:
            exclusions.append(
                ShotExclusion(
                    survey.shot,
                    f"no coil deliberately driven above {EXCITATION_CURRENT:.0f} A",
                )
            )
            continue
        if len(survey.field_channels) < minimum_probes:
            exclusions.append(
                ShotExclusion(
                    survey.shot,
                    f"only {len(survey.field_channels)} field probes present, "
                    f"below the {minimum_probes} needed to over-determine the fit",
                )
            )
            continue
        admitted.append(survey)

    if not admitted:
        raise CohortError("no shot in the survey satisfies the vacuum criteria")

    families = {survey.shot: survey.excitation_family for survey in admitted}
    withheld = tuple(sorted(set(held_out_families), key=_family_rank))
    unknown = [name for name in withheld if name not in set(families.values())]
    if unknown:
        raise CohortError(f"held-out families {unknown} occur in no admitted shot")

    held_out = [shot for shot in families if families[shot] in withheld]
    remaining = [shot for shot in families if shot not in set(held_out)]
    by_family: dict[str, list[int]] = {}
    for shot in remaining:
        by_family.setdefault(families[shot], []).append(shot)
    for family in sorted(by_family, key=_family_rank):
        members = sorted(by_family[family])
        take = (
            max(1, round(len(members) * held_out_fraction)) if len(members) > 1 else 0
        )
        held_out.extend(members[len(members) - take :] if take else [])
    held = set(held_out)
    cohort = VacuumCohort(
        training=tuple(sorted(shot for shot in families if shot not in held)),
        held_out=tuple(sorted(held)),
        families=families,
        exclusions=tuple(exclusions),
        held_out_families=withheld,
    )
    cohort.validate()
    return cohort


@dataclass(frozen=True)
class ExcludedWindow:
    """One time span dropped from a shot, and the physics that dropped it."""

    shot: int
    start: float
    stop: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "reason": self.reason,
            "shot": self.shot,
            "start": float(self.start),
            "stop": float(self.stop),
        }


@dataclass
class ShotWaveforms:
    """One shot's excitation and response on a common time base.

    ``scale_corrections`` records what the acquisition range setting of each probe
    channel was divided out by, one row per channel including the channels nothing
    was divided out of.  It is carried on the waveforms rather than applied
    silently because a channel read at unity because its scale was measured to be
    unity and one read at unity because nobody measured it are different
    statements, and only the first is a calibration.

    ``sample_mask`` marks the samples a fit may read.  It is the conjunction of
    every window test, and the windows it removed are listed in ``excluded`` so
    the boundary is visible without re-deriving it.

    ``baseline_mask`` marks the samples a probe's standing offset is measured in
    and deliberately survives the ramp-rate test.  The offset window sits before
    any coil was driven, so a test on how fast the coils are slewing has nothing
    to say about it, and letting that test consume the window would leave the
    zero of every channel unmeasurable on exactly the shots with the cleanest
    excitation.

    ``shape_mismatched_sensors`` names field arrays that exist in the archive but
    cannot be put on its field clock because their sample shapes differ.  They are
    deliberately absent from ``sensors``; retaining their identities keeps that
    refusal distinct from a channel the archive does not contain.
    """

    shot: int
    time: np.ndarray
    drives: Mapping[str, np.ndarray]
    probes: Mapping[str, np.ndarray]
    sensors: Mapping[str, np.ndarray]
    plasma_current: np.ndarray
    sample_mask: np.ndarray
    baseline_mask: np.ndarray
    excluded: tuple[ExcludedWindow, ...] = ()
    provenance: tuple[SignalProvenance, ...] = field(default_factory=tuple)
    scale_corrections: tuple[ScaleCorrection, ...] = field(default_factory=tuple)
    shape_mismatched_sensors: tuple[str, ...] = ()

    @property
    def sample_count(self) -> int:
        """Return how many time samples survived the window tests."""

        return int(np.count_nonzero(self.sample_mask))

    @property
    def scaled_channels(self) -> tuple[str, ...]:
        """Return the channels an acquisition scale was divided out of."""

        return tuple(
            sorted(
                row.channel
                for row in self.scale_corrections
                if row.applied and row.scale != 1.0
            )
        )

    @property
    def unscaled_channels(self) -> tuple[str, ...]:
        """Return the channels read raw because no scale warranted a correction."""

        return tuple(
            sorted(row.channel for row in self.scale_corrections if not row.applied)
        )


def _resample(
    time: np.ndarray,
    source_time: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Interpolate a channel onto the response time base.

    The excitation and the response are acquired on different clocks at
    different rates, and the response is the slower of the two, so the coil
    currents are read at the probe sample times.  Interpolation outside the
    excitation's own span returns a masked value rather than the end sample,
    because holding the last current flat asserts a drive that was not recorded.
    """

    finite = np.isfinite(source_time) & np.isfinite(values)
    if int(finite.sum()) < 2:
        return np.full(time.shape, np.nan)
    order = np.argsort(source_time[finite])
    src = source_time[finite][order]
    val = values[finite][order]
    result = np.interp(time, src, val, left=np.nan, right=np.nan)
    return np.where((time >= src[0]) & (time <= src[-1]), result, np.nan)


def read_shot_waveforms(
    shot: int,
    *,
    store: Path | str = SHOT_STORE,
    quiescent_ramp_fraction: float = 0.0,
    block_scale: ScaleReader | None = None,
) -> ShotWaveforms:
    """Read one shot's coil excitations and probe responses, windows marked.

    Currents arrive in kiloamperes on the excitation clock and are returned in
    amperes on the probe clock.  Two window tests run here: samples where the
    plasma-current channel exceeds the plasma-free threshold are dropped, and so
    are samples where any admitted channel is not finite.  A third test, on how
    fast the excitation is changing, is available through
    ``quiescent_ramp_fraction`` and is the one that suppresses induced vessel
    current; it is left off by default so the passive fit can see the transient
    the turn fit wants excluded.

    Probe channels are returned with their measured acquisition range setting
    divided out, because a channel recorded at twice its usual setting is not a
    measurement of twice the field.  ``block_scale`` names what supplies the setting
    and defaults to the machine's correction document; passing a
    :class:`~nova.imas.mast_block_scale.BlockScaleTable` with no blocks reads the
    archive exactly as published.  What was divided out of each
    channel -- including the channels nothing was divided out of, and why -- comes
    back in ``scale_corrections``.
    """

    import zarr

    root = Path(store)
    group = zarr.open_group(f"{root}/{shot}.zarr", mode="r")
    currents = group[CURRENT_GROUP]
    fields = group[FIELD_GROUP]
    current_keys = set(currents.keys())
    field_keys = set(fields.keys())
    current_identity = str(dict(currents.attrs).get("uuid", ""))
    field_identity = str(dict(fields.attrs).get("uuid", ""))

    time = np.asarray(fields["time"][...], dtype=float)
    source_time = np.asarray(currents["time"][...], dtype=float)
    provenance = [
        SignalProvenance(str(root), shot, FIELD_GROUP, "time", field_identity),
        SignalProvenance(str(root), shot, CURRENT_GROUP, "time", current_identity),
    ]

    drives: dict[str, np.ndarray] = {}
    for drive in COIL_DRIVES:
        if drive.channel not in current_keys:
            continue
        raw = np.asarray(currents[drive.channel][...], dtype=float) * KILO
        drives[drive.family] = _resample(time, source_time, raw)
        provenance.append(
            SignalProvenance(
                str(root), shot, CURRENT_GROUP, drive.channel, current_identity
            )
        )

    plasma = np.zeros_like(time)
    if "plasma_current" in current_keys:
        plasma = _resample(
            time,
            source_time,
            np.asarray(currents["plasma_current"][...], dtype=float) * KILO,
        )
        provenance.append(
            SignalProvenance(
                str(root), shot, CURRENT_GROUP, "plasma_current", current_identity
            )
        )

    sensors: dict[str, np.ndarray] = {}
    shape_mismatched_sensors: list[str] = []
    for channel in sorted(field_keys):
        if channel == "time":
            continue
        values = np.asarray(fields[channel][...], dtype=float)
        provenance.append(
            SignalProvenance(str(root), shot, FIELD_GROUP, channel, field_identity)
        )
        if values.shape != time.shape:
            shape_mismatched_sensors.append(channel)
            continue
        sensors[channel] = values
    table = promoted_block_scales() if block_scale is None else block_scale
    sensors, corrections = table.normalise(shot, sensors)
    probes = {
        channel: values
        for channel, values in sensors.items()
        if _CHANNEL_PATTERN.match(channel)
    }

    mask = np.ones(time.shape, dtype=bool)
    excluded: list[ExcludedWindow] = []
    for name, values in drives.items():
        bad = ~np.isfinite(values)
        if bad.any():
            excluded.extend(_windows(shot, time, bad, f"drive {name} not recorded"))
        mask &= ~bad
    plasma_bad = ~np.isfinite(plasma) | (np.abs(plasma) >= PLASMA_FREE_CURRENT)
    if plasma_bad.any():
        excluded.extend(
            _windows(
                shot,
                time,
                plasma_bad,
                f"plasma current reaches {PLASMA_FREE_CURRENT:.0f} A",
            )
        )
    mask &= ~plasma_bad

    quiet = mask.copy()
    for values in drives.values():
        quiet &= np.abs(np.nan_to_num(values)) < ENERGISED_CURRENT

    if quiescent_ramp_fraction > 0.0:
        ramping = _ramping(time, drives, quiescent_ramp_fraction)
        if ramping.any():
            excluded.extend(
                _windows(
                    shot,
                    time,
                    ramping,
                    "excitation slews fast enough to drive vessel current",
                )
            )
        mask &= ~ramping

    return ShotWaveforms(
        shot=shot,
        time=time,
        drives=drives,
        probes=probes,
        sensors=sensors,
        plasma_current=plasma,
        sample_mask=mask,
        baseline_mask=quiet,
        excluded=tuple(excluded),
        provenance=tuple(provenance),
        scale_corrections=corrections,
        shape_mismatched_sensors=tuple(shape_mismatched_sensors),
    )


@dataclass(frozen=True)
class RawArchiveReader:
    """Explicitly read field signals exactly as the archive published them."""

    store: Path | str = SHOT_STORE

    def read_shot_waveforms(
        self, shot: int, *, quiescent_ramp_fraction: float = 0.0
    ) -> ShotWaveforms:
        """Read a shot without applying any correction document entries.

        Field arrays without the field clock's shape remain refused and are named
        by the returned ``shape_mismatched_sensors`` provenance.
        """

        return read_shot_waveforms(
            shot,
            store=self.store,
            quiescent_ramp_fraction=quiescent_ramp_fraction,
            block_scale=BlockScaleTable(),
        )


RAW_ARCHIVE = RawArchiveReader()
"""The sole production entry point for archive-as-published field reads."""


VESSEL_TIME = 5.0e-3
"""Seconds of vessel current decay time used to scale the ramp-rate test.

An induced current is of order ``tau dI/dt``, so comparing ``tau dI/dt`` against
the coil's own peak current is what says whether the vessel is carrying a
noticeable fraction of the excitation.  The value is the order of magnitude
documented for MAST's reduced vessel model and is used only to set the window
boundary -- it is a screening scale, never a fitted parameter.
"""


def _ramping(
    time: np.ndarray,
    drives: Mapping[str, np.ndarray],
    fraction: float,
) -> np.ndarray:
    """Mark samples where a driven coil slews fast enough to load the vessel.

    The test runs only on channels that carry a real excitation.  A channel
    sitting at its noise floor has a rate dominated by that noise, so scaling
    against its own peak rate would flag the whole record -- including the
    pre-pulse window the probe offsets are measured in -- on the strength of a
    coil that was never driven.  The threshold compares the induced current
    scale ``tau dI/dt`` against the channel's peak current instead, which is a
    statement about the machine and not about the acquisition noise.
    """

    ramping = np.zeros(time.shape, dtype=bool)
    for values in drives.values():
        peak = _peak(values)
        if peak < ENERGISED_CURRENT:
            continue
        rate = np.gradient(np.nan_to_num(values), time)
        ramping |= VESSEL_TIME * np.abs(rate) > fraction * peak
    return ramping


def _windows(
    shot: int,
    time: np.ndarray,
    flag: np.ndarray,
    reason: str,
) -> list[ExcludedWindow]:
    """Collapse a per-sample rejection flag into contiguous time spans."""

    windows: list[ExcludedWindow] = []
    indices = np.flatnonzero(flag)
    if indices.size == 0:
        return windows
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate([[indices[0]], indices[breaks + 1]])
    stops = np.concatenate([indices[breaks], [indices[-1]]])
    for start, stop in zip(starts, stops, strict=True):
        windows.append(
            ExcludedWindow(shot, float(time[start]), float(time[stop]), reason)
        )
    return windows
