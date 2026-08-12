"""The per-slice inputs a MAST equilibrium solve reads, mapped onto the description.

A solve consumes three things from a shot: what current was driven through each
described conductor, what each described sensor read, and the plasma current a
Rogowski measured.  The archive publishes all three, and none of them arrives in
the form a solve wants.  The currents are in kiloamperes, several of them have
already been multiplied by a turn count, and one feeds two parallel circuits.  The
sensor channels are named by the array they belong to, while the description
orders its sensors by geometry, so which channel reads which sensor is a statement
somebody has to establish.  And every quantity carries the source's coordinate
convention, which is not the dictionary's.

This module states each of those once, per channel, as a
:class:`~nova.io.sourcemap.SourceSignal` the description can be checked against.
Three joins carry the weight, and each rests on different evidence.

**Conductor currents** come from the description itself.  The artifact's drive map
already says what one ampere of a channel drives in ampere turns, and its
``turns_with_sign`` says how many turns the conductor has, so the conductor current
one ampere of the channel implies is the ratio of the two.  Written that way the
map has no separate table of channel semantics to drift out of step -- and the one
conductor pair whose turn count no experiment fixed drops out of the served set by
arithmetic rather than by a special case, because a ratio with an unset
denominator is not a number.

**Probe channels** join to described probes by family block and channel number,
the join the vacuum cohort established.  The reconstruction's own input array
gives an independent read on it: its columns are ordered by the same static
geometry the description was built from, so matching a channel's trace against
those columns identifies the sensor without consulting a channel description.
Where the traces are distinct enough to discriminate, they confirm the block join.

**Flux-loop channels** cannot use their descriptions at all.  Twenty-seven channels
carry eleven distinct claimed positions between them, several families repeating
one position, so a position read out of a channel description is not evidence of
anything.  The channel *number*, on the other hand, is the reconstruction's own
loop index within a family block, which the trace match establishes and which then
reaches a described loop through the reconstruction's measured loop position.

Every clock is left alone.  The excitation and the response are acquired on
different clocks at different rates, and a map that resampled one onto the other
would be making a modelling choice on a consumer's behalf; the dictionary's
heterogeneous time lets each signal carry the clock it was measured on.

Pack totals are not served either, and for a reason the store settles rather than
leaves open.  Six coil sets publish a third current channel beside their conductor
and ampere-turn ones, and that channel is the sum of the coil's ampere turns and
the current its own case carries -- measured, to the last bit a 32-bit float
holds.  It is therefore a restatement of two quantities the map already reaches,
and it is not servable as either of them: the case current is induced, so the
total is not a fixed multiple of the coil, and the two terms sit in different
containers of the dictionary with no field spanning them.

Circuit currents are deliberately not served.  A series pair carries one current,
and the store publishes a feed channel for each of its two coils; on the pilot
shots those two agree to within a few per cent of peak, which is the coherence a
series connection predicts and also the reason a circuit row would be a choice
rather than a copy.  The coil rows already carry both measurements, so writing
either of them into a circuit field as well would put one measurement in two
places and hide which channel a residual came from.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray

from nova.imas.machine_drive import DriveMap
from nova.imas.machine_evidence import FieldEvidence
from nova.imas.mast_block_scale import (
    ScaleCorrection,
    ScaleReader,
)
from nova.imas.mast_vacuum_cohort import (
    COIL_DRIVES,
    CURRENT_GROUP,
    FIELD_GROUP,
    KILO,
    RawArchiveReader,
    SHOT_STORE,
    CohortError,
    SignalProvenance,
    parse_probe_channel,
    probe_channels,
    read_shot_waveforms,
)
from nova.io.cocos import IP_LIKE, ONE_LIKE, PSI_LIKE
from nova.io.sourcemap import (
    ACCEPTED,
    BlockedSignal,
    SourceSignal,
    SourceSignalMap,
    tensorize,
)
from nova.io.standardname import StandardNameResolver

SOURCE_CONVENTION = 3
"""The convention the MAST sources are written in, measured from the data."""

TARGET_CONVENTION = 17
"""The convention DD 4 is written in."""

MEASURED_FLUX_CONVENTION = 13
"""The source's own sign senses with the 2*pi already inside the flux.

A flux loop links the total flux in Weber while the reconstruction's flux function
is per radian, so the two differ in exactly the digit that says whether the 2*pi
is in the quantity -- and a convention differing from another in only that digit
is a convention ten places along the same table.  Declaring the measured loops
here rather than special-casing them is what lets one algebra hand the measured
channel a factor of one and the reconstructed flux function a factor of 2*pi.
The digit is measured, not read off the unit string: the ratio of a loop's fitted
flux to the reconstructed flux interpolated to that loop's own position is 2*pi.
"""

PACK_TOTAL_TOLERANCE = 1.2e-7
"""How far a pack total may sit from its coil plus its case, per unit of the terms.

The three channels are recorded as 32-bit floats, so a sum of two of them can
reproduce the third only to the last bit either carries, and the bound here is
that resolution rather than a physical allowance.  The measured worst case is
1.09e-07 of the summed terms over 4578 coil-set and shot pairs drawn across the
whole campaign, with a median of 9.7e-09; a store whose totals were acquired
independently of the two terms would not sit inside a float's own rounding.
"""

CURRENT_CLOCK = "current"
"""Clock the excitation channels are acquired on."""

FIELD_CLOCK = "field"
"""Clock the magnetic diagnostics are acquired on."""

RECONSTRUCTION_GROUP = "efm"
"""Store group whose fitted inputs identify which sensor a channel reads."""

COIL_CURRENT_NAME = "current_of_poloidal_field_coil"
PROBE_FIELD_NAME = "poloidal_magnetic_field_of_poloidal_magnetic_field_probe"
LOOP_FLUX_NAME = "poloidal_magnetic_flux_of_flux_loop"
PLASMA_CURRENT_NAME = "plasma_current"

FULL_LOOP_TYPE = 1
"""Dictionary loop-type index of a toroidally closed flux loop."""

LOOP_POSITION_TOLERANCE = 0.015
"""Metres a reconstruction loop may sit from a described one and still be it.

The registry's own diagnostic position tolerance, set by the spread between the
static setup variants rather than chosen here.  A reconstruction loop further than
this from every described loop is a loop the description does not carry, which is
a gap to report and not a match to loosen a threshold for.
"""

LOOP_FAMILY_SIZES: tuple[tuple[str, int], ...] = (
    ("cc", 10),
    ("p2u", 4),
    ("p3u", 4),
    ("p4u", 4),
    ("p5u", 4),
    ("p6u", 2),
    ("p2l", 4),
    ("p3l", 4),
    ("p4l", 4),
    ("p5l", 4),
    ("p6l", 2),
)
"""How many loops each named family contributes, in reconstruction index order.

Measured, not assumed.  Every published loop channel's trace matches exactly one
column of the reconstruction's fitted loop input, the matched column is
``block start + channel number - 1`` for all of them, and the family sizes above
are what those blocks are.  The block boundaries are visible in the numbering
itself: the centre-column family omits two of its ten channels in some shots and
the columns those channels would occupy are skipped rather than closed up, so the
channel number indexes a fixed layout rather than a per-shot list.
"""

RECONSTRUCTION_LOOP_COUNT = sum(size for _, size in LOOP_FAMILY_SIZES)
"""Loops the reconstruction carries; a store disagreeing is a different layout."""

_LOOP_CHANNEL_PATTERN = re.compile(r"^fl_(?:(cc)(\d{2})|(p\d[ul])_(\d))$")


class SolveInputError(ValueError):
    """Raised when a solve input cannot be mapped onto the description."""


def parse_loop_channel(channel: str) -> tuple[str, int]:
    """Split a flux-loop channel into its family and its number within the family."""

    match = _LOOP_CHANNEL_PATTERN.match(channel)
    if match is None:
        raise SolveInputError(f"unrecognised flux-loop channel {channel!r}")
    family = match[1] or match[3]
    return family, int(match[2] or match[4])


def loop_channel_name(family: str, number: int) -> str:
    """Return the store channel a family and number name.

    The centre-column family numbers its channels with two digits and the
    coil-mounted families with one, which is the archive's own convention and the
    reason a single format string does not cover both.
    """

    if family == "cc":
        return f"fl_{family}{number:02d}"
    return f"fl_{family}_{number}"


def reconstruction_loop_rows() -> dict[str, int]:
    """Return the reconstruction loop index each flux-loop channel reads."""

    rows: dict[str, int] = {}
    start = 0
    for family, size in LOOP_FAMILY_SIZES:
        for number in range(1, size + 1):
            rows[loop_channel_name(family, number)] = start + number - 1
        start += size
    return rows


@dataclass(frozen=True)
class DescribedMachine:
    """What the map needs to read out of the description it fills.

    Held as plain sequences rather than as open IDSs so the map can be built,
    digested and tested without an open pulse, and so the identity it was built
    against is fixed at read time instead of drifting under it.
    """

    dd_version: str
    coils: tuple[str, ...]
    turns: Mapping[str, float | None]
    probes: tuple[str, ...]
    probe_poses: np.ndarray
    loops: tuple[str, ...]
    loop_positions: np.ndarray
    passive_loops: tuple[str, ...]
    passive_elements: Mapping[str, int]
    drives: DriveMap

    def coil_index(self, name: str) -> int:
        """Return the position of one described coil."""

        return self.coils.index(name)

    def validate(self) -> None:
        """Reject a description whose sensor arrays and positions disagree."""

        if len(self.loops) != len(self.loop_positions):
            raise SolveInputError(
                f"{len(self.loops)} described loops carry "
                f"{len(self.loop_positions)} positions"
            )
        if len(self.probes) != len(self.probe_poses):
            raise SolveInputError(
                f"{len(self.probes)} described probes carry "
                f"{len(self.probe_poses)} poses"
            )
        self.drives.validate()


def describe_machine(
    ids: Mapping[str, Any],
    *,
    dd_version: str,
    drives: DriveMap,
) -> DescribedMachine:
    """Read the described conductors and sensors out of an opened machine description.

    Turn counts arrive as the dictionary's own empty value where no experiment
    fixed them, and are carried here as ``None`` rather than as that sentinel: a
    conductor whose turn count is unset has no conductor current to serve, and the
    difference between "unset" and "a very large number" is the difference between
    a blocked channel and a wrong column.
    """

    pf_active = ids["pf_active"]
    magnetics = ids["magnetics"]
    pf_passive = ids["pf_passive"]
    coils = tuple(str(coil.name) for coil in pf_active.coil)
    turns: dict[str, float | None] = {}
    for coil in pf_active.coil:
        values = [_finite_value(element.turns_with_sign) for element in coil.element]
        turns[str(coil.name)] = None if any(v is None for v in values) else values[0]
    loops = []
    positions = []
    for loop in magnetics.flux_loop:
        if int(loop.type.index) != FULL_LOOP_TYPE:
            continue
        loops.append(str(loop.name))
        positions.append([float(loop.position[0].r), float(loop.position[0].z)])
    machine = DescribedMachine(
        dd_version=str(dd_version),
        coils=coils,
        turns=turns,
        probes=tuple(str(probe.name) for probe in magnetics.b_field_pol_probe),
        probe_poses=np.asarray(
            [
                [
                    float(probe.position.r),
                    float(probe.position.z),
                    float(probe.poloidal_angle),
                ]
                for probe in magnetics.b_field_pol_probe
            ],
            dtype=float,
        ),
        loops=tuple(loops),
        loop_positions=np.asarray(positions, dtype=float).reshape(len(loops), 2),
        passive_loops=tuple(str(loop.name) for loop in pf_passive.loop),
        passive_elements={
            str(loop.name): len(loop.element) for loop in pf_passive.loop
        },
        drives=drives,
    )
    machine.validate()
    return machine


def _finite_value(value: Any) -> float | None:
    """Return a dictionary float, or None when it carries the empty value."""

    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number) or abs(number) > 1.0e30:
        return None
    return number


def reconstruction_loop_positions(
    shot: int,
    *,
    store: Path | str = SHOT_STORE,
) -> np.ndarray:
    """Return the reconstruction's own measured loop positions for one shot."""

    import zarr

    group = zarr.open_group(f"{Path(store)}/{shot}.zarr", mode="r")
    if RECONSTRUCTION_GROUP not in group:
        raise SolveInputError(
            f"shot {shot} carries no {RECONSTRUCTION_GROUP!r} group, so the loop "
            "channels have no measured position to reach a described loop through"
        )
    reconstruction = group[RECONSTRUCTION_GROUP]
    radius = np.asarray(reconstruction["silop_r"][...], dtype=float)
    height = np.asarray(reconstruction["silop_z"][...], dtype=float)
    if radius.size != RECONSTRUCTION_LOOP_COUNT:
        raise SolveInputError(
            f"shot {shot} carries {radius.size} reconstruction loops against the "
            f"{RECONSTRUCTION_LOOP_COUNT} the channel blocks account for"
        )
    return np.column_stack([radius, height])


def reconstruction_probe_poses(
    shot: int,
    *,
    store: Path | str = SHOT_STORE,
) -> np.ndarray:
    """Return the reconstruction's own probe positions and sensitive-axis angles.

    Columns are the major radius, the height and the poloidal angle in radians;
    the source states the angle in degrees.  This is the array the description's
    probe poses were built from, so comparing a channel's matched column against
    the described probe it is mapped to tests the join against geometry rather
    than against an index convention.
    """

    import zarr

    group = zarr.open_group(f"{Path(store)}/{shot}.zarr", mode="r")
    if RECONSTRUCTION_GROUP not in group:
        raise SolveInputError(
            f"shot {shot} carries no {RECONSTRUCTION_GROUP!r} group, so the probe "
            "channels have no independent pose to be checked against"
        )
    reconstruction = group[RECONSTRUCTION_GROUP]
    return np.column_stack(
        [
            np.asarray(reconstruction["magpr_r"][...], dtype=float),
            np.asarray(reconstruction["magpr_z"][...], dtype=float),
            np.deg2rad(np.asarray(reconstruction["magpr_ang"][...], dtype=float)),
        ]
    )


def loop_target_indices(
    machine: DescribedMachine,
    positions: np.ndarray,
    *,
    tolerance: float = LOOP_POSITION_TOLERANCE,
) -> dict[str, int | None]:
    """Return the described loop each channel reaches, or None where none does.

    The match is one-to-one and nearest-first: a reconstruction loop claims the
    described loop it is closest to, and a described loop already claimed is not
    claimed twice.  Two loops of a pair sit fifteen millimetres apart, so a
    many-to-one match would silently put two channels on one sensor.
    """

    rows = reconstruction_loop_rows()
    distances = [
        (
            float(
                np.hypot(
                    machine.loop_positions[:, 0] - positions[row, 0],
                    machine.loop_positions[:, 1] - positions[row, 1],
                ).min()
            ),
            channel,
            row,
        )
        for channel, row in rows.items()
    ]
    assigned: dict[str, int | None] = {channel: None for channel in rows}
    taken: set[int] = set()
    for distance, channel, row in sorted(distances):
        if distance > tolerance:
            continue
        separation = np.hypot(
            machine.loop_positions[:, 0] - positions[row, 0],
            machine.loop_positions[:, 1] - positions[row, 1],
        )
        order = np.argsort(separation)
        for index in order:
            if separation[index] > tolerance:
                break
            if int(index) in taken:
                continue
            assigned[channel] = int(index)
            taken.add(int(index))
            break
    return assigned


AMPERE_TURN_UNIT = "kA.turn"
"""Unit of a channel the archive multiplied by a turn count before publishing."""

CONDUCTOR_UNIT = "kA"
"""Unit of a channel measuring the current in one conductor of a winding."""


def _channel_unit(drive: Any, turns: float) -> str:
    """Return whether a channel publishes a conductor current or a turn product.

    Derived from the drive rather than from the channel's name.  One ampere in a
    conductor of an eight-turn winding drives eight ampere turns, so a channel of
    that winding whose ampere drives only one ampere turn must already have been
    multiplied.  For a single-turn conductor the two readings coincide and the
    distinction has nothing to distinguish.
    """

    if abs(turns) > 1.0 and math.isclose(
        float(drive.ampere_turns_per_ampere), 1.0, rel_tol=1e-9
    ):
        return AMPERE_TURN_UNIT
    return CONDUCTOR_UNIT


def coil_current_signals(
    machine: DescribedMachine,
) -> tuple[tuple[SourceSignal, ...], tuple[BlockedSignal, ...]]:
    """Return one row per excitation channel whose conductor current is derivable.

    The conductor current one ampere of a channel implies is the ampere turns it
    drives divided by the conductor's turn count.  That single relation covers all
    three channel kinds without naming any of them: a channel measuring one
    conductor drives its turn count and converts at one, a channel already
    multiplied drives one ampere turn and converts at the reciprocal of the turn
    count, and a channel feeding parallel circuits drives the turn count times its
    split.  It is also why an unset turn count blocks a channel rather than
    defaulting it -- the ratio has no denominator, and a weight of one in its place
    would be a fabricated column.
    """

    signals: list[SourceSignal] = []
    blocked: list[BlockedSignal] = []
    for drive in machine.drives.drives:
        if drive.container != "pf_active":
            continue
        conductor = drive.conductor
        target = f"pf_active/coil({conductor})/current/data"
        turns = machine.turns.get(conductor)
        unit = "kA" if turns is None else _channel_unit(drive, turns)
        if turns is None:
            blocked.append(
                BlockedSignal(
                    source_group=CURRENT_GROUP,
                    source_channel=drive.channel,
                    target_path=target,
                    reason=(
                        f"the store publishes this coil's excitation as ampere turns "
                        f"and the description leaves {conductor}'s turn count unset, "
                        "so the conductor current the channel implies is the ratio of "
                        "a known numerator to an unknown denominator"
                    ),
                    unmet=(
                        f"pf_active/coil({conductor})/element/turns_with_sign is not "
                        "sourced"
                    ),
                )
            )
            continue
        signals.append(
            SourceSignal(
                standard_name=COIL_CURRENT_NAME,
                catalog_status=ACCEPTED,
                source_group=CURRENT_GROUP,
                source_channel=drive.channel,
                source_unit=unit,
                target_path=target,
                target_unit="A",
                target_index=machine.coil_index(conductor),
                transformation=IP_LIKE,
                source_convention=SOURCE_CONVENTION,
                target_convention=TARGET_CONVENTION,
                unit_factor=KILO,
                channel_factor=float(drive.ampere_turns_per_ampere) / float(turns),
                time_base=CURRENT_CLOCK,
                evidence=drive.evidence,
                statement=(
                    f"one ampere of {drive.channel} drives "
                    f"{drive.ampere_turns_per_ampere:.6g} ampere turns through "
                    f"{conductor}, which carries {turns:.6g} turns, so the "
                    "conductor current it implies is "
                    f"{drive.ampere_turns_per_ampere / turns:.6g} ampere per "
                    "ampere of the channel"
                ),
            )
        )
    return tuple(signals), tuple(blocked)


def case_current_blocked(machine: DescribedMachine) -> tuple[BlockedSignal, ...]:
    """Return the measured case currents the dictionary's passive loop cannot hold.

    The description carries the coil cases as one passive family whose elements are
    the individual plates, because that is what the catalog measures, and the
    dictionary carries one current per passive loop rather than one per element.
    Eight measured enclosure currents therefore have one field between them.  The
    drive map already says which plates each channel reaches, so the currents are
    usable by a forward operator; what is unavailable is the dictionary field, and
    cutting the described family into eight loops to create eight fields would
    invent a topology no source states.
    """

    channels = {
        drive.channel: drive
        for drive in machine.drives.drives
        if drive.container == "pf_passive"
    }
    blocked = []
    for channel, drive in sorted(channels.items()):
        if drive.conductor not in machine.passive_loops:
            raise SolveInputError(
                f"channel {channel!r} drives {drive.conductor!r}, which the "
                "description carries no passive loop for"
            )
        plates = machine.passive_elements[drive.conductor]
        blocked.append(
            BlockedSignal(
                source_group=CURRENT_GROUP,
                source_channel=channel,
                target_path=f"pf_passive/loop({drive.conductor})/current",
                reason=(
                    f"{channel} measures the current in the plates "
                    f"{list(drive.elements)} of the described {drive.conductor} "
                    f"family, and the dictionary carries one current for the whole "
                    f"loop, so {len(channels)} measured enclosure currents and "
                    f"{plates} described plates have one field between them"
                ),
                unmet=(
                    "pf_passive/loop/current is per loop while the measured currents "
                    "are per enclosure; the drive map carries the per-element weights "
                    "instead"
                ),
            )
        )
    return tuple(blocked)


def probe_field_signals(
    machine: DescribedMachine,
    probe_families: Sequence[Mapping[str, Any]],
) -> tuple[SourceSignal, ...]:
    """Return one row per described poloidal field probe.

    The described probe's name carries its family and its position in the sensor
    array, so the join the vacuum cohort established is checked against the
    description rather than trusted: a channel whose family block does not land on
    a probe of that family is refused here instead of reporting a field from
    somewhere else.
    """

    signals = []
    for probe in probe_channels(probe_families):
        expected = f"{probe.family}_{probe.registry_index}"
        described = machine.probes[probe.registry_index]
        if described != expected:
            raise SolveInputError(
                f"channel {probe.channel!r} joins to described probe "
                f"{described!r} at index {probe.registry_index}, which is not the "
                f"{probe.family!r} probe {expected!r} the family block names"
            )
        signals.append(
            SourceSignal(
                standard_name=PROBE_FIELD_NAME,
                catalog_status=ACCEPTED,
                source_group=FIELD_GROUP,
                source_channel=probe.channel,
                source_unit="T",
                target_path=f"magnetics/b_field_pol_probe({described})/field/data",
                target_unit="T",
                target_index=probe.registry_index,
                transformation=ONE_LIKE,
                source_convention=SOURCE_CONVENTION,
                target_convention=TARGET_CONVENTION,
                unit_factor=1.0,
                channel_factor=1.0,
                time_base=FIELD_CLOCK,
                evidence=FieldEvidence.MEASURED,
                statement=(
                    "a probe reads the field component along its own sensitive axis "
                    "at its own position, which no convention rescales; the axis it "
                    "projects onto is the described probe's orientation and is not "
                    "carried by this channel"
                ),
            )
        )
    return tuple(signals)


def flux_loop_signals(
    machine: DescribedMachine,
    targets: Mapping[str, int | None],
) -> tuple[tuple[SourceSignal, ...], tuple[BlockedSignal, ...]]:
    """Return one row per flux-loop channel that reaches a described loop."""

    signals: list[SourceSignal] = []
    blocked: list[BlockedSignal] = []
    for channel, index in sorted(targets.items()):
        if index is None:
            blocked.append(
                BlockedSignal(
                    source_group=FIELD_GROUP,
                    source_channel=channel,
                    target_path="magnetics/flux_loop/flux/data",
                    reason=(
                        f"the reconstruction loop {channel} reads sits further than "
                        f"{LOOP_POSITION_TOLERANCE * 1e3:.0f} mm from every described "
                        "loop, so the description carries no sensor for it"
                    ),
                    unmet=(
                        "the described loop set does not cover this reconstruction "
                        "loop position"
                    ),
                )
            )
            continue
        described = machine.loops[index]
        signals.append(
            SourceSignal(
                standard_name=LOOP_FLUX_NAME,
                catalog_status=ACCEPTED,
                source_group=FIELD_GROUP,
                source_channel=channel,
                source_unit="Wb",
                target_path=f"magnetics/flux_loop({described})/flux/data",
                target_unit="Wb",
                target_index=index,
                transformation=PSI_LIKE,
                source_convention=MEASURED_FLUX_CONVENTION,
                target_convention=TARGET_CONVENTION,
                unit_factor=1.0,
                channel_factor=1.0,
                time_base=FIELD_CLOCK,
                evidence=FieldEvidence.MEASURED,
                statement=(
                    "a loop links the total flux through its own contour, so the 2*pi "
                    "the reconstruction's flux function needs is already in this "
                    "number and the target convention asks for the total flux too"
                ),
            )
        )
    return tuple(signals), tuple(blocked)


def plasma_current_signal() -> SourceSignal:
    """Return the Rogowski-derived plasma current row.

    The dictionary offers two homes and only one of them is authorable here.  A
    Rogowski coil is a described sensor with a contour, an area and a turn density,
    and none of those is in the description, so filling
    ``magnetics/rogowski_coil`` would invent a sensor to hold a number.  The plasma
    current array is a derived measurement carrying the method that produced it,
    which is what this channel is.
    """

    return SourceSignal(
        standard_name=PLASMA_CURRENT_NAME,
        catalog_status=ACCEPTED,
        source_group=CURRENT_GROUP,
        source_channel="plasma_current",
        source_unit="kA",
        target_path="magnetics/ip/data",
        target_unit="A",
        target_index=0,
        transformation=IP_LIKE,
        source_convention=SOURCE_CONVENTION,
        target_convention=TARGET_CONVENTION,
        unit_factor=KILO,
        channel_factor=1.0,
        time_base=CURRENT_CLOCK,
        evidence=FieldEvidence.MEASURED,
        statement=(
            "the analysed plasma-current channel is a Rogowski-derived measurement of "
            "the net toroidal current; its sign follows the toroidal sense the source "
            "and the target share, so the conversion is the unit alone"
        ),
    )


def toroidal_field_blocked() -> tuple[BlockedSignal, ...]:
    """Return the toroidal-field references a solve needs and cannot yet be served.

    Both routes are blocked by the description rather than by the source.  The
    measured feed current has no described conductor to belong to: the
    toroidal-field description carries no coil and no turn count, so there is
    nothing for an ampere to be an ampere of.  The reconstruction's vacuum field
    would reach the dictionary as the product of that field with a reference radius,
    and a product with a per-shot radius is not a fixed conversion factor -- it is a
    second signal, which is a different kind of map row than this one.
    """

    return (
        BlockedSignal(
            source_group=CURRENT_GROUP,
            source_channel="tf_current",
            target_path="tf/coil/current/data",
            reason=(
                "the measured toroidal feed current has no described conductor: the "
                "toroidal-field description carries no coil, so there is no turn "
                "count to convert a feed current into a conductor current with"
            ),
            unmet="tf/coil and tf/coil/turns are not sourced",
        ),
        BlockedSignal(
            source_group=RECONSTRUCTION_GROUP,
            source_channel="bvac_val",
            target_path="tf/b_field_phi_vacuum_r/data",
            reason=(
                "the dictionary carries the vacuum toroidal field as its product with "
                "a reference radius, and that radius is a per-shot reconstruction "
                "scalar rather than a fixed factor, so this row would be the product "
                "of two source signals"
            ),
            unmet=(
                "tf/r0 is not sourced and a fixed-factor conversion cannot express a "
                "product of two channels"
            ),
        ),
    )


def pack_total_channels(
    machine: DescribedMachine,
) -> tuple[tuple[str, str, str], ...]:
    """Return each coil set whose published total is its coil plus its own case.

    Read off the drive map rather than listed.  A set qualifies when the
    description reaches both terms of the sum -- an already-multiplied coil
    channel driving a conductor, and a case channel driving the enclosure around
    it -- because that pairing is what makes the total a restatement rather than a
    measurement of its own.  A set publishing only one of the two would leave a
    total carrying something no described conductor accounts for, which is a
    different refusal and belongs beside a different reason.

    Each row is the set's channel prefix followed by the two channels its total
    decomposes into, in that order.
    """

    coil_channels = {
        drive.channel
        for drive in machine.drives.drives
        if drive.container == "pf_active"
    }
    case_channels = {
        drive.channel
        for drive in machine.drives.drives
        if drive.container == "pf_passive"
    }
    suffix = "_coil_current"
    return tuple(
        (prefix, channel, f"{prefix}_case_current")
        for channel in sorted(coil_channels)
        if channel.endswith(suffix)
        for prefix in (channel[: -len(suffix)],)
        if f"{prefix}_case_current" in case_channels
    )


def pack_total_residuals(
    machine: DescribedMachine, shot: int, *, store: Path | str = SHOT_STORE
) -> dict[str, float]:
    """Return how far each pack total departs from its coil plus its case.

    The residual is scaled by the two terms rather than by the total, because the
    terms cancel to a few parts in a thousand of themselves on shots where the case
    opposes the coil, and dividing by what is left of the total after a
    cancellation measures the cancellation instead of the identity.
    """

    import zarr

    group = zarr.open_group(f"{Path(store)}/{shot}.zarr", mode="r")
    if CURRENT_GROUP not in group:
        raise SolveInputError(f"shot {shot} carries no {CURRENT_GROUP!r} group")
    current = group[CURRENT_GROUP]
    residuals = {}
    for prefix, coil_channel, case_channel in pack_total_channels(machine):
        channels = (f"{prefix}_current", coil_channel, case_channel)
        if any(channel not in current for channel in channels):
            continue
        traces = [
            np.asarray(current[channel][...], dtype=float) for channel in channels
        ]
        if len({trace.shape for trace in traces}) != 1:
            continue
        finite = np.all([np.isfinite(trace) for trace in traces], axis=0)
        if int(finite.sum()) < 10:
            continue
        total, coil, case = (trace[finite] for trace in traces)
        scale = float(np.max(np.abs(coil)) + np.max(np.abs(case)))
        if scale <= 0.0:
            continue
        residuals[prefix] = float(np.max(np.abs(total - coil - case)) / scale)
    return residuals


def unmapped_current_blocked(machine: DescribedMachine) -> tuple[BlockedSignal, ...]:
    """Return the remaining excitation channels and why none reaches a conductor.

    Kept so the count adds up.  The store's excitation group publishes more
    channels than the drive map covers, and a channel absent from a map is
    indistinguishable from one nobody looked at unless the reason it is absent is
    written down beside it.
    """

    blocked = [
        BlockedSignal(
            source_group=CURRENT_GROUP,
            source_channel="efps_current",
            target_path="pf_active/supply/current/data",
            reason=(
                "this is a supply current measured at a link board, and no supply is "
                "described: the description authors an empty supply array rather than "
                "a partial one"
            ),
            unmet="pf_active/supply is not sourced",
        ),
    ]
    for channel in ("error_field_a", "error_field_b"):
        blocked.append(
            BlockedSignal(
                source_group=CURRENT_GROUP,
                source_channel=channel,
                target_path="coils_non_axisymmetric/coil/current/data",
                reason=(
                    "an error-field coil is not axisymmetric and this description is, "
                    "so the conductor this channel drives is outside it"
                ),
                unmet="no non-axisymmetric conductor description exists",
            )
        )
    for prefix, side in (("p2l", "lower"), ("p2u", "upper")):
        conductors = tuple(
            name
            for name in machine.coils
            if name.startswith("p2_") and name.endswith(f"_{side}")
        )
        carried = (
            f"{len(conductors)} separate conductors, {', '.join(conductors)},"
            if conductors
            else "separate conductors,"
        )
        pack_channels = tuple(
            drive.channel
            for drive in machine.drives.drives
            if drive.conductor in conductors and drive.channel.endswith("_coil_current")
        )
        terms = ", ".join((*pack_channels, f"{prefix}_case_current"))
        blocked.append(
            BlockedSignal(
                source_group=CURRENT_GROUP,
                source_channel=f"{prefix}_current",
                target_path="pf_active/coil/current/data",
                reason=(
                    "this channel is the set total for a coil set the description "
                    f"carries as {carried}: it equals {terms} sample for sample, so "
                    "what it measures is already accounted for channel by channel "
                    "and no one conductor carries the sum"
                ),
                unmet=(
                    "the interconnection of the two packs of this coil set is not "
                    "sourced, so no described conductor carries what this channel "
                    "measures"
                ),
            )
        )
    for prefix, coil_channel, case_channel in pack_total_channels(machine):
        blocked.append(
            BlockedSignal(
                source_group=CURRENT_GROUP,
                source_channel=f"{prefix}_current",
                target_path="pf_active/coil/current/data",
                reason=(
                    f"this channel is the pack total: it equals {coil_channel} plus "
                    f"{case_channel} sample for sample, so it measures the coil's "
                    "ampere turns together with the current its own case carries, "
                    "and the description holds those two in different containers"
                ),
                unmet=(
                    "the case current is induced rather than driven, so the pack "
                    "total is not a fixed multiple of either term and the two terms "
                    "it decomposes into already carry the measurement"
                ),
            )
        )
    return tuple(blocked)


def solve_input_map(
    machine: DescribedMachine,
    probe_families: Sequence[Mapping[str, Any]],
    loop_targets: Mapping[str, int | None],
) -> SourceSignalMap:
    """Return every per-slice solve input the description can and cannot receive."""

    currents, current_blocked = coil_current_signals(machine)
    loops, loop_blocked = flux_loop_signals(machine, loop_targets)
    return SourceSignalMap.create(
        (
            *currents,
            *probe_field_signals(machine, probe_families),
            *loops,
            plasma_current_signal(),
        ),
        (
            *current_blocked,
            *case_current_blocked(machine),
            *loop_blocked,
            *toroidal_field_blocked(),
            *unmapped_current_blocked(machine),
        ),
    )


@dataclass(frozen=True)
class ShotSignals:
    """One shot's solve inputs, the raw samples behind them and their provenance.

    ``source_map`` is the map restricted to the channels this shot actually
    published, so a consumer reads what was served rather than what the
    configuration could serve: the store is missing individual channels on many
    shots, and the difference between the two is exactly the list of what a
    particular slice cannot constrain.

    ``scale_corrections`` records the acquisition range setting divided out of each
    probe channel on this shot.  It sits on the read rather than in the map because
    the setting is a property of the shot and the map is a property of the
    configuration: a channel's conversion to the description does not change between
    shots, but which range the acquisition happened to be on does.
    """

    shot: int
    dataset: xarray.Dataset
    source_map: SourceSignalMap
    samples: Mapping[str, np.ndarray]
    clocks: Mapping[str, np.ndarray]
    absent_channels: tuple[str, ...]
    misaligned_channels: tuple[str, ...]
    dropped_samples: Mapping[str, int]
    provenance: tuple[SignalProvenance, ...] = field(default_factory=tuple)
    scale_corrections: tuple[ScaleCorrection, ...] = field(default_factory=tuple)

    @property
    def sample_counts(self) -> dict[str, int]:
        """Return how many samples each clock carries."""

        return {name: int(clock.size) for name, clock in sorted(self.clocks.items())}

    @property
    def scaled_channels(self) -> tuple[str, ...]:
        """Return the channels an acquisition setting was divided out of."""

        return tuple(
            sorted(
                row.channel
                for row in self.scale_corrections
                if row.applied and row.scale != 1.0
            )
        )

    @property
    def unscaled_channels(self) -> tuple[str, ...]:
        """Return the channels served as published because no setting warranted one."""

        return tuple(
            sorted(row.channel for row in self.scale_corrections if not row.applied)
        )


_CLOCK_GROUPS = {CURRENT_CLOCK: CURRENT_GROUP, FIELD_CLOCK: FIELD_GROUP}


def _probe_channel(channel: str) -> bool:
    """Return whether a source channel names a poloidal field probe."""

    try:
        parse_probe_channel(channel)
    except CohortError:
        return False
    return True


def _solve_sensor_unit(channel: str) -> str:
    """Return the physical unit of a magnetic sensor admitted to a solve."""

    if _probe_channel(channel):
        return "T"
    if _LOOP_CHANNEL_PATTERN.match(channel):
        return "Wb"
    raise SolveInputError(f"{channel!r} is not a solve sensor channel")


@dataclass(frozen=True)
class CorrectedSolveInputs:
    """Dense corrected waveforms ready for slice-wise solve shard staging.

    Channel labels and units are shard metadata and the numeric arrays share their
    first dimension, one row per time slice.  ``corrections`` is aligned with
    ``sensor_channels`` and preserves the disposition that says whether each sensor
    value was corrected; the values alone cannot distinguish an applied unity
    correction from an unmeasured channel left as published.

    ``bytes_per_slice`` counts the dense numeric payload only: one time, every coil
    current, every sensor signal, and plasma current.  Channel labels, units,
    dispositions and provenance are written once per shard rather than repeated for
    every slice.
    """

    shot: int
    time_s: np.ndarray
    coil_channels: tuple[str, ...]
    coil_currents_a: np.ndarray
    sensor_channels: tuple[str, ...]
    sensor_signals: np.ndarray
    sensor_units: tuple[str, ...]
    plasma_current_a: np.ndarray
    corrections: tuple[ScaleCorrection, ...]
    provenance: tuple[SignalProvenance, ...] = field(default_factory=tuple)

    @property
    def slice_count(self) -> int:
        """Return the number of solve slices in the payload."""

        return int(self.time_s.size)

    @property
    def bytes_per_slice(self) -> int:
        """Return the dense numeric bytes one staged solve slice occupies."""

        if self.slice_count == 0:
            return 0
        total = (
            self.time_s.nbytes
            + self.coil_currents_a.nbytes
            + self.sensor_signals.nbytes
            + self.plasma_current_a.nbytes
        )
        return int(total // self.slice_count)

    @property
    def kilobytes_per_slice(self) -> float:
        """Return the dense numeric payload per slice in binary kilobytes."""

        return self.bytes_per_slice / 1024.0


def read_corrected_solve_inputs(
    shot: int,
    *,
    store: Path | str = SHOT_STORE,
) -> CorrectedSolveInputs:
    """Return the corrected, solve-ready contract consumed by shard staging.

    The fields are ``time_s`` in seconds; ``coil_currents_a`` in amperes with
    columns named by ``coil_channels``; ``sensor_signals`` with columns named by
    ``sensor_channels`` and per-column units in ``sensor_units`` (tesla for
    poloidal-field probes and weber for flux loops); and ``plasma_current_a`` in
    amperes.  All numeric arrays are float64 and have one row per field-clock slice.

    This entry point intentionally exposes no raw-reader or correction override.  It
    obtains every field through :func:`read_shot_waveforms`, the corrected archive
    door, and returns its applied values with one correction disposition per sensor.
    Coil currents and plasma current come from that same read already interpolated
    onto the field clock, so shard staging cannot assemble a partly corrected slice
    by opening archive groups independently.
    """

    waveforms = read_shot_waveforms(int(shot), store=store)
    coil_channels = tuple(
        drive.family for drive in COIL_DRIVES if drive.family in waveforms.drives
    )
    sensor_channels = tuple(
        channel
        for channel in sorted(waveforms.sensors)
        if _probe_channel(channel) or _LOOP_CHANNEL_PATTERN.match(channel)
    )
    corrections = {row.channel: row for row in waveforms.scale_corrections}
    missing_dispositions = sorted(set(sensor_channels) - corrections.keys())
    if missing_dispositions:
        raise SolveInputError(
            "corrected read returned no disposition for sensor channels "
            f"{missing_dispositions}"
        )

    time = np.asarray(waveforms.time, dtype=float)
    coil_currents = np.column_stack(
        [
            np.asarray(waveforms.drives[channel], dtype=float)
            for channel in coil_channels
        ]
    )
    sensor_signals = np.column_stack(
        [
            np.asarray(waveforms.sensors[channel], dtype=float)
            for channel in sensor_channels
        ]
    )
    plasma_current = np.asarray(waveforms.plasma_current, dtype=float)
    expected = time.shape
    arrays = {
        "coil currents": coil_currents,
        "sensor signals": sensor_signals,
        "plasma current": plasma_current,
    }
    mismatched = {
        name: values.shape
        for name, values in arrays.items()
        if values.shape[0] != time.size
    }
    if mismatched:
        raise SolveInputError(
            f"solve inputs do not share the field clock {expected}: {mismatched}"
        )
    return CorrectedSolveInputs(
        shot=int(shot),
        time_s=time,
        coil_channels=coil_channels,
        coil_currents_a=coil_currents,
        sensor_channels=sensor_channels,
        sensor_signals=sensor_signals,
        sensor_units=tuple(_solve_sensor_unit(channel) for channel in sensor_channels),
        plasma_current_a=plasma_current,
        corrections=tuple(corrections[channel] for channel in sensor_channels),
        provenance=waveforms.provenance,
    )


def read_solve_inputs(
    source_map: SourceSignalMap,
    shot: int,
    *,
    store: Path | str = SHOT_STORE,
    resolver: StandardNameResolver | None = None,
    block_scale: ScaleReader | None = None,
) -> ShotSignals:
    """Read one shot's mapped channels and convert them onto the description.

    A group places every channel's samples on one clock that spans more than the
    acquisition and pads the rest, so the clock is trimmed to the samples every
    admitted channel actually carries and how many were dropped is reported.
    Holding the padded samples would put a fill value into a solve input, and
    interpolating across them would assert an excitation that was never recorded.

    A handful of channels carry a sample count their own group's clock does not
    have.  There is no stated correspondence between such a channel's samples and
    the clock, and index-aligning them would silently move every sample in time, so
    the channel is refused for that shot and reported as misaligned rather than
    served on a guess.

    Probe channels are served with their measured acquisition range setting divided
    out, so a solve input means the same thing on every shot.  ``block_scale`` names
    the table and defaults to the promoted one; an empty table serves the archive
    exactly as published.  What was divided out of each channel, and on what warrant,
    comes back in ``scale_corrections`` and is summarised in the dataset attributes.
    """

    import zarr

    root = Path(store)
    waveforms = read_shot_waveforms(shot, store=store, block_scale=block_scale)
    clocks: dict[str, np.ndarray] = {}
    samples: dict[str, np.ndarray] = {}
    dropped: dict[str, int] = {}
    absent: list[str] = []
    misaligned: list[str] = []
    provenance: list[SignalProvenance] = []
    available: list[str] = []
    for base in (CURRENT_CLOCK, FIELD_CLOCK):
        wanted = sorted(
            {
                signal.source_channel
                for signal in source_map.signals
                if signal.time_base == base
            }
        )
        if not wanted:
            continue
        name = _CLOCK_GROUPS[base]
        if base == FIELD_CLOCK:
            raw_source = waveforms.sensors
            keys = set(raw_source)
            clock = waveforms.time
            identity = next(
                (
                    row.group_identity
                    for row in waveforms.provenance
                    if row.group == FIELD_GROUP
                ),
                "",
            )
        else:
            try:
                node = zarr.open_group(f"{root}/{shot}.zarr", mode="r")[CURRENT_GROUP]
            except Exception:  # noqa: BLE001 - group absence is reported below
                absent.extend(wanted)
                continue
            keys = set(node.keys())
            clock = np.asarray(node["time"][...], dtype=float)
            identity = str(dict(node.attrs).get("uuid", ""))
            raw_source = node
        present = [channel for channel in wanted if channel in keys]
        absent.extend(channel for channel in wanted if channel not in keys)
        raw = {}
        for channel in present:
            values = np.asarray(raw_source[channel], dtype=float)
            if values.shape != clock.shape:
                misaligned.append(channel)
                continue
            raw[channel] = values
        present = sorted(raw)
        mask = np.isfinite(clock)
        for values in raw.values():
            mask &= np.isfinite(values)
        dropped[base] = int(clock.size - np.count_nonzero(mask))
        clocks[base] = clock[mask]
        provenance.append(SignalProvenance(str(root), shot, name, "time", identity))
        for channel, values in raw.items():
            samples[channel] = values[mask]
            provenance.append(
                SignalProvenance(str(root), shot, name, channel, identity)
            )
        available.extend(present)
    corrections = tuple(
        row for row in waveforms.scale_corrections if row.channel in samples
    )
    selected = source_map.select(available)
    dataset = tensorize(
        selected,
        samples,
        clocks,
        resolver=resolver,
        attrs={
            "shot": int(shot),
            "store": str(root),
            "absent_channels": sorted(absent),
            "misaligned_channels": sorted(misaligned),
            "dropped_samples": [
                f"{base}:{count}" for base, count in sorted(dropped.items())
            ],
            "acquisition_scaled_channels": sorted(
                f"{row.channel}:{row.scale:.6g}"
                for row in corrections
                if row.applied and row.scale != 1.0
            ),
            "acquisition_unscaled_channels": sorted(
                f"{row.channel}:{row.disposition}"
                for row in corrections
                if not row.applied
            ),
        },
    )
    return ShotSignals(
        shot=int(shot),
        dataset=dataset,
        source_map=selected,
        samples=samples,
        clocks=clocks,
        absent_channels=tuple(sorted(absent)),
        misaligned_channels=tuple(sorted(misaligned)),
        dropped_samples=dropped,
        provenance=tuple(provenance),
        scale_corrections=corrections,
    )


def trace_matched_columns(
    shot: int,
    channels: Sequence[str],
    array_name: str,
    *,
    store: Path | str = SHOT_STORE,
) -> dict[str, tuple[int, float, float]]:
    """Identify each channel's sensor by matching its trace to a fitted input column.

    The reconstruction's fitted inputs are the same measurements, held in the index
    order of the static geometry the description was built from, so the column a
    channel's trace reproduces names the sensor it reads without any channel
    description being consulted.  Returned per channel as the matched column, its
    scaled residual and the runner-up's, because a match only identifies a sensor
    when the runner-up is clearly worse: adjacent probes in a dense array read
    nearly the same field, and there the comparison is silent rather than wrong.
    """

    import zarr

    group = zarr.open_group(f"{Path(store)}/{shot}.zarr", mode="r")
    if RECONSTRUCTION_GROUP not in group:
        raise SolveInputError(f"shot {shot} carries no {RECONSTRUCTION_GROUP!r} group")
    reconstruction = group[RECONSTRUCTION_GROUP]
    fitted = np.asarray(reconstruction[array_name][...], dtype=float)
    fitted_time = np.asarray(reconstruction["time"][...], dtype=float)
    waveforms = RawArchiveReader(store).read_shot_waveforms(shot)
    field = waveforms.sensors
    field_time = waveforms.time
    matches: dict[str, tuple[int, float, float]] = {}
    for channel in channels:
        if channel not in field:
            continue
        values = np.asarray(field[channel], dtype=float)
        if values.shape != field_time.shape:
            continue
        usable = np.isfinite(values) & np.isfinite(field_time)
        if int(usable.sum()) < 2:
            continue
        trace = np.interp(
            fitted_time,
            field_time[usable],
            values[usable],
            left=np.nan,
            right=np.nan,
        )
        residuals = np.array(
            [
                _scaled_residual(trace, fitted[:, column])
                for column in range(fitted.shape[1])
            ]
        )
        order = np.argsort(residuals)
        matches[channel] = (
            int(order[0]),
            float(residuals[order[0]]),
            float(residuals[order[1]]),
        )
    return matches


def _scaled_residual(trace: np.ndarray, column: np.ndarray) -> float:
    """Return the largest difference between two traces, scaled by the column."""

    usable = np.isfinite(trace) & np.isfinite(column)
    if int(usable.sum()) < 10:
        return float("inf")
    scale = max(float(np.max(np.abs(column[usable]))), 1.0e-12)
    return float(np.max(np.abs(trace[usable] - column[usable])) / scale)


def field_polarity(shot: int, *, store: Path | str = SHOT_STORE) -> dict[str, float]:
    """Return the signed peak of the plasma and toroidal channels for one shot.

    The two cohorts a map has to survive differ by running both of these reversed,
    and the invariant worth testing is not either sign on its own but that their
    relation is the same in both: a convention factor that flipped a current would
    move one cohort's signs and not the other's.
    """

    import zarr

    group = zarr.open_group(f"{Path(store)}/{shot}.zarr", mode="r")
    if CURRENT_GROUP not in group:
        raise SolveInputError(f"shot {shot} carries no {CURRENT_GROUP!r} group")
    current = group[CURRENT_GROUP]
    peaks = {}
    for channel in ("plasma_current", "tf_current"):
        if channel not in current:
            continue
        values = np.asarray(current[channel][...], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            peaks[channel] = float(finite[np.argmax(np.abs(finite))] * KILO)
    return peaks
