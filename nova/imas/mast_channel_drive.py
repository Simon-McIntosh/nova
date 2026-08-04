"""What one ampere of each published MAST current channel drives.

The catalog fixes the conductors, the vacuum cohort fixes their turn counts, and
neither says how the archive's current channels reach them.  That last step is
where a forward operator is silently lost, because the store publishes three
kinds of current channel side by side and they are not interchangeable.

``sol_current`` feeds the solenoid, which is wired as two parallel circuits, so
one ampere of it pushes half an ampere through every turn and the ampere turns it
drives are half the winding's turn count -- not one, and not the turn count.
``<coil>_coil_current`` has already been multiplied by the coil's turn count
before publication: its ampere turns are one per ampere, and a consumer that
multiplies it by ``turns_with_sign`` as well squares the turn count.
``<coil>_feed_current`` is the current in one conductor of the same coil, so
there the turn count is exactly the right multiplier.  The two channels differ by
a factor between eight and twenty-three depending on the coil, and nothing in the
name distinguishes them.

The coil cases are the other half.  Eight of the ten case groups have a measured
current channel of their own, so those conductors are driven rather than induced,
and a model that leaves them out puts their field into the residual.  They live
in ``pf_passive`` because that is what they are -- shorted single-turn
enclosures -- and the drive record says which plates of the single case family
each channel reaches, without cutting the family up to say it.

Every weight here is the total ampere turns per ampere of the named channel and
supersedes ``turns_with_sign`` for that channel.  Which channel a consumer holds
decides which weight it applies; applying both double counts.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import shapely

from nova.imas.machine_drive import (
    SECTION_AREA,
    SINGLE_ELEMENT,
    ChannelDrive,
    DriveMap,
)
from nova.imas.machine_evidence import (
    EvidenceRecord,
    FieldEvidence,
    SourceReference,
    Uncertainty,
)
from nova.imas.mast_fitted_parameters import (
    PUBLISHED_RATIO_SHOTS,
    PUBLISHED_TURN_RATIOS,
    fitted_turns,
)
from nova.imas.mast_passive_response import (
    CASE_FAMILY,
    CASE_PROXIMITY,
    case_side,
    passive_sections,
)
from nova.imas.mast_seed_parameters import CIRCUIT_RELATIONS
from nova.imas.mast_vacuum_cohort import (
    CASE_CURRENT_CHANNELS,
    COIL_DRIVES,
    DERIVED_TURN_CHANNELS,
)
from nova.imas.mast_vacuum_response import coil_sections

AMPERE_TURN_CHANNEL_WEIGHT = 1.0
"""Ampere turns one ampere of an already-multiplied channel drives."""

CASE_TURNS = 1.0
"""Turns a shorted axisymmetric coil case carries: it closes on itself once."""

MEASURED_AMPERE_TURN_RATIOS = {
    family: float(turns) for family, turns in PUBLISHED_TURN_RATIOS.items()
}
"""Constant the archive multiplies each conductor-current channel by.

Measured, not fitted: the store publishes both channels of the pair for these ten
coils, and their ratio is the same exact integer on every shot that carries both.
That is what makes the ampere-turn channel usable at unit weight -- the
multiplication has already happened, and this is the number that happened.

The same ratio does double duty, and the two readings must not be confused.  Read
as a property of the CHANNEL it says one ampere already drives one ampere turn,
which is what this table is for.  Read as a property of the COIL it states the turn
count, which is what promotes ten of the thirteen counts in
:mod:`nova.imas.mast_fitted_parameters` -- so the two live in one place rather than
being written down twice and allowed to drift apart.
"""

AMPERE_TURN_RATIO_SHOTS = PUBLISHED_RATIO_SHOTS
"""Shots the constant ratio holds on, for the least-carried of the ten pairs."""

_TWO_TERMINAL_CIRCUITS = frozenset({"P3", "P4", "P5", "P6"})
"""Circuits whose whole published relation is a junction between two coils.

A circuit joining more coils than this has an interconnection the sources do not
fix, and a circuit joining fewer has its relation inside a single coil, where a
matrix indexed by coils has no column to write it in.
"""


def _shot_store(locator: str) -> SourceReference:
    """Cite the level-1 shot store as the origin of a channel's semantics."""

    return SourceReference(
        title="MAST level-1 vacuum shot store",
        url="https://mastapp.site/",
        locator=locator,
        machine="mast",
        text_verified=True,
    )


def active_families(geometry: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the active components in the order the artifact authors them."""

    return tuple(sorted(geometry["active_components"]))


def _element_count(geometry: Mapping[str, Any], key: str, family: str) -> int:
    """Return how many outline polygons one component is authored as."""

    outline = shapely.from_wkb(bytes.fromhex(geometry[key][family]))
    parts = getattr(outline, "geoms", None)
    return 1 if parts is None else len(parts)


def enclosed_coil_sets(geometry: Mapping[str, Any]) -> tuple[str | None, ...]:
    """Return the coil set each case plate encloses, in the family's own order.

    The registry publishes the cases as one family of plates with no statement
    about which coil each belongs to, and the store publishes one current per coil
    set.  Enclosure settles the join without a fit: a case surrounds its own coil,
    so a plate belongs to the set it is nearest to, and a plate further than
    :data:`~nova.imas.mast_passive_response.CASE_PROXIMITY` from every coil takes
    ``None`` rather than being attached to the least distant one.  The sequence
    position is the element index the artifact writes that plate at.
    """

    plates = passive_sections(geometry).get(CASE_FAMILY)
    if not plates:
        raise KeyError(f"registry carries no {CASE_FAMILY!r} family")
    coils: dict[str, list[shapely.Polygon]] = {}
    for family, parts in coil_sections(geometry).items():
        coils.setdefault(case_side(family), []).extend(
            shapely.Polygon(vertices) for vertices in parts
        )
    enclosed: list[str | None] = []
    for vertices in plates:
        plate = shapely.Polygon(vertices)
        distances = {
            family: min(plate.distance(polygon) for polygon in polygons)
            for family, polygons in coils.items()
        }
        nearest = min(distances, key=lambda key: distances[key])
        enclosed.append(None if distances[nearest] > CASE_PROXIMITY else nearest)
    return tuple(enclosed)


def case_plate_channels(geometry: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return the case plates each measured case-current channel drives."""

    assignment: dict[str, list[int]] = {}
    for index, coil_set in enumerate(enclosed_coil_sets(geometry)):
        channel = CASE_CURRENT_CHANNELS.get(coil_set) if coil_set else None
        if channel is None:
            continue
        assignment.setdefault(channel, []).append(index)
    return {channel: tuple(rows) for channel, rows in sorted(assignment.items())}


def undriven_case_sets(geometry: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the coil sets whose case encloses plates but carries no channel."""

    return tuple(
        sorted(
            {
                coil_set
                for coil_set in enclosed_coil_sets(geometry)
                if coil_set is not None and coil_set not in CASE_CURRENT_CHANNELS
            }
        )
    )


def _active_drive_path(family: str, channel: str) -> str:
    return f"pf_active/coil({family})/current({channel})"


def _case_drive_path(coil_set: str, channel: str) -> str:
    return f"pf_passive/loop({CASE_FAMILY}_{coil_set})/current({channel})"


def _parallel_feed_drive(geometry: Mapping[str, Any], drive: Any) -> ChannelDrive:
    """Return a split-feed channel's weight: the turn count times the split.

    A winding fed as several parallel circuits carries a fraction of the feed
    current in each turn, so its ampere turns per feed ampere are the turn count
    scaled by that fraction.  The weight is derived from the authored turn count
    rather than from the fit's own multiplier, which keeps the two from drifting
    apart if the count is ever re-authored.
    """

    row = fitted_turns(drive.family)
    split = float(drive.turn_to_channel_current_ratio)
    elements = tuple(range(_element_count(geometry, "active_components", drive.family)))
    return ChannelDrive(
        channel=drive.channel,
        container="pf_active",
        conductor=drive.family,
        elements=elements,
        circuit=drive.circuit,
        ampere_turns_per_ampere=float(row.turns) * split,
        distribution=SINGLE_ELEMENT if len(elements) == 1 else SECTION_AREA,
        evidence=FieldEvidence.FITTED,
        path=_active_drive_path(drive.family, drive.channel),
        uncertainty=Uncertainty(
            lower=float(row.interval.lower) * split,
            upper=float(row.interval.upper) * split,
            unit="A.turn/A",
        ),
    )


def _ampere_turn_drive(
    geometry: Mapping[str, Any],
    drive: Any,
    channel: str,
) -> ChannelDrive:
    """Return a unit weight for a channel the archive already multiplied."""

    elements = tuple(range(_element_count(geometry, "active_components", drive.family)))
    return ChannelDrive(
        channel=channel,
        container="pf_active",
        conductor=drive.family,
        elements=elements,
        circuit=drive.circuit,
        ampere_turns_per_ampere=AMPERE_TURN_CHANNEL_WEIGHT,
        distribution=SINGLE_ELEMENT if len(elements) == 1 else SECTION_AREA,
        evidence=FieldEvidence.MEASURED,
        path=_active_drive_path(drive.family, channel),
        uncertainty=None,
    )


def _conductor_drive(geometry: Mapping[str, Any], drive: Any) -> ChannelDrive:
    """Return a conductor-current channel's weight: the coil's fitted turn count."""

    row = fitted_turns(drive.family)
    elements = tuple(range(_element_count(geometry, "active_components", drive.family)))
    return ChannelDrive(
        channel=drive.channel,
        container="pf_active",
        conductor=drive.family,
        elements=elements,
        circuit=drive.circuit,
        ampere_turns_per_ampere=float(row.turns),
        distribution=SINGLE_ELEMENT if len(elements) == 1 else SECTION_AREA,
        evidence=FieldEvidence.FITTED,
        path=_active_drive_path(drive.family, drive.channel),
        uncertainty=row.interval,
    )


def _case_drives(geometry: Mapping[str, Any]) -> list[ChannelDrive]:
    """Return one unit-turn drive per measured case-current channel."""

    inverse = {channel: coil_set for coil_set, channel in CASE_CURRENT_CHANNELS.items()}
    drives = []
    for channel, elements in case_plate_channels(geometry).items():
        drives.append(
            ChannelDrive(
                channel=channel,
                container="pf_passive",
                conductor=CASE_FAMILY,
                elements=elements,
                circuit="",
                ampere_turns_per_ampere=CASE_TURNS,
                distribution=SINGLE_ELEMENT if len(elements) == 1 else SECTION_AREA,
                evidence=FieldEvidence.GENERATED,
                path=_case_drive_path(inverse[channel], channel),
                uncertainty=Uncertainty(
                    lower=CASE_TURNS,
                    upper=CASE_TURNS,
                    unit="turn",
                ),
            )
        )
    return drives


def channel_drives(geometry: Mapping[str, Any]) -> DriveMap:
    """Return every published channel that drives a described conductor.

    An active coil whose archive publishes both channel kinds gets both, because
    which one a campaign holds is the campaign's business and the conversion is
    different for each.  The map is keyed by channel for exactly that reason: a
    consumer selects the channels it has and reads the weight beside each.
    """

    drives: list[ChannelDrive] = []
    for drive in COIL_DRIVES:
        if drive.reports_ampere_turns:
            drives.append(_ampere_turn_drive(geometry, drive, drive.channel))
            continue
        pair = DERIVED_TURN_CHANNELS.get(drive.family)
        if pair is not None:
            drives.append(_ampere_turn_drive(geometry, drive, pair[0]))
            drives.append(_conductor_drive(geometry, drive))
            continue
        if drive.turn_to_channel_current_ratio == 1.0:
            raise KeyError(
                f"channel {drive.channel!r} measures one conductor of {drive.family!r} "
                "and the archive publishes no ampere-turn partner, so what one of its "
                "amperes drives is not stated anywhere"
            )
        drives.append(_parallel_feed_drive(geometry, drive))
    drives += _case_drives(geometry)
    return DriveMap.create(drives)


def circuit_connections(geometry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return the node matrix for each circuit whose whole relation is sourced.

    The dictionary indexes the second dimension by supply and then by coil, one
    column each, and marks the terminal a node reaches with its sign.  No supply
    inventory is published, so no supply is authored and the matrix carries the
    coil columns alone: what it states is which coil terminals are joined to each
    other, which is exactly the published relation and nothing beyond it.

    A series pair shares one node, reached by the first coil's negative terminal
    and the second coil's positive one, so the current leaving one winding enters
    the next in the same sense.  An anti-series pair is joined negative to
    negative, which is what reverses the second winding and produces the opposing
    field its supply is there to drive.
    """

    families = active_families(geometry)
    column = {family: index for index, family in enumerate(families)}
    matrices: dict[str, np.ndarray] = {}
    for relation in CIRCUIT_RELATIONS:
        if relation.name not in _TWO_TERMINAL_CIRCUITS:
            continue
        first, second = sorted(relation.families)
        matrix = np.zeros((1, len(families)), dtype=np.int32)
        matrix[0, column[first]] = -1
        matrix[0, column[second]] = -1 if relation.connection == "anti-series" else 1
        matrices[relation.name] = matrix
    return matrices


_TERMINAL_SENSE = (
    "a coil's positive terminal is the one a positive current enters to produce "
    "the field its positive signed turn count predicts, which is what makes the "
    "difference between a series and an anti-series junction a difference between "
    "the terminals a node joins"
)

_NO_SUPPLY_INVENTORY = (
    "no supply is authored, so the matrix carries coil columns only and asserts "
    "nothing about terminals outside the coils it names"
)


def _connection_records(first_shot: int, last_shot: int) -> list[EvidenceRecord]:
    """Record which circuit topologies the published relations let us author."""

    records = [
        EvidenceRecord(
            path="pf_active/supply",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "no source lists the poloidal field supplies, so the supply array "
                "stays empty and the circuit node matrices index coils alone"
            ),
            assumptions=(
                "an empty supply array is a smaller claim than a partial one: it "
                "leaves the supply side of every circuit unstated rather than "
                "asserting that the supplies named are all there are",
                "the coil-to-coil junctions the sources do fix are authorable "
                "without any supply column, because a node lists the terminals that "
                "meet at it and a terminal nobody published is simply not listed",
            ),
        ),
        EvidenceRecord(
            path="pf_active/circuit(P1)/connections",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "the solenoid's two parallel circuits divide one coil, and the node "
                "matrix has one column per coil, so the relation has no column to be "
                "written in"
            ),
            assumptions=(
                "the catalog resolves the solenoid as a single winding pack, so the "
                "alternating layers of the two circuits are not separable conductors "
                "in this description and splitting the pack to make them separable "
                "would invent a geometry no source measures",
                "the electrical consequence of the relation is carried instead as "
                "the ampere turns one ampere of the feed channel drives, which is "
                "what a forward operator needs from it",
            ),
        ),
    ]
    for relation in CIRCUIT_RELATIONS:
        if relation.name not in _TWO_TERMINAL_CIRCUITS:
            continue
        first, second = sorted(relation.families)
        junction = (
            "negative terminal"
            if relation.connection == "anti-series"
            else "positive terminal"
        )
        records.append(
            EvidenceRecord(
                path=f"pf_active/circuit({relation.name})/connections",
                evidence=FieldEvidence.PUBLISHED,
                first_shot=first_shot,
                last_shot=last_shot,
                statement=(
                    f"one node joins the negative terminal of {first} to the "
                    f"{junction} of {second}, which is the {relation.connection} "
                    "connection the source states"
                ),
                assumptions=(_TERMINAL_SENSE, _NO_SUPPLY_INVENTORY),
                source=relation.source,
            )
        )
    return records


def _active_drive_records(
    geometry: Mapping[str, Any],
    first_shot: int,
    last_shot: int,
) -> list[EvidenceRecord]:
    """Record what one ampere of each active channel drives, and on what evidence."""

    records: list[EvidenceRecord] = []
    for drive in COIL_DRIVES:
        if drive.reports_ampere_turns:
            records.append(
                EvidenceRecord(
                    path=_active_drive_path(drive.family, drive.channel),
                    evidence=FieldEvidence.MEASURED,
                    first_shot=first_shot,
                    last_shot=last_shot,
                    statement=(
                        f"the store publishes this coil's excitation as ampere turns, "
                        f"so one ampere of {drive.channel} drives one ampere turn and "
                        "the column is fixed without a turn count"
                    ),
                    source=_shot_store(f"channel {drive.channel}"),
                )
            )
            continue
        pair = DERIVED_TURN_CHANNELS.get(drive.family)
        if pair is None:
            row = fitted_turns(drive.family)
            split = float(drive.turn_to_channel_current_ratio)
            records.append(
                EvidenceRecord(
                    path=_active_drive_path(drive.family, drive.channel),
                    evidence=FieldEvidence.FITTED,
                    first_shot=first_shot,
                    last_shot=last_shot,
                    statement=(
                        f"one ampere of {drive.channel} drives "
                        f"{row.turns * split:.3f} ampere turns: the winding carries "
                        f"{row.turns:.3f} turns and each takes {split:g} of the feed "
                        "current, because the feed splits between parallel circuits"
                    ),
                    assumptions=(
                        "the parallel circuits carry equal current, which follows "
                        "from their being alternating layers of one winding rather "
                        "than separately fed sections",
                        "the weight and the authored turn count therefore differ by "
                        "that split by construction, so a consumer applies one or the "
                        "other and never their product",
                    ),
                    source=_shot_store(
                        f"{row.shot_count} isolating shots, channel {drive.channel}"
                    ),
                    uncertainty=Uncertainty(
                        lower=float(row.interval.lower) * split,
                        upper=float(row.interval.upper) * split,
                        unit="A.turn/A",
                    ),
                )
            )
            continue
        ampere_turn_channel, feed_channel = pair
        ratio = MEASURED_AMPERE_TURN_RATIOS[drive.family]
        row = fitted_turns(drive.family)
        records.append(
            EvidenceRecord(
                path=_active_drive_path(drive.family, ampere_turn_channel),
                evidence=FieldEvidence.MEASURED,
                first_shot=first_shot,
                last_shot=last_shot,
                statement=(
                    f"{ampere_turn_channel} is {feed_channel} multiplied by "
                    f"{ratio:g} on every one of at least {AMPERE_TURN_RATIO_SHOTS} "
                    "shots that carry both, so it already reports ampere turns and "
                    "one ampere of it drives one ampere turn"
                ),
                source=_shot_store(
                    f"channel pair {ampere_turn_channel} and {feed_channel}"
                ),
            )
        )
        records.append(
            EvidenceRecord(
                path=_active_drive_path(drive.family, feed_channel),
                evidence=FieldEvidence.FITTED,
                first_shot=first_shot,
                last_shot=last_shot,
                statement=(
                    f"one ampere of {feed_channel} is one ampere in one conductor, so "
                    f"it drives the coil's {row.turns:.3f} ampere turns"
                ),
                assumptions=(
                    "the weight is the coil's own fitted turn count, so it agrees "
                    f"with the authored turns_with_sign and the archive's {ratio:g} "
                    "lies inside the interval the cohort supports",
                    "a channel reporting one conductor's current and one reporting "
                    "the product carry the same physics at different scales, so the "
                    "two weights for this coil must never both be applied",
                ),
                source=_shot_store(
                    f"{row.shot_count} isolating shots, channel {feed_channel}"
                ),
                uncertainty=row.interval,
            )
        )
    return records


def _case_drive_records(
    geometry: Mapping[str, Any],
    first_shot: int,
    last_shot: int,
) -> list[EvidenceRecord]:
    """Record which case plates each measured case channel drives, and which none do."""

    inverse = {channel: coil_set for coil_set, channel in CASE_CURRENT_CHANNELS.items()}
    records: list[EvidenceRecord] = []
    for channel, elements in case_plate_channels(geometry).items():
        coil_set = inverse[channel]
        records.append(
            EvidenceRecord(
                path=_case_drive_path(coil_set, channel),
                evidence=FieldEvidence.GENERATED,
                first_shot=first_shot,
                last_shot=last_shot,
                statement=(
                    f"{channel} is a measured current in the {coil_set} case, carried "
                    f"by the {len(elements)} plate(s) of the case family that enclose "
                    "that coil set at one turn between them"
                ),
                assumptions=(
                    "a case is a shorted axisymmetric enclosure, so it closes on "
                    "itself once and its turn count is one whatever subdivision the "
                    "catalog stored it in",
                    "the plates of one enclosure form one connected conductor, so the "
                    "measured current splits between them in proportion to poloidal "
                    "section area, which is what a uniform current density does",
                    "this current is measured rather than induced by the model, so a "
                    "description that leaves the case undriven puts its field into "
                    "the residual instead of into the prediction",
                ),
                uncertainty=Uncertainty(
                    lower=CASE_TURNS,
                    upper=CASE_TURNS,
                    unit="turn",
                ),
                source=_shot_store(f"channel {channel}"),
            )
        )
    for coil_set in undriven_case_sets(geometry):
        records.append(
            EvidenceRecord(
                path=_case_drive_path(coil_set, "case_current"),
                evidence=FieldEvidence.UNRESOLVED,
                first_shot=first_shot,
                last_shot=last_shot,
                statement=(
                    f"the {coil_set} case encloses plates the grouping resolves, but "
                    "the store publishes no case-current channel for this coil set"
                ),
                assumptions=(
                    "the case current is an induced quantity, so its absence leaves "
                    "the enclosure to the passive model rather than leaving a driven "
                    "column empty",
                    "reusing a neighbouring set's channel would drive one conductor "
                    "with another's measurement",
                ),
            )
        )
    return records


def electrical_records(
    geometry: Mapping[str, Any],
    *,
    first_shot: int,
    last_shot: int,
) -> list[EvidenceRecord]:
    """Return every record the artifact's electrical drive semantics contribute."""

    return [
        *_connection_records(first_shot, last_shot),
        *_active_drive_records(geometry, first_shot, last_shot),
        *_case_drive_records(geometry, first_shot, last_shot),
    ]
