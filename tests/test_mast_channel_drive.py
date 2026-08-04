"""Cover the electrical drive semantics the artifact publishes.

The geometry says where the conductors are and the cohort says how many turns
they carry.  Neither answers the question a forward operator actually asks: given
one ampere of a named channel, how many ampere turns flow in which conductor.
These tests pin that answer, its provenance, and the two ways of getting it
wrong -- driving an already-multiplied channel through a turn count, and leaving
a measured conductor out of the model altogether.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.machine_drive import (
    SECTION_AREA,
    SINGLE_ELEMENT,
    ChannelDrive,
    DriveError,
    DriveMap,
)
from nova.imas.machine_evidence import EvidenceLedger, FieldEvidence, Uncertainty
from nova.imas.mast_channel_drive import (
    AMPERE_TURN_CHANNEL_WEIGHT,
    CASE_TURNS,
    MEASURED_AMPERE_TURN_RATIOS,
    case_plate_channels,
    channel_drives,
    circuit_connections,
    electrical_records,
    undriven_case_sets,
)
from nova.imas.mast_fitted_parameters import VACUUM_FITTED_TURNS, fitted_turns
from nova.imas.mast_geometry import REPRESENTATIVE_SHOT
from nova.imas.mast_passive_response import case_grouping
from nova.imas.mast_seed_parameters import CIRCUIT_RELATIONS
from nova.imas.mast_vacuum_cohort import (
    CASE_CURRENT_CHANNELS,
    COIL_DRIVES,
    DERIVED_TURN_CHANNELS,
)

FIRST_SHOT = 11695
LAST_SHOT = 30473

DRIVEN_COLUMNS = 21
"""Conductor sets a campaign holding every published channel can drive."""


@pytest.fixture(scope="module")
def geometry():
    """Return the registry configuration the drive map is authored against."""

    registry = MachineGeometryRegistry.default()
    return registry.select(REPRESENTATIVE_SHOT).configuration.geometry


@pytest.fixture(scope="module")
def drives(geometry):
    """Return the published drive map."""

    return channel_drives(geometry)


@pytest.fixture(scope="module")
def ledger(geometry):
    """Return the electrical records as a validated ledger."""

    return EvidenceLedger.create(
        electrical_records(geometry, first_shot=FIRST_SHOT, last_shot=LAST_SHOT)
    )


def _drive(drives: DriveMap, channel: str) -> ChannelDrive:
    return drives.for_channel(channel)


def test_every_driven_conductor_is_reachable(drives):
    """The campaign's whole excitation reaches a described conductor."""

    channels = set(drives.channels())
    for row in COIL_DRIVES:
        expected = (
            DERIVED_TURN_CHANNELS[row.family][0]
            if row.family in DERIVED_TURN_CHANNELS
            else row.channel
        )
        assert expected in channels
    assert set(CASE_CURRENT_CHANNELS.values()) <= channels
    assert len(drives.columns()) == DRIVEN_COLUMNS


def test_the_solenoid_drives_half_its_turns(drives):
    """Two parallel circuits share the feed, so one ampere drives half the winding."""

    row = fitted_turns("sol")
    drive = _drive(drives, "sol_current")
    assert drive.conductor == "sol"
    assert drive.evidence is FieldEvidence.FITTED
    assert drive.ampere_turns_per_ampere == pytest.approx(0.5 * row.turns)
    assert drive.ampere_turns_per_ampere == pytest.approx(344.656565)
    assert drive.uncertainty is not None
    assert drive.uncertainty.contains(drive.ampere_turns_per_ampere)
    assert drive.uncertainty.upper - drive.uncertainty.lower == pytest.approx(
        2.0 * 0.5 * row.half_width * row.turns_per_multiplier
    )


def test_an_ampere_turn_channel_drives_one_turn(drives):
    """A channel the archive already multiplied is never multiplied again."""

    for family, (ampere_turn_channel, _) in DERIVED_TURN_CHANNELS.items():
        drive = _drive(drives, ampere_turn_channel)
        assert drive.conductor == family
        assert drive.ampere_turns_per_ampere == AMPERE_TURN_CHANNEL_WEIGHT
        assert drive.evidence is FieldEvidence.MEASURED
        assert drive.uncertainty is None


def test_a_conductor_channel_drives_the_authored_turn_count(drives):
    """The feed-channel weight is the coil's own turn count, not a second number."""

    for family, (_, feed_channel) in DERIVED_TURN_CHANNELS.items():
        drive = _drive(drives, feed_channel)
        row = fitted_turns(family)
        assert drive.conductor == family
        assert drive.evidence is FieldEvidence.FITTED
        assert drive.ampere_turns_per_ampere == pytest.approx(row.turns)


def test_the_two_channel_weights_of_one_coil_differ_by_the_archive_ratio(drives):
    """Both routes carry the same physics, so their weights differ by that ratio."""

    for family, (ampere_turn_channel, feed_channel) in DERIVED_TURN_CHANNELS.items():
        ratio = MEASURED_AMPERE_TURN_RATIOS[family]
        product = _drive(drives, ampere_turn_channel).ampere_turns_per_ampere
        conductor = _drive(drives, feed_channel).ampere_turns_per_ampere
        assert conductor / product == pytest.approx(ratio, rel=0.06)
        assert fitted_turns(family).interval.contains(ratio)


def test_the_unidentified_coils_are_still_drivable(drives):
    """An unresolved turn count does not block a coil published as ampere turns."""

    for family in ("p6_lower", "p6_upper"):
        row = fitted_turns(family)
        assert not row.identified
        drive = _drive(drives, row.channel)
        assert drive.conductor == family
        assert drive.evidence is FieldEvidence.MEASURED
        assert drive.ampere_turns_per_ampere == AMPERE_TURN_CHANNEL_WEIGHT


def test_the_measured_case_currents_reach_their_own_plates(drives, geometry):
    """Each case channel drives the plates enclosing its coil set and no others."""

    grouped = case_grouping(geometry)
    reached: list[int] = []
    for coil_set, channel in sorted(CASE_CURRENT_CHANNELS.items()):
        drive = _drive(drives, channel)
        assert drive.container == "pf_passive"
        assert drive.conductor == "coil_cases"
        assert drive.ampere_turns_per_ampere == CASE_TURNS
        assert drive.evidence is FieldEvidence.GENERATED
        assert len(drive.elements) == len(grouped[f"coil_cases_{coil_set}"])
        reached += list(drive.elements)
    assert len(reached) == len(set(reached))


def test_a_case_group_with_no_channel_is_left_undriven(geometry, ledger):
    """A case nobody measured stays passive rather than borrowing a neighbour."""

    assert undriven_case_sets(geometry) == ("p6_lower", "p6_upper")
    unresolved = ledger.paths_with_state(FieldEvidence.UNRESOLVED)
    for coil_set in ("p6_lower", "p6_upper"):
        path = f"pf_passive/loop(coil_cases_{coil_set})/current(case_current)"
        assert path in unresolved


def test_the_plate_assignment_agrees_with_the_published_grouping(geometry):
    """The channel map and the enclosure grouping split the same plates the same way."""

    grouped = case_grouping(geometry)
    assigned = case_plate_channels(geometry)
    for coil_set, channel in CASE_CURRENT_CHANNELS.items():
        assert len(assigned[channel]) == len(grouped[f"coil_cases_{coil_set}"])
    assert sum(len(rows) for rows in assigned.values()) < sum(
        len(rows) for rows in grouped.values()
    )


def test_a_multi_plate_group_splits_by_section_area(drives):
    """One connected conductor spreads its current, a single element takes it whole."""

    for drive in drives.drives:
        expected = SINGLE_ELEMENT if len(drive.elements) == 1 else SECTION_AREA
        assert drive.distribution == expected


def test_every_weight_cites_a_record_the_ledger_carries(drives, ledger):
    """A weight without provenance is a number a whole vacuum field scales by."""

    paths = {record.path for record in ledger.records}
    for drive in drives.drives:
        assert drive.path in paths


def test_the_sourced_circuit_junctions_are_authored(geometry):
    """A two-coil circuit's whole published relation becomes one node."""

    families = sorted(geometry["active_components"])
    matrices = circuit_connections(geometry)
    assert sorted(matrices) == ["P3", "P4", "P5", "P6"]
    for name, matrix in matrices.items():
        relation = next(row for row in CIRCUIT_RELATIONS if row.name == name)
        assert matrix.shape == (1, len(families))
        columns = np.flatnonzero(matrix[0])
        assert [families[index] for index in columns] == sorted(relation.families)
        signs = sorted(matrix[0][columns].tolist())
        assert signs == ([-1, -1] if relation.connection == "anti-series" else [-1, 1])


def test_the_unsourced_circuit_topologies_stay_unset(geometry, ledger):
    """A circuit the sources leave open keeps an empty matrix and a stated reason."""

    assert "P1" not in circuit_connections(geometry)
    assert "P2" not in circuit_connections(geometry)
    unresolved = ledger.paths_with_state(FieldEvidence.UNRESOLVED)
    assert "pf_active/circuit(P1)/connections" in unresolved
    assert "pf_active/supply" in unresolved


def test_the_records_stay_inside_the_machines_the_sources_describe(ledger):
    """Every citation names original MAST or names it alongside its upgrade."""

    for record in ledger.records:
        if record.source is not None:
            assert record.source.machine in {"mast", "mast-and-mast-u"}


def test_the_channel_ratio_and_the_turn_count_stay_one_number():
    """The ratio a channel carries and the count a coil has are the same integer.

    They are two readings of one measurement, so writing them down separately
    would let them drift apart while both looked authoritative.
    """

    recorded = {
        row.family: row.published_turns for row in VACUUM_FITTED_TURNS if row.published
    }
    assert len(recorded) == len(MEASURED_AMPERE_TURN_RATIOS)
    for family, turns in recorded.items():
        assert MEASURED_AMPERE_TURN_RATIOS[family] == float(turns)


def test_a_drive_map_round_trips_through_canonical_json(drives):
    """The published map decodes back to itself, weights and provenance intact."""

    restored = DriveMap.from_list(drives.as_list())
    assert restored == drives
    assert restored.digest == drives.digest


def test_a_zero_weight_is_not_a_drive():
    """A conductor a channel does not drive carries no drive, not a weight of zero."""

    with pytest.raises(DriveError, match="zero weight"):
        DriveMap.create(
            [
                ChannelDrive(
                    channel="p3l_coil_current",
                    container="pf_active",
                    conductor="p3_lower",
                    elements=(0,),
                    circuit="P3",
                    ampere_turns_per_ampere=0.0,
                    distribution=SINGLE_ELEMENT,
                    evidence=FieldEvidence.MEASURED,
                    path="pf_active/coil(p3_lower)/current(p3l_coil_current)",
                )
            ]
        )


def test_an_unresolved_drive_is_an_absent_drive():
    """There is no way to publish a weight the evidence does not support."""

    with pytest.raises(DriveError, match="carries no drive at all"):
        DriveMap.create(
            [
                ChannelDrive(
                    channel="p6l_current",
                    container="pf_active",
                    conductor="p6_lower",
                    elements=(0,),
                    circuit="P6",
                    ampere_turns_per_ampere=1.0,
                    distribution=SINGLE_ELEMENT,
                    evidence=FieldEvidence.UNRESOLVED,
                    path="pf_active/coil(p6_lower)/current(p6l_current)",
                )
            ]
        )


def test_one_channel_cannot_carry_two_weights():
    """A channel with two weights is a channel whose meaning is undecided."""

    drive = ChannelDrive(
        channel="sol_current",
        container="pf_active",
        conductor="sol",
        elements=(0,),
        circuit="P1",
        ampere_turns_per_ampere=344.0,
        distribution=SINGLE_ELEMENT,
        evidence=FieldEvidence.FITTED,
        path="pf_active/coil(sol)/current(sol_current)",
        uncertainty=Uncertainty(lower=333.0, upper=356.0, unit="A.turn/A"),
    )
    with pytest.raises(DriveError, match="two drive weights"):
        DriveMap.create([drive, drive])


def test_a_fitted_weight_must_bound_itself():
    """A fitted number without an interval reports a precision nobody measured."""

    with pytest.raises(DriveError, match="must bound its weight"):
        DriveMap.create(
            [
                ChannelDrive(
                    channel="sol_current",
                    container="pf_active",
                    conductor="sol",
                    elements=(0,),
                    circuit="P1",
                    ampere_turns_per_ampere=344.0,
                    distribution=SINGLE_ELEMENT,
                    evidence=FieldEvidence.FITTED,
                    path="pf_active/coil(sol)/current(sol_current)",
                )
            ]
        )


def test_a_campaigns_channel_set_selects_its_own_drives(drives):
    """A consumer reads the weights for the channels it holds and no others."""

    held = ("sol_current", "p3l_coil_current", "p2l_case_current")
    selected = drives.select(held)
    assert selected.channels() == tuple(sorted(held))
    assert len(selected.columns()) == 3
