"""Cover the documented MAST circuit relations and nominal material seeds.

The seeds are only as good as the sources behind them, so these tests check the
citations themselves: that no source describing the upgraded machine alone is
admitted, that a material is either named by a source or explicitly assigned by
association, and that a seeded resistance really is resistivity times toroidal
path length over the measured section. They also pin which fields stay
unresolved, because a shrinking unresolved set is a claim that needs evidence.
"""

from __future__ import annotations

import math

import imas
import pytest

from nova.catalog.mast_geometry import MachineGeometryRegistry, physical_digest
from nova.imas.machine_evidence import FieldEvidence
from nova.imas.mast_seed_parameters import (
    CIRCUIT_RELATIONS,
    INCONEL,
    MEASURED_DECAY_BAND,
    NOMINAL_SLOWEST_MODE,
    MAST_MACHINE_SCOPES,
    PROPOSED_STANDARD_NAMES,
    STAINLESS_STEEL,
    loop_sections,
    material_is_named_in_source,
    passive_material,
    seed_evidence,
)

FIRST_SHOT = 11695
LAST_SHOT = 30473


@pytest.fixture(scope="module")
def geometry():
    """Return the packaged physical configuration."""

    registry = MachineGeometryRegistry.default()
    return next(iter(registry.configurations.values())).geometry


@pytest.fixture(scope="module")
def ledger(geometry):
    """Return the seeded evidence ledger for the packaged configuration."""

    return seed_evidence(geometry, first_shot=FIRST_SHOT, last_shot=LAST_SHOT)


def test_every_citation_describes_the_original_machine(ledger) -> None:
    sources = [record.source for record in ledger.records if record.source is not None]

    assert sources
    for source in sources:
        source.validate()
        assert source.machine in MAST_MACHINE_SCOPES
    assert "mast-u" not in {source.machine for source in sources}


def test_citations_record_whether_the_wording_was_read(ledger) -> None:
    unverified = {
        source.title
        for record in ledger.records
        if (source := record.source) is not None and not source.text_verified
    }

    assert len(unverified) == 1
    assert "Review of Scientific Instruments" in next(iter(unverified))


def test_documented_circuits_cover_every_active_family(geometry) -> None:
    families = {
        family for relation in CIRCUIT_RELATIONS for family in relation.families
    }

    assert families == set(geometry["active_components"])
    assert sum(len(relation.families) for relation in CIRCUIT_RELATIONS) == len(
        families
    )
    connections = {relation.name: relation.connection for relation in CIRCUIT_RELATIONS}
    assert connections["P6"] == "anti-series"
    assert {connections[name] for name in ("P2", "P3", "P4", "P5")} == {"series"}


def test_solenoid_turn_current_is_half_the_documented_feed_current() -> None:
    solenoid = next(
        relation for relation in CIRCUIT_RELATIONS if relation.families == ("sol",)
    )

    assert solenoid.turn_to_feed_current_ratio == 0.5
    assert "two parallel circuits" in solenoid.connection
    assert solenoid.source.locator == "p. 9"


def test_material_is_named_by_a_source_or_assigned_by_association() -> None:
    assert material_is_named_in_source("coil_cases")
    assert material_is_named_in_source("incon")
    assert not material_is_named_in_source("vertw")
    assert passive_material("coil_cases") is STAINLESS_STEEL
    assert passive_material("incon") is INCONEL
    assert passive_material("vertw") is STAINLESS_STEEL
    with pytest.raises(KeyError, match="unknown passive family"):
        passive_material("not_a_family")


def test_an_ambiguous_conductor_family_gets_no_material(ledger) -> None:
    assert passive_material("rodgr") is None
    record = next(
        row
        for row in ledger.records
        if row.path == "pf_passive/loop(rodgr)/resistivity"
    )

    assert record.evidence is FieldEvidence.UNRESOLVED
    assert record.uncertainty is None
    assert "copper" in " ".join(record.assumptions)


def test_inconel_is_more_resistive_than_stainless_with_disjoint_intervals() -> None:
    steel = STAINLESS_STEEL.resistivity_interval()
    inconel = INCONEL.resistivity_interval()

    assert steel.upper < inconel.lower
    assert steel.contains(STAINLESS_STEEL.resistivity)
    assert inconel.contains(INCONEL.resistivity)
    assert steel.unit == inconel.unit == "ohm.m"


def test_loop_resistance_is_resistivity_times_path_length_over_section(
    geometry,
) -> None:
    section = loop_sections(geometry)["vertw"]
    expected = (
        STAINLESS_STEEL.resistivity
        * 2.0
        * math.pi
        * section.major_radius
        / section.area
    )

    resistance = STAINLESS_STEEL.loop_resistance(section.area, section.major_radius)
    interval = STAINLESS_STEEL.resistance_interval(section.area, section.major_radius)

    assert resistance == pytest.approx(expected)
    assert resistance == pytest.approx(1.1232e-4, rel=1e-3)
    assert interval.contains(resistance)
    assert interval.lower < resistance < interval.upper
    assert interval.unit == "ohm"


def test_only_connected_sections_get_a_seeded_resistance(geometry, ledger) -> None:
    sections = loop_sections(geometry)
    multi_part = {
        family for family, section in sections.items() if not section.is_single_loop
    }
    unresolved_resistance = {
        row.path.removeprefix("pf_passive/loop(").removesuffix(")/resistance")
        for row in ledger.records
        if row.path.endswith(")/resistance")
        and row.evidence is FieldEvidence.UNRESOLVED
    }

    assert multi_part == {"coil_cases", "mid", "ring", "rodgr"}
    assert unresolved_resistance == multi_part | {"rodgr"}
    assert sections["coil_cases"].parts == 24


def test_section_measurement_rejects_a_degenerate_loop() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        STAINLESS_STEEL.loop_resistance(0.0, 1.0)


def test_seeded_ledger_states_every_field_it_touches(ledger) -> None:
    assert ledger.state_counts() == {
        "measured": 8,
        "published": 9,
        "generated": 29,
        "fitted": 0,
        "unresolved": 15,
    }
    assert ledger.paths_with_state(FieldEvidence.UNRESOLVED) == (
        "magnetics/b_field_phi_probe/toroidal_angle",
        "magnetics/b_field_pol_probe/area",
        "magnetics/b_field_pol_probe/position/phi",
        "magnetics/b_field_pol_probe/toroidal_angle",
        "magnetics/flux_loop(saddle)/traversal_sign",
        "pf_active/circuit/connections",
        "pf_active/coil/element/turns_with_sign",
        "pf_passive/loop(coil_cases)/resistance",
        "pf_passive/loop(mid)/resistance",
        "pf_passive/loop(ring)/resistance",
        "pf_passive/loop(rodgr)/resistance",
        "pf_passive/loop(rodgr)/resistivity",
        "tf/coil/conductor/elements",
        "tf/coils_n",
        "tf/r0",
    )


def test_the_decay_calibration_records_its_negative_result(ledger) -> None:
    """The nominal resistance carries the reason a fit did not replace it.

    A negative result that lives only in a report is invisible to every reader of
    the description, and the next pass repeats the work.  So the record states that
    the fit was run, what it could not identify, and the corroboration that makes
    the seed a tested value: the predicted slowest mode sits inside the band the
    probe decays measure.
    """

    record = next(
        row for row in ledger.records if row.path == "pf_passive/loop/resistance"
    )
    assumptions = " ".join(record.assumptions)

    assert record.evidence is FieldEvidence.GENERATED
    assert "could not identify a resistivity" in record.statement
    assert MEASURED_DECAY_BAND[0] < NOMINAL_SLOWEST_MODE <= MEASURED_DECAY_BAND[1]
    assert "71.9 ms" in assumptions
    assert "ramps slower than" in assumptions
    assert "does not generalise" in assumptions
    assert "profile open" in assumptions


def test_the_p2_packs_are_published_as_separately_fed(ledger) -> None:
    """Two currents measured on one shot settle the pack interconnection.

    A series connection cannot carry nine kiloamperes in one pack and thirty
    amperes in the other, so this is excluded by measurement rather than argued
    from plausibility -- which is why the record is published rather than fitted.
    """

    record = next(
        row for row in ledger.records if row.path == "pf_active/circuit(P2)/connections"
    )
    assert record.evidence is FieldEvidence.PUBLISHED
    assert "no common current" in record.statement
    assert record.source is not None
    assert "feed-current" in record.source.locator


def test_the_supply_inventory_stays_unauthorable(ledger) -> None:
    """Knowing the packs are separate says nothing about how many supplies exist."""

    record = next(
        row for row in ledger.records if row.path == "pf_active/circuit/connections"
    )
    assert record.evidence is FieldEvidence.UNRESOLVED
    assert any("controllable outputs" in row for row in record.assumptions)


def test_turns_are_the_only_axisymmetric_forward_model_blocker(ledger) -> None:
    assert ledger.forward_model_blockers() == (
        "pf_active/coil/element/turns_with_sign",
    )
    turns = next(
        row
        for row in ledger.records
        if row.path == "pf_active/coil/element/turns_with_sign"
    )

    assert turns.evidence is FieldEvidence.UNRESOLVED
    assert "filament or element count" in " ".join(turns.assumptions)


def test_discrete_choices_keep_both_candidates(ledger) -> None:
    bank = next(
        row
        for row in ledger.records
        if row.path == "magnetics/b_field_pol_probe/position/phi"
    )
    saddle = next(
        row
        for row in ledger.records
        if row.path == "magnetics/flux_loop(saddle)/traversal_sign"
    )

    assert len(bank.candidates) == 2
    assert "150" in bank.candidates[0] and "330" in bank.candidates[1]
    assert saddle.candidates == ("negative traversal", "positive traversal")
    assert not bank.blocks_axisymmetric_forward_model
    assert not saddle.blocks_axisymmetric_forward_model


def test_toroidal_field_detail_is_absent_without_blocking_the_forward_model(
    ledger,
) -> None:
    topology = next(row for row in ledger.records if row.path == "tf")
    reference_radius = next(row for row in ledger.records if row.path == "tf/r0")

    assert topology.evidence is FieldEvidence.PUBLISHED
    assert "central solenoid" in topology.statement
    assert reference_radius.evidence is FieldEvidence.UNRESOLVED
    assert not reference_radius.blocks_axisymmetric_forward_model
    assert "one over major radius" in " ".join(reference_radius.assumptions)


def test_seeding_leaves_the_physical_identity_untouched(geometry, ledger) -> None:
    assert physical_digest(dict(geometry)) == "ca06c8f64481114f"
    assert ledger.records
    assert physical_digest(dict(geometry)) == "ca06c8f64481114f"


def test_proposed_standard_names_point_at_real_dictionary_paths() -> None:
    factory = imas.IDSFactory(version="4.1.1")

    assert PROPOSED_STANDARD_NAMES
    for proposal in PROPOSED_STANDARD_NAMES:
        ids_name, _, path = proposal.dd_path.partition("/")
        assert factory.exists(ids_name), proposal.dd_path
        metadata = factory.new(ids_name).metadata
        assert metadata[path] is not None, proposal.dd_path
        assert proposal.name == proposal.name.lower()
        assert proposal.note
