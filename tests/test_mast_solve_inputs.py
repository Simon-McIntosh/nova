"""The MAST solve-input map: what it serves, what it refuses, and what it recovers.

Two kinds of test.  The ones that need no store build a small described machine by
hand, because a synthetic conductor with a known turn count checks the conversion
arithmetic more sharply than a recorded one -- the answer is known in advance and
every refusal can be given the exact case it exists for.  The ones that do need the
store round-trip real shots of both field polarities against the published
description, and skip where the level-1 mirror is absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.machine_drive import ChannelDrive, DriveMap
from nova.imas.machine_evidence import FieldEvidence, Uncertainty
from nova.imas.mast_block_scale import BRACKETED, MEASURED, ScaleCorrection
from nova.imas.mast_geometry import REPRESENTATIVE_SHOT, publish_refined_artifact
from nova.imas.mast_solve_input_ids import (
    build_solve_input_map,
    open_verified_description,
    preferred_signals,
    round_trip_shot,
)
from nova.imas.mast_solve_inputs import (
    COIL_CURRENT_NAME,
    CURRENT_GROUP,
    FIELD_GROUP,
    LOOP_FLUX_NAME,
    LOOP_POSITION_TOLERANCE,
    MEASURED_FLUX_CONVENTION,
    PACK_TOTAL_TOLERANCE,
    PLASMA_CURRENT_NAME,
    PROBE_FIELD_NAME,
    RECONSTRUCTION_LOOP_COUNT,
    SHOT_STORE,
    SOURCE_CONVENTION,
    TARGET_CONVENTION,
    DescribedMachine,
    SolveInputError,
    case_current_blocked,
    coil_current_signals,
    field_polarity,
    flux_loop_signals,
    loop_channel_name,
    loop_target_indices,
    pack_total_channels,
    pack_total_residuals,
    parse_loop_channel,
    probe_field_signals,
    read_corrected_solve_inputs,
    read_solve_inputs,
    reconstruction_loop_positions,
    reconstruction_loop_rows,
    reconstruction_probe_poses,
    solve_input_map,
    toroidal_field_blocked,
    trace_matched_columns,
    unmapped_current_blocked,
)
from nova.imas.mast_vacuum_cohort import ShotWaveforms
from nova.io.cocos import IP_LIKE, ONE_LIKE, PSI_LIKE
from nova.utilities.importmanager import mark_import

with mark_import(
    "imas-standard-names", "imas-standard-names-catalog"
) as _needs_standard_names:
    import imas_standard_names  # noqa: F401
    import imas_standard_names_catalog  # noqa: F401

FORWARD_SHOTS = (11766, 15000, 24000, 28000)
"""Pilot shots running the plasma current positive and the toroidal field negative."""

REVERSED_SHOTS = (13500, 13600, 22500)
"""Pilot shots running both of those reversed -- the same machine, both polarities."""

JOIN_SHOT = 15000
"""The shot the loop join is taken from; that it is the same for any is a test below."""

PROBE_POSITION_TOLERANCE = 0.015
"""Metres a channel's own measured position may sit from the described probe's."""

DISCRIMINATING_RESIDUAL = 0.1
"""Scaled residual below which a trace match is close enough to mean something."""

DISCRIMINATING_MARGIN = 3.0
"""How much worse the runner-up must be before a match identifies a sensor.

Adjacent sensors in a dense array read nearly the same field, so a match that beats
its runner-up narrowly identifies nothing.  Those channels are passed over rather
than counted either way: silence is the honest outcome of an ambiguous comparison.
"""

PROBE_FAMILIES = ({"family": "ccbv"}, {"family": "obr"}, {"family": "obv"})
"""One probe per family, matching the synthetic description's probe array."""

_needs_store = pytest.mark.skipif(
    not Path(SHOT_STORE).is_dir(),
    reason=f"MAST level-1 store not present at {SHOT_STORE}",
)


def _machine(**overrides) -> DescribedMachine:
    """Return a two-coil, two-probe, two-loop description with known turn counts."""

    drives = DriveMap.create(
        [
            ChannelDrive(
                channel="c1_feed_current",
                container="pf_active",
                conductor="c1",
                elements=(0,),
                circuit="C",
                ampere_turns_per_ampere=8.0,
                distribution="single",
                evidence=FieldEvidence.FITTED,
                path="pf_active/coil(c1)/current(c1_feed_current)",
                uncertainty=Uncertainty(lower=7.5, upper=8.5, unit="turn"),
            ),
            ChannelDrive(
                channel="c1_coil_current",
                container="pf_active",
                conductor="c1",
                elements=(0,),
                circuit="C",
                ampere_turns_per_ampere=1.0,
                distribution="single",
                evidence=FieldEvidence.MEASURED,
                path="pf_active/coil(c1)/current(c1_coil_current)",
            ),
            ChannelDrive(
                channel="c2_current",
                container="pf_active",
                conductor="c2",
                elements=(0,),
                circuit="C",
                ampere_turns_per_ampere=1.0,
                distribution="single",
                evidence=FieldEvidence.MEASURED,
                path="pf_active/coil(c2)/current(c2_current)",
            ),
        ]
    )
    row = {
        "dd_version": "4.1.1",
        "coils": ("c1", "c2"),
        "turns": {"c1": 8.0, "c2": None},
        "probes": ("ccbv_0", "obr_1", "obv_2"),
        "probe_poses": np.array(
            [[0.18, 1.4, math.pi / 2], [1.4, 0.2, 0.0], [1.4, 0.2, math.pi / 2]]
        ),
        "loops": ("flux_loop_0", "flux_loop_1"),
        "loop_positions": np.array([[0.1785, -1.2381], [0.1785, 1.2349]]),
        "passive_loops": ("coil_cases",),
        "passive_elements": {"coil_cases": 24},
        "drives": drives,
    }
    row.update(overrides)
    return DescribedMachine(**row)


# --- the reconstruction loop layout ----------------------------------------


def test_the_loop_channel_blocks_account_for_every_reconstruction_loop():
    """The family sizes are a layout, so they have to add up to the array they index."""

    rows = reconstruction_loop_rows()
    assert len(rows) == RECONSTRUCTION_LOOP_COUNT == 46
    assert sorted(rows.values()) == list(range(46))


@pytest.mark.parametrize(
    ("channel", "row"),
    [
        ("fl_cc01", 0),
        ("fl_cc10", 9),
        ("fl_p2u_1", 10),
        ("fl_p3u_4", 17),
        ("fl_p4u_4", 21),
        ("fl_p6u_1", 26),
        ("fl_p2l_1", 28),
        ("fl_p5l_4", 43),
        ("fl_p6l_2", 45),
    ],
)
def test_a_loop_channel_indexes_a_fixed_layout(channel, row):
    """The channel number is a position in the layout, not a position in a shot's list.

    Which is why an absent channel leaves its column empty rather than closing the
    gap: the two centre-column channels missing from many shots do not shift the
    twenty-four channels numbered after them.
    """

    assert reconstruction_loop_rows()[channel] == row


def test_a_loop_channel_name_survives_being_parsed():
    """Both of the archive's numbering conventions have to come back unchanged."""

    for channel in reconstruction_loop_rows():
        family, number = parse_loop_channel(channel)
        assert loop_channel_name(family, number) == channel


def test_an_unrecognised_loop_channel_is_refused():
    """A channel outside the layout has no index to be given."""

    with pytest.raises(SolveInputError, match="unrecognised flux-loop channel"):
        parse_loop_channel("fl_p7x_1")


# --- the conductor-current conversion --------------------------------------


def test_a_conductor_current_is_the_ampere_turns_over_the_turns():
    """One relation covers all three channel kinds without naming any of them."""

    signals, blocked = coil_current_signals(_machine())
    factors = {row.source_channel: row.channel_factor for row in signals}
    # a channel measuring one conductor converts at one, whatever the turn count
    assert factors["c1_feed_current"] == pytest.approx(1.0)
    # a channel already multiplied converts at the reciprocal of the turn count
    assert factors["c1_coil_current"] == pytest.approx(1.0 / 8.0)
    assert {row.source_channel for row in blocked} == {"c2_current"}


def test_an_unset_turn_count_blocks_a_channel_rather_than_defaulting_it():
    """A ratio with no denominator is not a number; a one in its place is invention."""

    _, blocked = coil_current_signals(_machine())
    row = next(iter(blocked))
    assert row.source_channel == "c2_current"
    assert "turns_with_sign is not sourced" in row.unmet
    assert row.target_path == "pf_active/coil(c2)/current/data"


def test_a_current_row_carries_the_source_convention_and_is_unscaled_by_it():
    """The toroidal sense the source and the target share leaves a current alone."""

    signals, _ = coil_current_signals(_machine())
    row = next(r for r in signals if r.source_channel == "c1_feed_current")
    assert row.transformation == IP_LIKE
    assert row.source_convention == SOURCE_CONVENTION
    assert row.target_convention == TARGET_CONVENTION
    assert row.convention_factor == pytest.approx(1.0)
    assert row.factor == pytest.approx(1.0e3)


def test_the_direct_measurement_fills_the_dictionary_field():
    """One field takes one value, and the route that avoids a fitted number wins."""

    signals, _ = coil_current_signals(_machine())
    source_map = solve_input_map(_machine(), PROBE_FAMILIES, {})
    both = source_map.for_target("pf_active/coil(c1)/current/data")
    assert {row.source_channel for row in both} == {
        "c1_feed_current",
        "c1_coil_current",
    }
    chosen = preferred_signals(source_map)
    picked = [row for row in chosen if row.standard_name == COIL_CURRENT_NAME]
    assert [row.source_channel for row in picked] == ["c1_feed_current"]
    assert len(signals) == 2


# --- the sensor rows -------------------------------------------------------


def test_a_probe_field_is_untouched_by_the_conventions():
    """A probe reads a local component along its own axis; no convention rescales it."""

    signals = probe_field_signals(_machine(), PROBE_FAMILIES)
    assert [row.source_channel for row in signals] == ["ccbv01", "obr01", "obv01"]
    assert [row.target_index for row in signals] == [0, 1, 2]
    for row in signals:
        assert row.transformation == ONE_LIKE
        assert row.factor == pytest.approx(1.0)
        assert row.source_group == FIELD_GROUP


def test_a_probe_channel_landing_on_another_family_is_refused():
    """A silently shifted probe reports a field from the wrong place."""

    machine = _machine(probes=("ccbv_0", "obv_1", "obr_2"))
    with pytest.raises(SolveInputError, match="which is not the 'obr' probe"):
        probe_field_signals(machine, PROBE_FAMILIES)


def test_a_measured_loop_flux_carries_no_further_two_pi():
    """The measured loop is declared in the convention that already has the 2*pi."""

    signals, blocked = flux_loop_signals(
        _machine(), {"fl_cc01": 0, "fl_cc02": 1, "fl_p6u_1": None}
    )
    assert [row.source_channel for row in signals] == ["fl_cc01", "fl_cc02"]
    for row in signals:
        assert row.transformation == PSI_LIKE
        assert row.source_convention == MEASURED_FLUX_CONVENTION
        assert row.convention_factor == pytest.approx(1.0)
        assert row.factor == pytest.approx(1.0)
        assert row.source_unit == row.target_unit == "Wb"
    assert [row.source_channel for row in blocked] == ["fl_p6u_1"]
    assert "no sensor for it" in blocked[0].reason


def test_the_plasma_current_row_scales_the_unit_and_nothing_else():
    """A Rogowski-derived current is a measurement in the shared toroidal sense."""

    source_map = solve_input_map(_machine(), PROBE_FAMILIES, {})
    row = next(r for r in source_map.signals if r.standard_name == PLASMA_CURRENT_NAME)
    assert row.source_group == CURRENT_GROUP
    assert row.target_path == "magnetics/ip/data"
    assert row.factor == pytest.approx(1.0e3)
    assert "Rogowski" in row.statement


def test_the_shard_contract_carries_corrected_values_and_their_dispositions(
    monkeypatch, tmp_path
):
    """The staging door returns corrected sensors and the warrant for each value."""

    time = np.array([0.0, 1.0])
    raw_probe = np.array([4.0, 8.0])
    waveforms = ShotWaveforms(
        shot=42,
        time=time,
        drives={
            "sol": np.array([100.0, 200.0]),
            "p2_inner_lower": np.array([300.0, 400.0]),
        },
        probes={"obr01": raw_probe / 2.0},
        sensors={
            "fl_cc01": np.array([0.03, 0.04]),
            "obr01": raw_probe / 2.0,
            "timesec": time,
        },
        plasma_current=np.array([500.0, 600.0]),
        sample_mask=np.ones(2, dtype=bool),
        baseline_mask=np.ones(2, dtype=bool),
        scale_corrections=(
            ScaleCorrection("fl_cc01", 42, 1.0, BRACKETED),
            ScaleCorrection("obr01", 42, 2.0, MEASURED),
            ScaleCorrection("timesec", 42, 1.0, BRACKETED),
        ),
    )
    calls = []

    def corrected_door(shot, *, store):
        calls.append((shot, store))
        return waveforms

    monkeypatch.setattr(
        "nova.imas.mast_solve_inputs.read_shot_waveforms", corrected_door
    )
    inputs = read_corrected_solve_inputs(42, store=tmp_path)

    assert calls == [(42, tmp_path)]
    assert inputs.coil_channels == ("sol", "p2_inner_lower")
    assert inputs.sensor_channels == ("fl_cc01", "obr01")
    assert inputs.sensor_units == ("Wb", "T")
    assert np.array_equal(inputs.sensor_signals[:, 1], raw_probe / 2.0)
    assert not np.array_equal(inputs.sensor_signals[:, 1], raw_probe)
    assert [row.disposition for row in inputs.corrections] == [BRACKETED, MEASURED]
    assert [row.applied for row in inputs.corrections] == [True, True]
    assert inputs.bytes_per_slice == 6 * np.dtype(float).itemsize
    assert inputs.kilobytes_per_slice == pytest.approx(0.046875)


@_needs_store
def test_a_recorded_shot_reports_a_dense_corrected_payload_per_slice():
    """The archive contract is dense, aligned and sized for shard planning."""

    inputs = read_corrected_solve_inputs(JOIN_SHOT)
    assert inputs.slice_count == inputs.coil_currents_a.shape[0]
    assert inputs.slice_count == inputs.sensor_signals.shape[0]
    assert inputs.slice_count == inputs.plasma_current_a.size
    assert inputs.coil_currents_a.shape[1] == len(inputs.coil_channels)
    assert inputs.sensor_signals.shape[1] == len(inputs.sensor_channels)
    assert [row.channel for row in inputs.corrections] == list(inputs.sensor_channels)
    assert inputs.kilobytes_per_slice == pytest.approx(inputs.bytes_per_slice / 1024.0)
    assert inputs.kilobytes_per_slice > 0.0


# --- the loop position join ------------------------------------------------


def test_a_loop_reaches_the_described_loop_it_sits_on():
    """The join is by measured position: the channel descriptions are placeholders."""

    machine = _machine()
    positions = np.full((RECONSTRUCTION_LOOP_COUNT, 2), 99.0)
    positions[0] = machine.loop_positions[1]
    positions[1] = machine.loop_positions[0]
    targets = loop_target_indices(machine, positions)
    assert targets["fl_cc01"] == 1
    assert targets["fl_cc02"] == 0
    assert targets["fl_cc03"] is None


def test_two_channels_never_claim_one_described_loop():
    """Loops of a pair sit millimetres apart, so many-to-one would hide a sensor."""

    machine = _machine()
    positions = np.full((RECONSTRUCTION_LOOP_COUNT, 2), 99.0)
    positions[0] = machine.loop_positions[0]
    positions[1] = machine.loop_positions[0]
    targets = loop_target_indices(machine, positions)
    claimed = [row for row in targets.values() if row is not None]
    assert claimed == [0]
    assert len(claimed) == len(set(claimed))


def test_a_loop_beyond_the_tolerance_is_a_gap_and_not_a_match():
    """A described set that does not cover a loop is reported, not stretched to fit."""

    machine = _machine()
    positions = np.full((RECONSTRUCTION_LOOP_COUNT, 2), 99.0)
    positions[0] = machine.loop_positions[0] + np.array(
        [2.0 * LOOP_POSITION_TOLERANCE, 0.0]
    )
    assert loop_target_indices(machine, positions)["fl_cc01"] is None


# --- what the description cannot receive ------------------------------------


def test_every_blocked_channel_states_an_unmet_condition():
    """A blocked row exists to be actionable, so it names what would unblock it."""

    machine = _machine()
    blocked = (
        *case_current_blocked(machine),
        *toroidal_field_blocked(),
        *unmapped_current_blocked(machine),
    )
    for row in blocked:
        row.validate()
        assert row.reason and row.unmet


def test_the_measured_case_currents_share_one_dictionary_field():
    """Eight enclosure currents and one loop field is a dictionary limit."""

    machine = _machine(
        drives=DriveMap.create(
            [
                ChannelDrive(
                    channel=f"p{index}l_case_current",
                    container="pf_passive",
                    conductor="coil_cases",
                    elements=(index,),
                    circuit="",
                    ampere_turns_per_ampere=1.0,
                    distribution="single",
                    evidence=FieldEvidence.GENERATED,
                    path=f"pf_passive/loop(coil_cases)/current(p{index}l)",
                    uncertainty=Uncertainty(lower=1.0, upper=1.0, unit="turn"),
                )
                for index in (2, 3)
            ]
        )
    )
    blocked = case_current_blocked(machine)
    assert [row.source_channel for row in blocked] == [
        "p2l_case_current",
        "p3l_case_current",
    ]
    assert "one current for the whole loop" in blocked[0].reason
    assert "24 described plates" in blocked[0].reason
    assert "drive map carries the per-element weights" in blocked[0].unmet


def _paired_machine(cased: tuple[str, ...] = ("c1",)) -> DescribedMachine:
    """Return a machine whose named sets publish a coil channel and a case one."""

    drives = [
        ChannelDrive(
            channel=f"{name}_coil_current",
            container="pf_active",
            conductor=name,
            elements=(0,),
            circuit="C",
            ampere_turns_per_ampere=1.0,
            distribution="single",
            evidence=FieldEvidence.MEASURED,
            path=f"pf_active/coil({name})/current({name}_coil_current)",
        )
        for name in ("c1", "c2")
    ]
    drives += [
        ChannelDrive(
            channel=f"{name}_case_current",
            container="pf_passive",
            conductor="coil_cases",
            elements=(index,),
            circuit="",
            ampere_turns_per_ampere=1.0,
            distribution="single",
            evidence=FieldEvidence.GENERATED,
            path=f"pf_passive/loop(coil_cases)/current({name})",
            uncertainty=Uncertainty(lower=1.0, upper=1.0, unit="turn"),
        )
        for index, name in enumerate(cased)
    ]
    return _machine(turns={"c1": 8.0, "c2": 12.0}, drives=DriveMap.create(drives))


def test_a_pack_total_is_paired_with_the_two_terms_it_sums():
    """Read off the drive map, so each row names the channels it decomposes into."""

    rows = pack_total_channels(_paired_machine())
    assert rows == (("c1", "c1_coil_current", "c1_case_current"),)
    blocked = {
        row.source_channel: row for row in unmapped_current_blocked(_paired_machine())
    }
    assert "c1_current" in blocked
    assert "c1_coil_current plus c1_case_current" in blocked["c1_current"].reason
    assert "not a fixed multiple" in blocked["c1_current"].unmet


def test_a_set_without_a_measured_case_carries_no_pack_total_row():
    """A total whose second term is unmeasured is a different refusal, not this one."""

    assert pack_total_channels(_machine()) == ()
    paired = pack_total_channels(_paired_machine(cased=("c1", "c2")))
    assert [row[0] for row in paired] == ["c1", "c2"]


def test_the_toroidal_field_is_blocked_by_the_description_not_the_source():
    """Both toroidal routes want a described conductor or a described radius."""

    blocked = {row.source_channel: row for row in toroidal_field_blocked()}
    assert set(blocked) == {"tf_current", "bvac_val"}
    assert "tf/coil" in blocked["tf_current"].unmet
    assert "tf/r0" in blocked["bvac_val"].unmet


# --- the published description, both polarities ----------------------------


@pytest.fixture(scope="module")
def description(tmp_path_factory):
    """Publish and open the refined description the solve inputs are served against."""

    artifact = publish_refined_artifact(
        tmp_path_factory.mktemp("described") / "refined"
    )
    return open_verified_description(artifact)


@pytest.fixture(scope="module")
def probe_families():
    """Return the registry probe sequence the channel join is built from."""

    registry = MachineGeometryRegistry.default()
    geometry = registry.select(REPRESENTATIVE_SHOT).configuration.geometry
    return geometry["magnetics"]["poloidal_probes"]


@pytest.fixture(scope="module")
def published_map(description, probe_families):
    """Build the solve-input map against the published description."""

    return build_solve_input_map(description, probe_families, shot=JOIN_SHOT)


def test_the_description_leaves_exactly_one_coil_pair_unserved(
    description, published_map
):
    """The one gap in the served currents is the turn count no experiment fixed."""

    blocked = {
        row.source_channel
        for row in published_map.blocked
        if row.target_path.startswith("pf_active/coil(p6")
    }
    assert blocked == {"p6l_current", "p6u_current"}
    assert description.machine.turns["p6_lower"] is None
    assert description.machine.turns["p6_upper"] is None
    served = {row.conductor for row in description.machine.drives.drives}
    assert {"p6_lower", "p6_upper"} <= served


def test_the_served_rows_cover_every_described_probe(description, published_map):
    """A described sensor with no channel is one a solve cannot use."""

    probes = [
        row for row in published_map.signals if row.standard_name == PROBE_FIELD_NAME
    ]
    assert len(probes) == 78
    assert [row.target_index for row in probes] == list(range(78))
    for row in probes:
        assert (
            description.machine.probes[row.target_index]
            == (row.target_path.split("(")[1].split(")")[0])
        )


def test_a_set_level_channel_names_the_conductors_it_spans(description, published_map):
    """The obstruction is read off the description, so it names the packs it spans."""

    blocked = {row.source_channel: row for row in published_map.blocked}
    reason = blocked["p2l_current"].reason
    assert "p2_inner_lower" in reason
    assert "p2_outer_lower" in reason
    assert "interconnection" in blocked["p2l_current"].unmet


def test_a_set_level_channel_names_the_rows_that_already_carry_it(published_map):
    """A refusal that rests on redundancy has to name what makes it redundant.

    The set total is not unexplained: each pack publishes its own already-multiplied
    channel, which the map serves, and the set publishes its case, which is refused
    for the separate reason that the dictionary holds one current per passive loop.
    Naming all three is what separates this refusal from one nobody has looked into.
    """

    blocked = {row.source_channel: row for row in published_map.blocked}
    served = {row.source_channel for row in published_map.signals}
    for prefix, side in (("p2l", "lower"), ("p2u", "upper")):
        reason = blocked[f"{prefix}_current"].reason
        packs = (f"p2i{side[0]}_coil_current", f"p2o{side[0]}_coil_current")
        case = f"{prefix}_case_current"
        for term in (*packs, case):
            assert term in reason, (prefix, term)
        assert set(packs) <= served
        assert case in blocked and case not in served


@_needs_store
@pytest.mark.parametrize("shot", (FORWARD_SHOTS[1], REVERSED_SHOTS[0]))
def test_a_set_total_is_its_two_packs_plus_its_case_to_the_last_bit(shot):
    """The redundancy the refusal rests on is measured, not assumed."""

    import zarr

    group = zarr.open_group(f"{SHOT_STORE}/{shot}.zarr", mode="r")[CURRENT_GROUP]
    for prefix, side in (("p2l", "lower"), ("p2u", "upper")):
        names = (
            f"{prefix}_current",
            f"p2i{side[0]}_coil_current",
            f"p2o{side[0]}_coil_current",
            f"{prefix}_case_current",
        )
        traces = [np.asarray(group[name][...], dtype=float) for name in names]
        finite = np.all([np.isfinite(trace) for trace in traces], axis=0)
        total, inner, outer, case = (trace[finite] for trace in traces)
        scale = sum(float(np.max(np.abs(term))) for term in (inner, outer, case))
        residual = float(np.max(np.abs(total - inner - outer - case))) / scale
        assert residual < PACK_TOTAL_TOLERANCE, (prefix, residual)


@_needs_store
def test_the_loop_join_is_a_property_of_the_configuration(description):
    """The join must not depend on which shot's measured positions it was taken from."""

    joins = {
        shot: loop_target_indices(
            description.machine, reconstruction_loop_positions(shot)
        )
        for shot in FORWARD_SHOTS + REVERSED_SHOTS
    }
    reference = joins[JOIN_SHOT]
    for shot, join in joins.items():
        assert join == reference, shot
    served = [row for row in reference.values() if row is not None]
    assert len(served) == len(set(served)) == 43


@_needs_store
@pytest.mark.parametrize("shot", (JOIN_SHOT, REVERSED_SHOTS[0]))
def test_the_loop_channel_numbering_is_the_reconstruction_index(shot):
    """The layout is measured: each channel reproduces the column its number names."""

    rows = reconstruction_loop_rows()
    matches = trace_matched_columns(shot, sorted(rows), "silop_x")
    discriminating = {
        channel: match
        for channel, match in matches.items()
        if match[1] < DISCRIMINATING_RESIDUAL
        and match[2] > DISCRIMINATING_MARGIN * match[1]
    }
    assert len(discriminating) >= 20
    violations = {
        channel: (match[0], rows[channel])
        for channel, match in discriminating.items()
        if match[0] != rows[channel]
    }
    assert not violations


@_needs_store
@pytest.mark.parametrize("shot", (JOIN_SHOT, REVERSED_SHOTS[0]))
def test_each_probe_channel_reads_the_sensor_it_is_mapped_to(
    shot, description, published_map
):
    """Checked against the reconstruction's own poses, not against an index convention.

    The outer arrays are co-located pairs -- a radial and an axial probe at each of
    nineteen positions -- so an index correspondence alone would not distinguish
    them.  Position and sensitive axis together do.  The reconstruction angle
    increases counter-clockwise while DDv4 ``poloidal_angle`` increases clockwise,
    so the same directed axis has the opposite numerical angle in the description.
    """

    source_poses = reconstruction_probe_poses(shot)
    described = description.machine.probe_poses
    rows = {
        row.source_channel: row.target_index
        for row in published_map.signals
        if row.standard_name == PROBE_FIELD_NAME
    }
    matches = trace_matched_columns(shot, sorted(rows), "magpr_x")
    checked = 0
    for channel, match in matches.items():
        column, residual, runner_up = match
        if residual >= DISCRIMINATING_RESIDUAL:
            continue
        if runner_up <= DISCRIMINATING_MARGIN * residual:
            continue
        target = described[rows[channel]]
        assert (
            np.hypot(*(source_poses[column, :2] - target[:2]))
            <= PROBE_POSITION_TOLERANCE
        )
        assert -source_poses[column, 2] == pytest.approx(target[2], abs=1e-4)
        checked += 1
    assert checked >= 30


@_needs_store
@_needs_standard_names
@pytest.mark.parametrize("shot", (FORWARD_SHOTS[1], REVERSED_SHOTS[0]))
def test_a_shot_round_trips_through_the_description(
    shot, description, published_map, tmp_path
):
    """The whole path: store to dataset to pulse and back to the source samples."""

    trip = round_trip_shot(description, published_map, shot, work_directory=tmp_path)
    assert trip.dataset_residual < 1e-12
    assert trip.read_back_residual < 1e-12
    assert trip.description_preserved
    assert len(trip.served_channels) > 80
    assert trip.sample_counts["current"] > 0
    assert trip.sample_counts["field"] > 0


@_needs_store
@_needs_standard_names
def test_both_field_polarities_reach_the_description_unflipped(
    description, published_map, tmp_path
):
    """A factor touching a current would move one cohort and not the other.

    The invariant is not either sign but their product: the two cohorts run the
    plasma current and the toroidal field both reversed, so a map that preserved
    the relation in one and not the other would be scaling something it should not.
    """

    seen = {}
    for shot in (FORWARD_SHOTS[1], REVERSED_SHOTS[0]):
        trip = round_trip_shot(
            description,
            published_map,
            shot,
            work_directory=tmp_path / f"polarity_{shot}",
        )
        peaks = trip.polarity
        seen[shot] = np.sign(peaks["plasma_current"]) * np.sign(peaks["tf_current"])
        assert trip.read_back_residual < 1e-12
    forward, reversed_ = seen[FORWARD_SHOTS[1]], seen[REVERSED_SHOTS[0]]
    assert forward == reversed_ == -1
    assert field_polarity(FORWARD_SHOTS[1])["plasma_current"] > 0
    assert field_polarity(REVERSED_SHOTS[0])["plasma_current"] < 0


@_needs_store
@_needs_standard_names
def test_a_coil_publishing_two_channels_measures_its_own_turn_count(
    description, published_map, tmp_path
):
    """A coil reached through two channels must agree with itself on its turns.

    Every coil carrying a published turn count resolves to the same exact integer
    on the feed route and the coil route, so the two reconstructions of its current
    agree to floating point.  The comparison runs over recorded waveforms, so the
    two routes close to about a part in a million rather than exactly.
    """

    trip = round_trip_shot(
        description, published_map, FORWARD_SHOTS[1], work_directory=tmp_path
    )
    counted = [
        value
        for path, value in trip.redundancy.items()
        if any(name in path for name in ("p2_", "p3_", "p4_", "p5_"))
    ]
    assert counted and max(counted) < 1e-6


@_needs_store
@_needs_standard_names
def test_a_channel_off_its_own_clock_is_refused_rather_than_aligned(published_map):
    """Aligning a channel to a clock it does not fit moves every sample in time."""

    signals = read_solve_inputs(published_map, REVERSED_SHOTS[0])
    assert "fl_cc10" in signals.misaligned_channels
    assert "fl_cc10" not in signals.samples
    assert "fl_cc10" not in signals.source_map.channels()


@_needs_store
def test_the_excitation_group_is_fully_accounted_for(published_map):
    """A channel absent from a map with no reason cannot be told from an oversight."""

    import zarr

    group = zarr.open_group(f"{SHOT_STORE}/{JOIN_SHOT}.zarr", mode="r")[CURRENT_GROUP]
    published = {
        name for name in group.keys() if name not in ("time", "timesec", "status")
    }
    served = {
        row.source_channel
        for row in published_map.signals
        if row.source_group == CURRENT_GROUP
    }
    blocked = {
        row.source_channel
        for row in published_map.blocked
        if row.source_group == CURRENT_GROUP
    }
    assert served | blocked == published
    assert not served & blocked
    assert len(served) == 22
    assert len(blocked) == 22


@_needs_store
@pytest.mark.parametrize("shot", (FORWARD_SHOTS[1], REVERSED_SHOTS[0]))
def test_a_pack_total_is_its_coil_plus_its_case_to_the_last_bit(shot, description):
    """The identity is the channel's whole content, so it is read from the store."""

    residuals = pack_total_residuals(description.machine, shot)
    assert set(residuals) == {"p3l", "p3u", "p4l", "p4u", "p5l", "p5u"}
    assert max(residuals.values()) < PACK_TOTAL_TOLERANCE


@_needs_store
def test_a_pack_total_reaches_neither_term_by_a_fixed_factor(description):
    """The refusal rests on the case being induced: a spread, not an offset."""

    import zarr

    group = zarr.open_group(f"{SHOT_STORE}/{FORWARD_SHOTS[1]}.zarr", mode="r")[
        CURRENT_GROUP
    ]
    spreads = {}
    for prefix, coil_channel, _ in pack_total_channels(description.machine):
        coil = np.asarray(group[coil_channel][...], dtype=float)
        total = np.asarray(group[f"{prefix}_current"][...], dtype=float)
        driven = np.isfinite(coil) & np.isfinite(total)
        driven &= np.abs(coil) > 0.2 * np.max(np.abs(coil[np.isfinite(coil)]))
        ratio = total[driven] / coil[driven]
        spreads[prefix] = float(np.ptp(ratio))
    assert min(spreads.values()) > 0.1


@_needs_store
def test_every_pack_total_names_the_two_channels_it_sums(description, published_map):
    """A refusal that states a mechanism has to state it in the description's terms."""

    blocked = {row.source_channel: row for row in published_map.blocked}
    for prefix, coil_channel, case_channel in pack_total_channels(description.machine):
        row = blocked[f"{prefix}_current"]
        assert coil_channel in row.reason
        assert case_channel in row.reason
        assert coil_channel in {
            signal.source_channel for signal in published_map.signals
        }
        assert case_channel in blocked


@_needs_store
def test_the_loop_flux_rows_serve_the_measured_total_flux(published_map):
    """Every served loop keeps the Weber it was measured in."""

    rows = [row for row in published_map.signals if row.standard_name == LOOP_FLUX_NAME]
    assert len(rows) == 43
    assert {row.factor for row in rows} == {1.0}
