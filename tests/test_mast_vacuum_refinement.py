"""The vacuum refinement's identifiability rules and what it authored.

The fit itself reads a shot store that is not present everywhere, so the tests
that need waveforms build them: a synthetic shot with a known turn count is a
stronger check than a recorded one anyway, because the answer is known in advance
and every screening rule can be given the exact case it exists to refuse.

The tests that need no waveforms -- the channel joins, the authored artifact, the
evidence ledger -- run against the registry and the pinned fit results directly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.machine_evidence import EvidenceLedger, FieldEvidence
from nova.imas.mast_fitted_parameters import (
    AXIAL_PROBE_FAMILIES,
    RADIAL_PROBE_FAMILY,
    WORST_PUBLISHED_OFFSET,
    VACUUM_FITTED_TURNS,
    authored_turns,
    fitted_evidence,
    fitted_turns,
    refined_evidence,
)
from nova.imas.mast_geometry import (
    REPRESENTATIVE_SHOT,
    author_refined_ids,
    publish_refined_artifact,
)
from nova.imas.mast_passive_response import (
    CASE_FAMILY,
    PassiveError,
    case_grouping,
    case_side,
    passive_coupling,
    passive_groups,
)
from nova.imas.mast_seed_parameters import seed_evidence
from nova.imas.mast_vacuum_cohort import (
    COIL_DRIVES,
    ENERGISED_CURRENT,
    EXCITATION_CURRENT,
    PLASMA_FREE_CURRENT,
    SUSTAINED_HOLD,
    CohortError,
    ShotSurvey,
    ShotWaveforms,
    parse_probe_channel,
    probe_channels,
    select_vacuum_cohort,
)
from nova.imas.mast_vacuum_response import (
    MAXIMUM_CORRELATION,
    MINIMUM_STANDOFF,
    ResponseError,
    ResponseModel,
    aggregate_turns,
    coil_response_matrix,
    coil_sections,
    per_shot_estimates,
    probe_standoff,
    score_prediction,
)

REFINED_SEMANTIC_IDENTITY = (
    "sha256:ec2f7c6015ac4b193e2f9bb4594b2a90e7ff9650bb5566d214c3aec39f179c88"
)
"""Semantic address of the refined revision this worktree authors.

It moves whenever the evidence ledger gains or restates a record, which is the
behaviour the identity split exists for: what the description asserts about a
field has changed while the conductor geometry has not, so
:data:`PHYSICAL_DIGEST` and :data:`REGISTRY_DIGEST` are pinned beside it and a
change in either is a fault rather than an update.
"""

PHYSICAL_DIGEST = "ca06c8f64481114f"
"""Conductor geometry the refinement must not disturb."""

REGISTRY_DIGEST = "7083e8029c879310d4b811ecc58f5eefdd40b2bfe01b4a1714b177b03a307366"
"""Registry identity the refinement must not disturb."""

DRIVEN_COLUMNS = 21
"""Conductor sets the campaign's published current channels reach."""


@pytest.fixture(scope="module")
def geometry():
    """Return the registry configuration the refinement is authored against."""

    registry = MachineGeometryRegistry.default()
    return registry.select(REPRESENTATIVE_SHOT).configuration.geometry


@pytest.fixture(scope="module")
def probes(geometry):
    """Return the registry's poloidal probe sequence."""

    return geometry["magnetics"]["poloidal_probes"]


@pytest.fixture(scope="module")
def channels(probes):
    """Return the store channel names joined onto the registry probe blocks."""

    return probe_channels(probes)


@pytest.fixture(scope="module")
def model(geometry, probes, channels):
    """Return the response model under the fitted sensitive-axis assignment."""

    return ResponseModel.build(
        geometry,
        probes,
        channels,
        radial_families=frozenset({RADIAL_PROBE_FAMILY}),
    )


def synthetic_shot(
    model: ResponseModel,
    turns: dict[str, float],
    *,
    shot: int = 900001,
    samples: int = 400,
    offset: float = 3.0e-4,
    noise: float = 0.0,
    seed: int = 0,
) -> ShotWaveforms:
    """Build a shot whose probes read exactly what ``turns`` implies.

    The waveform holds a flat top so the shot passes the sustained test, and each
    probe carries a standing offset so the baseline subtraction is exercised
    rather than assumed away.
    """

    generator = np.random.default_rng(seed)
    time = np.linspace(-0.2, 1.0, samples)
    ramp = np.clip(time / 0.05, 0.0, 1.0) * np.clip((0.8 - time) / 0.05, 0.0, 1.0)
    drives = {family: np.zeros_like(time) for family in model.families}
    for family, excitation in turns.items():
        drives[family] = excitation["current"] * ramp

    probes_out: dict[str, np.ndarray] = {}
    for row, target in enumerate(model.targets):
        signal = np.full_like(time, offset * (1 + row % 5))
        for column, family in enumerate(model.families):
            count = turns.get(family, {}).get("turns", 0.0)
            signal = signal + model.response[row, column] * count * drives[family]
        if noise > 0.0:
            signal = signal + generator.normal(0.0, noise, size=time.shape)
        probes_out[target.channel] = signal

    return ShotWaveforms(
        shot=shot,
        time=time,
        drives=drives,
        probes=probes_out,
        plasma_current=np.zeros_like(time),
        sample_mask=np.ones(time.shape, dtype=bool),
        baseline_mask=time < -0.05,
    )


def drive(current: float, turns: float) -> dict[str, float]:
    """Describe one coil's excitation and the turn count that scales it."""

    return {"current": current, "turns": turns}


# --- channel identity -------------------------------------------------


def test_probe_blocks_join_in_channel_order(probes, channels):
    """Each family's channels number from one across its registry block."""

    assert len(channels) == len(probes)
    for channel in channels:
        assert probes[channel.registry_index]["family"] == channel.family
        assert channel.channel == f"{channel.family}{channel.number:02d}"
    numbers = [c.number for c in channels if c.family == "ccbv"]
    assert numbers == list(range(1, 41))


def test_unrecognised_channel_is_refused():
    """A channel name outside the known families is rejected, not guessed at."""

    assert parse_probe_channel("obr07") == ("obr", 7)
    with pytest.raises(CohortError):
        parse_probe_channel("xyz01")


def test_every_active_component_has_one_drive_channel(geometry):
    """The drive table covers each registry coil exactly once."""

    families = sorted(geometry["active_components"])
    assert sorted(row.family for row in COIL_DRIVES) == families
    assert len({row.channel for row in COIL_DRIVES}) == len(COIL_DRIVES)
    for row in COIL_DRIVES:
        row.validate()


# --- cohort selection -------------------------------------------------


def survey(shot: int, **peaks: float) -> ShotSurvey:
    """Build a survey record with sustained excitation on the named coils."""

    return ShotSurvey(
        shot=shot,
        plasma_current_peak=peaks.pop("plasma", 0.0),
        toroidal_current_peak=0.0,
        coil_peaks=dict(peaks),
        coil_hold_times={name: 1.0 for name in peaks},
        turn_multipliers={},
        absent_groups=(),
        absent_channels=(),
        field_channels=tuple(f"ccbv{i:02d}" for i in range(1, 41)),
    )


def test_plasma_shots_are_refused_with_their_reason():
    """A shot carrying plasma current leaves the cohort and says why."""

    rows = [
        survey(1, p3_upper=2.0e4),
        survey(2, p3_upper=2.0e4, plasma=PLASMA_FREE_CURRENT * 2),
        survey(3, p3_upper=2.0e4),
        survey(4, p4_upper=2.0e4),
        survey(5, p4_upper=2.0e4),
    ]
    cohort = select_vacuum_cohort(rows, held_out_fraction=0.5)
    assert 2 not in cohort.shots
    reasons = {row.shot: row.reason for row in cohort.exclusions}
    assert "plasma current" in reasons[2]


def test_idle_coils_do_not_label_a_shot():
    """A coil holding a standing current is not a deliberate excitation."""

    quiet = survey(1, p6_upper=ENERGISED_CURRENT * 2)
    assert quiet.energised_families == ("p6_upper",)
    assert quiet.excited_families == ()
    with pytest.raises(CohortError):
        select_vacuum_cohort([quiet])


def test_split_never_shares_a_shot_or_a_withheld_family():
    """Training and held-out arms are disjoint and honour a withheld family."""

    rows = [survey(index, p3_upper=2.0e4) for index in range(1, 7)]
    rows += [survey(index, p4_upper=2.0e4) for index in range(7, 13)]
    cohort = select_vacuum_cohort(rows, held_out_families=["P4"])
    assert not set(cohort.training) & set(cohort.held_out)
    assert all(cohort.families[shot] != "P4" for shot in cohort.training)
    assert {cohort.families[shot] for shot in cohort.held_out} >= {"P4"}
    cohort.validate()


def test_series_partner_and_hold_time_gate_isolation():
    """A coil is isolating only when its partner is quiet and it is held."""

    both = survey(1, p3_upper=2.0e4, p3_lower=2.0e4)
    assert both.asymmetric_coils() == ()
    alone = survey(2, p3_upper=2.0e4, p3_lower=1.0e2)
    assert alone.asymmetric_coils() == ("p3_upper",)
    pulsed = ShotSurvey(
        shot=3,
        plasma_current_peak=0.0,
        toroidal_current_peak=0.0,
        coil_peaks={"p3_upper": 2.0e4},
        coil_hold_times={"p3_upper": SUSTAINED_HOLD / 10.0},
        turn_multipliers={},
        absent_groups=(),
        absent_channels=(),
        field_channels=(),
    )
    assert pulsed.asymmetric_coils() == ("p3_upper",)
    assert pulsed.sustained_coils() == ()


# --- response model ---------------------------------------------------


def test_response_scales_with_current_and_turns(geometry, model):
    """The predicted field is linear in both the turn count and the current."""

    response = coil_response_matrix(geometry, model.targets)
    assert response.shape == (len(model.targets), len(model.families))
    assert np.all(np.isfinite(response))
    single = synthetic_shot(model, {"p3_upper": drive(1.0e4, 8.0)}, shot=900010)
    double = synthetic_shot(model, {"p3_upper": drive(2.0e4, 8.0)}, shot=900011)
    quadruple = synthetic_shot(model, {"p3_upper": drive(2.0e4, 16.0)}, shot=900012)
    channel = model.targets[0].channel
    base = single.probes[channel] - single.probes[channel][0]
    assert np.allclose(double.probes[channel] - double.probes[channel][0], 2.0 * base)
    assert np.allclose(
        quadruple.probes[channel] - quadruple.probes[channel][0], 4.0 * base
    )


def test_standoff_uses_the_pack_width_not_its_diagonal(geometry, model):
    """A long thin coil does not make every probe in the machine near-field."""

    standoff = probe_standoff(geometry, model.targets)
    column = model.families.index("sol")
    sections = coil_sections(geometry)["sol"]
    bounds = shapely.Polygon(sections[0]).bounds
    width = min(bounds[2] - bounds[0], bounds[3] - bounds[1])
    diagonal = math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])
    assert diagonal > 10.0 * width
    # Scaled by the width, the outboard probes stand many widths clear; scaled by
    # the diagonal every probe in the machine would sit inside one.
    assert standoff[:, column].max() > 5.0


def test_probes_inside_a_pack_are_never_admitted(model):
    """A probe closer than the cut is dropped for every coil excited that shot."""

    keep = model.admissible_probes(["p4_upper"])
    column = model.families.index("p4_upper")
    assert keep.sum() < len(model.targets)
    assert np.all(model.standoff[keep, column] >= MINIMUM_STANDOFF)


def test_axis_assignment_changes_the_predicted_component(geometry, probes, channels):
    """Reassigning a family's sensitive axis changes what the model predicts."""

    radial = ResponseModel.build(
        geometry, probes, channels, radial_families=frozenset({"obr"})
    )
    axial = ResponseModel.build(geometry, probes, channels, radial_families=frozenset())
    rows = [i for i, t in enumerate(radial.targets) if t.family == "obr"]
    assert not np.allclose(radial.response[rows], axial.response[rows])


def test_sensitive_axis_is_a_unit_vector(model):
    """Every probe carries a normalised axis, so no gain hides in it."""

    for target in model.targets:
        target.validate()
        assert math.isclose(
            math.hypot(target.radial_cosine, target.axial_sine), 1.0, abs_tol=1e-12
        )


# --- identifiability --------------------------------------------------


def test_a_clean_single_coil_shot_recovers_its_turn_count(model):
    """A synthetic shot with a known turn count is fitted back exactly."""

    shot = synthetic_shot(model, {"p3_upper": drive(2.0e4, 8.0)})
    estimates = [
        row for row in per_shot_estimates([shot], model) if row.family == "p3_upper"
    ]
    assert len(estimates) == 1
    assert estimates[0].identified
    assert estimates[0].multiplier == pytest.approx(8.0, abs=1e-6)


def test_polarity_is_recovered_with_the_turn_count(model):
    """A coil wired the other way round comes back with a negative count."""

    shot = synthetic_shot(model, {"p3_upper": drive(2.0e4, -8.0)})
    estimate = next(
        row for row in per_shot_estimates([shot], model) if row.family == "p3_upper"
    )
    assert estimate.multiplier == pytest.approx(-8.0, abs=1e-6)
    assert estimate.identified


def test_an_up_down_pair_driven_together_still_separates(model):
    """A series pair is separable because the probe array is not symmetric.

    Sharing a waveform is not by itself degeneracy.  The upper and lower coil of a
    set sit at opposite ends of an array that samples both, so their patterns over
    the probes differ and the solve can divide the pair's total between them.  The
    correlation screen has to leave this case alone, and this test is what says it
    does.
    """

    shot = synthetic_shot(
        model,
        {"p3_upper": drive(2.0e4, 8.0), "p3_lower": drive(2.0e4, 6.0)},
        noise=1.0e-6,
    )
    estimates = {
        row.family: row
        for row in per_shot_estimates([shot], model)
        if row.family.startswith("p3")
    }
    assert set(estimates) == {"p3_upper", "p3_lower"}
    assert all(row.identified for row in estimates.values())
    assert estimates["p3_upper"].multiplier == pytest.approx(8.0, abs=0.05)
    assert estimates["p3_lower"].multiplier == pytest.approx(6.0, abs=0.05)


def test_two_packs_sharing_a_waveform_and_a_place_are_refused(model):
    """Coils that are neither separated in space nor in time get no number.

    The inner and outer winding packs of one coil set sit a few centimetres apart,
    so a probe metres away cannot tell their fields apart; driven with the same
    waveform they are degenerate in both senses at once.  The solve still returns
    two numbers, and the correlation between them is what refuses them.
    """

    shot = synthetic_shot(
        model,
        {
            "p2_inner_lower": drive(2.0e4, 12.0),
            "p2_outer_lower": drive(2.0e4, 8.0),
        },
        noise=2.0e-5,
    )
    estimates = {
        row.family: row
        for row in per_shot_estimates([shot], model)
        if row.family.startswith("p2")
    }
    assert set(estimates) == {"p2_inner_lower", "p2_outer_lower"}
    assert all(abs(row.correlation) > MAXIMUM_CORRELATION for row in estimates.values())
    assert not any(row.identified for row in estimates.values())


def test_an_unseen_coil_is_refused_rather_than_fitted(model):
    """A coil contributing almost nothing does not get a number."""

    shot = synthetic_shot(
        model,
        {"p4_upper": drive(2.0e4, 23.0), "p6_upper": drive(EXCITATION_CURRENT, 1.0)},
    )
    estimates = {row.family: row for row in per_shot_estimates([shot], model)}
    assert estimates["p4_upper"].identified
    assert not estimates["p6_upper"].identified
    assert estimates["p6_upper"].leverage < 0.02


def test_aggregate_reports_spread_across_shots(model):
    """Shots that disagree widen the interval rather than the mean."""

    shots = [
        synthetic_shot(
            model,
            {"p4_upper": drive(2.0e4, count)},
            shot=900100 + index,
            noise=1.0e-5,
            seed=index,
        )
        for index, count in enumerate((22.0, 24.0))
    ]
    disposition = next(
        row
        for row in aggregate_turns(per_shot_estimates(shots, model))
        if row.family == "p4_upper"
    )
    assert disposition.multiplier == pytest.approx(23.0, abs=0.2)
    assert disposition.spread == pytest.approx(1.0, abs=0.1)
    assert not disposition.resolves_an_integer
    assert disposition.interval[0] < 23.0 < disposition.interval[1]


def test_fitted_turns_predict_a_shot_they_were_not_fitted_on(model):
    """A held-out multi-coil shot is reproduced by single-coil turn counts."""

    training = [
        synthetic_shot(model, {"p3_upper": drive(2.0e4, 8.0)}, shot=900200),
        synthetic_shot(model, {"p4_upper": drive(1.5e4, 23.0)}, shot=900201),
    ]
    counts = {
        row.family: row.multiplier
        for row in aggregate_turns(per_shot_estimates(training, model))
        if row.identified
    }
    assert set(counts) == {"p3_upper", "p4_upper"}
    held_out = synthetic_shot(
        model,
        {"p3_upper": drive(1.1e4, 8.0), "p4_upper": drive(0.9e4, 23.0)},
        shot=900202,
    )
    score = score_prediction([held_out], model, counts)
    assert score.variance_explained > 0.999


def test_a_shot_without_a_baseline_window_is_refused(model):
    """Probe offsets are measured, so a shot with no quiet window is refused."""

    shot = synthetic_shot(model, {"p3_upper": drive(2.0e4, 8.0)})
    shot.baseline_mask = np.zeros_like(shot.baseline_mask)
    with pytest.raises(ResponseError):
        model.design(shot)


# --- passive grouping -------------------------------------------------


def test_case_plates_group_onto_the_coil_set_they_enclose(geometry):
    """Every case plate joins one coil set and none is left unassigned."""

    grouped = case_grouping(geometry)
    plates = sum(len(rows) for rows in grouped.values())
    outline = shapely.from_wkb(
        bytes.fromhex(geometry["passive_components"][CASE_FAMILY])
    )
    assert plates == len(outline.geoms)
    assert all(name.startswith(f"{CASE_FAMILY}_") for name in grouped)
    assert "coil_cases_unassigned" not in grouped


def test_case_grouping_is_up_down_symmetric(geometry):
    """A coil set and its mirror carry the same number of case plates."""

    grouped = case_grouping(geometry)
    for name, rows in grouped.items():
        mirror = name.replace("_upper", "_UP").replace("_lower", "_upper")
        mirror = mirror.replace("_UP", "_lower")
        assert len(grouped[mirror]) == len(rows), name


def test_case_side_folds_winding_packs_onto_one_case():
    """The inner and outer packs of a set share the case that encloses them."""

    assert case_side("p2_inner_upper") == case_side("p2_outer_upper") == "p2_upper"
    assert case_side("p3_lower") == "p3_lower"


def test_passive_coupling_covers_every_group(geometry, model):
    """Each grouped conductor produces a finite, non-zero probe pattern."""

    groups = passive_groups(geometry)
    coupling = passive_coupling(groups, model.targets)
    assert coupling.shape == (len(model.targets), len(groups))
    assert np.all(np.isfinite(coupling))
    assert np.all(np.linalg.norm(coupling, axis=0) > 0.0)


def test_a_group_without_a_section_is_refused(geometry, model):
    """An empty conductor group raises rather than contributing nothing."""

    groups = list(passive_groups(geometry))
    groups[0] = type(groups[0])(name="hollow", family="hollow", sections=())
    with pytest.raises(PassiveError):
        passive_coupling(groups, model.targets)


# --- what was authored ------------------------------------------------


def test_each_coil_states_which_route_established_its_count():
    """A published count, a fitted one and an unreachable one stay distinguishable."""

    published = [row for row in VACUUM_FITTED_TURNS if row.published]
    fitted = [
        row for row in VACUUM_FITTED_TURNS if row.identified and not row.published
    ]
    absent = [row for row in VACUUM_FITTED_TURNS if not row.identified]
    assert {row.family for row in absent} == {"p6_lower", "p6_upper"}
    assert len(published) == 10
    assert {row.family for row in fitted} == {"sol"}
    for row in published:
        assert row.turns == float(row.published_turns)
        assert row.turns == float(round(row.turns))


def test_the_published_count_is_corroborated_by_an_independent_fit():
    """Each published integer is reached again by measuring the field.

    The two routes are independent: one divides two published current channels,
    the other predicts the probes from the registry outlines and the measured
    current.  Agreement is therefore a check on the description and not a
    restatement of the archive.
    """

    covered = [row for row in VACUUM_FITTED_TURNS if row.published]
    assert covered
    for row in covered:
        assert abs(row.corroboration) <= WORST_PUBLISHED_OFFSET + 1.0e-6


def test_a_published_interval_is_never_narrower_than_its_disagreement():
    """An exact integer is not reported with an interval that asserts agreement.

    The archive's integer is exact as a statement; what is uncertain is whether this
    description reproduces the field it implies.  So the interval has to be at least
    as wide as the distance the fit lands from it -- a zero-width interval on an
    exact integer would claim the agreement rather than measure it.
    """

    for row in VACUUM_FITTED_TURNS:
        if not row.published:
            continue
        published = float(row.published_turns)
        half = 0.5 * (row.interval.upper - row.interval.lower)
        assert half >= abs(row.fitted_turns - published) - 1.0e-9
        assert row.interval.contains(published)
        assert row.interval.contains(row.fitted_turns)


def test_one_coil_disagrees_and_the_record_says_so():
    """P5 lower rounds to twenty-four, and that is kept visible.

    Its experiments split by campaign rather than scatter, so the disagreement is
    a systematic no term in this description carries.  Recording it as agreement
    would hide the largest unexplained term this route has found.
    """

    row = fitted_turns("p5_lower")
    assert row.published_turns == 23
    assert not row.agrees_with_published
    assert round(row.fitted_turns) == 24
    assert row.turns == 23.0
    others = [
        item
        for item in VACUUM_FITTED_TURNS
        if item.published and item.family != "p5_lower"
    ]
    assert all(item.agrees_with_published for item in others)


def test_the_solenoid_turn_count_doubles_its_fitted_multiplier():
    """Two parallel circuits mean one turn carries half the feed current."""

    row = fitted_turns("sol")
    assert row.turns_per_multiplier == 2.0
    assert row.turns == pytest.approx(2.0 * row.multiplier)
    assert row.interval.contains(row.turns)


def test_every_identified_coil_has_positive_polarity():
    """The cohort found one polarity for every coil it could see."""

    assert all(row.multiplier > 0.0 for row in VACUUM_FITTED_TURNS if row.identified)


def test_authored_turns_omit_the_coils_no_shot_could_see():
    """An unmeasured coil gets no turn count rather than a plausible one."""

    authored = authored_turns()
    assert "p6_lower" not in authored
    assert "p6_upper" not in authored
    assert len(authored) == 11


def test_fitted_evidence_is_a_valid_ledger():
    """Every record the refinement contributes validates on its own terms."""

    ledger = EvidenceLedger.create(fitted_evidence())
    ledger.validate()
    counts = ledger.state_counts()
    assert counts["published"] == 10
    assert counts["fitted"] == 4
    assert counts["measured"] == 6
    assert counts["unresolved"] == 10


def test_refinement_narrows_the_forward_model_blockers(geometry):
    """Only the coils the cohort could not see still block the forward model."""

    seed = seed_evidence(geometry, first_shot=11695, last_shot=30473)
    assert "pf_active/coil/element/turns_with_sign" in seed.forward_model_blockers()
    refined = EvidenceLedger.create(refined_evidence(seed.records))
    assert refined.forward_model_blockers() == (
        "pf_active/coil(p6_lower)/element/turns_with_sign",
        "pf_active/coil(p6_upper)/element/turns_with_sign",
    )


def test_the_blanket_probe_angle_claim_is_narrowed(geometry):
    """The seed's measured angle for all probes does not survive beside the fit."""

    seed = seed_evidence(geometry, first_shot=11695, last_shot=30473)
    refined = EvidenceLedger.create(refined_evidence(seed.records))
    paths = {row.path for row in refined.records}
    assert "magnetics/b_field_pol_probe/poloidal_angle" not in paths
    assert f"magnetics/b_field_pol_probe({RADIAL_PROBE_FAMILY})/poloidal_angle" in paths
    assert (
        f"magnetics/b_field_pol_probe({AXIAL_PROBE_FAMILIES})/poloidal_angle" in paths
    )


def test_the_measured_saddle_geometry_survives_the_refinement(geometry):
    """Refining the traversal sign does not delete the measured path geometry."""

    seed = seed_evidence(geometry, first_shot=11695, last_shot=30473)
    refined = EvidenceLedger.create(refined_evidence(seed.records))
    path = "magnetics/flux_loop(saddle)/position"
    record = next(row for row in refined.records if row.path == path)
    assert record.evidence is FieldEvidence.MEASURED


def test_refined_ids_carry_the_fitted_turns_and_axis():
    """The authored IDSs hold what the cohort measured and nothing it did not."""

    registry = MachineGeometryRegistry.default()
    selection = registry.select(REPRESENTATIVE_SHOT)
    bundle = author_refined_ids(selection, first_shot=11695, last_shot=30473)
    turns = authored_turns()
    for coil in bundle.ids["pf_active"].coil:
        name = str(coil.name)
        for element in coil.element:
            if name in turns:
                assert float(element.turns_with_sign) == pytest.approx(turns[name])
            else:
                assert not element.turns_with_sign.has_value
    assert bundle.unset_turns == ("p6_lower", "p6_upper")

    angles = {
        str(probe.name).rsplit("_", 1)[0]: float(probe.poloidal_angle)
        for probe in bundle.ids["magnetics"].b_field_pol_probe
        if str(probe.name)
    }
    assert angles[RADIAL_PROBE_FAMILY] == pytest.approx(0.0)
    assert angles["ccbv"] == pytest.approx(-math.pi / 2.0, abs=1e-4)


def test_refined_artifact_keeps_the_machine_it_describes(tmp_path):
    """The refined revision changes its semantics, never its conductor identity."""

    artifact = publish_refined_artifact(tmp_path / "refined")
    manifest = artifact.manifest
    assert manifest.physical_digest == PHYSICAL_DIGEST
    assert manifest.registry_digest == REGISTRY_DIGEST
    assert manifest.dd_version == "4.1.1"
    assert manifest.semantic_identity() == REFINED_SEMANTIC_IDENTITY
    assert not manifest.complete


def test_the_refined_artifact_carries_its_own_drive_semantics(tmp_path):
    """The published artifact says what one ampere of every channel drives."""

    manifest = publish_refined_artifact(tmp_path / "driven").manifest
    drives = manifest.drive_map
    drives.validate()

    assert len(manifest.driven_columns()) == DRIVEN_COLUMNS
    assert drives.for_channel("sol_current").ampere_turns_per_ampere == pytest.approx(
        344.656565
    )
    assert drives.for_channel("p3u_coil_current").ampere_turns_per_ampere == 1.0
    assert drives.for_channel("p2l_case_current").container == "pf_passive"
    ledger_paths = {record.path for record in manifest.evidence.records}
    assert {drive.path for drive in drives.drives} <= ledger_paths


def test_the_refined_ids_carry_the_sourced_circuit_junctions():
    """A two-coil circuit's node matrix is authored; an open one stays empty."""

    registry = MachineGeometryRegistry.default()
    selection = registry.select(REPRESENTATIVE_SHOT)
    bundle = author_refined_ids(selection, first_shot=11695, last_shot=30473)
    pf_active = bundle.ids["pf_active"]
    coils = [str(coil.name) for coil in pf_active.coil]
    written = {}
    for circuit in pf_active.circuit:
        if circuit.connections.has_value:
            written[str(circuit.name)] = np.asarray(circuit.connections)
    assert sorted(written) == ["P3", "P4", "P5", "P6"]
    assert len(pf_active.supply) == 0
    for matrix in written.values():
        assert matrix.shape == (1, len(coils))
    junction = written["P3"][0]
    assert [coils[index] for index in np.flatnonzero(junction)] == [
        "p3_lower",
        "p3_upper",
    ]
    assert sorted(junction[np.flatnonzero(junction)].tolist()) == [-1, 1]
    opposed = written["P6"][0]
    assert sorted(opposed[np.flatnonzero(opposed)].tolist()) == [-1, -1]


def test_refined_artifact_is_reproducible(tmp_path):
    """Authoring twice gives one semantic address, whatever the stored bytes do."""

    first = publish_refined_artifact(tmp_path / "one").manifest
    second = publish_refined_artifact(tmp_path / "two").manifest
    assert first.semantic_identity() == second.semantic_identity()
