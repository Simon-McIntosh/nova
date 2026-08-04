"""What the calibration classes admit, and the amplitude screen that guards them.

The rules under test all exist to refuse something, so each one is given the exact
case it was written for: a ratio carrying the store's own float noise, a coil pair
driven in lockstep, a shot held for milliseconds rather than for two thirds of a
second, and a shot whose probes read half what its currents imply.  A rule that
cannot be shown refusing its own case is not a rule.

The waveforms are synthetic for the same reason the earlier refinement's are: the
answer is known in advance, so a screen that lets the wrong shot through is
visible as a wrong number rather than as a plausible one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_calibration_cohort import (
    RATIO_INTEGER_TOLERANCE,
    SYMMETRIC_PAIR_CONTRAST,
    CalibrationError,
    ExperimentClass,
    ampere_turn_ratio_support,
    calibration_experiments,
    class_counts,
    classify_experiment,
    identifiability_map,
    integer_ampere_turn_ratios,
    select_calibration_cohort,
)
from nova.imas.mast_fitted_parameters import RADIAL_PROBE_FAMILY
from nova.imas.mast_sensor_noise import (
    MINIMUM_NOISE_SAMPLES,
    NoiseError,
    measure_noise_envelope,
    measure_repeat_scatter,
    repeat_groups,
)
from nova.imas.mast_vacuum_cohort import (
    ERROR_FIELD_CHANNELS,
    EXCITATION_CURRENT,
    PLASMA_FREE_CURRENT,
    ShotSurvey,
    ShotWaveforms,
    probe_channels,
)
from nova.imas.mast_vacuum_response import (
    ADMISSIBLE_SCALE,
    ResponseError,
    ResponseModel,
    published_turn_scale,
)

REPRESENTATIVE_SHOT = 11766
"""Registry selection the calibration machinery is exercised against."""

FULL_PROBE_ARRAY = tuple(f"ccbv{index:02d}" for index in range(1, 41))
"""Enough recorded channels to over-determine a coil amplitude."""


@pytest.fixture(scope="module")
def geometry():
    """Return the registry configuration the calibration is authored against."""

    registry = MachineGeometryRegistry.default()
    return registry.select(REPRESENTATIVE_SHOT).configuration.geometry


@pytest.fixture(scope="module")
def model(geometry):
    """Return the response model under the measured sensitive-axis assignment."""

    probes = geometry["magnetics"]["poloidal_probes"]
    return ResponseModel.build(
        geometry,
        probes,
        probe_channels(probes),
        radial_families=frozenset({RADIAL_PROBE_FAMILY}),
    )


def survey(
    shot: int,
    *,
    hold: float = 1.0,
    plasma: float = 0.0,
    toroidal: float = 0.0,
    ratios: dict[str, float] | None = None,
    probes: tuple[str, ...] = FULL_PROBE_ARRAY,
    absent: tuple[str, ...] = (),
    error_field: dict[str, float] | None = None,
    **peaks: float,
) -> ShotSurvey:
    """Build a survey record whose excitation shape is stated outright."""

    return ShotSurvey(
        shot=shot,
        plasma_current_peak=plasma,
        toroidal_current_peak=toroidal,
        coil_peaks=dict(peaks),
        coil_hold_times={name: hold for name in peaks},
        turn_multipliers=dict(ratios or {}),
        absent_groups=absent,
        absent_channels=(),
        field_channels=probes,
        error_field_peaks=dict(error_field or {}),
    )


def synthetic_shot(
    model: ResponseModel,
    excitation: dict[str, tuple[float, float]],
    *,
    shot: int = 910001,
    samples: int = 400,
    amplitude: float = 1.0,
    offset: float = 2.0e-4,
    noise: float = 0.0,
    seed: int = 0,
) -> ShotWaveforms:
    """Build a shot whose probes read ``amplitude`` times what its turns imply.

    ``amplitude`` is what the screen exists to catch.  Setting it to one half
    reproduces the store's matched pairs -- identical currents, every probe reading
    a factor of two low, the field's shape across the array untouched -- without
    depending on those shots being present.
    """

    generator = np.random.default_rng(seed)
    time = np.linspace(-0.2, 1.0, samples)
    ramp = np.clip(time / 0.05, 0.0, 1.0) * np.clip((0.8 - time) / 0.05, 0.0, 1.0)
    drives = {family: np.zeros_like(time) for family in model.families}
    for family, (current, _) in excitation.items():
        drives[family] = current * ramp

    probes: dict[str, np.ndarray] = {}
    for row, target in enumerate(model.targets):
        signal = np.full_like(time, offset * (1 + row % 5))
        for column, family in enumerate(model.families):
            turns = excitation.get(family, (0.0, 0.0))[1]
            signal = signal + amplitude * (
                model.response[row, column] * turns * drives[family]
            )
        if noise > 0.0:
            signal = signal + generator.normal(0.0, noise, size=time.shape)
        probes[target.channel] = signal

    return ShotWaveforms(
        shot=shot,
        time=time,
        drives=drives,
        probes=probes,
        plasma_current=np.zeros_like(time),
        sample_mask=np.ones(time.shape, dtype=bool),
        baseline_mask=time < -0.05,
    )


# --- the archive's own turn counts -------------------------------------


def test_float_noise_does_not_split_a_published_ratio():
    """A ratio the store spells 23.000001 is still the integer twenty-three.

    The store derives each ampere-turn channel by multiplying a conductor current,
    so dividing them back returns the integer plus rounding noise of about one part
    in ten million.  Grouping the results by decimal spelling reports the family as
    varying and discards the archive's statement of the count entirely, which is
    what left four of the ten counts unrecognised.
    """

    surveys = [
        survey(1, ratios={"p4_lower": 23.0}, p4_lower=1.0e4),
        survey(2, ratios={"p4_lower": 23.000001}, p4_lower=1.0e4),
        survey(3, ratios={"p4_lower": 22.999999}, p4_lower=1.0e4),
    ]
    assert integer_ampere_turn_ratios(surveys) == {"p4_lower": 23}
    assert ampere_turn_ratio_support(surveys) == {"p4_lower": 3}


def test_a_genuinely_varying_ratio_is_not_promoted():
    """Two different integers across the store are not averaged into one."""

    surveys = [
        survey(1, ratios={"p4_lower": 23.0}, p4_lower=1.0e4),
        survey(2, ratios={"p4_lower": 12.0}, p4_lower=1.0e4),
    ]
    assert integer_ampere_turn_ratios(surveys) == {}


def test_the_tolerance_sits_far_below_a_neighbouring_integer():
    """The tolerance cannot admit a ratio that is closer to another count."""

    assert RATIO_INTEGER_TOLERANCE < 0.5
    drifted = [survey(1, ratios={"p3_lower": 8.4}, p3_lower=1.0e4)]
    assert integer_ampere_turn_ratios(drifted) == {}


# --- experiment classes -----------------------------------------------


def test_a_sustained_lone_coil_measures_its_own_turns():
    """One coil held above the excitation floor identifies itself."""

    row = classify_experiment(survey(11, p4_lower=1.5e4))
    assert row.experiment is ExperimentClass.SUSTAINED_SINGLE_COIL
    assert row.identifies == ("p4_lower",)
    assert row.identifies_sum == ()
    assert row.measures_turns


def test_a_lockstep_pair_measures_only_its_sum():
    """Two coils carrying the same current fix the total and not the split."""

    row = classify_experiment(survey(12, p4_lower=1.5e4, p4_upper=1.5e4))
    assert row.experiment is ExperimentClass.SUSTAINED_SYMMETRIC_PAIR
    assert row.identifies == ()
    assert row.identifies_sum == ("p4_lower", "p4_upper")
    assert not row.measures_turns


def test_a_pair_with_contrast_separates_its_members():
    """A pair driven unevenly enough is identifiable member by member."""

    uneven = classify_experiment(
        survey(13, p4_lower=1.5e4, p4_upper=0.5 * SYMMETRIC_PAIR_CONTRAST * 1.5e4)
    )
    assert uneven.identifies == ("p4_lower", "p4_upper")
    assert uneven.identifies_sum == ()


def test_a_pulsed_shot_cannot_measure_a_turn_count():
    """A few-millisecond pulse reads the coil together with its induced currents."""

    row = classify_experiment(survey(14, hold=6.0e-3, p4_lower=1.5e4))
    assert row.experiment is ExperimentClass.PULSED_EXCITATION
    assert row.identifies == ()
    assert not row.measures_turns


def test_a_shot_below_the_excitation_floor_is_quiescent():
    """Standing current in the vertical coils is not an experiment."""

    row = classify_experiment(survey(15, p6_lower=300.0, p6_upper=300.0))
    assert row.experiment is ExperimentClass.QUIESCENT
    assert row.measures_noise


def test_the_toroidal_field_alone_measures_the_sensors():
    """A shot holding only the toroidal field carries no poloidal excitation."""

    row = classify_experiment(survey(16, toroidal=1.0e5))
    assert row.experiment is ExperimentClass.TOROIDAL_FIELD_ONLY
    assert row.measures_noise
    assert row.identifies == ()


def test_a_plasma_shot_is_not_a_vacuum_experiment():
    """A plasma disqualifies a shot whatever its coils were doing."""

    row = classify_experiment(
        survey(17, plasma=2.0 * PLASMA_FREE_CURRENT, p4_lower=1.5e4)
    )
    assert row.experiment is ExperimentClass.PLASMA
    assert not row.measures_turns
    assert not row.measures_noise


def test_a_missing_store_group_is_unreadable_rather_than_empty():
    """An absent group is recorded as unreadable, never as a zero reading."""

    row = classify_experiment(survey(18, absent=("amb",), p4_lower=1.5e4))
    assert row.experiment is ExperimentClass.UNREADABLE
    thin = classify_experiment(survey(19, probes=FULL_PROBE_ARRAY[:5], p4_lower=1.5e4))
    assert thin.experiment is ExperimentClass.UNREADABLE


def test_class_counts_cover_every_classed_shot():
    """Every shot lands in exactly one class."""

    surveys = [
        survey(21, p4_lower=1.5e4),
        survey(22, p4_lower=1.5e4, p4_upper=1.5e4),
        survey(23, hold=1.0e-3, p3_lower=1.5e4),
        survey(24, toroidal=1.0e5),
        survey(25),
    ]
    experiments = calibration_experiments(surveys)
    assert sum(class_counts(experiments).values()) == len(surveys)
    assert [row.shot for row in experiments] == [21, 22, 23, 24, 25]


# --- identifiability --------------------------------------------------


def test_identifiability_separates_alone_from_sum_only_from_unreachable():
    """The map says, per coil, on what terms the archive can reach it."""

    surveys = [
        survey(31, p4_lower=1.5e4),
        survey(32, p5_lower=1.5e4, p5_upper=1.5e4),
    ]
    rows = {
        row.family: row for row in identifiability_map(calibration_experiments(surveys))
    }
    assert rows["p4_lower"].identifiable
    assert not rows["p4_lower"].sum_only
    assert rows["p5_lower"].sum_only
    assert not rows["p5_lower"].identifiable
    assert not rows["p3_lower"].identifiable
    assert not rows["p3_lower"].sum_only
    assert rows["p4_lower"].strongest == pytest.approx(1.5e4)


# --- the declared split -----------------------------------------------


def test_the_split_holds_out_shots_from_every_measurable_coil():
    """A coil with several experiments is challenged as well as trained."""

    surveys = [survey(40 + index, p4_lower=1.5e4) for index in range(8)]
    surveys += [survey(60 + index, p3_lower=1.5e4) for index in range(4)]
    cohort = select_calibration_cohort(surveys, held_out_fraction=0.25)
    assert set(cohort.training) & set(cohort.held_out) == set()
    assert len(cohort.held_out) == 3
    held = set(cohort.held_out)
    assert held & {40 + index for index in range(8)}
    assert held & {60 + index for index in range(4)}


def test_a_cohort_that_holds_nothing_back_is_refused():
    """One shot for one coil cannot be both trained on and challenged with."""

    with pytest.raises(CalibrationError):
        select_calibration_cohort([survey(70, p4_lower=1.5e4)])


def test_a_noise_shot_never_also_fits_a_turn_count():
    """The arms and the noise set are disjoint by construction."""

    surveys = [survey(80 + index, p4_lower=1.5e4) for index in range(4)]
    surveys += [survey(90 + index, toroidal=1.0e5) for index in range(3)]
    cohort = select_calibration_cohort(surveys)
    assert set(cohort.noise_shots) == {90, 91, 92}
    assert not set(cohort.noise_shots) & set(cohort.shots)
    cohort.validate()


def test_the_cohort_record_states_its_classes_and_published_ratios():
    """The record carries what a later stage must not be free to re-choose."""

    surveys = [
        survey(100 + index, ratios={"p4_lower": 23.0}, p4_lower=1.5e4)
        for index in range(4)
    ]
    surveys += [survey(110, toroidal=1.0e5)]
    payload = select_calibration_cohort(surveys).as_dict()
    assert payload["published_ampere_turn_ratios"] == {"p4_lower": 23}
    assert payload["class_sizes"][str(ExperimentClass.SUSTAINED_SINGLE_COIL)] == 4
    assert payload["noise_shots"] == [110]


# --- the excitation the census had not recorded ------------------------


def test_a_shot_with_no_error_field_reading_never_claims_one():
    """An unrecorded non-axisymmetric channel means unmeasured, not undriven."""

    assert not survey(120, p4_lower=1.5e4).error_field_driven
    quiet = survey(121, error_field={name: 20.0 for name in ERROR_FIELD_CHANNELS})
    assert not quiet.error_field_driven
    driven = survey(
        122, error_field={ERROR_FIELD_CHANNELS[0]: 2.0 * EXCITATION_CURRENT}
    )
    assert driven.error_field_driven


# --- the amplitude screen ---------------------------------------------


def test_a_shot_matching_its_published_turns_reads_unit_amplitude(model):
    """A shot built from the published counts measures an amplitude of one."""

    waveforms = synthetic_shot(model, {"p4_lower": (1.5e4, 23.0)})
    measured = published_turn_scale(waveforms, model, {"p4_lower": 23.0})
    assert measured.scale == pytest.approx(1.0, rel=1.0e-6)
    assert measured.admissible
    assert measured.probe_spread == pytest.approx(0.0, abs=1.0e-6)
    assert measured.variance_explained == pytest.approx(1.0, abs=1.0e-9)


def test_a_half_scale_shot_is_refused_rather_than_fitted(model):
    """Probes reading half the implied field cannot measure a turn count.

    This is the store's matched-pair case.  Left in a pool of correctly scaled
    shots it does not look wrong -- the solve reports a small error and high
    leverage on a count near half the true one -- so it has to be refused on the
    amplitude and not on the fit's own confidence.
    """

    waveforms = synthetic_shot(model, {"p4_lower": (1.5e4, 23.0)}, amplitude=0.5)
    measured = published_turn_scale(waveforms, model, {"p4_lower": 23.0})
    assert measured.scale == pytest.approx(0.5, rel=1.0e-6)
    assert not measured.admissible
    assert measured.probe_spread == pytest.approx(0.0, abs=1.0e-6)


def test_a_uniform_amplitude_leaves_the_field_shape_untouched(model):
    """The screen separates an amplitude from a geometry error by the spread.

    An acquisition scale moves every probe by the same factor, so the ratio taken
    probe by probe has no scatter; a wrong conductor position moves the probes
    differently and shows up there instead.  Reporting both is what stops the two
    being confused.
    """

    scaled = published_turn_scale(
        synthetic_shot(model, {"p4_lower": (1.5e4, 23.0)}, amplitude=0.5),
        model,
        {"p4_lower": 23.0},
    )
    mixed = published_turn_scale(
        synthetic_shot(model, {"p4_lower": (1.5e4, 11.5)}, noise=2.0e-3, seed=3),
        model,
        {"p4_lower": 23.0},
    )
    assert scaled.probe_spread < mixed.probe_spread


def test_the_admissible_interval_admits_neither_a_half_nor_a_double():
    """The screen's interval is set well clear of the anomaly it refuses."""

    lower, upper = ADMISSIBLE_SCALE
    assert lower > 0.5
    assert upper < 2.0
    assert lower < 1.0 < upper


def test_a_shot_driving_no_published_coil_is_refused_not_scored(model):
    """The screen answers for the published coils or it declines to answer."""

    waveforms = synthetic_shot(model, {"sol": (1.5e4, 345.0)})
    with pytest.raises(ResponseError):
        published_turn_scale(waveforms, model, {"p4_lower": 23.0})


# --- the sensor floor -------------------------------------------------


def quiet_shot(
    model: ResponseModel,
    *,
    shot: int,
    scatter: float,
    drift: float,
    samples: int = 2000,
    seed: int = 0,
) -> ShotWaveforms:
    """Build a shot with no excitation, a known scatter and a known drift."""

    generator = np.random.default_rng(seed)
    time = np.linspace(-2.0, 2.0, samples)
    probes = {
        target.channel: drift * time + generator.normal(0.0, scatter, size=time.shape)
        for target in model.targets
    }
    return ShotWaveforms(
        shot=shot,
        time=time,
        drives={family: np.zeros_like(time) for family in model.families},
        probes=probes,
        plasma_current=np.zeros_like(time),
        sample_mask=np.ones(time.shape, dtype=bool),
        baseline_mask=time < -1.0,
    )


def test_the_floor_is_the_scatter_and_not_the_drift(model):
    """A wandering integrator zero is reported separately, not as noise.

    A drift of one millitesla per second over a four-second record swamps a
    hundred-microtesla scatter, so reporting the raw standard deviation would put
    the floor an order too high and make every model look noise-limited.
    """

    envelope = measure_noise_envelope(
        [quiet_shot(model, shot=1, scatter=1.0e-4, drift=1.0e-3)]
    )
    row = envelope.channel(model.targets[0].channel)
    assert row.scatter == pytest.approx(1.0e-4, rel=0.1)
    assert row.drift_rate == pytest.approx(1.0e-3, rel=0.1)
    assert envelope.pooled_scatter == pytest.approx(1.0e-4, rel=0.1)


def test_the_pooled_floor_is_a_quadratic_mean(model):
    """The pooled floor is the same kind of average as a pooled residual."""

    envelope = measure_noise_envelope(
        [quiet_shot(model, shot=2, scatter=2.0e-4, drift=0.0)]
    )
    values = envelope.scatter
    assert envelope.pooled_scatter == pytest.approx(float(np.sqrt(np.mean(values**2))))
    assert envelope.pooled_scatter >= float(np.mean(values))


def test_a_family_floor_pools_only_its_own_channels(model):
    """Each probe family reports its own floor."""

    envelope = measure_noise_envelope(
        [quiet_shot(model, shot=3, scatter=1.5e-4, drift=0.0)]
    )
    families = envelope.family_scatter()
    assert set(families) == {"ccbv", "obr", "obv"}
    for value in families.values():
        assert value == pytest.approx(1.5e-4, rel=0.15)


def test_a_channel_with_too_few_samples_reports_no_floor(model):
    """A dropped-out channel is left out rather than given a meaningless number."""

    waveforms = quiet_shot(
        model, shot=4, scatter=1.0e-4, drift=0.0, samples=MINIMUM_NOISE_SAMPLES // 2
    )
    with pytest.raises(NoiseError):
        measure_noise_envelope([waveforms])


def test_an_empty_envelope_is_refused(model):
    """An envelope with no channel in it is not a measurement."""

    with pytest.raises(NoiseError):
        measure_noise_envelope([])


# --- repeat experiments -----------------------------------------------


def test_repeats_group_by_coil_and_current(model):
    """Two shots holding one coil at one current are the archive's repetition."""

    surveys = [
        survey(200, p4_lower=1.40e4),
        survey(201, p4_lower=1.41e4),
        survey(202, p4_lower=7.0e3),
    ]
    groups = repeat_groups(calibration_experiments(surveys))
    assert len(groups) == 1
    family, shots, peak = groups[0]
    assert family == "p4_lower"
    assert shots == (200, 201)
    assert peak == pytest.approx(1.405e4, rel=1.0e-3)


def test_two_shots_a_campaign_apart_are_not_one_repetition():
    """A repeat is a back-to-back re-firing, not two similar shots years apart.

    Pooling distant shots reports the archive's whole variability as shot-to-shot
    scatter: on this store it collects twenty-five solenoid shots spanning fifteen
    thousand shot numbers and claims seventy-four percent disagreement, where the
    genuine back-to-back pairs agree to a fraction of a percent.
    """

    surveys = [survey(300, p4_lower=1.4e4), survey(9300, p4_lower=1.4e4)]
    assert repeat_groups(calibration_experiments(surveys)) == ()
    close = [survey(300, p4_lower=1.4e4), survey(305, p4_lower=1.4e4)]
    assert len(repeat_groups(calibration_experiments(close))) == 1


def test_an_amplitude_refused_shot_never_enters_a_repeat():
    """A shot recorded at half amplitude disagrees with its twin by that half."""

    surveys = [survey(310 + index, p4_lower=1.4e4) for index in range(3)]
    experiments = calibration_experiments(surveys)
    assert repeat_groups(experiments)[0][1] == (310, 311, 312)
    screened = repeat_groups(experiments, exclude={312})
    assert screened[0][1] == (310, 311)


def test_repeat_scatter_divides_out_the_supply_and_keeps_the_rest(model):
    """A shot delivering less current is not counted as sensor disagreement."""

    first = synthetic_shot(model, {"p4_lower": (1.4e4, 23.0)}, shot=210)
    second = synthetic_shot(model, {"p4_lower": (1.2e4, 23.0)}, shot=211)
    measured = measure_repeat_scatter("p4_lower", (210, 211), {210: first, 211: second})
    assert measured.relative_scatter == pytest.approx(0.0, abs=1.0e-6)
    assert measured.shots == (210, 211)

    louder = synthetic_shot(model, {"p4_lower": (1.4e4, 23.0)}, shot=212, amplitude=1.1)
    disagreeing = measure_repeat_scatter(
        "p4_lower", (210, 212), {210: first, 212: louder}
    )
    assert disagreeing.relative_scatter > 0.04
    assert math.isfinite(disagreeing.absolute_scatter)


def test_repeat_scatter_needs_two_readable_shots(model):
    """One shot cannot disagree with itself."""

    first = synthetic_shot(model, {"p4_lower": (1.4e4, 23.0)}, shot=220)
    with pytest.raises(NoiseError):
        measure_repeat_scatter("p4_lower", (220, 221), {220: first})
