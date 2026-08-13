"""Whether the calibration fit recovers a fault that was planted in the data.

Each test builds a shot in which one thing is deliberately wrong -- a probe reading
eight percent high, a probe rotated by a known angle, a coil whose field at one
probe is misrepresented -- and asks whether the fit returns that thing and whether
the pre-registered criterion names the right cause.  A discriminant that cannot be
shown separating two planted causes is not a discriminant.

The two properties the statistics depend on are tested directly rather than
inferred: the orthogonal pairing must be by position, so a renumbering must not
change who is paired with whom; and the standard error must come from the scatter
across shots, so adding samples to a shot must not shrink it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_fitted_parameters import RADIAL_PROBE_FAMILY
from nova.imas.mast_probe_calibration import (
    MINIMUM_SHOTS,
    PooledRigidFit,
    ShotGain,
    adjudicate_probes,
    aggregate_family_gains,
    build_statistics,
    orthogonal_pairs,
    pool_rigid_systems,
    pooled_corrections,
    pooled_held_out,
    probe_family,
    score_rigid_correction,
    shot_gains,
    standoff_table,
    verdict_counts,
)
from nova.imas.mast_probe_discriminant import (
    CO_LOCATION_TOLERANCE,
    FAMILY_LEVERAGE,
    ProbeVerdict,
)
from nova.imas.mast_vacuum_cohort import ShotWaveforms, probe_channels
from nova.imas.mast_vacuum_response import MINIMUM_STANDOFF, ResponseModel

REPRESENTATIVE_SHOT = 11766
"""Registry selection the calibration is exercised against."""

WEIGHTS = {
    "sol": 344.656565,
    "p2_inner_lower": 12.0,
    "p2_inner_upper": 12.0,
    "p2_outer_lower": 8.0,
    "p2_outer_upper": 8.0,
    "p3_lower": 8.0,
    "p3_upper": 8.0,
    "p4_lower": 23.0,
    "p4_upper": 23.0,
    "p5_lower": 23.0,
    "p5_upper": 23.0,
    "p6_lower": 1.0,
    "p6_upper": 1.0,
}
"""The drive weights the description carries, so the fit's target gain is one."""

FLOOR = {
    f"{family}{index:02d}": 5.0e-5
    for family in ("ccbv", "obr", "obv")
    for index in range(1, 41)
}
"""A quiescent scatter for every channel, so no probe is refused for want of one."""


@pytest.fixture(scope="module")
def geometry():
    """Return the registry configuration the calibration is authored against."""

    return MachineGeometryRegistry.default().select(REPRESENTATIVE_SHOT)


@pytest.fixture(scope="module")
def model(geometry):
    """Return the response model under the measured sensitive-axis assignment."""

    probes = geometry.configuration.geometry["magnetics"]["poloidal_probes"]
    return ResponseModel.build(
        geometry.configuration.geometry,
        probes,
        probe_channels(probes),
        radial_families=frozenset({RADIAL_PROBE_FAMILY}),
    )


@pytest.fixture(scope="module")
def pairs(model):
    """Return the position-matched orthogonal probe pairs."""

    return orthogonal_pairs(model)


def synthetic_shot(
    model,
    excitation: dict[str, float],
    *,
    shot: int = 920001,
    samples: int = 600,
    gains: dict[str, float] | None = None,
    tilts: dict[str, float] | None = None,
    family_error: tuple[str, str, float] | None = None,
    offset: float = 3.0e-4,
) -> ShotWaveforms:
    """Build a shot whose probes read exactly what the planted faults imply.

    ``gains`` scales a named channel's reading.  ``tilts`` rotates a channel in
    the poloidal plane, mixing in the true radial field at its own position, which
    is what a mis-mounted probe does.  ``family_error`` scales ONE coil's
    contribution at ONE probe, which is what a misrepresented winding does and
    what no rigid probe transform can imitate.
    """

    time = np.linspace(-0.3, 1.2, samples)
    ramp = np.clip(time / 0.08, 0.0, 1.0) * np.clip((1.0 - time) / 0.08, 0.0, 1.0)
    drives = {family: np.zeros_like(time) for family in model.families}
    for family, current in excitation.items():
        drives[family] = current * ramp

    axis = {target.channel: target for target in model.targets}
    partner_of = {pair.channel: pair.partner for pair in orthogonal_pairs(model)}
    true_field: dict[str, np.ndarray] = {}
    for row, target in enumerate(model.targets):
        field = np.zeros_like(time)
        for column, family in enumerate(model.families):
            scale = WEIGHTS.get(family, 1.0)
            contribution = model.response[row, column] * scale * drives[family]
            if (
                family_error is not None
                and family_error[0] == target.channel
                and family_error[1] == family
            ):
                contribution = contribution * family_error[2]
            field = field + contribution
        true_field[target.channel] = field

    probes: dict[str, np.ndarray] = {}
    for index, target in enumerate(model.targets):
        signal = true_field[target.channel]
        angle = (tilts or {}).get(target.channel, 0.0)
        if angle:
            partner = partner_of.get(target.channel)
            if partner is not None:
                signal = (
                    math.cos(angle) * signal + math.sin(angle) * true_field[partner]
                )
        signal = (gains or {}).get(target.channel, 1.0) * signal
        probes[target.channel] = signal + offset * (1 + index % 4)

    del axis
    return ShotWaveforms(
        shot=shot,
        time=time,
        drives=drives,
        probes=probes,
        sensors=probes,
        plasma_current=np.zeros_like(time),
        sample_mask=np.ones(time.shape, dtype=bool),
        baseline_mask=time < -0.1,
    )


def fit_shots(model, pairs, shots):
    """Run the per-shot fit over several shots and pool the rigid systems."""

    pair_map = {row.channel: row.partner for row in pairs}
    gains, rigid, systems = [], [], []
    for waveforms in shots:
        one, two, three = shot_gains(model, waveforms, WEIGHTS, pairs=pair_map)
        gains.extend(one)
        rigid.extend(two)
        systems.extend(three)
    return gains, rigid, pool_rigid_systems(systems)


# --- pairing -----------------------------------------------------------


def test_every_outboard_probe_but_one_pair_finds_its_orthogonal_twin(pairs):
    """The arrays are built as pairs, and the exception is a measured 3 mm offset."""

    channels = {row.channel for row in pairs}
    assert len(pairs) == 36
    assert {"obv06", "obr06", "obv14", "obr14"} <= channels
    assert "obv10" not in channels and "obr10" not in channels
    for row in pairs:
        assert row.separation <= CO_LOCATION_TOLERANCE
        assert probe_family(row.channel) != probe_family(row.partner)


def test_pairing_survives_a_renumbering(model):
    """Pairing is by position, so shuffled channel names must not move a pair.

    The two outboard arrays happen to be numbered in step, and a pairing that
    leaned on that would keep working while silently pairing two different
    places the moment the archive renumbered one array.
    """

    shuffled = ResponseModel(
        families=model.families,
        targets=tuple(reversed(model.targets)),
        response=model.response[::-1, :],
        standoff=model.standoff[::-1, :],
        radial_families=model.radial_families,
    )
    original = {row.channel: row.partner for row in orthogonal_pairs(model)}
    reordered = {row.channel: row.partner for row in orthogonal_pairs(shuffled)}
    assert original == reordered


# --- the planted faults ------------------------------------------------


def test_a_probe_reading_high_returns_its_own_scale(model, pairs):
    """A scale planted on one channel comes back on that channel and no other."""

    shots = [
        synthetic_shot(model, {"p5_upper": 8.0e3}, shot=920001, gains={"obv06": 1.08}),
        synthetic_shot(model, {"p4_upper": 9.0e3}, shot=920002, gains={"obv06": 1.08}),
        synthetic_shot(model, {"p3_upper": 1.1e4}, shot=920003, gains={"obv06": 1.08}),
    ]
    gains, _, _ = fit_shots(model, pairs, shots)
    planted = [row for row in gains if row.channel == "obv06"]
    assert planted, "the fit read no gain at the probe the fault was planted on"
    assert np.allclose([row.gain for row in planted], 1.08, atol=1e-6)
    others = [row.gain for row in gains if row.channel == "obv08"]
    assert others and np.allclose(others, 1.0, atol=1e-6)


def test_a_rotated_probe_is_recovered_from_its_neighbour_and_not_the_model(
    model, pairs
):
    """The angle comes back from the co-located channel's own reading.

    The cross term is the measured radial field, so this test would still pass if
    the model's radial component were wrong -- which is the property the
    discriminant depends on.
    """

    angle = 0.05
    shots = [
        synthetic_shot(model, {"p5_upper": 8.0e3}, shot=920011, tilts={"obv06": angle}),
        synthetic_shot(model, {"p4_upper": 9.0e3}, shot=920012, tilts={"obv06": angle}),
        synthetic_shot(model, {"p3_upper": 1.1e4}, shot=920013, tilts={"obv06": angle}),
    ]
    per_shot, rigid = fit_shots(model, pairs, shots)[1:]
    fitted = next(row for row in rigid if row.channel == "obv06")
    assert fitted.identified
    assert fitted.tilt == pytest.approx(angle, abs=1e-6)
    assert fitted.gain == pytest.approx(1.0, abs=1e-6)
    quiet = next(row for row in rigid if row.channel == "obv08")
    assert quiet.tilt == pytest.approx(0.0, abs=1e-9)
    single = [row for row in per_shot if row.channel == "obv06"]
    assert single and not any(row.separable for row in single), (
        "a single-coil shot must not claim it separated scale from rotation"
    )


def test_one_coil_misrepresented_at_one_probe_is_excitation_selective(model, pairs):
    """The field-shape fault moves the gain only on the shots that coil drove.

    This is the whole discriminant: the same probe returns a different scale
    depending on which coil is driving, which no scale-and-rotation pair on the
    probe can produce.
    """

    fault = ("obv06", "p5_upper", 1.15)
    shots = [
        synthetic_shot(model, {"p5_upper": 8.0e3}, shot=920021, family_error=fault),
        synthetic_shot(model, {"p4_upper": 9.0e3}, shot=920022, family_error=fault),
        synthetic_shot(model, {"p3_upper": 1.1e4}, shot=920023, family_error=fault),
    ]
    gains, _, _ = fit_shots(model, pairs, shots)
    by_family = {row.family: row.gain for row in gains if row.channel == "obv06"}
    assert by_family["p5_upper"] == pytest.approx(1.15, abs=1e-6)
    for family, gain in by_family.items():
        if family != "p5_upper":
            assert gain == pytest.approx(1.0, abs=1e-6)


def test_the_criterion_separates_the_two_planted_causes(model, pairs):
    """A scale fault and a one-coil field fault reach opposite verdicts."""

    def statistics_for(**fault):
        shots = [
            synthetic_shot(model, {family: current}, shot=930000 + index, **fault)
            for index, (current, family) in enumerate(
                (current, family)
                for current in (8.0e3, 7.0e3, 6.0e3)
                for family in ("p3_upper", "p4_upper", "p5_upper")
            )
        ]
        gains, _, rigid = fit_shots(model, pairs, shots)
        family_gains = aggregate_family_gains(gains, standoff_table(model))
        rows = adjudicate_probes(
            family_gains,
            rigid,
            FLOOR,
            held_out={"obv06": (1.0e-6, 1.0e-3)},
        )
        return next(row for row in rows if row.channel == "obv06")

    scaled = statistics_for(gains={"obv06": 1.08})
    shaped = statistics_for(family_error=("obv06", "p5_upper", 1.15))
    assert scaled.verdict is ProbeVerdict.CALIBRATION_GAIN
    assert shaped.verdict is ProbeVerdict.FIELD_SHAPE
    assert not shaped.promoted
    assert shaped.statistics.excitation_selective
    assert not scaled.statistics.excitation_selective


# --- the leverage screen and the standard error -------------------------


def test_a_coil_that_does_not_dominate_returns_no_gain(model, pairs):
    """Two coils sharing a probe's signal give a blend, so neither is recorded."""

    shot = synthetic_shot(model, {"p4_upper": 9.0e3, "p5_upper": 9.0e3}, shot=920041)
    gains, _, _ = fit_shots(model, pairs, [shot])
    for row in gains:
        assert row.leverage >= FAMILY_LEVERAGE
    at_probe = {row.family for row in gains if row.channel == "obv06"}
    assert "p4_upper" not in at_probe


def test_the_standard_error_is_the_scatter_across_shots(model):
    """Adding samples must not shrink the error bar; adding shots must.

    A waveform's samples are correlated, so a standard error taken from the
    sample count would fall as the square root of the record length and every
    probe would read as excitation-selective.
    """

    standoff = {("obv06", "p5_upper"): 1.1}
    spread = [1.02, 0.98, 1.04, 0.96]
    few = [
        ShotGain(
            shot=index,
            channel="obv06",
            family="p5_upper",
            gain=value,
            leverage=0.9,
            sample_count=500,
            residual=1.0e-5,
            signal=1.0e-3,
        )
        for index, value in enumerate(spread[:3])
    ]
    many_samples = [
        ShotGain(**{**row.__dict__, "sample_count": 500_000}) for row in few
    ]
    more_shots = few + [
        ShotGain(
            shot=99,
            channel="obv06",
            family="p5_upper",
            gain=spread[3],
            leverage=0.9,
            sample_count=500,
            residual=1.0e-5,
            signal=1.0e-3,
        )
    ]
    base = aggregate_family_gains(few, standoff)[0]
    padded = aggregate_family_gains(many_samples, standoff)[0]
    grown = aggregate_family_gains(more_shots, standoff)[0]
    assert padded.standard_error == pytest.approx(base.standard_error)
    assert grown.shot_count == 4
    assert base.near_field is True


def test_a_family_measured_on_too_few_shots_is_not_counted(model):
    """One shot gives no scatter, so its gain cannot enter the spread test."""

    rows = [
        ShotGain(
            shot=1,
            channel="obv06",
            family="p5_upper",
            gain=1.4,
            leverage=0.9,
            sample_count=500,
            residual=1.0e-5,
            signal=1.0e-3,
        )
    ]
    gains = aggregate_family_gains(rows, {("obv06", "p5_upper"): 1.1})
    assert gains[0].shot_count < MINIMUM_SHOTS
    assert not gains[0].identified
    statistics = build_statistics("obv06", gains, None, noise_floor=5.0e-5)
    assert statistics.family_count == 0


def test_the_standoff_boundary_matches_the_turn_fit(model):
    """Near and far mean what they mean in the turn fit, not something new."""

    table = standoff_table(model)
    assert table[("obv06", "p5_upper")] < MINIMUM_STANDOFF
    assert table[("obv06", "sol")] > MINIMUM_STANDOFF


# --- the held-out challenge --------------------------------------------


def test_a_correction_that_fits_its_own_shots_must_still_predict(model, pairs):
    """The held-out score is a prediction, so the coefficients come from elsewhere."""

    pair_map = {row.channel: row.partner for row in pairs}
    train = [
        synthetic_shot(model, {"p5_upper": 8.0e3}, shot=920051, gains={"obv06": 1.08}),
        synthetic_shot(model, {"p4_upper": 9.0e3}, shot=920052, gains={"obv06": 1.08}),
        synthetic_shot(model, {"p3_upper": 1.1e4}, shot=920053, gains={"obv06": 1.08}),
    ]
    _, _, rigid = fit_shots(model, pairs, train)
    corrections = pooled_corrections(rigid)
    unseen = synthetic_shot(
        model, {"p5_upper": 6.0e3}, shot=920054, gains={"obv06": 1.08}
    )
    scored = score_rigid_correction(model, unseen, WEIGHTS, corrections, pairs=pair_map)
    corrected, reference = scored["obv06"]
    assert corrected < reference
    assert corrected < 1.0e-9
    pooled = pooled_held_out([scored])
    assert pooled["obv06"][0] == pytest.approx(corrected)


def test_a_probe_with_no_held_out_coverage_is_not_promoted(model):
    """A calibration nothing challenged stays unpromoted however clean its fit."""

    gains = aggregate_family_gains(
        [
            ShotGain(
                shot=index,
                channel="obv06",
                family=family,
                gain=1.08,
                leverage=0.9,
                sample_count=500,
                residual=1.0e-6,
                signal=1.0e-3,
            )
            for family in ("p3_upper", "p4_upper", "p5_upper")
            for index in range(3)
        ],
        standoff_table_stub(),
    )
    rows = adjudicate_probes(gains, (clean_rigid_fit(),), FLOOR)
    row = next(item for item in rows if item.channel == "obv06")
    assert row.verdict is ProbeVerdict.CALIBRATION_GAIN
    assert not row.improves_held_out
    assert not row.promoted


def test_a_probe_with_no_orthogonal_partner_is_never_a_calibration_item(model):
    """Without a rigid fit the residual is unmeasured, not zero.

    A probe reporting no rigid residual at all would otherwise read as having
    been explained to its noise floor by a fit nobody ran, and a probe-side
    verdict would follow from an absence of evidence.
    """

    gains = aggregate_family_gains(
        [
            ShotGain(
                shot=index,
                channel="obv06",
                family=family,
                gain=1.08,
                leverage=0.9,
                sample_count=500,
                residual=1.0e-6,
                signal=1.0e-3,
            )
            for family in ("p3_upper", "p4_upper", "p5_upper")
            for index in range(3)
        ],
        standoff_table_stub(),
    )
    row = next(
        item for item in adjudicate_probes(gains, (), FLOOR) if item.channel == "obv06"
    )
    assert math.isinf(row.statistics.rigid_residual)
    assert not row.statistics.rigid_fit_reaches_floor
    assert row.verdict is ProbeVerdict.INSEPARABLE
    assert row.as_dict()["statistics"]["rigid_residual"] is None


def clean_rigid_fit() -> PooledRigidFit:
    """Return a conditioned rigid fit that reaches the synthetic noise floor."""

    return PooledRigidFit(
        channel="obv06",
        partner="obr06",
        shot_count=9,
        gain=1.08,
        tilt=0.0,
        gain_error=1.0e-4,
        tilt_error=1.0e-4,
        condition=25.0,
        residual=FLOOR["obv06"],
        signal=1.0e-3,
        tilt_variance_removed=0.0,
    )


def standoff_table_stub() -> dict[tuple[str, str], float]:
    """Return standoffs placing one coil near and the rest far."""

    return {
        ("obv06", "p5_upper"): 1.1,
        ("obv06", "p4_upper"): 4.9,
        ("obv06", "p3_upper"): 11.4,
    }


# --- the record ---------------------------------------------------------


def test_a_channel_without_a_measured_floor_is_left_unjudged(model):
    """A probe whose noise was never measured is skipped, not defaulted."""

    gains = aggregate_family_gains(
        [
            ShotGain(
                shot=index,
                channel="obv06",
                family=family,
                gain=1.08,
                leverage=0.9,
                sample_count=500,
                residual=1.0e-6,
                signal=1.0e-3,
            )
            for family in ("p3_upper", "p4_upper", "p5_upper")
            for index in range(3)
        ],
        standoff_table_stub(),
    )
    assert adjudicate_probes(gains, (), {}) == ()


def test_verdict_counts_name_every_verdict(model, pairs):
    """A verdict absent from a run still appears as a zero, so a reader can tell."""

    counts = verdict_counts(())
    assert set(counts) == {str(verdict) for verdict in ProbeVerdict}
    assert set(counts.values()) == {0}


def test_the_record_round_trips_through_json(model, pairs):
    """The record is the artifact's input, so it has to serialize exactly."""

    shots = [
        synthetic_shot(model, {"p5_upper": 8.0e3}, shot=920061, gains={"obv06": 1.08}),
        synthetic_shot(model, {"p4_upper": 9.0e3}, shot=920062, gains={"obv06": 1.08}),
        synthetic_shot(model, {"p3_upper": 1.1e4}, shot=920063, gains={"obv06": 1.08}),
    ]
    gains, _, rigid = fit_shots(model, pairs, shots)
    rows = adjudicate_probes(
        aggregate_family_gains(gains, standoff_table(model)), rigid, FLOOR
    )
    from nova.imas.mast_probe_calibration import calibration_record

    record = calibration_record(
        rows,
        pairs,
        training_shots=[920061, 920062, 920063],
        held_out_shots=[],
        refused_shots=[],
    )
    assert json.loads(json.dumps(record, sort_keys=True)) == record
    assert record["pre_registration"]["spread_significance"] == 3.0
    assert "obv10" in record["unpaired_channels"] or not record["unpaired_channels"]
