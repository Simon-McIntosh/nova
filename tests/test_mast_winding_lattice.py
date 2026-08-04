"""What the turn layout derives from geometry, and whether the fit can read it.

Two halves.  The first pins the geometry: the chamfer must vacate exactly one grid
position on every one of these coils, because that is what makes the layout a
prediction of the published turn count rather than a parameter fitted to it, and a
change that broke it would otherwise pass unnoticed as a small shift in a residual.

The second half asks whether the estimator can find a layout that is really there.
The waveforms are synthetic and built FROM a chosen layout, a chosen set of channel
gains and a chosen per-shot amplitude, so the answer is known before the fit runs.
That matters more here than in most places: this measurement returns a null result
on the archive, and a null result is only worth reading if the same machinery
returns a positive one when the effect is present.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_vacuum_cohort import (
    ERROR_FIELD_CHANNELS,
    EXCITATION_CURRENT,
    ShotWaveforms,
    probe_channels,
)
from nova.imas.mast_vacuum_response import ResponseModel, coil_sections
from nova.imas.mast_winding_lattice import (
    FILL_BOUNDS,
    LATTICE_SHAPES,
    LatticeError,
    TurnLattice,
    admissible_shapes,
    amplitude_reach,
    baseline_columns,
    calibrate_array,
    channel_deltas,
    error_field_quiescent,
    lattice_column,
    passes_error_field_screen,
    profile_displacement,
    profile_fill,
    reduce_shot,
    score_hypothesis,
    search_shapes,
    section_column,
    translated_section,
    uniform_column,
)

REPRESENTATIVE_SHOT = 11766
"""Registry selection the layout is authored against."""

WOUND_FAMILIES = ("p4_lower", "p4_upper", "p5_lower", "p5_upper")
"""Coils the archive publishes twenty-three turns for."""

PUBLISHED_TURNS = 23
"""Turns the archive publishes for each of them, on more than fifteen thousand shots."""


@pytest.fixture(scope="module")
def geometry():
    """Return the registry configuration the layout is authored against."""

    return MachineGeometryRegistry.default().select(REPRESENTATIVE_SHOT)


@pytest.fixture(scope="module")
def outlines(geometry):
    """Return each wound coil's winding-pack outline."""

    sections = coil_sections(geometry.configuration.geometry)
    return {family: sections[family][0] for family in WOUND_FAMILIES}


@pytest.fixture(scope="module")
def model(geometry):
    """Return the uniform-density response every layout is compared against."""

    probes = geometry.configuration.geometry["magnetics"]["poloidal_probes"]
    return ResponseModel.build(
        geometry.configuration.geometry, probes, probe_channels(probes)
    )


@pytest.fixture(scope="module")
def baseline(model, geometry):
    """Return the uniform-density column of every coil."""

    return baseline_columns(model, geometry.configuration.geometry)


def synthetic_shot(
    model: ResponseModel,
    columns: dict[str, np.ndarray],
    excitation: dict[str, float],
    *,
    shot: int = 920001,
    samples: int = 300,
    amplitude: float = 1.0,
    turns: float = PUBLISHED_TURNS,
    gains: dict[str, float] | None = None,
    missing: dict[str, slice] | None = None,
    offset: float = 3.0e-4,
    noise: float = 0.0,
    seed: int = 0,
) -> ShotWaveforms:
    """Build a shot whose probes read the field ``columns`` predicts.

    ``turns`` is the ampere-turn weight the drive channel carries, so a shot built
    here and reduced with the same weight returns unit amplitude.  ``gains``
    multiplies one channel's reading and ``missing`` blanks a span of one channel's
    samples, which are the two ways a real shot departs from the model in a manner
    that could be mistaken for a layout.
    """

    generator = np.random.default_rng(seed)
    time = np.linspace(-0.2, 1.0, samples)
    ramp = np.clip(time / 0.05, 0.0, 1.0) * np.clip((0.8 - time) / 0.05, 0.0, 1.0)
    drives = {family: np.zeros_like(time) for family in model.families}
    for family, current in excitation.items():
        drives[family] = current * ramp

    probes: dict[str, np.ndarray] = {}
    for row, target in enumerate(model.targets):
        signal = np.full_like(time, offset * (1 + row % 4))
        for family, current in excitation.items():
            signal = signal + amplitude * turns * columns[family][row] * drives[family]
        if gains and target.channel in gains:
            signal = (signal - offset * (1 + row % 4)) * gains[
                target.channel
            ] + offset * (1 + row % 4)
        if noise > 0.0:
            signal = signal + generator.normal(0.0, noise, size=time.shape)
        if missing and target.channel in missing:
            signal = signal.copy()
            signal[missing[target.channel]] = np.nan
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


def scatter_for(model: ResponseModel, value: float = 1.0e-4) -> dict[str, float]:
    """Return a flat noise envelope, so whitening cannot pick a favourite."""

    return {target.channel: value for target in model.targets}


# --- what the chamfer derives ------------------------------------------


@pytest.mark.parametrize("family", WOUND_FAMILIES)
@pytest.mark.parametrize("shape", LATTICE_SHAPES)
def test_chamfer_vacates_exactly_one_position(outlines, family, shape):
    """A grid of twenty-four positions loses one to the chamfer, leaving twenty-three.

    This is the whole reason the layout is a claim about the machine and not a knob:
    the count the archive publishes falls out of an outline that was already carried
    as identity, with nothing fitted.  If a geometry change ever moved the chamfer
    enough to cut a second position -- or none -- the layout would stop reproducing
    the published count and this is where that shows up.
    """

    lattice = TurnLattice(*shape)
    assert lattice.positions == PUBLISHED_TURNS + 1
    assert lattice.turn_count(outlines[family]) == PUBLISHED_TURNS


@pytest.mark.parametrize("family", WOUND_FAMILIES)
def test_the_vacated_position_is_the_outboard_corner_off_the_midplane(outlines, family):
    """The cross-over sits where a wound pack takes it, and mirrors about z = 0.

    Both P4 coils and both P5 coils vacate their outboard corner on the side away
    from the midplane, which is the same physical corner under the up-down mirror.
    An implementation that placed the vacancy on the inboard side, or failed to
    mirror, would still return twenty-three turns and would move the current
    centroid the wrong way.
    """

    outline = outlines[family]
    lattice = TurnLattice(6, 4)
    grid_r, grid_z = lattice.grid(outline)
    keep = lattice.occupied(outline)
    vacant_r, vacant_z = float(grid_r[~keep][0]), float(grid_z[~keep][0])
    middle_r = 0.5 * (grid_r.min() + grid_r.max())
    assert vacant_r > middle_r
    assert abs(vacant_z) > abs(0.5 * (grid_z.min() + grid_z.max()))


def test_only_three_grids_reproduce_the_count_on_every_coil(outlines):
    """Searched over every grid to twelve a side, three shapes work on all four.

    The offered set is the intersection, not any one coil's list: the four coils are
    identically wound, so a shape that describes one and fails another describes
    neither.  P4 upper alone would also admit three by eight, and that is precisely
    the kind of coil-by-coil freedom the intersection removes.  The set is a measured
    result rather than a choice, so it is pinned -- widening it would let the fit
    choose a layout that contradicts the count, and narrowing it would hide a
    hypothesis the data is entitled to prefer.
    """

    per_coil = {
        family: search_shapes(outline, PUBLISHED_TURNS)
        for family, outline in outlines.items()
    }
    shared = set(per_coil[WOUND_FAMILIES[0]])
    for found in per_coil.values():
        shared &= set(found)
    assert tuple(sorted(shared)) == tuple(sorted(LATTICE_SHAPES))
    for family, found in per_coil.items():
        assert set(LATTICE_SHAPES) <= set(found), family
        assert admissible_shapes(outlines[family], PUBLISHED_TURNS) == LATTICE_SHAPES
        assert admissible_shapes(outlines[family], PUBLISHED_TURNS + 1) == ()


@pytest.mark.parametrize("family", WOUND_FAMILIES)
def test_insulation_thickness_cannot_change_the_turn_count(outlines, family):
    """Contracting the layout does not pull the vacated position back inside.

    The chamfer test runs on the uncontracted grid on purpose.  If the fill fraction
    were allowed to decide which positions exist, the turn count would become a
    function of the fitted parameter and the fit could buy a twenty-fourth turn by
    tightening the insulation.
    """

    counts = {
        TurnLattice(6, 4, fill).turn_count(outlines[family])
        for fill in (1.0, 0.95, 0.9, 0.85, 0.8, 0.5)
    }
    assert counts == {PUBLISHED_TURNS}


@pytest.mark.parametrize("family", WOUND_FAMILIES)
def test_the_layout_moves_the_current_inboard_and_toward_the_midplane(outlines, family):
    """Vacating the outer corner pulls the current centroid off the outline's own.

    The direction is the measurable claim -- it is what a probe beside the pack sees
    -- and it is opposite in sign to what contracting the layout does, so the two
    parameters are not degenerate.
    """

    outline = outlines[family]
    centre = shapely.Polygon(outline).centroid
    lattice_r, lattice_z = TurnLattice(6, 4).centroid(outline)
    assert lattice_r < centre.x
    assert abs(lattice_z) < abs(centre.y)
    assert 0.5e-3 < math.hypot(lattice_r - centre.x, lattice_z - centre.y) < 5.0e-3


def test_one_section_is_the_uniform_outline(model, outlines):
    """A single-section layout is the uniform column, exactly.

    The two paths through the coupling must agree where they describe the same
    thing, or a layout comparison is reading a difference between two integrators
    rather than between two current distributions.
    """

    outline = outlines["p5_upper"]
    assert section_column(model.targets, (outline,)) == pytest.approx(
        uniform_column(model.targets, outline), rel=1.0e-12
    )


def test_a_displaced_outline_is_not_a_layout(model, outlines):
    """Displacing the outline moves the field; it is the control, not the model."""

    outline = outlines["p5_upper"]
    moved = translated_section(outline, (2.0e-3, 0.0))
    assert shapely.Polygon(moved).centroid.x == pytest.approx(
        shapely.Polygon(outline).centroid.x + 2.0e-3
    )
    assert not np.allclose(
        uniform_column(model.targets, moved), uniform_column(model.targets, outline)
    )
    with pytest.raises(LatticeError):
        translated_section(outline, (1.0, 2.0, 3.0))


def test_a_layout_is_invisible_beyond_two_pack_widths(model, outlines, baseline):
    """Outside the exclusion radius every layout predicts the same field.

    This is why the turn fit was entitled to use the uniform outline on the probes
    it kept, and it is also the limit on what this node can do: the probes that
    could settle a layout are exactly the ones a turn count may not be fitted on.
    """

    family = "p5_upper"
    column = model.families.index(family)
    layout = lattice_column(model.targets, outlines[family], TurnLattice(6, 4))
    uniform = baseline[family]
    far = model.standoff[:, column] >= 2.0
    carries_signal = np.abs(uniform) > 1.0e-8
    fraction = np.abs(
        (layout[far & carries_signal] - uniform[far & carries_signal])
        / uniform[far & carries_signal]
    )
    assert np.median(fraction) < 1.0e-3
    near = model.standoff[:, column] < 1.5
    assert np.max(np.abs((layout[near] - uniform[near]) / uniform[near])) > 5.0e-3


# --- the estimator, on data whose answer is known ----------------------


@pytest.mark.parametrize("family", WOUND_FAMILIES)
def test_a_layout_cannot_move_a_pooled_amplitude_by_a_tenth_of_a_percent(
    model, baseline, outlines, family
):
    """No offered layout shifts a far-field amplitude by more than a tenth of a percent.

    This is the bound that keeps a layout out of every amplitude argument.  A
    response-scale or turn-count measurement is a pooled amplitude over the probes
    standing clear of the coil, and rearranging the turns inside the pack cannot
    move it by even a tenth of a percent -- so a reported discrepancy of a few
    percent, and a fortiori one that differs between campaigns, is not a winding
    layout however the layout is drawn.
    """

    column = model.families.index(family)
    clear = model.standoff[:, column] >= 2.0
    reach = amplitude_reach(model.targets, outlines[family], baseline[family], clear)
    assert reach < 1.0e-3


def test_an_exact_model_leaves_no_residual(model, baseline, outlines):
    """A shot built from the model returns unit amplitude and nothing left over."""

    waveforms = synthetic_shot(model, baseline, {"p5_upper": 8.0e3})
    moments = reduce_shot(
        waveforms, model, "p5_upper", {"p5_upper": 23.0}, scatter_for(model)
    )
    score = score_hypothesis(moments, baseline)
    assert score.amplitude == pytest.approx(1.0, rel=1.0e-9)
    assert score.residual < 1.0e-6 * score.signal


def test_missing_samples_do_not_bias_the_amplitude(model, baseline):
    """A channel with a third of its samples blank still reports amplitude one.

    Folding absent samples in as zeroes drags the fitted amplitude down by exactly
    the missing fraction, which on this array is a thirty percent error sitting on
    top of a one percent measurement -- large enough to invert the sign of a layout
    verdict, and silent because every channel moves together.
    """

    blanked = {"obv06": slice(100, 200), "obr10": slice(50, 150)}
    waveforms = synthetic_shot(model, baseline, {"p5_upper": 8.0e3}, missing=blanked)
    moments = reduce_shot(
        waveforms, model, "p5_upper", {"p5_upper": 23.0}, scatter_for(model)
    )
    assert set(blanked) <= set(moments.channels)
    counts = dict(zip(moments.channels, moments.samples_used, strict=True))
    assert counts["obv06"] < counts["ccbv01"]
    score = score_hypothesis(moments, baseline)
    assert score.amplitude == pytest.approx(1.0, rel=1.0e-9)


def test_a_planted_layout_is_recovered(model, baseline, outlines):
    """A cohort built from a known fill fraction returns that fill fraction.

    The estimator has to be shown finding a layout that is present before its
    failure to find one in the archive means anything about the archive.
    """

    family = "p5_upper"
    planted = 0.88
    columns = dict(baseline)
    columns[family] = lattice_column(
        model.targets, outlines[family], TurnLattice(6, 4, planted)
    )
    moments = [
        reduce_shot(
            synthetic_shot(
                model, columns, {family: current}, shot=930000 + index, seed=index
            ),
            model,
            family,
            {family: 23.0},
            scatter_for(model),
        )
        for index, current in enumerate((6.0e3, 8.0e3, 1.2e4))
    ]
    profile = profile_fill(
        moments, model.targets, outlines[family], baseline, shape=(6, 4)
    )
    assert profile.fill == pytest.approx(planted, abs=0.01)
    assert profile.improvement > 0.5
    assert FILL_BOUNDS[0] <= profile.fill <= FILL_BOUNDS[1]


def test_a_planted_displacement_is_recovered(model, baseline, outlines):
    """A coil whose outline really sits two millimetres inboard is found there.

    The displacement scan bounds what any intra-pack redistribution could buy, so
    it has to be shown resolving a displacement of the size the sources disagree
    by.
    """

    family = "p5_lower"
    planted = (-2.0e-3, 1.0e-3)
    columns = dict(baseline)
    columns[family] = uniform_column(
        model.targets, translated_section(outlines[family], planted)
    )
    moments = [
        reduce_shot(
            synthetic_shot(
                model, columns, {family: current}, shot=940000 + index, seed=index
            ),
            model,
            family,
            {family: 23.0},
            scatter_for(model),
        )
        for index, current in enumerate((8.0e3, 1.4e4))
    ]
    profile = profile_displacement(
        moments, model.targets, outlines[family], baseline, reach=4.0e-3, steps=9
    )
    assert profile.offset[0] == pytest.approx(planted[0], abs=1.0e-3)
    assert profile.offset[1] == pytest.approx(planted[1], abs=1.0e-3)
    assert profile.improvement > 0.5


def test_planted_channel_gains_are_recovered_off_the_far_field(model, baseline):
    """Gains solved where no probe stands close return the numbers that were put in.

    A gain and a layout are the same parameter on one coil's shots, so the layout
    is only measurable once the gains are known.  They are recovered from the shots
    where each channel stands clear of everything driven, and that separation is
    what this pins.
    """

    planted = {"obv06": 1.25, "obr17": 0.5, "obv14": 0.9}
    moments = []
    for index, (family, current) in enumerate(
        (("p4_lower", 8.0e3), ("p4_upper", 9.0e3), ("p5_lower", 1.0e4))
    ):
        waveforms = synthetic_shot(
            model,
            baseline,
            {family: current},
            shot=950000 + index,
            amplitude=1.1,
            gains=planted,
            seed=index,
        )
        moments.append(
            reduce_shot(
                waveforms, model, family, {family: 23.0}, scatter_for(model)
            )
        )
    calibration = calibrate_array(moments, baseline)
    for channel, value in planted.items():
        assert calibration.gains[channel] == pytest.approx(value, rel=0.02)
    for level in calibration.amplitudes.values():
        assert level == pytest.approx(1.1, rel=0.02)


def test_channel_deltas_report_every_probe_both_ways(model, baseline, outlines):
    """A layout change is reported per channel, signed, so a trade shows up.

    A promotion that helps the two probes it was aimed at by hurting six others is
    not an improvement, and the only way to see that is a per-channel ledger.
    """

    family = "p5_upper"
    proposed = dict(baseline)
    proposed[family] = lattice_column(
        model.targets, outlines[family], TurnLattice(6, 4, 0.9)
    )
    moments = [
        reduce_shot(
            synthetic_shot(model, baseline, {family: 8.0e3}),
            model,
            family,
            {family: 23.0},
            scatter_for(model),
        )
    ]
    deltas = channel_deltas(moments, baseline, proposed, select="all")
    assert len(deltas) == len(moments[0].channels)
    assert all({"before", "after", "delta"} <= set(row) for row in deltas.values())
    assert sum(row["delta"] > 0.0 for row in deltas.values()) > 0


# --- the error-field screen --------------------------------------------


def test_a_driven_error_field_channel_refuses_the_shot():
    """A bank above the excitation floor is driven, and the shot is refused."""

    state = error_field_quiescent(
        {name: 0.0 for name in ERROR_FIELD_CHANNELS} | {"error_field_02": 5.0e3}
    )
    assert state["error_field_02"] == "driven"
    assert not passes_error_field_screen(state)


def test_an_unrecorded_error_field_channel_is_never_quiescent():
    """A channel nobody looked at is unmeasured, which is not the same as quiet.

    Treating an absent channel as zero would silently admit every shot from a
    campaign that did not record the bank at all, which is most of them.
    """

    state = error_field_quiescent({"efps_current": 10.0}, absent=("error_field_02",))
    assert state["error_field_02"] == "unmeasured"
    assert state["error_field_05"] == "unmeasured"
    assert state["efps_current"] == "quiescent"
    assert not passes_error_field_screen(state)


def test_a_quiet_bank_admits_the_shot():
    """Every channel measured and below the floor is the only admitting case."""

    state = error_field_quiescent(
        {name: 0.1 * EXCITATION_CURRENT for name in ERROR_FIELD_CHANNELS}
    )
    assert set(state.values()) == {"quiescent"}
    assert passes_error_field_screen(state)
    assert not passes_error_field_screen({})


# --- refusals ----------------------------------------------------------


def test_a_layout_needs_a_coil_the_response_carries(model, baseline):
    """A family the response has no column for cannot be reduced."""

    waveforms = synthetic_shot(model, baseline, {"p5_upper": 8.0e3})
    with pytest.raises(LatticeError):
        reduce_shot(waveforms, model, "p9_upper", {"p9_upper": 1.0}, scatter_for(model))


def test_a_coil_with_no_promoted_weight_is_refused(model, baseline):
    """Every energised coil needs a weight, or its field is silently dropped."""

    waveforms = synthetic_shot(model, baseline, {"p5_upper": 8.0e3, "p4_lower": 6.0e3})
    with pytest.raises(LatticeError):
        reduce_shot(waveforms, model, "p5_upper", {"p5_upper": 23.0}, scatter_for(model))


def test_a_degenerate_grid_or_fill_is_refused():
    """A grid with no positions, or a non-positive fill, is not a winding."""

    with pytest.raises(LatticeError):
        TurnLattice(0, 4)
    with pytest.raises(LatticeError):
        TurnLattice(6, 4, 0.0)
    with pytest.raises(LatticeError):
        TurnLattice(6, 4, float("nan"))
