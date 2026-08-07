"""Whether currents planted in a synthetic machine come back out of the sensors.

The manufactured truth is a machine with conductors at known positions carrying known
currents, read by sensors at known positions.  The design is built from the conductor
geometry alone, the sensors are given exactly what those currents produce, and the
solve has to return the currents.  Nothing here reads a drive map, which is the whole
point of inverting rather than regressing: the answer must come from the measurement.

The spectrum is tested where it matters, which is not on a well-posed design.  Two
conductors placed almost on top of each other produce almost the same field at every
sensor, so their difference is a direction the sensors cannot see; the weakest mode
has to name that pair.  A spectrum that reported a comfortable condition number there
would be telling a consumer it can measure something nothing measured.

The rest are the traps the arithmetic hides.  A sensor with no measured floor must not
be admitted at unit weight, because an uncharacterised sensor would then set the
solve.  A conductor whose current is published is a known drive and leaves the
measurement; solving for it instead lets it absorb a neighbour's error.  And a ratio
taken through zero current is a quotient of two noise floors, so the ratio is taken
only over the samples where the conductor was actually driven.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.calibrate.inversion import (
    InversionError,
    conductor_ratio,
    identifiability,
    solve_currents,
    subtract_known_drives,
    truncation_ladder,
    whiten,
)
from nova.calibrate.localize import axial_projection, loop_field

SENSOR_COUNT = 36
"""Sensors, comfortably more than the conductors they are asked to resolve."""

SAMPLES = 400
"""Time samples in a manufactured window."""

CONDUCTORS = {
    "inner_lower": (0.9, -1.2),
    "inner_upper": (0.9, 1.2),
    "outer_lower": (1.7, -0.6),
    "outer_upper": (1.7, 0.6),
    "solenoid": (0.3, 0.0),
}
"""Five conductors at well-separated positions in the poloidal plane."""


def sensors():
    """Return sensor positions and the axis each is sensitive to."""

    angle = np.linspace(0.0, 2.0 * np.pi, SENSOR_COUNT, endpoint=False)
    return (
        1.4 + 0.4 * np.cos(angle),
        1.0 * np.sin(angle),
        np.cos(1.7 * angle),
        np.sin(1.7 * angle),
    )


def design_matrix(positions):
    """Return each conductor's sensor response per unit current."""

    radius, height, cosine, sine = sensors()
    return np.column_stack(
        [
            axial_projection(*loop_field(radius, height, r0, z0), cosine, sine)
            for r0, z0 in positions.values()
        ]
    )


def histories(count, seed=3):
    """Return one current history per conductor over the manufactured window."""

    time = np.linspace(0.0, 1.0, SAMPLES)
    generator = np.random.default_rng(seed)
    return np.stack(
        [
            (1.0e3 + 4.0e3 * generator.random())
            * np.sin((1.0 + index) * 2.0 * np.pi * time + generator.random())
            for index in range(count)
        ]
    )


def test_the_planted_currents_come_back_out_of_the_sensors():
    design = design_matrix(CONDUCTORS)
    truth = histories(len(CONDUCTORS))
    solution = solve_currents(design, design @ truth, tuple(CONDUCTORS))
    assert np.allclose(solution.currents, truth, rtol=1e-8)
    assert solution.residual < 1e-12 * solution.signal
    assert solution.held_out_residual < 1e-12 * solution.held_out_signal
    assert np.allclose(solution.current("solenoid"), truth[4], rtol=1e-8)


def test_asking_for_a_conductor_the_solve_did_not_carry_is_refused():
    design = design_matrix(CONDUCTORS)
    solution = solve_currents(design, design @ histories(5), tuple(CONDUCTORS))
    with pytest.raises(InversionError, match="not a column"):
        solution.current("nonexistent")


def test_the_held_out_residual_reports_reconstruction_and_the_fit_residual_does_not():
    """Adding sensors that carry an undescribed field is invisible to the fit residual.

    The fit gives every conductor the current that suits it best, so a fault it can
    partly absorb leaves a small residual; the sensors withheld from the solve are what
    say whether the currents predict anything they were not fitted to.
    """

    design = design_matrix(CONDUCTORS)
    truth = histories(len(CONDUCTORS))
    radius, height, cosine, sine = sensors()
    undescribed = axial_projection(
        *loop_field(radius, height, 2.3, 1.6), cosine, sine
    )
    observed = design @ truth + np.outer(undescribed, 400.0 * truth[0])
    solution = solve_currents(design, observed, tuple(CONDUCTORS))
    assert solution.held_out_residual > solution.residual


def test_two_conductors_the_sensors_cannot_tell_apart_name_themselves():
    degenerate = dict(CONDUCTORS)
    degenerate["twin"] = (0.9005, -1.2005)
    design = design_matrix(degenerate)
    spectrum = identifiability(design, tuple(degenerate))
    weakest = spectrum.modes[-1]
    named = {name for name, _ in weakest.dominant}
    assert {"inner_lower", "twin"} <= named
    assert spectrum.condition_number > 1.0e3
    assert weakest.relative < 1.0e-3


def test_a_well_separated_set_carries_every_direction_the_sensors_are_asked_for():
    """No mode of a well-separated set falls to the degenerate pair's thousandth.

    The weakest direction is still the solenoid, which sits furthest inside the sensor
    ring and is therefore always the least strongly seen; what distinguishes this set
    from a degenerate one is that being weakest costs it two orders and not five.
    """

    spectrum = identifiability(design_matrix(CONDUCTORS), tuple(CONDUCTORS))
    assert spectrum.condition_number < 1.0e3
    assert spectrum.unresolved(1.0e-3) == ()
    assert len(spectrum.modes) == len(CONDUCTORS)
    assert spectrum.modes[-1].dominant[0][0] == "solenoid"


def test_a_sensor_with_no_measured_floor_is_given_no_say():
    design = design_matrix(CONDUCTORS)
    floors = np.full(SENSOR_COUNT, 1.0e-4)
    floors[7] = np.inf
    whitened = whiten(design, floors)
    assert np.allclose(whitened[7], 0.0)
    assert np.allclose(whitened[0], design[0] * 1.0e4)


def test_floors_arriving_by_channel_name_are_ordered_against_the_design():
    design = design_matrix(CONDUCTORS)
    names = [f"probe{index:02d}" for index in range(SENSOR_COUNT)]
    floors = {name: 2.0e-4 for name in names[:-1]}
    whitened = whiten(design, floors, rows=names)
    assert np.allclose(whitened[-1], 0.0)
    assert np.allclose(whitened[0], design[0] * 5.0e3)


def test_floors_given_as_a_mapping_without_row_names_are_refused():
    with pytest.raises(InversionError, match="row channel names"):
        whiten(design_matrix(CONDUCTORS), {"probe00": 1.0})


def test_a_published_current_leaves_the_measurement_rather_than_being_solved():
    design = design_matrix(CONDUCTORS)
    truth = histories(len(CONDUCTORS))
    observed = design @ truth
    columns = tuple(CONDUCTORS)
    reduced = subtract_known_drives(
        observed, design, columns, {"solenoid": truth[4]}
    )
    solution = solve_currents(
        design[:, :4], reduced, columns[:4]
    )
    assert np.allclose(solution.currents, truth[:4], rtol=1e-8)


def test_removing_a_drive_with_no_column_in_the_design_is_refused():
    design = design_matrix(CONDUCTORS)
    with pytest.raises(InversionError, match="no column in the design"):
        subtract_known_drives(
            design @ histories(5),
            design,
            tuple(CONDUCTORS),
            {"absent": np.zeros(SAMPLES)},
        )


def test_the_ladder_at_full_rank_is_the_full_solve():
    design = design_matrix(CONDUCTORS)
    truth = histories(len(CONDUCTORS))
    observed = design @ truth
    ladder = truncation_ladder(design, observed, (3, len(CONDUCTORS)))
    assert set(ladder) == {3, len(CONDUCTORS)}
    assert np.allclose(ladder[len(CONDUCTORS)], truth, rtol=1e-8)
    assert not np.allclose(ladder[3], truth, rtol=1e-2)


def test_a_rank_the_design_does_not_have_is_dropped_rather_than_faked():
    design = design_matrix(CONDUCTORS)
    assert truncation_ladder(design, design @ histories(5), (0, 99)) == {}


def test_a_conductor_ratio_is_turns_per_recorded_ampere():
    turns, feed = 23.0, histories(1)[0]
    fit = conductor_ratio(feed, turns * feed, excited=200.0)
    assert fit.slope == pytest.approx(turns, rel=1e-12)
    assert fit.variance_explained == pytest.approx(1.0, abs=1e-12)


def test_a_ratio_is_taken_only_where_the_conductor_was_driven():
    """Samples through zero are a quotient of two noise floors and must not vote."""

    generator = np.random.default_rng(17)
    feed = np.concatenate([np.full(300, 1.0e3), np.zeros(300)])
    solved = 23.0 * feed + generator.normal(0.0, 5.0, feed.size)
    assert conductor_ratio(feed, solved, excited=200.0).slope == pytest.approx(
        23.0, rel=1e-3
    )


def test_a_conductor_never_driven_says_nothing():
    quiet = np.full(SAMPLES, 10.0)
    assert conductor_ratio(quiet, 23.0 * quiet, excited=200.0) is None


def test_a_design_and_a_measurement_of_different_heights_are_refused():
    design = design_matrix(CONDUCTORS)
    with pytest.raises(InversionError, match="against"):
        solve_currents(design, np.zeros((SENSOR_COUNT - 1, SAMPLES)), tuple(CONDUCTORS))
