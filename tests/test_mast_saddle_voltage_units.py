"""Pin acquisition gain recovery and refusal of unvalidated voltage units."""

import numpy as np
import pytest

from benchmarks.mast_saddle_voltage_units import (
    GainEstimate,
    drive_scale,
    estimate_gain,
    signal_clock,
    validate_known_units,
    voltage_scale,
)
from nova.imas.mast_vacuum_cohort import CoilDrive


def test_ampere_turn_current_is_not_multiplied_by_winding_turns():
    drive = CoilDrive("vertical", "vertical_current", "vertical", True)
    assert drive_scale("kA * turn", drive, {"vertical": None}) == 1000
    with pytest.raises(ValueError, match="expected kiloampere-turns"):
        drive_scale("kA", drive, {"vertical": 9})
    feed = CoilDrive("winding", "feed", "winding", False, 0.5)
    assert drive_scale("kA", feed, {"winding": 20}) == 10000
    with pytest.raises(ValueError, match="unknown feed-current unit"):
        drive_scale("kA * turn", feed, {"winding": 20})


def test_doubled_gain_is_recovered_without_promoting_sign():
    prediction = np.linspace(-3.0, 4.0, 101) ** 3
    nominal = estimate_gain(prediction, prediction + 7)
    doubled = estimate_gain(prediction, 2 * prediction - 11)
    reversed_loop = estimate_gain(prediction, -2 * prediction + 50)
    assert nominal.raw_units_per_volt == pytest.approx(1)
    assert doubled.raw_units_per_volt == pytest.approx(2)
    assert reversed_loop.raw_units_per_volt == pytest.approx(2)
    assert doubled.relative_residual < 1e-14
    assert doubled.samples == 101


def test_known_unit_gain_miss_above_two_percent_is_refused():
    prediction = np.linspace(-2.0, 2.0, 100)
    fits = [
        estimate_gain(prediction, prediction),
        estimate_gain(prediction, 1.021 * prediction),
    ]
    verdict = validate_known_units(fits, [1, 1])
    assert not verdict["accepted"]
    assert verdict["maximum_relative_gain_error"] == pytest.approx(0.021)


def test_known_unit_pass_requires_every_shot_and_waveform():
    exact = GainEstimate(1, 0, 100)
    assert validate_known_units([exact, GainEstimate(1.019, 0.01, 100)], [1, 1])[
        "accepted"
    ]
    assert not validate_known_units([exact, GainEstimate(1, 0.021, 100)], [1, 1])[
        "accepted"
    ]
    assert not validate_known_units(
        [exact, GainEstimate(float("nan"), 0, 100)], [1, 1]
    )["accepted"]
    with pytest.raises(ValueError, match="at least two"):
        validate_known_units([exact], [1])


def test_millivolt_gain_is_checked_in_recorded_units():
    fits = [GainEstimate(1000, 0, 100), GainEstimate(1000, 0, 100)]
    assert validate_known_units(fits, [1000, 1000])["accepted"]
    assert not validate_known_units(fits, [1, 1])["accepted"]


@pytest.mark.parametrize(
    "prediction,recorded",
    [
        (np.zeros(10), np.arange(10)),
        (np.arange(10), np.zeros(10)),
        (np.arange(7), np.arange(7)),
        (np.arange(10), np.full(10, np.nan)),
        (np.arange(10), np.arange(11)),
    ],
)
def test_unidentifiable_gain_is_refused(prediction, recorded):
    with pytest.raises(ValueError):
        estimate_gain(prediction, recorded)


@pytest.mark.parametrize(
    "units,expected", [("Volt", 1), ("V", 1), ("mV", 0.001), ("", None), ("Arb", None)]
)
def test_only_explicit_voltage_units_supply_a_gain(units, expected):
    assert voltage_scale({"units": units, "label": "V"}) == expected


def test_signal_uses_its_declared_clock():
    class Array:
        def __init__(self, values, attrs):
            self.values, self.attrs = values, attrs

        def __array__(self, dtype=None, copy=None):
            return np.asarray(self.values, dtype=dtype)

    group = {
        "sad_out_m01": Array(np.arange(10), {"_ARRAY_DIMENSIONS": ["sec"]}),
        "sec": Array(np.arange(10) / 1000, {"units": "S"}),
        "time": Array(np.zeros(10), {"units": "S"}),
    }
    clock, values = signal_clock(group, "sad_out_m01")
    np.testing.assert_array_equal(clock, np.arange(10) / 1000)
    np.testing.assert_array_equal(values, np.arange(10))
    group["sec"].attrs["units"] = "Arb"
    with pytest.raises(ValueError, match="unknown clock unit"):
        signal_clock(group, "sad_out_m01")
