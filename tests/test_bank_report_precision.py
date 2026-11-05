"""The paired precision comparison accepts one bound and refuses each of the four."""

import numpy as np
import pytest

from scripts.prototypes.bank_solve_cycle.report import PAIR_LIMITS, compare, pair_gaps


def receipt(offset_flux=0.0, offset_axis=0.0, offset_saddle=0.0, offset_boundary=0.0):
    """Two-lane receipt shifted from zero by the named amount on every lane."""
    roots = np.array(
        [
            [offset_axis, 0.0],
            [0.0, offset_saddle],
        ]
    )
    return {
        "lanes": 2,
        "call": 0,
        "rows": [
            {
                "lanes": 2,
                "call": 0,
                "span": 1.0,
                "minor_radius": 1.0,
                "flux": np.array([offset_flux, offset_flux]),
                "axis": np.array([[offset_axis, 0.0], [offset_axis, 0.0]]),
                "saddle": np.array([[0.0, offset_saddle], [0.0, offset_saddle]]),
                "boundary_flux": np.array([offset_boundary, offset_boundary]),
                "oracle_roots": roots,
                "oracle": {"boundary_ratio": 0.0},
                "residual": np.array([1e-12, 1e-12]),
                "converged": [True, True],
            }
        ],
    }


def gaps_for(head):
    values, limits = pair_gaps(receipt()["rows"][0], head["rows"][0])
    return {key: float(np.max(value)) for key, value in values.items()}, limits


def test_inside_every_limit_passes_and_reports_zero_gaps():
    values, limits = gaps_for(receipt())
    assert compare(receipt(), receipt()) == [
        {
            "lanes": 2,
            "call": 0,
            "flux": 0.0,
            "axis": 0.0,
            "saddle": 0.0,
            "boundary_flux": 0.0,
            "residual": 1e-12,
            "oracle_axis": 0.0,
            "oracle_saddle": 0.0,
            "oracle_boundary": 0.0,
        }
    ]
    for key in ("flux", "axis", "saddle", "boundary_flux"):
        assert values[key] == 0.0
        assert limits[key] == PAIR_LIMITS[key]


def test_flux_beyond_its_limit_is_refused():
    values, limits = gaps_for(receipt(offset_flux=2e-8))
    assert values["flux"] > limits["flux"]
    assert values["flux"] == pytest.approx(2e-8)
    for key in ("axis", "saddle", "boundary_flux"):
        assert values[key] <= limits[key]
    with pytest.raises(AssertionError):
        compare(receipt(), receipt(offset_flux=2e-8))


def test_axis_beyond_its_limit_is_refused():
    values, limits = gaps_for(receipt(offset_axis=6e-4))
    assert values["axis"] > limits["axis"]
    assert values["axis"] == pytest.approx(6e-4)
    assert values["head_oracle_axis"] <= limits["head_oracle_axis"]
    for key in ("flux", "saddle", "boundary_flux"):
        assert values[key] <= limits[key]
    with pytest.raises(AssertionError):
        compare(receipt(), receipt(offset_axis=6e-4))


def test_saddle_beyond_its_limit_is_refused():
    values, limits = gaps_for(receipt(offset_saddle=6e-4))
    assert values["saddle"] > limits["saddle"]
    assert values["saddle"] == pytest.approx(6e-4)
    assert values["head_oracle_saddle"] <= limits["head_oracle_saddle"]
    for key in ("flux", "axis", "boundary_flux"):
        assert values[key] <= limits[key]
    with pytest.raises(AssertionError):
        compare(receipt(), receipt(offset_saddle=6e-4))


def test_boundary_flux_beyond_its_limit_is_refused():
    values, limits = gaps_for(receipt(offset_boundary=2e-4))
    assert values["boundary_flux"] > limits["boundary_flux"]
    assert values["boundary_flux"] == pytest.approx(2e-4)
    assert values["head_oracle_boundary"] <= limits["head_oracle_boundary"]
    for key in ("flux", "axis", "saddle"):
        assert values[key] <= limits[key]
    with pytest.raises(AssertionError):
        compare(receipt(), receipt(offset_boundary=2e-4))
