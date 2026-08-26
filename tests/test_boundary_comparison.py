"""Tests for the shared closed-boundary comparison contract."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from nova.equilibrium.boundary_comparison import (
    BoundaryMode,
    classify_boundary_mode,
    compare_closed_boundaries,
)


def _ring(count: int, *, offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack((np.cos(angle), np.sin(angle))) + offset


def _compare(predicted: object, reference: object, **overrides):
    arguments = {
        "class_margin": 0.25,
        "reference_mode": BoundaryMode.DIVERTED,
        "predicted_saddle_rz_m": np.array([1.0, -1.0]),
        "reference_x_points_rz_m": np.array([[1.1, -1.0]]),
        "sample_count": 400,
    }
    arguments.update(overrides)
    return compare_closed_boundaries(predicted, reference, **arguments)


def test_identical_and_translated_rings_have_physical_distances():
    ring = _ring(96)
    identical = _compare(ring, ring)
    translated = _compare(ring + (0.2, 0.0), ring)

    assert identical.symmetric_sup_distance_m == pytest.approx(0.0, abs=1.0e-15)
    assert identical.symmetric_rms_distance_m == pytest.approx(0.0, abs=1.0e-15)
    assert translated.symmetric_sup_distance_m == pytest.approx(0.2, abs=1.0e-3)
    assert 0.0 < translated.symmetric_rms_distance_m < 0.2


def test_fixed_arc_length_sampling_removes_vertex_density_and_closure_forms():
    sparse = _ring(24)
    dense = _ring(24 * 7)
    repeated_density = np.repeat(sparse, 5, axis=0)

    density_result = _compare(sparse, repeated_density)
    closure_result = _compare(np.vstack((dense, dense[0])), dense)

    assert density_result.symmetric_sup_distance_m == pytest.approx(0.0, abs=1.0e-15)
    assert density_result.symmetric_rms_distance_m == pytest.approx(0.0, abs=1.0e-15)
    assert closure_result.symmetric_sup_distance_m == pytest.approx(0.0, abs=1.0e-15)
    assert closure_result.symmetric_rms_distance_m == pytest.approx(0.0, abs=1.0e-15)


def test_distance_targets_segments_instead_of_only_vertices():
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    midpoint_ring = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    result = _compare(square, midpoint_ring, sample_count=4)

    assert result.symmetric_sup_distance_m == pytest.approx(np.sqrt(0.5))
    assert result.symmetric_rms_distance_m == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("margin", "expected"),
    [
        (np.inf, BoundaryMode.DIVERTED),
        (2.0, BoundaryMode.DIVERTED),
        (0.0, BoundaryMode.DIVERTED),
        (-2.0, BoundaryMode.LIMITED),
        (-np.inf, BoundaryMode.LIMITED),
    ],
)
def test_classification_follows_margin_including_infinities(margin, expected):
    assert classify_boundary_mode(margin) is expected


def test_classification_refuses_nan_and_comparison_fails_closed():
    with pytest.raises(ValueError, match="indeterminate"):
        classify_boundary_mode(np.nan)

    result = _compare(_ring(16), _ring(16), class_margin=np.nan)
    assert result.achieved_mode is None
    assert result.topology_class_agreement is None
    assert result.failures == ("missing_achieved_topology_class",)


@pytest.mark.parametrize(
    ("override", "failure", "null_fields"),
    [
        (
            {"predicted": None},
            "missing_predicted_closed_boundary",
            ("symmetric_sup_distance_m", "symmetric_rms_distance_m"),
        ),
        (
            {"reference": None},
            "missing_reference_closed_boundary",
            ("symmetric_sup_distance_m", "symmetric_rms_distance_m"),
        ),
        (
            {"class_margin": None},
            "missing_achieved_topology_class",
            ("achieved_mode", "topology_class_agreement"),
        ),
        (
            {"reference_mode": None},
            "missing_reference_topology_class",
            ("reference_mode", "topology_class_agreement"),
        ),
        (
            {"predicted_saddle_rz_m": None},
            "missing_predicted_saddle",
            ("x_point_distance_m",),
        ),
        (
            {"reference_x_points_rz_m": None},
            "missing_reference_x_points",
            ("x_point_distance_m",),
        ),
    ],
)
def test_each_missing_input_has_a_stable_failure_and_json_null(
    override, failure, null_fields
):
    predicted = override.pop("predicted", _ring(20))
    reference = override.pop("reference", _ring(20))
    result = _compare(predicted, reference, **override)

    assert result.failures == (failure,)
    for field in null_fields:
        assert getattr(result, field) is None
    payload = json.loads(json.dumps(asdict(result), allow_nan=False))
    for field in null_fields:
        assert payload[field] is None


def test_x_distance_is_invariant_to_unordered_one_and_two_point_sets():
    ring = _ring(20)
    saddle = np.array([1.0, -1.0])
    near = np.array([1.0, -0.75])
    far = np.array([-2.0, 3.0])

    singleton = _compare(
        ring,
        ring,
        predicted_saddle_rz_m=saddle,
        reference_x_points_rz_m=near[None, :],
    )
    forward = _compare(
        ring,
        ring,
        predicted_saddle_rz_m=saddle,
        reference_x_points_rz_m=np.stack((near, far)),
    )
    reversed_result = _compare(
        ring,
        ring,
        predicted_saddle_rz_m=saddle,
        reference_x_points_rz_m=np.stack((far, near)),
    )

    assert singleton.x_point_distance_m == pytest.approx(0.25)
    assert forward.x_point_distance_m == singleton.x_point_distance_m
    assert reversed_result.x_point_distance_m == singleton.x_point_distance_m


def test_open_leg_extension_cannot_enter_the_closed_boundary_api():
    closed = _ring(40)
    baseline = _compare(closed, closed)
    open_legs = np.array([[1.0, 0.0], [1.0, -2.0], [1.0, -2000.0]])

    with pytest.raises(TypeError, match="open_legs_rz_m"):
        compare_closed_boundaries(
            closed,
            closed,
            class_margin=0.25,
            reference_mode=BoundaryMode.DIVERTED,
            predicted_saddle_rz_m=np.array([1.0, -1.0]),
            reference_x_points_rz_m=np.array([[1.1, -1.0]]),
            open_legs_rz_m=open_legs,
        )
    assert baseline == _compare(closed, closed)
    assert np.ptp(open_legs[:, 1]) > 1000.0


def test_invalid_boundaries_and_empty_finite_x_set_fail_closed():
    ring = _ring(20)
    result = _compare(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        ring,
        reference_x_points_rz_m=np.array([[np.nan, 0.0], [1.0, np.inf]]),
    )

    assert result.failures == (
        "missing_predicted_closed_boundary",
        "missing_reference_x_points",
    )
    assert result.symmetric_sup_distance_m is None
    assert result.symmetric_rms_distance_m is None
    assert result.x_point_distance_m is None
