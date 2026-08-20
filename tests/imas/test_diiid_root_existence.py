"""Contracts for labelled-map free-boundary residual attribution."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "diiid_root_existence.py"
SPEC = importlib.util.spec_from_file_location("diiid_root_existence", MODULE_PATH)
root_existence = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = root_existence
SPEC.loader.exec_module(root_existence)


def test_preregistration_precedes_scoring_and_pins_the_nearby_root_criterion(
    tmp_path,
):
    path = root_existence.write_preregistration(tmp_path)
    declaration = json.loads(path.read_text())
    assert declaration["selection"]["frames"] == 5
    assert declaration["nearby_root_criterion"][
        "maximum_free_boundary_fractional_rms"
    ] == pytest.approx(0.05)
    assert declaration["attribution"]["dominant_component_fraction"] == (
        pytest.approx(2.0 / 3.0)
    )
    assert declaration["coefficients_fitted"] == 0
    assert declaration["coil_currents_adjusted"] is False
    assert root_existence.require_preregistration(path)


def test_changed_or_missing_preregistration_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="preregistered"):
        root_existence.require_preregistration(tmp_path / "missing.json")
    path = root_existence.write_preregistration(tmp_path)
    record = json.loads(path.read_text())
    record["nearby_root_criterion"]["maximum_free_boundary_fractional_rms"] = 1.0
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="does not match"):
        root_existence.require_preregistration(path)


def test_regions_partition_every_node_and_residual_fractions_close():
    core = np.zeros((9, 9), dtype=bool)
    core[2:7, 2:7] = True
    regions = root_existence.region_masks(core)
    membership = sum(mask.astype(int) for mask in regions.values())
    np.testing.assert_array_equal(membership, np.ones_like(membership))
    residual = np.arange(81, dtype=float).reshape(9, 9)
    metrics = root_existence.residual_metrics(residual, 100.0, regions)
    assert metrics["regional_fraction_sum"] == pytest.approx(1.0)
    assert sum(metrics["regional_squared_residual_fractions"].values()) == (
        pytest.approx(1.0)
    )


def test_boundary_discrepancy_reports_literal_and_gauged_values():
    labelled = np.arange(25, dtype=float).reshape(5, 5)
    coil = labelled - 3.0
    measured = root_existence.boundary_discrepancy(labelled, coil, 3.0)
    assert measured["before_gauge"]["signed_mean_wb_per_radian"] == (pytest.approx(3.0))
    assert measured["after_gauge"]["rms_wb_per_radian"] == pytest.approx(0.0)
    assert len(measured["after_gauge_samples_wb_per_radian"]) == 16


def test_border_coordinates_follow_radius_height_array_order():
    radius = np.asarray([1.0, 1.5, 2.0])
    height = np.asarray([-1.0, 0.0])
    coordinates = root_existence.border_coordinates(radius, height)
    expected = np.asarray(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.5, 0.0],
            [2.0, 0.0],
            [2.0, -1.0],
            [1.5, -1.0],
        ]
    )
    np.testing.assert_array_equal(coordinates, expected)


@pytest.mark.parametrize(
    ("fixed", "vacuum", "boundary", "expected"),
    (
        (0.01, 0.10, 0.10, "vacuum-field-dominated"),
        (0.10, 0.01, 0.01, "flux-function-dominated"),
        (0.05, 0.05, 0.10, "both"),
    ),
)
def test_attribution_verdict_uses_the_declared_numeric_split(
    fixed, vacuum, boundary, expected
):
    result = root_existence.component_verdict(fixed, vacuum, boundary)
    assert result["verdict"] == expected
    assert result["share_sum"] == pytest.approx(1.0)


def test_benchmark_source_contains_no_root_find_route():
    source = MODULE_PATH.read_text()
    forbidden = (
        "newton_krylov",
        "host_krylov",
        "ForwardProfile",
        "scipy.optimize.root",
    )
    for token in forbidden:
        assert token not in source
    assert '"root_find_performed": False' in source
    assert '"no_root_find": True' in source
