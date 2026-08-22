from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "benchmarks" / "diiid_circuit_driven_forward_validation.py"
SPEC = importlib.util.spec_from_file_location(
    "diiid_circuit_driven_forward_validation", MODULE_PATH
)
study = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)

RECEIPT_PATH = (
    ROOT
    / "docs/figures/coil-circuit-discovery/circuit-driven-forward-validation"
    / study.RECEIPT_NAME
)
CONTOUR_RECEIPT_PATH = RECEIPT_PATH.with_name(study.CONTOUR_RECEIPT_NAME)


def _receipt() -> dict:
    return json.loads(RECEIPT_PATH.read_text())


def test_selection_is_strictly_outside_calibration_and_polarity_banks() -> None:
    receipt = _receipt()
    selection = receipt["selection"]
    selected = {(item["shot"], item["frame"]) for item in selection["selected_frames"]}
    calibration_frames, calibration_shots, _bank = study.calibration_population()
    affected = study.polarity_population()

    assert len(selected) >= 5
    assert len({shot for shot, _frame in selected}) >= 3
    assert not selected & calibration_frames
    assert not {shot for shot, _frame in selected} & calibration_shots
    assert not {shot for shot, _frame in selected} & affected
    assert selection["calibration_bank"]["strictly_outside"] is True
    assert selection["calibration_bank"]["frame_count"] == 60
    assert selection["calibration_bank"]["shot_count"] == 20
    assert selection["all_finite_diverted"] is True
    assert selection["all_polarity_screened"] is True
    assert selection["all_recorded_plasma_currents_qualified"] is True
    assert selection["recorded_plasma_current_floor_abs_a"] == 50_000.0
    assert all(
        abs(item["recorded_plasma_current_a"]) >= 50_000.0
        for item in selection["selected_frames"]
    )

    excluded = selection["degenerate_current_audit"]
    assert len(excluded) == 6
    assert all(item["qualified"] is False for item in excluded)
    assert all(item["absolute_current_a"] < 50_000.0 for item in excluded)
    failures = selection["guard_failure_rows"]
    assert failures
    assert all(item["amplitude_guard_triggered"] is True for item in failures)
    assert all(item["fixed_point_relative_residual"] is None for item in failures)
    assert all(item["iterations"] == 0 for item in failures)
    assert {item["shot"] for item in failures} >= {
        "d3d_shot_00bb25ac45.parquet",
        "d3d_shot_00f71cf526.parquet",
    }


def test_circuit_arm_has_complete_label_free_current_authority() -> None:
    receipt = _receipt()
    audit = receipt["current_path_audit"]

    assert audit["label_derived_current_reads"] == 0
    assert audit["per_frame_current_fits"] == 0
    assert audit["least_squares_updates"] == 0
    assert audit["unknown_current_parameters"] == 0
    assert audit["plasma_current_channel"] == "magnetics_plasma_current"
    assert len(audit["competition_current_channels"]) == 19
    for frame in receipt["frames"]:
        assert abs(frame["target_plasma_current"]["value_a"]) >= 50_000.0
        assert "shipped" in frame["target_plasma_current"]["authority"]
        current = frame["circuit_driven"]["current_receipt"]
        assert current["complete_count"] == 24
        assert current["unknown_parameter_count"] == 0
        assert current["all_finite"] is True
        assert len(current["response_order"]) == 24
        assert len(current["conductors"]) == 24
        assert all(np.isfinite(item["value_a_turn"]) for item in current["conductors"])
        assert all(
            "label" not in item["authority"].lower() for item in current["conductors"]
        )
        assert frame["shipped_only"]["current_receipt"]["complete_count"] == 19


def test_receipt_carries_route_coverage_and_convergence_qualified_metrics() -> None:
    receipt = _receipt()

    assert receipt["aggregate"]["frame_count"] >= 5
    assert receipt["aggregate"]["shot_count"] >= 3
    assert (
        receipt["comparison"]["label_representability_ceiling_fractional_rms"] == 0.0429
    )
    assert (
        receipt["caveats"]["e89_systematic"][
            "E89UP_effective_gain_minus_integer_wiring"
        ]
        == 0.04569475694961733
    )
    for frame in receipt["frames"]:
        for arm in ("circuit_driven", "shipped_only"):
            routes = frame[arm]["routes"]
            assert set(routes) == set(study.ROUTE_NAMES)
            for route_name, solve in routes.items():
                assert isinstance(solve["converged"], bool)
                assert solve["iterations"] >= 0
                assert isinstance(solve["residual_trajectory"], list)
                assert solve["metrics"]["fractional_flux_rms"] is not None
                if solve["fixed_point_relative_residual"] is None:
                    assert solve["lambda_guard_triggered"] is True
                    assert solve["iterations"] == 0
                    assert solve["residual_trajectory"] == []
                if solve["converged"]:
                    assert solve["metrics"]["boundary_mean_separation_m"] is not None
                    assert solve["metrics"]["x_point_separation_m"] is not None
                else:
                    assert solve["metrics"]["boundary_mean_separation_m"] is None
                    assert solve["metrics"]["x_point_separation_m"] is None
                if route_name.endswith("_portfolio"):
                    assert set(solve["portfolio_branches"]) == {
                        "limited",
                        "diverted",
                    }
                    policy = solve["branch_selection"]["policy"]
                    assert policy["cold_start_class"] == "diverted"
                    assert solve["branch_selection"]["admissibility"] == {
                        "limited": False,
                        "diverted": True,
                    }
                else:
                    if solve["lambda_guard_triggered"]:
                        match = re.search(
                            r"profile amplitude ([^ ]+)", solve["termination"]
                        )
                        assert match is not None
                        termination_amplitude = float(match.group(1))
                        guard_value = solve["lambda_guard_value"]
                        assert (guard_value is None) == (
                            not np.isfinite(termination_amplitude)
                        )
                        if guard_value is not None:
                            assert guard_value == termination_amplitude
                    else:
                        assert solve["lambda_guard_value"] is None
        best_route = frame["circuit_driven"]["best_converged_route"]
        if best_route is not None:
            assert frame["circuit_driven"]["routes"][best_route]["converged"] is True

    failure_table = receipt["verdict"]["per_route_failure_table"]
    assert set(failure_table) == set(study.ROUTE_NAMES)
    for counts in failure_table.values():
        assert counts["circuit_converged_frames"] + counts["circuit_failed_frames"] == 5
        assert (
            counts["shipped_only_converged_frames"]
            + counts["shipped_only_failed_frames"]
            == 5
        )
    score_table = receipt["aggregate"]["qualified_route_score_table"]
    assert len(score_table) == 2 * len(study.ROUTE_NAMES)
    assert {(row["arm"], row["route"]) for row in score_table} == {
        (arm, route)
        for arm in ("circuit_driven", "shipped_only")
        for route in study.ROUTE_NAMES
    }
    figure = ROOT / receipt["artifacts"]["overlay_figure"]
    assert figure.is_file()
    assert figure.stat().st_size > 10_000


def test_contour_figures_reproduce_banked_converged_rows() -> None:
    sidecar = json.loads(CONTOUR_RECEIPT_PATH.read_text())
    receipt = _receipt()
    expected = {
        (frame["shot"], frame["frame"], route_name)
        for frame in receipt["frames"]
        for route_name, route in frame["circuit_driven"]["routes"].items()
        if route["converged"]
    }

    policy = sidecar["guard_policy"]
    assert policy["state_derived_science_metric_relative_tolerance"] == 1.0e-6
    assert policy["route_diagnostic_residual_relative_tolerance"] == 1.0e-2
    assert "5.3204e-4 relative" in policy["route_diagnostic_note"]
    assert "original banked validation run" in policy["route_diagnostic_note"]
    assert sidecar["all_banked_values_reproduced"] is True
    assert sidecar["figure_count"] == len(expected) == 3
    assert {
        (item["shot"], item["frame"], item["route"]) for item in sidecar["figures"]
    } == expected
    for item in sidecar["figures"]:
        levels = np.asarray(item["shared_contour_levels_wb"], dtype=float)
        assert len(levels) == 17
        assert np.all(np.isfinite(levels))
        assert np.all(np.diff(levels) > 0.0)
        assert "passed verbatim" in item["shared_level_identity"]
        for metric_name, comparison in item["banked_comparison"].items():
            expected_tolerance = (
                1.0e-2 if metric_name == "fixed_point_relative_residual" else 1.0e-6
            )
            assert comparison["relative_tolerance"] == expected_tolerance
            assert comparison["matched"] is True
            np.testing.assert_allclose(
                comparison["reproduced"],
                comparison["banked"],
                rtol=expected_tolerance,
                atol=0.0,
            )
        figure = ROOT / item["figure"]
        assert figure.is_file()
        assert figure.stat().st_size > 10_000
