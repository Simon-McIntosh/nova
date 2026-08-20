import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "diiid_forward_gs_match.py"
SPEC = importlib.util.spec_from_file_location("diiid_forward_gs_match", MODULE_PATH)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def test_registered_bar_is_tied_to_the_measured_label_ceiling(tmp_path):
    path = gate.write_preregistration(tmp_path)
    record = json.loads(path.read_text())
    basis = record["score"]["bar_basis"]
    assert basis["discretisation_floor_used"] is False
    assert basis["strict_gs_residual_attributed_to_irreducible_non_gs_content"] == (
        0.9968
    )
    assert record["score"]["registered_median_interior_r_squared_bar"] == (
        pytest.approx(0.95 * 0.949)
    )
    assert record["score"]["coefficients_fitted"] == 0
    assert record["selection"]["frames"] == 3
    assert record["solver"]["route"] == "host"
    assert record["solver"]["solver_tolerance"] == pytest.approx(1.0e-8)
    assert record["solver"]["maximum_evaluations"] == 1701
    assert record["solver"]["initial_relaxation"] == pytest.approx(0.2)
    assert record["solver"]["minimum_relaxation"] == pytest.approx(1.0e-6)
    assert record["solver"]["relaxation_reduction_interval"] == 100
    assert record["solver"]["relative_residual_tolerance"] == pytest.approx(1.0e-5)
    assert "144-turn assumption" in record["q95"]["assumption_scope"]
    assert "do not depend" in record["q95"]["assumption_scope"]
    assert gate.require_preregistration(path)


def test_scoring_refuses_a_changed_or_missing_preregistration(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="preregistered"):
        gate.require_preregistration(missing)
    path = gate.write_preregistration(tmp_path)
    record = json.loads(path.read_text())
    record["score"]["registered_median_interior_r_squared_bar"] = 0.0
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="does not match"):
        gate.require_preregistration(path)


def test_pseudo_wall_encloses_the_released_grid_and_expands_outward():
    radius = np.linspace(0.84, 2.54, 65)
    height = np.linspace(-1.6, 1.6, 65)
    baseline = gate.pseudo_wall(radius, height, 0.0, points_per_side=8)
    expanded = gate.pseudo_wall(radius, height, 0.05, points_per_side=8)
    assert baseline.shape == (32, 2)
    assert baseline[:, 0].min() == pytest.approx(radius[0])
    assert baseline[:, 0].max() == pytest.approx(radius[-1])
    assert baseline[:, 1].min() == pytest.approx(height[0])
    assert baseline[:, 1].max() == pytest.approx(height[-1])
    assert expanded[:, 0].min() < baseline[:, 0].min()
    assert expanded[:, 0].max() > baseline[:, 0].max()
    assert expanded[:, 1].min() < baseline[:, 1].min()
    assert expanded[:, 1].max() > baseline[:, 1].max()


def test_frame_selection_uses_diverted_finite_labels_without_score_filtering():
    row = {
        "efit_times": [0.0, 1.0, 2.0, 3.0, 4.0],
        "efit_q95": [4.0] * 5,
        "efit_lcfs_n": [12] * 5,
        "magnetics_dsep_times": [0.0, 1.0, 2.0, 3.0, 4.0],
        "magnetics_dsep": [-0.1, 0.1, 0.2, 0.3, -0.2],
        "efit_psirz": [np.ones((3, 3)).tolist() for _ in range(5)],
    }
    assert gate._eligible_frame(row) == 2
    row["efit_psirz"][2][1][1] = float("nan")
    assert gate._eligible_frame(row) == 1


def test_flux_score_removes_only_the_additive_gauge():
    labelled = np.array([[1.0, 2.0], [4.0, 8.0]])
    predicted = labelled - 7.0
    interior = np.ones_like(labelled, dtype=bool)
    r_squared, fractional_rms, gauge, aligned = gate.gauge_metrics(
        labelled, predicted, interior
    )
    assert r_squared == pytest.approx(1.0)
    assert fractional_rms == pytest.approx(0.0)
    assert gauge == pytest.approx(7.0)
    np.testing.assert_allclose(aligned, labelled)


def test_contour_separation_is_symmetric_and_reported_in_millimetres():
    labelled = np.array([[1.0, 0.0], [1.0, 1.0]])
    predicted = labelled + np.array([0.002, 0.0])
    mean, maximum = gate.contour_separation(predicted, labelled)
    assert mean == pytest.approx(2.0)
    assert maximum == pytest.approx(2.0)


def _frame(*, r_squared: float, converged: bool, expansion: float = 0.02):
    metrics = gate.MatchMetrics(
        interior_r_squared=r_squared,
        interior_fractional_rms=np.sqrt(max(0.0, 1.0 - r_squared)),
        additive_gauge_wb=0.0,
        separatrix_mean_radial_separation_mm=2.0,
        separatrix_maximum_radial_separation_mm=4.0,
        magnetic_axis_displacement_mm=1.0,
        predicted_q95_nova=-4.2,
        labelled_q95_nova=-4.0,
        signed_relative_q95_error=-0.05,
    )
    return gate.FrameResult(
        shot="shot.parquet",
        frame=4,
        time_ms=100.0,
        geometry_digest="digest",
        reliable_flux_surfaces=19,
        pseudo_wall_expansion=expansion,
        pseudo_wall_statement="pseudo-wall",
        fixed_point_relative_residual=1.0e-7 if converged else 1.0e-3,
        residual_tolerance=gate.REGISTERED_RESIDUAL_TOLERANCE,
        finite=True,
        diverted=True,
        converged=converged,
        convergence_criterion="declared criterion",
        solver_termination="returned within budget",
        metrics=metrics,
    )


def test_summary_keeps_nonconvergence_visible_and_fail_closed():
    results = [_frame(r_squared=0.95, converged=True) for _ in range(2)]
    results.append(_frame(r_squared=0.95, converged=False))
    sensitivity = [
        _frame(r_squared=0.950, converged=True, expansion=0.02),
        _frame(r_squared=0.948, converged=True, expansion=0.05),
    ]
    summary = gate.summarize(results, sensitivity, "hash")
    assert summary["convergence"]["converged_frames"] == 2
    assert summary["convergence"]["nonconverged_frames"] == [
        {"shot": "shot.parquet", "frame": 4}
    ]
    assert summary["pseudo_wall"]["maximum_absolute_r_squared_move"] == (
        pytest.approx(0.002)
    )
    assert "144-turn" in summary["metric_assumptions"]["q95"]
    assert summary["passed"] is False
