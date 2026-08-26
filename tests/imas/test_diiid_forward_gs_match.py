import importlib.util
import inspect
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
    assert record["solver"]["route"] == "newton_krylov"
    assert record["solver"]["profile_evaluations"] == 180
    assert record["solver"]["newton_steps"] == 24
    assert record["solver"]["gmres_iterations"] == 24
    assert record["solver"]["warmup"] == 8
    assert record["solver"]["relaxation"] == pytest.approx(0.5)
    assert record["solver"]["step_cap"] == pytest.approx(10.0)
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


def test_valid_closed_branch_rows_feed_shared_metre_comparator_only():
    controls = np.array(
        [
            [[0.0, 0.0], [1 / 3, 0.0], [2 / 3, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1 / 3], [1.0, 2 / 3], [1.0, 1.0]],
            [[1.0, 1.0], [2 / 3, 1.0], [1 / 3, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 2 / 3], [0.0, 1 / 3], [0.0, 0.0]],
            np.full((4, 2), 1000.0),
        ]
    )
    closed = gate._sample_cubic_branch(
        controls, np.array([True, True, True, True, False])
    )
    reference = closed + np.array([0.002, 0.0])
    comparison = gate.compare_closed_boundaries(
        closed,
        reference,
        class_margin=0.25,
        reference_mode=gate.BoundaryMode.DIVERTED,
        predicted_saddle_rz_m=np.array([0.5, 0.0]),
        reference_x_points_rz_m=np.array([[0.503, 0.004]]),
    )

    assert comparison.symmetric_sup_distance_m == pytest.approx(0.002)
    assert comparison.symmetric_rms_distance_m == pytest.approx(np.sqrt(2.0e-6))
    assert comparison.x_point_distance_m == pytest.approx(0.005)
    assert comparison.topology_class_agreement is True
    assert comparison.failures == ()
    assert np.max(np.abs(closed)) == pytest.approx(1.0)


def test_scoring_uses_gradient_reference_x_and_margin_classification():
    source = inspect.getsource(gate.solve_frame)
    assert "boundary_gradient_minimum(" in source
    assert "reference_x_points_rz_m=labelled_x_point[None, :]" in source
    assert "class_margin=float(topology.class_margin)" in source
    assert "topology.diverted" not in source


def test_assembled_closed_and_open_plotting_geometry_stay_separate(monkeypatch):
    closed = np.array([[[0.0, 0.0]] * 4, [[1.0, 0.0]] * 4, [[0.0, 0.0]] * 4])
    open_controls = np.zeros((2, 3, 4, 2))
    open_controls[0, 0] = np.array([[2.0, 0.0], [1.5, 0.0], [1.0, 0.0], [0.5, 0.0]])
    monkeypatch.setattr(
        gate,
        "assemble_separatrix_branches",
        lambda *args, **kwargs: {
            "closed_controls_rz": closed,
            "closed_valid": np.array([True, True, False]),
            "open_controls_rz": open_controls,
            "open_valid": np.array([[True, False, False], [False, False, False]]),
            "open_branch_valid": np.array([True, False]),
        },
    )

    sampled_closed, sampled_open = gate._assembled_boundary_geometry(
        np.zeros((3, 3)),
        np.arange(3.0),
        np.arange(3.0),
        0.0,
        np.array([0.5, 0.5]),
    )

    assert len(sampled_closed) == 17
    assert len(sampled_open) == 1
    assert len(sampled_open[0]) == 9
    assert np.max(sampled_open[0][:, 0]) == pytest.approx(2.0)


def _frame(*, r_squared: float, converged: bool, expansion: float = 0.02):
    metrics = gate.MatchMetrics(
        interior_r_squared=r_squared,
        interior_fractional_rms=np.sqrt(max(0.0, 1.0 - r_squared)),
        additive_gauge_wb=0.0,
        closed_boundary_symmetric_sup_distance_m=0.004,
        closed_boundary_symmetric_rms_distance_m=0.002,
        polished_saddle_to_nearest_efit_x_m=0.001,
        topology_class_agreement=True,
        boundary_comparison_failures=(),
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
        achieved_topology_class="diverted",
        converged=converged,
        convergence_criterion="declared criterion",
        solver_termination="returned within budget",
        residual_history=(1.0e-3, 1.0e-5, 1.0e-7),
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
