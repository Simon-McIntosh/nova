import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "diiid_forward_gs_match.py"
SPEC = importlib.util.spec_from_file_location(
    "diiid_forward_gs_match_material", MODULE_PATH
)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def test_publication_artifact_records_are_stable_and_strict(tmp_path):
    receipt = {}

    publication_artifacts = gate._record_publication_artifacts(receipt, tmp_path)

    assert publication_artifacts == {
        "poloidal_frame_comparison": {
            "filesystem_path": str(tmp_path / "frame_flux_comparison.png"),
            "publication_path": (
                "/nova/figures/diiid-forward-onboarding/forward-gs/"
                "frame_flux_comparison.png"
            ),
        },
        "cohort_match_summary": {
            "filesystem_path": str(tmp_path / "cohort_match_summary.png"),
            "publication_path": (
                "/nova/figures/diiid-forward-onboarding/forward-gs/"
                "cohort_match_summary.png"
            ),
        },
        "strict_forward_match_receipt": {
            "filesystem_path": str(tmp_path / "forward_gs_receipt.json"),
            "publication_path": (
                "/nova/figures/diiid-forward-onboarding/forward-gs/"
                "forward_gs_receipt.json"
            ),
        },
    }
    assert all(
        Path(artifact["publication_path"]).is_absolute()
        for artifact in publication_artifacts.values()
    )
    assert receipt == {"publication_artifacts": publication_artifacts}
    json.dumps(receipt, allow_nan=False)


def _diagnostic(class_margin):
    return {
        "class_margin": jnp.asarray(class_margin),
        "axis_flux": jnp.asarray(0.0),
        "outward_flux_span": jnp.asarray(1.0),
        "typed_candidates": jnp.zeros((1, 4)),
        "typed_candidate_present": jnp.zeros(1, dtype=bool),
        "selected_typed_candidate_index": jnp.asarray(0),
        "connectivity_candidates": jnp.zeros((1, 4)),
        "connectivity_candidate_present": jnp.zeros(1, dtype=bool),
        "connectivity_candidate_admitted": jnp.zeros(1, dtype=bool),
        "connectivity_candidate_resolved": jnp.zeros(1, dtype=bool),
        "connectivity_candidate_state": jnp.zeros(1, dtype=int),
        "connectivity_candidate_confidence": jnp.zeros(1),
        "connectivity_candidate_class_margin": jnp.zeros(1),
        "connectivity_candidate_boundary_snr": jnp.zeros(1),
        "connectivity_candidate_root_support_cell": jnp.zeros(1),
        "selected_typed_candidate": jnp.zeros(4),
        "selected_typed_candidate_present": jnp.asarray(False),
        "selected_x_normalized_flux_operand": jnp.asarray(jnp.nan),
        "wall_candidate": jnp.zeros(3),
        "wall_candidate_present": jnp.asarray(False),
        "wall_normalized_flux_operand_before_shadow": jnp.asarray(jnp.nan),
        "wall_normalized_flux_operand": jnp.asarray(jnp.nan),
        "wall_shadowed": jnp.asarray(False),
        "typed_candidate_count": jnp.asarray(0),
        "connectivity_admitted_slot_count": jnp.asarray(0),
        "connectivity_candidate_count_before_capacity": jnp.asarray(0),
        "connectivity_candidate_overflow": jnp.asarray(False),
        "connectivity_discarded_score_upper_bound": jnp.asarray(jnp.nan),
    }


def _inputs():
    coordinate = np.array([[1.0, -1.0], [1.0, 1.0], [2.0, -1.0], [2.0, 1.0]])
    repaired_material = jnp.array([True, False, True, True])
    axis = jnp.array([1.5, 0.0])
    calls = []

    def connectivity_axis_seed(received_axis):
        calls.append(received_axis)
        return jnp.asarray(2), repaired_material

    operator = SimpleNamespace(
        physical_node_number=6,
        grid=SimpleNamespace(coordinate=coordinate),
        wall=SimpleNamespace(coordinate=jnp.array([[0.8, -1.0], [0.8, 1.0]])),
        topology=SimpleNamespace(
            split_flux_map=lambda physical: (physical[:4], physical[4:])
        ),
        _fixed_design_topology=SimpleNamespace(
            grid=lambda grid_flux: (jnp.empty((0, 4)), jnp.empty((0, 4)))
        ),
        inside_material=jnp.array([False, False, False, False]),
        connectivity_axis_seed=connectivity_axis_seed,
    )
    topology = SimpleNamespace(
        axis=axis,
        wall_point=jnp.array([0.8, 0.0]),
        wall_point_flux=jnp.asarray(0.5),
        class_margin=0.25,
    )
    profile = SimpleNamespace(operator=operator)
    state = jnp.arange(6.0)
    return profile, state, topology, repaired_material, calls


def test_terminal_diagnostic_uses_axis_seeded_connectivity_material(monkeypatch):
    profile, state, topology, repaired_material, calls = _inputs()
    captured = {}

    def diagnostics(*args):
        captured["material"] = args[3]
        return _diagnostic(topology.class_margin)

    monkeypatch.setattr(gate, "traced_margin_candidate_diagnostics", diagnostics)

    serialized = gate._terminal_xpoint_diagnostics(profile, state, topology)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], topology.axis)
    expected_material = repaired_material.reshape(2, 2).T
    np.testing.assert_array_equal(captured["material"], expected_material)
    assert serialized["class_margin_from_operands"] == topology.class_margin


def test_terminal_diagnostic_refuses_any_class_margin_mismatch(monkeypatch):
    profile, state, topology, _repaired_material, _calls = _inputs()
    changed_margin = topology.class_margin + 0.25
    monkeypatch.setattr(
        gate,
        "traced_margin_candidate_diagnostics",
        lambda *args: _diagnostic(changed_margin),
    )

    with pytest.raises(RuntimeError, match="class-margin operand"):
        gate._terminal_xpoint_diagnostics(profile, state, topology)


def _failed_frame(frame, *, finite_target=False):
    metrics = gate.MatchMetrics(
        interior_r_squared=0.5,
        interior_fractional_rms=0.25,
        additive_gauge_wb=float("nan"),
        closed_boundary_symmetric_sup_distance_m=None,
        closed_boundary_symmetric_rms_distance_m=None,
        polished_saddle_to_nearest_efit_x_m=None,
        topology_class_agreement=False,
        boundary_comparison_failures=("missing_predicted_closed_boundary",),
        magnetic_axis_displacement_mm=1.5,
        predicted_q95_nova=4.0 if finite_target else float("nan"),
        labelled_q95_nova=4.2,
        signed_relative_q95_error=-0.05,
    )
    return gate.FrameResult(
        shot="qualified-failure.parquet",
        frame=frame,
        time_ms=100.0 + frame,
        geometry_digest="geometry-digest",
        reliable_flux_surfaces=0,
        pseudo_wall_expansion=gate.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
        pseudo_wall_statement="physical wall",
        fixed_point_relative_residual=float("inf"),
        residual_tolerance=gate.GATE_RESIDUAL_TOLERANCE,
        finite=False,
        achieved_topology_class=None,
        converged=False,
        convergence_criterion="finite and converged",
        solver_termination="nonconverged within budget",
        residual_history=(1.0, float("nan"), float("inf")),
        iterations=24,
        target_current_a=200_000.0 if finite_target else float("nan"),
        achieved_current_a=float("-inf"),
        metrics=metrics,
    )


def _nonfinite_paths(value, path="receipt"):
    if isinstance(value, dict):
        return [
            nested
            for key, item in value.items()
            for nested in _nonfinite_paths(item, f"{path}.{key}")
        ]
    if isinstance(value, list | tuple):
        return [
            nested
            for index, item in enumerate(value)
            for nested in _nonfinite_paths(item, f"{path}[{index}]")
        ]
    if isinstance(value, float | np.floating) and not np.isfinite(value):
        return [path]
    return []


def test_failed_receipt_normalization_is_strict_and_lossless():
    frames = [_failed_frame(index, finite_target=index == 0) for index in range(5)]
    sensitivity = [
        _failed_frame(20, finite_target=True),
        gate.FrameResult(
            **{
                **frames[0].__dict__,
                "frame": 21,
                "pseudo_wall_expansion": 0.05,
            }
        ),
    ]
    preregistration_hash = "sha256:unchanged-preregistration"
    unnormalized = {
        "preregistration": {"sha256": preregistration_hash},
        "result": gate.summarize(frames, sensitivity, preregistration_hash),
    }

    assert _nonfinite_paths(unnormalized)
    receipt = gate._strict_json_value(unnormalized)

    json.dumps(receipt, allow_nan=False)
    assert _nonfinite_paths(receipt) == []
    assert len(receipt["result"]["frame_records"]) == 5
    assert len(receipt["result"]["per_frame_gate"]) == 5
    assert all(row["verdict"] == "FAIL" for row in receipt["result"]["per_frame_gate"])
    assert all(
        row["metrics"]["boundary_comparison_failures"]
        == ["missing_predicted_closed_boundary"]
        for row in receipt["result"]["frame_records"]
    )
    first = receipt["result"]["frame_records"][0]
    assert first["target_current_a"] == 200_000.0
    assert first["metrics"]["predicted_q95_nova"] == 4.0
    assert first["fixed_point_relative_residual"] is None
    assert first["achieved_current_a"] is None
    assert first["residual_history"] == [1.0, None, None]
    assert receipt["result"]["registered_median_interior_r_squared_bar"] == (
        gate.REGISTERED_MEDIAN_INTERIOR_R2_BAR
    )
    assert receipt["result"]["preregistration_sha256"] == preregistration_hash
    assert receipt["preregistration"]["sha256"] == preregistration_hash
    assert receipt["result"]["passed"] is False


@pytest.mark.parametrize(
    "exception_type",
    (gate.NoQualifiedAxisError, gate.ConstraintViolationError),
)
def test_named_solve_failure_retains_all_frame_identities(monkeypatch, exception_type):
    selected = [
        SimpleNamespace(path=Path(f"shot-{index}.parquet"), frame=index)
        for index in range(gate.EXECUTION_FRAME_COUNT)
    ]
    calls = []

    def synthetic_solve(row, frame, expansion):
        calls.append(frame)
        if frame == 2:
            raise exception_type("synthetic qualification failure")
        result = _failed_frame(frame, finite_target=True)
        metrics = gate.MatchMetrics(
            **{
                **result.metrics.__dict__,
                "topology_class_agreement": True,
                "boundary_comparison_failures": (),
            }
        )
        return (
            gate.FrameResult(
                **{
                    **result.__dict__,
                    "shot": Path(row["_source_path"]).name,
                    "finite": True,
                    "achieved_topology_class": "diverted",
                    "converged": True,
                    "metrics": metrics,
                }
            ),
            {"radius": np.ones(1)},
        )

    monkeypatch.setattr(gate, "solve_frame", synthetic_solve)
    monkeypatch.setattr(gate, "geometry_digest", lambda row: "geometry-digest")

    results = []
    fields = []
    for item in selected:
        row = {
            "_source_path": str(item.path),
            "efit_times": np.arange(gate.EXECUTION_FRAME_COUNT, dtype=float),
        }
        result, frame_fields = gate._solve_frame_retaining_failure(
            row, item.frame, gate.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION
        )
        results.append(result)
        fields.append(frame_fields)

    sensitivity = [
        results[0],
        gate.FrameResult(
            **{
                **results[0].__dict__,
                "pseudo_wall_expansion": gate.PSEUDO_WALL_EXPANSIONS[1],
            }
        ),
    ]
    receipt = gate._strict_json_value(
        gate.summarize(results, sensitivity, "sha256:synthetic")
    )
    receipt_rows = receipt["frame_records"]
    failed = receipt_rows[2]

    assert calls == list(range(gate.EXECUTION_FRAME_COUNT))
    assert [(row["shot"], row["frame"]) for row in receipt_rows] == [
        (item.path.name, item.frame) for item in selected
    ]
    assert failed["converged"] is False
    assert failed["solve_exception_class"] == exception_type.__name__
    assert failed["fixed_point_relative_residual"] is None
    assert failed["metrics"]["interior_r_squared"] is None
    assert failed["metrics"]["topology_class_agreement"] is None
    assert receipt["per_frame_gate"][2]["verdict"] == "FAIL"
    assert receipt["per_frame_gate"][2]["solve_exception_class"] == (
        exception_type.__name__
    )
    assert fields[2] == {"plot_unavailable_reason": exception_type.__name__}


def test_frame_solves_release_compilation_state_between_identities(monkeypatch):
    selected = [
        SimpleNamespace(path=Path(f"shot-{index}.parquet"), frame=index)
        for index in range(gate.EXECUTION_FRAME_COUNT)
    ]
    events = []

    def read_frame(path, _columns):
        events.append(("read", path.name))
        return {}

    def solve_frame(row, frame, expansion):
        events.append(("solve", frame, expansion, row["_source_path"]))
        return _failed_frame(frame, finite_target=True), {"frame": frame}

    monkeypatch.setattr(gate, "_read", read_frame)
    monkeypatch.setattr(gate, "_solve_frame_retaining_failure", solve_frame)
    monkeypatch.setattr(gate.jax, "clear_caches", lambda: events.append(("clear",)))
    monkeypatch.setattr(gate.gc, "collect", lambda: events.append(("collect",)))

    results, fields = gate._solve_selected_frames(
        selected, gate.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION
    )

    assert [result.frame for result in results] == list(
        range(gate.EXECUTION_FRAME_COUNT)
    )
    assert fields == [{"frame": index} for index in range(gate.EXECUTION_FRAME_COUNT)]
    expected_events = []
    for index in range(gate.EXECUTION_FRAME_COUNT):
        expected_events.extend(
            (
                ("read", f"shot-{index}.parquet"),
                (
                    "solve",
                    index,
                    gate.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
                    str(selected[index].path),
                ),
                ("clear",),
                ("collect",),
            )
        )
    assert events == expected_events
