import dataclasses
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "diiid_forward_gs_match.py"
SPEC = importlib.util.spec_from_file_location(
    "diiid_frame_checkpoint_material", MODULE_PATH
)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _frame(index: int) -> gate.FrameResult:
    metrics = gate.MatchMetrics(
        interior_r_squared=0.5 + 0.01 * index,
        interior_fractional_rms=0.25,
        additive_gauge_wb=0.0,
        closed_boundary_symmetric_sup_distance_m=0.02,
        closed_boundary_symmetric_rms_distance_m=0.01,
        polished_saddle_to_nearest_efit_x_m=0.03,
        topology_class_agreement=True,
        boundary_comparison_failures=(),
        magnetic_axis_displacement_mm=1.5,
        predicted_q95_nova=4.0,
        labelled_q95_nova=4.2,
        signed_relative_q95_error=-0.05,
    )
    return gate.FrameResult(
        shot=f"shot-{index}.parquet",
        frame=index,
        time_ms=100.0 + index,
        geometry_digest=f"geometry-{index}",
        reliable_flux_surfaces=1,
        pseudo_wall_expansion=gate.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
        pseudo_wall_statement="physical wall",
        fixed_point_relative_residual=1.0e-7,
        residual_tolerance=gate.GATE_RESIDUAL_TOLERANCE,
        finite=True,
        achieved_topology_class="diverted",
        converged=True,
        convergence_criterion="finite and converged",
        solver_termination="converged",
        residual_history=(1.0, 1.0e-7),
        metrics=metrics,
        iterations=4,
        active_set_iterations=3,
        active_set_residuals=(1.0, 0.1, 1.0e-7),
        active_set_mask_differences=(4, 2, 0),
        active_set_cycle_damping_activations=(0, 1, 0),
    )


def _strict_load(path: Path):
    def reject(value):
        raise ValueError(f"non-finite constant: {value}")

    return json.loads(path.read_text(), parse_constant=reject)


def _install_fixture(monkeypatch):
    monkeypatch.setattr(gate, "_read", lambda path, columns: {})
    monkeypatch.setattr(
        gate,
        "_solve_frame_retaining_failure",
        lambda row, frame, expansion: (_frame(frame), {"frame": frame}),
    )
    monkeypatch.setattr(gate.jax, "clear_caches", lambda: None)
    monkeypatch.setattr(gate.gc, "collect", lambda: None)


def test_each_completed_frame_is_atomically_checkpointed(monkeypatch, tmp_path):
    _install_fixture(monkeypatch)
    selected = [
        SimpleNamespace(path=Path(f"shot-{index}.parquet"), frame=index)
        for index in range(2)
    ]
    checkpoint = tmp_path / gate.PARTIAL_RECEIPT_NAME
    snapshots = []
    replacements = []
    replace = os.replace

    def observed_replace(source, target):
        source = Path(source)
        target = Path(target)
        assert source.parent == target.parent == tmp_path
        if target.exists():
            replacements.append(_strict_load(target))
        replace(source, target)

    monkeypatch.setattr(gate.os, "replace", observed_replace)

    def write_checkpoint(results):
        gate._write_frame_checkpoint(checkpoint, results, len(selected), "registered")
        snapshots.append(_strict_load(checkpoint))

    results, _fields = gate._solve_selected_frames(
        selected,
        gate.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
        on_frame_completed=write_checkpoint,
    )

    first = snapshots[0]
    assert len(results) == 2
    assert first["receipt_kind"] == "partial_frame_checkpoint"
    assert first["cohort_status"] == {
        "complete": False,
        "declared_frame_count": 2,
        "attempted_frame_count": 1,
        "unattempted_frame_count": 1,
        "unattempted_row_policy": (
            "rows not yet attempted are absent rather than fabricated"
        ),
    }
    assert len(first["frame_records"]) == 1
    assert set(first["frame_records"][0]) == set(dataclasses.asdict(results[0]))
    assert first["frame_records"][0]["active_set_residuals"] == [1.0, 0.1, 1.0e-7]
    assert first["frame_records"][0]["active_set_mask_differences"] == [4, 2, 0]
    assert first["frame_records"][0]["active_set_cycle_damping_activations"] == [
        0,
        1,
        0,
    ]
    assert len(replacements) == 1
    assert replacements[0] == first
    assert snapshots[-1]["cohort_status"]["attempted_frame_count"] == 2
    assert not list(tmp_path.glob(f".{gate.PARTIAL_RECEIPT_NAME}.*.tmp"))


def test_checkpointing_does_not_change_complete_receipt_schema(monkeypatch, tmp_path):
    _install_fixture(monkeypatch)
    selected = [
        SimpleNamespace(path=Path(f"shot-{index}.parquet"), frame=index)
        for index in range(gate.EXECUTION_FRAME_COUNT)
    ]
    baseline_results, baseline_fields = gate._solve_selected_frames(
        selected, gate.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION
    )
    checkpoint = tmp_path / gate.PARTIAL_RECEIPT_NAME
    checkpointed_results, checkpointed_fields = gate._solve_selected_frames(
        selected,
        gate.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
        on_frame_completed=lambda results: gate._write_frame_checkpoint(
            checkpoint, results, len(selected), "registered"
        ),
    )
    sensitivity = [
        baseline_results[0],
        dataclasses.replace(
            baseline_results[0],
            pseudo_wall_expansion=gate.PSEUDO_WALL_EXPANSIONS[1],
        ),
    ]
    baseline_receipt = gate._strict_json_value(
        {"result": gate.summarize(baseline_results, sensitivity, "registered")}
    )
    checkpointed_receipt = gate._strict_json_value(
        {"result": gate.summarize(checkpointed_results, sensitivity, "registered")}
    )
    checkpoint_relative = Path(os.path.relpath(checkpoint, Path.cwd())).as_posix()
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"?? {checkpoint_relative}\n M caller-owned.txt\n"
        ),
    )

    assert gate._strict_json_value(
        [dataclasses.asdict(result) for result in checkpointed_results]
    ) == gate._strict_json_value(
        [dataclasses.asdict(result) for result in baseline_results]
    )
    assert checkpointed_fields == baseline_fields
    assert checkpointed_receipt == baseline_receipt
    assert set(checkpointed_receipt) == {"result"}
    assert "cohort_status" not in checkpointed_receipt["result"]
    assert "receipt_kind" not in checkpointed_receipt["result"]
    assert gate._worktree_status_without_checkpoint(checkpoint) == [
        " M caller-owned.txt"
    ]
    json.dumps(checkpointed_receipt, allow_nan=False)
