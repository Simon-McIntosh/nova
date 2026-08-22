"""Contracts for the MAST same-shot warm-neighbour stall-lift measure."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from benchmarks import diiid_constrained_cold_start as cold_start
from benchmarks import efit_parity_warm_neighbour as warm_neighbour


def _frame(*, row: int = 50, count: int = 100) -> warm_neighbour._MastFrame:
    selected = warm_neighbour._MastSelection(
        frame=row,
        recorded_plasma_current_a=9.0e5,
        path=Path("/shot.zarr"),
    )
    return warm_neighbour._MastFrame(
        selected=selected,
        row={"efit_times": np.arange(count, dtype=float)},
        profile=object(),
        current=np.zeros(3),
        seed=np.zeros(3),
    )


def _metrics_stub() -> dict[str, Any]:
    return {
        "flux_map": {
            "sup_fraction_of_reference_span": 0.01,
            "rms_fraction_of_reference_span": 0.005,
        },
        "magnetic_axis": {
            "distance_m": 0.01,
            "registered_bound_m": 0.1,
            "passes": True,
        },
        "lcfs": {
            "symmetric_mean_distance_m": 0.02,
            "registered_bound_m": 0.2,
            "passes": True,
        },
        "x_point": {
            "distance_m": 0.03,
            "registered_bound_m": 0.3,
            "passes": True,
        },
        "topology": {
            "agreement": True,
            "registered_bound": True,
            "passes": True,
        },
        "plasma_current": {"signed_relative_deviation": 0.0},
        "poloidal_beta": {"signed_relative_deviation": 0.0},
        "internal_inductance": {"signed_relative_deviation": 0.0},
    }


def _record(
    *,
    outcome_class: str,
    residual: float,
    target_current: float = 9.0e5,
) -> dict[str, Any]:
    converged = outcome_class == "converged_plasma_root"
    nonzero = outcome_class != "vacuum_collapse"
    current = target_current if nonzero else 0.0
    return {
        "forward_branch_receipt": {
            "converged": converged,
            "residual": residual,
            "achieved_class": "diverted",
            "topology_consistent": True,
        },
        "terminal_state": {
            "plasma_current_a": current,
            "nonzero_current": nonzero,
            "profile_amplitude": 1.0,
        },
        "registered_parity_metrics": _metrics_stub(),
        "residual_trajectory": [residual],
        "iterations": 3,
        "termination": "test terminal",
        "outcome_class": outcome_class,
        "target_current_relative_error": 0.0 if nonzero else 1.0,
    }


def test_offset_ladder_is_the_imported_diiid_constant() -> None:
    """The declared ladder is reused, not redeclared or retuned."""
    assert warm_neighbour.NEIGHBOUR_FRAME_OFFSETS is cold_start.NEIGHBOUR_FRAME_OFFSETS
    assert warm_neighbour.NEIGHBOUR_FRAME_OFFSETS == (
        -1,
        1,
        -2,
        2,
        -4,
        4,
        -8,
        8,
        -16,
        16,
        -32,
        32,
    )


def test_shared_candidate_and_solve_helpers_are_imported_unchanged() -> None:
    assert warm_neighbour._neighbour_candidates is cold_start._neighbour_candidates
    assert warm_neighbour._solve_public_seam is cold_start._solve_public_seam


def test_mast_frame_exposes_exact_shared_helper_surface() -> None:
    assert [field.name for field in fields(warm_neighbour._MastSelection)] == [
        "frame",
        "recorded_plasma_current_a",
        "path",
    ]
    assert [field.name for field in fields(warm_neighbour._MastFrame)] == [
        "selected",
        "row",
        "profile",
        "current",
        "seed",
    ]


def test_existing_measurement_retains_partial_references(tmp_path: Path) -> None:
    measured = [{"shot": 21978, "slice_index": 35}]
    references = [{"reference": measured[0]}]
    receipt = {
        "aggregate": {"measured_references": measured},
        "references": references,
    }
    path = tmp_path / warm_neighbour.RECEIPT_NAME
    path.write_text(json.dumps(receipt))

    loaded, segments = warm_neighbour._existing_measurement(tmp_path, resume=True)

    assert loaded == references
    assert segments[0]["measured_references"] == measured
    assert len(segments[0]["source_receipt_sha256"]) == 64


def test_candidate_rows_are_earlier_offsets_first_and_in_bounds() -> None:
    frame = _frame(row=35, count=40)
    rows = warm_neighbour._candidate_rows(frame)
    expected = [
        35 + offset
        for offset in warm_neighbour.NEIGHBOUR_FRAME_OFFSETS
        if 0 <= 35 + offset < 40
    ]
    assert rows == expected
    assert 39 in rows
    assert all(row < 40 for row in rows)


@pytest.mark.parametrize(
    ("residual", "achieved_current", "diverted", "expected"),
    [
        (1.0e-9, 9.0e5, True, "converged_plasma_root"),
        (1.0e-9, 0.0, True, "vacuum_collapse"),
        (1.0e-3, 9.0e5, True, "bounded_non_convergence"),
    ],
)
def test_record_outcome_classification(
    monkeypatch,
    residual: float,
    achieved_current: float,
    diverted: bool,
    expected: str,
) -> None:
    operator = SimpleNamespace(
        read=lambda state: (None, SimpleNamespace(diverted=diverted))
    )
    profile = SimpleNamespace(
        operator=operator,
        observe=lambda state, current, target_current: object(),
    )
    frame = warm_neighbour._MastFrame(
        selected=warm_neighbour._MastSelection(
            frame=0,
            recorded_plasma_current_a=9.0e5,
            path=Path("/shot.zarr"),
        ),
        row={"efit_times": [0.0]},
        profile=profile,
        current=np.zeros(3),
        seed=np.zeros(3),
    )
    outcome = SimpleNamespace(
        state=np.ones(3),
        residual=residual,
        achieved_current_a=achieved_current,
        amplitude=1.0,
        residual_trajectory=(),
        iterations=2,
        termination="test",
    )
    monkeypatch.setattr(
        warm_neighbour, "_pinned_metrics", lambda *args: _metrics_stub()
    )
    record, _equilibrium = warm_neighbour._record_outcome(
        frame, {"group": object(), "row": 0, "reference_flux": np.zeros(3)}, outcome
    )
    assert record["outcome_class"] == expected
    assert record["registered_parity_metrics"] == _metrics_stub()


def test_local_warm_walk_stops_at_first_converged_candidate(monkeypatch) -> None:
    target = _frame()
    call_order: list[int] = []

    def prepare_side_effect(store, shot, row, cache_box):
        call_order.append(row)
        frame = _frame(row=row)
        mast_case = {"reference": {"time_s": float(row)}}
        return frame, mast_case, {}

    def solve_side_effect(frame, seed):
        return SimpleNamespace(state=np.full(3, frame.selected.frame))

    def record_side_effect(frame, context, outcome):
        converged = frame.selected.frame == 48
        outcome_class = (
            "converged_plasma_root" if converged else "bounded_non_convergence"
        )
        return _record(outcome_class=outcome_class, residual=1.0e-3), object()

    monkeypatch.setattr(warm_neighbour, "_prepare_frame", prepare_side_effect)
    monkeypatch.setattr(warm_neighbour, "_solve_public_seam", solve_side_effect)
    monkeypatch.setattr(warm_neighbour, "_record_outcome", record_side_effect)

    checks, source = warm_neighbour._find_mast_warm_source(
        Path("/unused"), 1, target, [None]
    )

    assert call_order == [49, 51, 48]
    assert [check["row"] for check in checks] == call_order
    assert source is not None
    source_row, _frame_value, _record_value, source_outcome = source
    assert source_row == 48
    np.testing.assert_array_equal(source_outcome.state, np.full(3, 48))


def test_local_warm_walk_reports_every_tried_row_without_a_source(
    monkeypatch,
) -> None:
    target = _frame()

    def prepare_side_effect(store, shot, row, cache_box):
        return _frame(row=row), {"reference": {"time_s": float(row)}}, {}

    monkeypatch.setattr(warm_neighbour, "_prepare_frame", prepare_side_effect)
    monkeypatch.setattr(
        warm_neighbour,
        "_solve_public_seam",
        lambda frame, seed: SimpleNamespace(state=seed),
    )
    monkeypatch.setattr(
        warm_neighbour,
        "_record_outcome",
        lambda frame, context, outcome: (
            _record(
                outcome_class="bounded_non_convergence",
                residual=1.0e-3,
            ),
            object(),
        ),
    )

    checks, source = warm_neighbour._find_mast_warm_source(
        Path("/unused"), 1, target, [None]
    )

    assert source is None
    assert [check["row"] for check in checks] == warm_neighbour._candidate_rows(target)


def test_measure_reference_skips_warm_search_after_cold_convergence(
    monkeypatch,
) -> None:
    frame = _frame(row=43)
    mast_case = {"reference": {"time_s": 3.0}}
    cold_record = _record(
        outcome_class="converged_plasma_root",
        residual=2.9e-16,
    )
    monkeypatch.setattr(
        warm_neighbour,
        "_prepare_frame",
        lambda store, shot, row, cache_box: (frame, mast_case, {}),
    )
    monkeypatch.setattr(
        warm_neighbour,
        "_solve_public_seam",
        lambda frame, seed: SimpleNamespace(kind="cold"),
    )
    monkeypatch.setattr(
        warm_neighbour,
        "_record_outcome",
        lambda frame, context, outcome: (cold_record, object()),
    )
    monkeypatch.setattr(
        warm_neighbour,
        "_metric_qualification",
        lambda metrics, residual: {"all": True},
    )

    def fail_walk(*args, **kwargs):
        raise AssertionError("warm search must not run after cold convergence")

    monkeypatch.setattr(warm_neighbour, "_find_mast_warm_source", fail_walk)
    reference = warm_neighbour.measure_reference(
        Path("/unused"),
        shot=22086,
        row=43,
        cache_box=[None],
        banked_control={"fixed_point_residual": 2.9e-16},
    )
    assert reference["cold_control"]["outcome_class"] == "converged_plasma_root"
    assert reference["warm_neighbour_search"]["triggered"] is False
    assert reference["warm_solve"] is None


def test_measure_reference_lifts_a_bounded_stall_via_warm_seed(
    monkeypatch,
) -> None:
    target = _frame(row=35)
    mast_case = {"reference": {"time_s": 5.0}}
    warm_seed = np.asarray([1.0, 2.0, 3.0])

    monkeypatch.setattr(
        warm_neighbour,
        "_prepare_frame",
        lambda store, shot, row, cache_box: (target, mast_case, {}),
    )

    def solve_side_effect(frame, seed):
        kind = "cold" if np.array_equal(seed, frame.seed) else "warm"
        return SimpleNamespace(kind=kind, state=np.asarray(seed))

    def record_side_effect(frame, context, outcome):
        if outcome.kind == "cold":
            return (
                _record(
                    outcome_class="bounded_non_convergence",
                    residual=1.0e-2,
                ),
                object(),
            )
        return (
            _record(outcome_class="converged_plasma_root", residual=1.0e-9),
            object(),
        )

    source_outcome = SimpleNamespace(state=warm_seed)
    source_record = _record(
        outcome_class="converged_plasma_root",
        residual=1.0e-9,
    )
    monkeypatch.setattr(warm_neighbour, "_solve_public_seam", solve_side_effect)
    monkeypatch.setattr(warm_neighbour, "_record_outcome", record_side_effect)
    monkeypatch.setattr(
        warm_neighbour,
        "_find_mast_warm_source",
        lambda store, shot, frame, cache_box: (
            [{"row": 34, "time_s": 4.0, "converged": True}],
            (34, _frame(row=34), source_record, source_outcome),
        ),
    )
    monkeypatch.setattr(
        warm_neighbour,
        "_metric_qualification",
        lambda metrics, residual: {"all": True},
    )

    reference = warm_neighbour.measure_reference(
        Path("/unused"),
        shot=21978,
        row=35,
        cache_box=[None],
        banked_control={"fixed_point_residual": 1.0e-2},
    )
    assert reference["cold_control"]["reproduces_banked_control"] is True
    assert reference["warm_neighbour_search"]["selected_source"]["row"] == 34
    assert reference["warm_solve"]["lifted_to_converged_plasma_root"] is True
    assert reference["warm_solve"]["target_current_relative_error"] == 0.0
