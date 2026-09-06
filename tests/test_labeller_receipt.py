"""Adjudication tables of the forward-labeller completion receipt.

The receipt aggregates per-shot manifests; the adjudication tables imas-ambix
asked for are computed over the per-shot diagnostics npz that the labeller
shard writes (one record per admitted slice).  These tests build a two-shot
synthetic sessions root and pin the table shapes, the invariant that every
table's bin counts sum to the written slice count, the pin displacement
summary on hand-computed inputs, and the per-shot free and conditioned slice
counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.labeller_batch import receipt


# --------------------------------------------------------------------------
# synthetic corpus
# --------------------------------------------------------------------------

#: (shot, times, conditioned, requested_class, free_error_m, cond_error_m)
_SHOTS = [
    (
        41001,
        np.asarray([0.000, 0.025, 0.050, 0.075]),
        np.asarray([True, False, True, False]),
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.010, 0.020, 0.010, 0.030]),
        np.asarray([0.002, 0.004, 0.001, 0.003]),
    ),
    (
        41002,
        np.asarray([0.010, 0.020, 0.030]),
        np.asarray([True, False, True]),
        np.asarray([1, 0, 0]),
        np.asarray([0.015, 0.005, 0.025]),
        np.asarray([0.006, 0.001, 0.002]),
    ),
]

TOTAL_SLICES = sum(int(len(times)) for _, times, *_ in _SHOTS)  # 7
TOTAL_CONDITIONED = sum(int(conditioned.sum()) for _, _, conditioned, *_ in _SHOTS)  # 4


@pytest.fixture(autouse=True)
def _seed_topology_class_names():
    """Name the two classes without importing the jax-hosting enum module.

    The armed values equal ``TopologyClass`` for the classes the labeller
    requests; seeding the receipt's lazy cache keeps this unit table fast and
    free of a jax import.
    """
    receipt._TOPOLOGY_NAMES = {0: "limited", 1: "diverted"}
    yield


def _slice_record(
    row: int,
    time: float,
    conditioned: bool,
    cls: int | None,
    free_error: float,
    conditioned_error: float,
) -> dict:
    record = {
        "row": row,
        "time": time,
        "written": True,
        "excluded": False,
        "geometry_masked": False,
        "converged": True,
        "qualified": True,
        "terminal_residual": 1e-7,
        "trips": 1,
        "newton_steps": 1,
        "free_trips": 1,
        "conditioned_trips": 1,
        "wall_seconds": 0.05,
        "termination": "converged",
        "conditioned": conditioned,
        "conditioning_flag": conditioned,
        "conditioning_target_source": "efm/current_centrd_z" if conditioned else None,
        "free_converged": True,
        "conditioned_converged": True if conditioned else None,
        "free_branch_guard_ok": True,
        "conditioned_branch_guard_ok": True if conditioned else None,
        "free_centroid_error_m": free_error,
        "conditioned_centroid_error_m": conditioned_error,
        "achieved_current_centroid_r": 0.9,
        "achieved_current_centroid_z": 0.2,
        "target_current_centroid_z": 0.2,
        "centroid_error_m": 0.0,
        "target_source": "efm/current_centrd_z",
        "branch_guard_ok": True,
    }
    if cls is not None:
        record["requested_class"] = int(cls)
    return record


def _write_shot(
    root,
    shot: int,
    times,
    conditioned,
    classes,
    free_errors,
    conditioned_errors,
    *,
    with_sessions: bool,
) -> None:
    rows = np.arange(len(times), dtype=int)
    slices = [
        _slice_record(int(row), float(time), bool(flag), cls, float(free), float(cond))
        for row, time, flag, cls, free, cond in zip(
            rows,
            times,
            conditioned,
            classes,
            free_errors,
            conditioned_errors,
            strict=True,
        )
    ]
    npz_path = root / f"{shot}.npz"
    np.savez_compressed(
        npz_path,
        row=np.asarray(rows, dtype=np.int32),
        time=np.asarray(times, dtype=np.float64),
        conditioned=np.asarray(conditioned, dtype=bool),
        conditioning_target_source=np.asarray(
            ["efm/current_centrd_z" if flag else "none" for flag in conditioned],
            dtype=str,
        ),
        free_guard_evaluated=np.ones(len(times), dtype=bool),
        free_branch_guard_ok=np.ones(len(times), dtype=bool),
        conditioned_guard_evaluated=np.asarray(conditioned, dtype=bool),
        conditioned_branch_guard_ok=np.asarray(conditioned, dtype=bool),
        free_centroid_error_m=np.asarray(free_errors, dtype=np.float64),
        conditioned_centroid_error_m=np.asarray(conditioned_errors, dtype=np.float64),
    )
    session_path = root / f"{shot}.nc"
    if with_sessions:
        session_path.write_text("", encoding="utf-8")
    manifest = {
        "schema": "nova-forward-labeller-shot",
        "shot": shot,
        "status": "complete",
        "session": str(session_path.resolve()),
        "companion": str(npz_path.resolve()),
        "nova_revision": "synthetic",
        "carrier_identity": "synthetic",
        "policy_digest": "synthetic",
        "policy": {"qualification_tolerance": 1e-4},
        "constraint": {"mode": "diagnostic_branch_guard"},
        "include_raster": False,
        "setup_wall_seconds": 0.0,
        "shot_wall_seconds": 1.0,
        "slice_count": len(times),
        "admitted_slice_count": len(times),
        "written_slice_count": len(times),
        "converged_slice_count": sum(1 for s in slices if s["converged"]),
        "unconverged_slice_count": 0,
        "excluded_slice_count": 0,
        "companion_slice_count": len(times),
        "flux_function_grid_points": 100,
        "slices": slices,
    }
    (root / f"{shot}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _build_root(tmp_path, *, with_sessions=False, with_classes=True):
    manifests = []
    for shot, times, conditioned, classes, free, cond in _SHOTS:
        cls_series = (
            classes if with_classes else np.full(len(times), None, dtype=object)
        )
        manifests.append(
            _write_shot(
                tmp_path,
                int(shot),
                times,
                conditioned,
                cls_series,
                free,
                cond,
                with_sessions=with_sessions,
            )
        )
    return manifests


def _aggregate(tmp_path, **kwargs):
    return receipt.aggregate(tmp_path, **kwargs)


# --------------------------------------------------------------------------
# helpers shared by the assertions
# --------------------------------------------------------------------------


def _bin_slice_totals(receipt_dict):
    """Return (slices, conditioned) per sub-table so the sum invariant holds."""
    time_table = receipt_dict["conditioned_by_time_in_shot"]
    decile_slices = sum(bin_["slices"] for bin_ in time_table["deciles"]["bins"])
    decile_conditioned = sum(
        bin_["conditioned"] for bin_ in time_table["deciles"]["bins"]
    )
    bin_slices = sum(bin_["slices"] for bin_ in time_table["absolute_50ms"]["bins"])
    bin_conditioned = sum(
        bin_["conditioned"] for bin_ in time_table["absolute_50ms"]["bins"]
    )
    class_table = receipt_dict["conditioned_by_topology_class"]["classes"]
    class_slices = sum(entry["slices"] for entry in class_table.values())
    class_conditioned = sum(entry["conditioned"] for entry in class_table.values())
    return {
        "deciles": (decile_slices, decile_conditioned),
        "absolute_50ms": (bin_slices, bin_conditioned),
        "classes": (class_slices, class_conditioned),
    }


# --------------------------------------------------------------------------
# table shape and sum invariants over the manifest-requested-class source
# --------------------------------------------------------------------------


def test_tables_sum_to_the_written_slice_count(tmp_path):
    _build_root(tmp_path)
    result = _aggregate(tmp_path)

    totals = _bin_slice_totals(result)
    assert totals["deciles"] == (TOTAL_SLICES, TOTAL_CONDITIONED)
    assert totals["absolute_50ms"] == (TOTAL_SLICES, TOTAL_CONDITIONED)
    assert totals["classes"] == (TOTAL_SLICES, TOTAL_CONDITIONED)

    deciles = result["conditioned_by_time_in_shot"]["deciles"]["bins"]
    assert [bin_["decile"] for bin_ in deciles] == list(range(10))
    assert result["conditioned_by_topology_class"]["source"] == (
        "manifest_requested_class"
    )
    assert "reason" not in result["conditioned_by_topology_class"]


def test_conditioned_by_time_in_shot_deciles_on_known_inputs(tmp_path):
    _build_root(tmp_path)
    table = _aggregate(tmp_path)["conditioned_by_time_in_shot"]
    bins = {bin_["decile"]: bin_ for bin_ in table["deciles"]["bins"]}
    expected = {
        0: (2, 2),
        3: (1, 0),
        5: (1, 0),
        6: (1, 1),
        9: (2, 1),
    }
    for decile, (count, conditioned) in expected.items():
        assert (bins[decile]["slices"], bins[decile]["conditioned"]) == (
            count,
            conditioned,
        )
        assert bins[decile]["fraction"] == pytest.approx(conditioned / count)
    for decile in (1, 2, 4, 7, 8):
        assert (bins[decile]["slices"], bins[decile]["conditioned"]) == (0, 0)


def test_conditioned_by_time_in_shot_absolute_50ms_bins(tmp_path):
    _build_root(tmp_path)
    bins = _aggregate(tmp_path)["conditioned_by_time_in_shot"]["absolute_50ms"]["bins"]
    by_start = {bin_["bin_ms"]: bin_ for bin_ in bins}
    # shot 41001 offsets 0, 25, 50, 75 ms; shot 41002 offsets 0, 10, 20 ms.
    assert by_start[0]["slices"] == 5
    assert by_start[0]["conditioned"] == 3
    assert by_start[50]["slices"] == 2
    assert by_start[50]["conditioned"] == 1
    assert by_start[0]["fraction"] == pytest.approx(3 / 5)
    assert by_start[50]["fraction"] == pytest.approx(1 / 2)


def test_conditioned_by_topology_class_bins(tmp_path):
    _build_root(tmp_path)
    classes = _aggregate(tmp_path)["conditioned_by_topology_class"]["classes"]
    assert classes["limited"]["slices"] == 4
    assert classes["limited"]["conditioned"] == 2
    assert classes["diverted"]["slices"] == 3
    assert classes["diverted"]["conditioned"] == 2


def test_pin_displacement_summary_on_known_inputs(tmp_path):
    _build_root(tmp_path)
    pin = _aggregate(tmp_path)["pin_displacement_mm"]
    assert pin["unit"] == "mm"
    overall = pin["overall"]
    assert overall["count"] == 7
    assert overall["median"] == pytest.approx(-9.0)
    assert overall["maximum"] == pytest.approx(-4.0)
    assert overall["p90"] == pytest.approx(
        np.percentile([-8, -16, -9, -27, -9, -4, -23], 90)
    )
    limited = pin["by_topology_class"]["limited"]
    assert limited["count"] == 4
    assert limited["median"] == pytest.approx(-12.0)
    assert limited["maximum"] == pytest.approx(-4.0)
    diverted = pin["by_topology_class"]["diverted"]
    assert diverted["count"] == 3
    assert diverted["median"] == pytest.approx(-9.0)
    assert diverted["maximum"] == pytest.approx(-9.0)


def test_per_shot_free_and_conditioned_counts(tmp_path):
    _build_root(tmp_path)
    shots = {entry["shot"]: entry for entry in _aggregate(tmp_path)["shots"]}
    assert shots[41001]["free_slices"] == 2
    assert shots[41001]["conditioned_slices"] == 2
    assert shots[41002]["free_slices"] == 1
    assert shots[41002]["conditioned_slices"] == 2
    for entry in shots.values():
        assert (
            entry["free_slices"] + entry["conditioned_slices"]
            == (entry["admitted_slices"])
        )


def test_aggregate_baseline_keys_unchanged(tmp_path):
    _build_root(tmp_path)
    result = _aggregate(tmp_path)
    for key in (
        "slices",
        "qualified_slices",
        "converged_fraction",
        "companion_shots",
        "shots",
        "failed",
    ):
        assert key in result
    assert result["slices"] == TOTAL_SLICES
    assert result["completed_shots"] == 2


# --------------------------------------------------------------------------
# class-source resolution: session frames, then unavailable
# --------------------------------------------------------------------------


def test_class_source_falls_back_to_session_frames(tmp_path, monkeypatch):
    _build_root(tmp_path, with_sessions=True)
    # read_session/frames_from_session cannot run against the empty stand-in
    # sessions, so the frame reader is replaced with a per-path stub.
    by_path = {
        tmp_path / "41001.nc": [1, 1, 0, 0],
        tmp_path / "41002.nc": [0, 1, 0],
    }

    def fake_classes(session_path):
        return by_path.get(Path(session_path))

    monkeypatch.setattr(receipt, "_classes_from_session", fake_classes)
    result = _aggregate(tmp_path)
    classes = result["conditioned_by_topology_class"]
    assert classes["source"] == "session_frame"
    assert classes["classes"]["diverted"]["slices"] == 3
    assert classes["classes"]["diverted"]["conditioned"] == 1
    assert classes["classes"]["limited"]["slices"] == 4
    assert classes["classes"]["limited"]["conditioned"] == 3
    assert _bin_slice_totals(result)["classes"] == (TOTAL_SLICES, TOTAL_CONDITIONED)


def test_class_source_unavailable_when_nothing_records_a_class(tmp_path):
    _build_root(tmp_path, with_classes=False)
    classes = _aggregate(tmp_path)["conditioned_by_topology_class"]
    assert classes["source"] == "unavailable"
    assert "reason" in classes
    assert classes["classes"] == {}
    # the time tables remain populated even when the class dimension is not
    totals = _bin_slice_totals(_aggregate(tmp_path))
    assert totals["deciles"] == (TOTAL_SLICES, TOTAL_CONDITIONED)
    assert totals["absolute_50ms"] == (TOTAL_SLICES, TOTAL_CONDITIONED)
