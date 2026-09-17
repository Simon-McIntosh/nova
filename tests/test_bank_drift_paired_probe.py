"""Contracts of the paired operand-solve probe's join and campaign merge.

The campaign is only a twelve-arm attribution if all twelve arms are present,
so the merge must count what it found and report what is missing rather than
returning a shorter table that reads as complete.  The join must likewise carry
an arm that only one tree emitted, without inventing a comparison for an arm
neither tree emitted.  These are pinned on synthetic emissions so the contracts
hold independently of any allocation, cache or solve.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.bank_drift_paired_probe import _compare, _merge, _panel

STAGES = {
    "profile_support": {"digest": "aaaa"},
    "partition_structure": {"digest": "bbbb"},
    "partition_values": {"digest": "cccc"},
}
FLAT_STAGES = {
    "profile_support": "aaaa",
    "partition_structure": "bbbb",
    "partition_values": "cccc",
    "exception": None,
}


def _arm(residual: float, converged: bool) -> dict:
    return {
        "converged": converged,
        "terminal_residual": residual,
        "termination_reason": "converged" if converged else "active_set_cycle_detected",
        "map_image": {"flux": "dddd", "axis": "eeee", "class_margin": "ffff"},
        "stage_exception": None,
    }


def _emission(label: str, arms: dict[str, dict]) -> dict:
    return {
        "tree_label": label,
        "tree_root": f"/tmp/{label}",
        "compile_cache": {"directory": "/tmp/cache"},
        "rows": [
            {
                "identity": "21978/35",
                "stages": {key: dict(value) for key, value in STAGES.items()},
                "arms": arms,
                "exception": None,
            }
        ],
    }


def _receipt(identity: str, arm: str, residual: float) -> dict:
    return {
        "generated_at": "2026-09-17T00:00:00+00:00",
        "left": {"label": "fae50f15", "root": "/tmp/old"},
        "right": {"label": "main", "root": "/tmp/new"},
        "rows": [
            {
                "identity": identity,
                "arm": arm,
                "residual_left": 1e-16,
                "residual_right": residual,
                "residual_delta": residual - 1e-16,
                "residual_moved": True,
                "residual_ratio": residual / 1e-16,
                "converged_left": True,
                "converged_right": False,
                "stages_left": dict(FLAT_STAGES),
                "stages_right": dict(FLAT_STAGES),
                "first_differing_stage": "partition_structure",
                "note": None,
            }
        ],
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_merge_counts_present_and_missing_rows(tmp_path: Path) -> None:
    """A campaign short of its expected rows reports the shortfall and fails."""

    arms_dir = tmp_path / "arms"
    arms_dir.mkdir()
    for index in range(10):
        _write(
            arms_dir / f"receipt-x{index:02d}.json", _receipt("21978/35", "pure", 0.1)
        )
    out = tmp_path / "receipt.json"

    status = _merge(arms_dir, out, expected_rows=12)

    summary = json.loads(out.read_text())["summary"]
    assert status == 1
    assert summary["rows_present"] == 10
    assert summary["rows_missing"] == 2
    assert summary["rows_unexpected"] == 0


def test_merge_complete_campaign_succeeds(tmp_path: Path) -> None:
    """Every expected row present is the passing control for the count."""

    arms_dir = tmp_path / "arms"
    arms_dir.mkdir()
    for index in range(12):
        _write(
            arms_dir / f"receipt-x{index:02d}.json", _receipt("21978/35", "pure", 0.1)
        )
    out = tmp_path / "receipt.json"

    status = _merge(arms_dir, out, expected_rows=12)

    summary = json.loads(out.read_text())["summary"]
    assert status == 0
    assert summary["rows_present"] == 12
    assert summary["rows_missing"] == 0
    assert summary["rows_unexpected"] == 0


def test_merge_flags_more_rows_than_expected(tmp_path: Path) -> None:
    """The count fails in both directions, not only when rows are absent."""

    arms_dir = tmp_path / "arms"
    arms_dir.mkdir()
    for index in range(13):
        _write(
            arms_dir / f"receipt-x{index:02d}.json", _receipt("21978/35", "pure", 0.1)
        )
    out = tmp_path / "receipt.json"

    status = _merge(arms_dir, out, expected_rows=12)

    summary = json.loads(out.read_text())["summary"]
    assert status == 1
    assert summary["rows_present"] == 13
    assert summary["rows_missing"] == 0
    assert summary["rows_unexpected"] == 1


def test_compare_carries_an_arm_only_one_tree_emitted(tmp_path: Path) -> None:
    """An arm one side is missing becomes a row with no comparison, not silence."""

    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    out = tmp_path / "receipt.json"
    _write(
        left,
        _emission("fae50f15", {"pure": _arm(1e-16, True), "mixed": _arm(2e-16, True)}),
    )
    _write(right, _emission("main", {"pure": _arm(2.45e-3, False)}))

    status = _compare(left, right, out)

    rows = {row["arm"]: row for row in json.loads(out.read_text())["rows"]}
    assert status == 0
    assert set(rows) == {"pure", "mixed"}
    assert rows["pure"]["residual_right"] == 2.45e-3
    assert rows["pure"]["residual_moved"] is True
    assert rows["mixed"]["residual_right"] is None
    assert rows["mixed"]["residual_moved"] is False
    assert rows["mixed"]["converged_right"] is None


def test_compare_does_not_invent_a_row_for_an_arm_neither_tree_ran(
    tmp_path: Path,
) -> None:
    """An arm absent from both sides is absent, not a zero row in the campaign."""

    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    out = tmp_path / "receipt.json"
    _write(left, _emission("fae50f15", {"pure": _arm(1e-16, True)}))
    _write(right, _emission("main", {"pure": _arm(2.45e-3, False)}))

    _compare(left, right, out)

    rows = json.loads(out.read_text())["rows"]
    assert [row["arm"] for row in rows] == ["pure"]


def _resolve_archive(path: Path, residual: float, converged: bool) -> None:
    """Write one resolve archive holding a terminal state whole."""

    import numpy as np

    radius = np.linspace(0.80, 2.20, 9)
    height = np.linspace(-2.00, 2.00, 11)
    radius_grid, height_grid = np.meshgrid(radius, height)
    peak = 1.0 - ((radius_grid - 1.35) ** 2 + (height_grid / 2.2) ** 2)
    wall = np.asarray(
        [
            [1.00, -2.00],
            [2.10, -2.00],
            [2.10, 2.00],
            [1.00, 2.00],
            [1.00, -2.00],
        ],
        dtype=float,
    )
    np.savez(
        path,
        tree_label=np.array("synthetic"),
        converged=np.asarray(converged),
        terminal_residual=np.asarray(residual),
        radius=np.asarray(radius, dtype=float),
        height=np.asarray(height, dtype=float),
        flux=np.asarray(peak, dtype=float),
        wall=wall,
        axis=np.asarray([1.35, 0.0], dtype=float),
        selected_x=np.asarray([1.45, -1.30], dtype=float),
    )


def test_panel_draws_both_states_from_their_resolve_archives(tmp_path: Path) -> None:
    """The pair renders from the two archives on one shared level array.

    This pins the archive keys the panel reads and the shared-level array, which
    is what a rename or a shape swap in the driver would break."""

    old = tmp_path / "old.npz"
    new = tmp_path / "new.npz"
    out = tmp_path / "panels.png"
    _resolve_archive(old, residual=3.857344e-16, converged=True)
    _resolve_archive(new, residual=2.451e-03, converged=False)

    status = _panel(old, new, out, 8)

    assert status == 0
    assert out.stat().st_size > 0
