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
import os
from pathlib import Path

import numpy as np

from benchmarks.bank_drift_paired_probe import (
    CHECKOUT_ROOT,
    _compare,
    _merge,
    _panel,
    _repo_relative,
)

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


def test_recorded_root_resolves_from_the_checkout_at_any_worktree_depth(
    tmp_path: Path,
) -> None:
    """A stated root re-resolves from the checkout, not from this worktree.

    A worktree root sits at the depth it was created at, so a root stated
    relative to it resolves only from a checkout of that same depth.  Stating
    it from the checkout root makes the value re-resolve from the checkout
    wherever the record is read.
    """

    outside = tmp_path / "tree"
    outside.mkdir()

    stated = _repo_relative(outside)

    assert not Path(stated).is_absolute()
    assert (CHECKOUT_ROOT / stated).resolve() == outside.resolve()
    assert os.path.relpath(outside.resolve(), CHECKOUT_ROOT) == stated


def test_merge_records_the_compile_log_it_counted_from(tmp_path: Path) -> None:
    """The count is evidence only beside the capture and digest it came from.

    The synthetic capture holds a known present marker so a zero cannot read as
    a cheap compile, and one of its lines is the solve marker so the two counts
    are shown to be different questions."""

    arms_dir = tmp_path / "arms"
    arms_dir.mkdir()
    for index in range(12):
        _write(
            arms_dir / f"receipt-x{index:02d}.json", _receipt("21978/35", "pure", 0.1)
        )
    log = tmp_path / "compile-capture.txt"
    log.write_text(
        "Compiling jit(convert_element_type) with global shapes\n"
        "Compiling jit(solve) with global shapes\n"
        "unrelated line\n"
        "Compiling jit(squeeze) with global shapes\n"
    )
    out = tmp_path / "receipt.json"

    status = _merge(arms_dir, out, expected_rows=12, compile_log=log)

    evidence = json.loads(out.read_text())["compile_evidence"]
    assert status == 0
    assert evidence["compile_lines"] == 3
    assert evidence["solve_compiles"] == 1
    assert evidence["log"] == _repo_relative(log)
    assert len(evidence["sha256"]) == 64


def test_merge_refuses_a_capture_that_is_not_a_compile_log(tmp_path: Path) -> None:
    """A capture with no compile line refuses rather than reporting zero."""

    import pytest

    arms_dir = tmp_path / "arms"
    arms_dir.mkdir()
    _write(arms_dir / "receipt-x00.json", _receipt("21978/35", "pure", 0.1))
    log = tmp_path / "not-a-capture.txt"
    log.write_text("nothing to do with compiles\n")

    with pytest.raises(SystemExit):
        _merge(arms_dir, tmp_path / "receipt.json", expected_rows=1, compile_log=log)


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


def _resolve_archive(
    path: Path,
    residual: float,
    converged: bool,
    wall: list[list[float]] | None = None,
    radius: np.ndarray | None = None,
    height: np.ndarray | None = None,
    identity: str = "21978/35",
) -> None:
    """Write one resolve archive holding a terminal state whole."""

    if radius is None:
        radius = np.linspace(0.80, 2.20, 9)
    if height is None:
        height = np.linspace(-2.00, 2.00, 11)
    radius_grid, height_grid = np.meshgrid(radius, height)
    peak = 1.0 - ((radius_grid - 1.35) ** 2 + (height_grid / 2.2) ** 2)
    if wall is None:
        wall = [
            [1.00, -2.00],
            [2.10, -2.00],
            [2.10, 2.00],
            [1.00, 2.00],
            [1.00, -2.00],
        ]
    np.savez(
        path,
        tree_label=np.array("synthetic"),
        identity=np.array(identity),
        arm=np.array("pure"),
        converged=np.asarray(converged),
        terminal_residual=np.asarray(residual),
        radius=np.asarray(radius, dtype=float),
        height=np.asarray(height, dtype=float),
        flux=np.asarray(peak, dtype=float),
        wall=np.asarray(wall, dtype=float),
        axis=np.asarray([1.35, 0.0], dtype=float),
        selected_x=np.asarray([1.45, -1.30], dtype=float),
    )


def _marker_lines(axes) -> list:
    """Every line that carries a marker and no stroke."""

    return [line for line in axes.lines if line.get_marker() not in ("", "None", None)]


def _wall_lines(axes) -> list:
    """Every stroked polyline, which in a poloidal panel is the wall."""

    return [line for line in axes.lines if line.get_marker() in ("", "None", None)]


NEW_WALL = [
    [1.00, -1.90],
    [2.05, -1.90],
    [2.05, 1.90],
    [1.00, 1.90],
    [1.00, -1.90],
]


def test_panel_draws_each_archive_s_own_wall_and_grid(tmp_path: Path) -> None:
    """Each panel carries its own archive's wall, not the producer's.

    The two archives are given different walls and different grids, so a panel
    that borrows the producer archive's geometry draws a wall the reader can
    tell apart from the one its own state was solved against, and a grid swap
    raises rather than rendering the wrong map.
    """

    import numpy as np

    from benchmarks.bank_drift_paired_probe import _panel_figure

    old = tmp_path / "old.npz"
    new = tmp_path / "new.npz"
    _resolve_archive(old, residual=3.857344e-16, converged=True)
    _resolve_archive(
        new,
        residual=2.451e-03,
        converged=False,
        wall=NEW_WALL,
        radius=np.linspace(0.90, 2.05, 7),
        height=np.linspace(-1.90, 1.90, 8),
    )

    figure, evidence = _panel_figure(old, new, 8)

    assert evidence["wall_bit_identical"] is False
    assert evidence["grid_bit_identical"] is False
    producer, current = figure.axes[0], figure.axes[1]
    for axes, wall in (
        (producer, [[1.0, -2.0], [2.1, -2.0], [2.1, 2.0], [1.0, 2.0], [1.0, -2.0]]),
        (current, NEW_WALL),
    ):
        drawn = _wall_lines(axes)
        assert len(drawn) == 1
        assert np.array_equal(
            np.column_stack(drawn[0].get_data()), np.asarray(wall, dtype=float)
        )

    figure.clear()


def test_panel_keeps_one_shared_level_array_across_both_panels(tmp_path: Path) -> None:
    """One physical level array reaches both contours and both null sets draw.

    Contours chosen per panel can make any two maps agree, so the shared array
    is the quantity under test; both null sets are checked by their marker
    vocabulary and by which of the two is hollowed."""

    import numpy as np

    from benchmarks.bank_drift_paired_probe import _panel_figure
    from nova.media.poloidal import contour_levels

    old = tmp_path / "old.npz"
    new = tmp_path / "new.npz"
    _resolve_archive(old, residual=3.857344e-16, converged=True)
    _resolve_archive(new, residual=2.451e-03, converged=False)

    figure, evidence = _panel_figure(old, new, 8)

    with np.load(old, allow_pickle=False) as archive:
        expected = contour_levels(np.asarray(archive["flux"], dtype=float), count=8)
    assert evidence["wall_bit_identical"] is True
    assert evidence["grid_bit_identical"] is True
    for axes in figure.axes:
        contours = axes.collections
        assert len(contours) == 1
        assert np.array_equal(np.asarray(contours[0].levels, dtype=float), expected)
        markers = _marker_lines(axes)
        assert {line.get_marker() for line in markers} == {"^", "x", "o", "s"}
        own = [line for line in markers if line.get_marker() == "^"][0]
        counterpart = [line for line in markers if line.get_marker() == "o"][0]
        assert own.get_markerfacecolor() != "none"
        assert counterpart.get_markerfacecolor() == "none"
        assert counterpart.get_color() == "#666666"
    figure.clear()


def test_panel_renders_and_records_its_inputs(tmp_path: Path) -> None:
    """The pair renders to a figure and the record beside it states its inputs."""

    old = tmp_path / "old.npz"
    new = tmp_path / "new.npz"
    out = tmp_path / "panels.png"
    _resolve_archive(old, residual=3.857344e-16, converged=True)
    _resolve_archive(new, residual=2.451e-03, converged=False)

    status = _panel(old, new, out, 8)

    assert status == 0
    assert out.exists()
    record = json.loads(out.with_suffix(".json").read_text())
    assert record["levels"] == 8
    assert len(record["shared_levels"]) >= 1
    assert record["wall_bit_identical"] is True
    assert out.stat().st_size > 0
