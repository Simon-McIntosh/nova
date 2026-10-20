"""Contracts for the XLA autotune cache measurement harness."""

from __future__ import annotations

import pytest

from benchmarks import xla_autotune_cache_arms as harness


def receipt(arm_a: int, arm_c: int) -> dict:
    return {
        "arms": {
            arm: {"files_after_cold": {"autotune_entries": count}}
            for arm, count in (("a", arm_a), ("c", arm_c))
        }
    }


def test_negative_control_holds_when_default_writes_and_disabled_arm_does_not() -> None:
    assert harness.negative_control_failure(receipt(3, 0)) is None


def test_negative_control_fails_when_disabled_arm_writes_entries() -> None:
    failure = harness.negative_control_failure(receipt(3, 1))

    assert failure is not None
    assert "arm (c) created 1 autotune entries" in failure


def test_negative_control_fails_when_default_arm_does_not_write_entries() -> None:
    failure = harness.negative_control_failure(receipt(0, 0))

    assert failure is not None
    assert "arm (a) created 0 autotune entries" in failure


def test_missing_data_ink_style_sheet_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import matplotlib.pyplot as plt

    def missing_style(_: str) -> None:
        raise OSError("style sheet not found")

    monkeypatch.setattr(plt.style, "use", missing_style)
    plotted_receipt = {
        "arms": {
            arm: {"cold_compile_seconds": 1.0, "steady_seconds": [1.0, 1.1, 0.9]}
            for arm in harness.ARMS
        }
    }

    with pytest.raises(OSError, match="style sheet not found"):
        harness.figure_from_receipt(plotted_receipt, tmp_path / "figures")
