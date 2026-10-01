"""Contracts of the anchor-treatment bisect receipt's verdict.

A receipt that reports "the override did not take effect" for an arm that never
landed reads as a measurement of the hook, and a receipt that names the anchor
partition the carrier of a difference at the round-off floor reads the summation
order as physics.  Both readings are pinned here on synthetic arm records, so
the contracts hold independently of any allocation, cache or solve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.bank_drift_anchor_bisect import (
    ROUND_OFF_FLOOR,
    _arm_absence,
    _arm_paths,
    _load_arm,
    _receipt,
    _residual_gap_below_round_off,
)

IDENTITY = "21978/35"
TRACED_RESIDUAL = 2.2041968427410595e-16
CONSTANT_RESIDUAL = 3.0996518101046146e-16


def _arm(
    residual: float | None,
    hook_names: list[str],
    exception: str | None = None,
) -> dict:
    return {
        "identity": IDENTITY,
        "solve_class": "pure",
        "treatment": "traced" if hook_names else "constant",
        "hook_names": hook_names,
        "anchors": {},
        "profile_support": {"digest": "profile"},
        "partition": {
            "structure_digest": "structure",
            "values_digest": "values",
            "summary": {"leaf_count": 76 if hook_names else 73},
        },
        "carrier_identity": "mast:21978:35:carrier",
        "target_current": 1.0,
        "exception": exception,
        "terminal_residual": residual,
        "termination_reason": "converged",
        "converged": True,
        "trip_trace": {},
    }


def _write_arms(out_dir: Path, traced: dict, constant: dict) -> None:
    """Write one arm record per treatment and solve class, at the driver's paths."""

    out_dir.mkdir(parents=True, exist_ok=True)
    for solve_class in ("pure", "mixed"):
        for treatment, record in (("traced", traced), ("constant", constant)):
            json_path, _array_path = _arm_paths(
                out_dir, IDENTITY, solve_class, treatment
            )
            json_path.write_text(json.dumps(record))


def _verdicts(out_dir: Path, tmp_path: Path) -> dict:
    out = tmp_path / "receipt.json"
    _receipt(out_dir, IDENTITY, tmp_path / "no-bank", out)
    payload = json.loads(out.read_text())
    return {
        solve_class: record["verdict"]
        for solve_class, record in payload["classes"].items()
    }


def test_a_missing_arm_file_is_reported_as_absent(tmp_path: Path) -> None:
    """An arm directory with no arms compares nothing; it is not a measurement."""

    out_dir = tmp_path / "arms"
    out_dir.mkdir()
    verdicts = _verdicts(out_dir, tmp_path)
    for solve_class in ("pure", "mixed"):
        verdict = verdicts[solve_class]
        assert set(verdict["arms_absent"]) == {"traced", "constant"}
        assert verdict["finding"].startswith("absent arms:")
        assert "did not take effect" not in verdict["finding"]
        assert verdict["terminal_residual_gap"] is None
        assert verdict["terminal_residual_gap_below_round_off"] is None


def test_a_raised_arm_is_reported_as_absent_with_its_exception(
    tmp_path: Path,
) -> None:
    """An arm whose solve raised is persisted with its traceback and no residual."""

    traced = _arm(None, ["declared_axis_flux"], exception="ValueError: bad operand")
    constant = _arm(CONSTANT_RESIDUAL, [])
    out_dir = tmp_path / "arms"
    _write_arms(out_dir, traced, constant)
    verdict = _verdicts(out_dir, tmp_path)["pure"]
    assert list(verdict["arms_absent"]) == ["traced"]
    assert "ValueError: bad operand" in verdict["arms_absent"]["traced"]
    assert "did not take effect" not in verdict["finding"]


def test_a_round_off_gap_does_not_name_a_carrier(tmp_path: Path) -> None:
    """Two residuals inside one epsilon are the same state, not a carrier."""

    out_dir = tmp_path / "arms"
    _write_arms(
        out_dir,
        _arm(TRACED_RESIDUAL, ["declared_axis_flux"]),
        _arm(CONSTANT_RESIDUAL, []),
    )
    verdict = _verdicts(out_dir, tmp_path)["pure"]
    assert verdict["hook_names_differ"] is True
    assert verdict["terminal_residual_differ"] is True
    assert verdict["terminal_residual_gap"] == abs(TRACED_RESIDUAL - CONSTANT_RESIDUAL)
    assert verdict["terminal_residual_gap_below_round_off"] is True
    assert "not the carrier" in verdict["finding"]
    assert f"{ROUND_OFF_FLOOR:.3g}" in verdict["finding"]


def test_a_separation_above_the_floor_still_names_the_carrier(
    tmp_path: Path,
) -> None:
    """The floor is a floor: a gap larger than it keeps the carrier reading."""

    out_dir = tmp_path / "arms"
    _write_arms(
        out_dir,
        _arm(2.4506488565449e-03, ["declared_axis_flux"]),
        _arm(CONSTANT_RESIDUAL, []),
    )
    verdict = _verdicts(out_dir, tmp_path)["pure"]
    assert verdict["terminal_residual_gap_below_round_off"] is False
    assert verdict["finding"].startswith("the anchor partition is the carrier:")


def test_identical_residuals_do_not_name_a_carrier(tmp_path: Path) -> None:
    """The flatten comparison survives: the hook moved the partition, not the state."""

    out_dir = tmp_path / "arms"
    _write_arms(
        out_dir,
        _arm(2.4506488565449e-03, ["declared_axis_flux"]),
        _arm(2.4506488565449e-03, []),
    )
    verdict = _verdicts(out_dir, tmp_path)["pure"]
    assert verdict["terminal_residual_differ"] is False
    assert verdict["terminal_residual_gap"] == 0.0
    assert verdict["finding"].startswith("the anchor partition is not the carrier:")


def test_the_absent_stub_carries_no_terminal_state(tmp_path: Path) -> None:
    """`_load_arm` on a path that does not exist answers an absence reason."""

    stub = _load_arm(tmp_path, IDENTITY, "pure", "traced")
    assert "terminal_residual" not in stub
    assert "missing arm file" in _arm_absence(stub)
    assert _arm_absence(_arm(TRACED_RESIDUAL, ["declared_axis_flux"])) is None


def test_the_floor_is_one_machine_epsilon() -> None:
    """The floor is the residual's own resolution, not a tuned tolerance."""

    assert ROUND_OFF_FLOOR == float(np.finfo(np.float64).eps)
    assert _residual_gap_below_round_off(None, 1.0) is None
    assert _residual_gap_below_round_off(1.0, None) is None
    assert _residual_gap_below_round_off(TRACED_RESIDUAL, CONSTANT_RESIDUAL) is True
    assert _residual_gap_below_round_off(0.0, ROUND_OFF_FLOOR * 2.0) is False
