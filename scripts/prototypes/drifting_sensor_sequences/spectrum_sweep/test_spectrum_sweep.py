"""Guards for the sweep's analyse gate and the study's wall write.

The analyse step is the only thing that refuses a wrong table, and the limiter
wall is written beside its name because several imas-efit arms share one study
directory.  Both were verified by hand when they landed; the tests below make
them redden on their own.  A driver that stops validating, or a wall write that
goes back to unlinking the name and saving into it, fails here rather than
quietly returning to the behaviour the repair round removed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sweep

TRUTH = {
    "axis": [6.288633, 0.498205],
    "flux_boundary": 2.811282,
    "flux_span": 76.703791,
    "q95": 2.955747,
    "plasma_current": -13.705,
    "sigma_loop": 83.969,
    "sigma_probe": 0.6731,
    "b0": 5.3,
}


NOISE = np.random.default_rng(11).normal(0.0, 0.01, sweep.SLICES)


def slice_rows(count: int, offset: float = 0.0) -> list[dict]:
    """Rows of one arm, scattered about truth so every measure is defined."""
    return [
        {
            "slice": index,
            "axis": [TRUTH["axis"][0] + offset + float(NOISE[index]), TRUTH["axis"][1]],
            "flux_boundary": TRUTH["flux_boundary"] + offset + float(NOISE[index]),
            "q95": TRUTH["q95"] + offset + float(NOISE[index]),
            "plasma_current": TRUTH["plasma_current"],
            "converged": True,
            "wall_seconds": 12.0,
        }
        for index in range(count)
    ]


def write_study(root: Path, arms) -> None:
    (root / "generating.json").write_text(json.dumps(TRUTH))
    for arm in arms:
        path = root / ("efit-" + sweep.label(arm) + ".jsonl")
        offset = 0.0 if arm["factor"] is None else 3.0 * arm["factor"]
        rows = slice_rows(sweep.SLICES, offset)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def receipts(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for sequence in ("white", "drift"):
        rows = slice_rows(2)
        (root / ("nova-%s-3-0.0100.jsonl" % sequence)).write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    return root


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    """A complete study in a temporary directory, wired into the driver."""
    write_study(tmp_path, sweep.arms())
    monkeypatch.setattr(sweep, "STUDY", tmp_path)
    monkeypatch.setattr(sweep, "NOVA_SOURCE", receipts(tmp_path / "nova"))
    return tmp_path


def test_analyse_publishes_a_complete_study(prepared) -> None:
    """Every declared arm present: the analyse step writes and returns zero."""
    assert sweep.main(["analyse"]) == 0
    payload = json.loads((prepared / "spectrum-summary.json").read_text())
    assert set(payload["arms"]) == {arm["name"] for arm in sweep.arms()}


def test_analyse_refuses_a_missing_arm(prepared) -> None:
    """An arm that never completed stops the payload being written."""
    missing = prepared / ("efit-" + sweep.label(sweep.arms()[0]) + ".jsonl")
    missing.unlink()
    assert sweep.main(["analyse"]) == 1
    assert not (prepared / "spectrum-summary.json").exists()


def test_analyse_refuses_a_short_arm(prepared) -> None:
    """An arm cut short of the declared slice count stops the payload."""
    arm = sweep.arms()[1]
    path = prepared / ("efit-" + sweep.label(arm) + ".jsonl")
    rows = slice_rows(sweep.SLICES // 2)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert sweep.main(["analyse"]) == 1
    assert not (prepared / "spectrum-summary.json").exists()


def test_analyse_refuses_a_control_that_lost_its_correlation(
    prepared, monkeypatch
) -> None:
    """A generator control that stopped shaping the noise reddens the gate."""

    original = sweep.generator_control

    def broken(study, truth, analyse):
        control = original(study, truth, analyse)
        control["index1_lag1"] = analyse.DRIFT_LAG_ONE_FLOOR - 0.1
        return control

    monkeypatch.setattr(sweep, "generator_control", broken)
    assert sweep.main(["analyse"]) == 1
    assert not (prepared / "spectrum-summary.json").exists()


def test_sweep_and_analysis_agree_on_the_white_limit() -> None:
    """One lag-one limit decides both the sweep check and the analysis clause."""
    _, analyse = sweep.import_study()
    payload = {
        "arms": {
            name: {
                "slices": sweep.SLICES,
                "slices_with_boundary": sweep.SLICES,
                "factor": None,
                "axis_lag1": 0.0,
                "flux_lag1": 0.0,
                "q95_lag1": 0.0,
                "axis_rms": 1.0,
                "q95_rms": 1.0,
            }
            for name in (arm["name"] for arm in sweep.arms())
        },
        "generator_control": {
            "slices": 2000,
            "index0_minus_white": 0.0,
            "zero_lag1_max_abs": 0.0,
            "index1_lag1": analyse.DRIFT_LAG_ONE_FLOOR + 0.1,
        },
        "crossing": {
            key: {"factor": 0.05, "value": limit + 1.0}
            for key, limit in sweep.CROSSINGS.items()
        },
        "nova": {"white": {"source_exists": True, "metrics": {"axis_rms": 1.0}}},
    }
    limits = (analyse.LAG_ONE_LIMIT - 0.01, analyse.LAG_ONE_LIMIT + 0.01)
    for value in limits:
        white = payload["arms"]["white-reference"]
        white["axis_lag1"] = value
        white["flux_lag1"] = value
        white["q95_lag1"] = value
        failures, _ = sweep.validate(payload, analyse)
        clause = analyse.thresholds((white, white))["white_lag1_pass"]
        inside = abs(value) <= analyse.LAG_ONE_LIMIT
        assert bool(failures) == (not inside)
        assert clause is inside


def test_wall_write_leaves_no_partial_file(monkeypatch, tmp_path) -> None:
    """A wall write interrupted mid-save leaves no file at the final name."""
    from nova.equilibrium.wall_mask import WallUnit

    sentinel = WallUnit(
        r=np.array([1.0, 2.0, 3.0, 1.0]),
        z=np.array([0.0, 1.0, 0.0, 0.0]),
        kind="material",
        closed=True,
        name="sentinel",
    )

    class Entry:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, name):
            return object()

        def close(self):
            pass

    monkeypatch.setattr("imas.DBEntry", Entry)
    monkeypatch.setattr(
        "nova.equilibrium.wall_mask.wall_units_from_ids", lambda wall: (sentinel,)
    )

    study, _ = sweep.import_study()

    def interrupted(target, **kwargs):
        if hasattr(target, "write"):
            target.write(b"partial")
        else:
            Path(target).write_bytes(b"partial")
        raise RuntimeError("interrupted save")

    monkeypatch.setattr(study.np, "savez", interrupted)
    path = tmp_path / "wall.npz"
    with pytest.raises(RuntimeError):
        study.wall_npz(path)
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))
