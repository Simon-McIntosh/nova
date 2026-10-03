"""The published shape control must preserve its measured counterexample."""

import copy
import json
import os
from pathlib import Path

from PIL import Image
import pytest


FIGURES = Path(
    os.environ.get(
        "NOVA_SHAPE_FIGURE_DIRECTORY",
        Path(__file__).resolve().parents[1]
        / "docs/figures/playable-forward-solve/shape-inverse",
    )
)


def _receipts():
    arm = json.loads((FIGURES / "shape-inverse-receipt.json").read_text())
    control = json.loads((FIGURES / "all-prescribed-negative-control.json").read_text())
    return arm, control


def test_historical_control_comparison_and_titles():
    from benchmarks.playable_shape_inverse_receipt import _historical_comparison

    arm, control = _receipts()
    rows = arm["historical_control_comparison"]
    assert _historical_comparison(arm["arms"], control["arms"]) == rows
    assert len(rows) == 2
    assert [row["arm"] for row in rows] == [
        "upper-point-plus-20mm",
        "elongation-plus-5pct",
    ]
    for index, row in enumerate(rows):
        assert row["arm_error_m"] == pytest.approx(
            arm["arms"][index]["final_turning_point_error_m"]
        )
        assert row["control_error_m"] == pytest.approx(
            control["arms"][index]["final_turning_point_error_m"]
        )
        assert row["control_minus_arm_error_m"] == pytest.approx(
            row["control_error_m"] - row["arm_error_m"]
        )
        assert row["comparison_admissible"] is False
        assert row["control_free_circuits"] == 101
        assert row["arm_free_circuits"] == 13
    assert rows[0]["control_minus_arm_error_m"] > 0
    assert rows[1]["control_minus_arm_error_m"] < 0

    for filename, key in (
        ("shape-inverse-receipt.png", "arm_figure_titles"),
        ("all-prescribed-negative-control.png", "control_figure_titles"),
    ):
        with Image.open(FIGURES / filename) as image:
            assert json.loads(image.info["Description"])["titles"] == arm[key]
        assert len(arm[key]) == 2
        assert all("mm" in title for title in arm[key])
    assert "66.68 mm" in arm["control_figure_titles"][1]
    assert "79.68 mm" in arm["arm_figure_titles"][1]
    assert "unconverged" in arm["control_figure_titles"][1]


def test_shape_leak_into_control_is_refused():
    from benchmarks.playable_shape_inverse_receipt import _historical_comparison

    arm, control = _receipts()
    altered = copy.deepcopy(control)
    altered["arms"][1]["achieved_turning_points_m"] = arm["arms"][1][
        "achieved_turning_points_m"
    ]
    with pytest.raises(ValueError, match="stored turning-point error"):
        _historical_comparison(arm["arms"], altered["arms"])
