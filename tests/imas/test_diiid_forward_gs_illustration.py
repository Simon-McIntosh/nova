from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from benchmarks import diiid_forward_gs_illustration as illustration


def _frame(shot: str, frame: int, fractional_rms: float) -> illustration.FrameMaps:
    radius = np.linspace(1.0, 2.0, 9)
    height = np.linspace(-0.8, 0.8, 9)
    radius_map, height_map = np.meshgrid(radius, height, indexing="ij")
    labelled = (radius_map - 1.5) ** 2 + height_map**2
    predicted = labelled + fractional_rms * np.ptp(labelled)
    predicted[[0, -1], :] = labelled[[0, -1], :]
    predicted[:, [0, -1]] = labelled[:, [0, -1]]
    contour = np.array([[1.2, 0.0], [1.5, 0.5], [1.8, 0.0], [1.2, 0.0]])
    measured, r_squared = illustration.field_metrics(labelled, predicted)
    metrics = illustration.FrameMetrics(
        shot=shot,
        frame=frame,
        time_ms=100.0 * frame,
        route=illustration.ROUTE,
        converged=True,
        convergence_criterion="relative update <= 1e-8",
        iterations=12,
        final_relative_update=8.0e-9,
        final_fixed_point_relative_update=4.0e-8,
        final_relaxation=0.2,
        reliable_extraction_surfaces=19,
        interior_fractional_rms=measured,
        representability_ceiling=illustration.LABEL_REPRESENTABILITY_CEILING,
        fractional_rms_to_ceiling_ratio=(
            measured / illustration.LABEL_REPRESENTABILITY_CEILING
        ),
        interior_r_squared=r_squared,
    )
    return illustration.FrameMaps(
        metrics=metrics,
        radius=radius,
        height=height,
        predicted=predicted,
        labelled=labelled,
        labelled_lcfs=contour,
    )


def test_named_cohort_and_context_are_quantitatively_locked():
    assert len(illustration.NAMED_FRAMES) >= 3
    assert len(set(illustration.NAMED_FRAMES)) == len(illustration.NAMED_FRAMES)
    assert illustration.LABEL_REPRESENTABILITY_CEILING == pytest.approx(0.0429)
    assert round(illustration.FREE_BOUNDARY_BOUNDARY_FIELD_SHARE, 4) == 0.8499
    assert "demonstration, not a passed production gate" in illustration.CAPTION


def test_field_metric_uses_the_interior_and_labelled_flux_range():
    labelled = np.arange(25, dtype=float).reshape(5, 5)
    predicted = labelled.copy()
    predicted[1:-1, 1:-1] += 2.0
    fractional_rms, r_squared = illustration.field_metrics(labelled, predicted)
    assert fractional_rms == pytest.approx(2.0 / 24.0)
    assert r_squared < 1.0


def test_boundary_share_is_read_from_the_root_existence_receipt(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "result": {
                    "pooled": {
                        "attribution": {
                            "vacuum_difference_share": (
                                illustration.FREE_BOUNDARY_BOUNDARY_FIELD_SHARE
                            )
                        }
                    }
                }
            }
        )
    )
    assert round(illustration.boundary_field_share(path), 4) == 0.8499
    changed = json.loads(path.read_text())
    changed["result"]["pooled"]["attribution"]["vacuum_difference_share"] = 0.5
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="has changed"):
        illustration.boundary_field_share(path)


def test_receipt_refuses_to_present_the_illustration_as_a_gate_pass():
    frames = [_frame(f"shot_{index}.parquet", index, 0.01) for index in range(3)]
    record = illustration.receipt(
        frames, illustration.FREE_BOUNDARY_BOUNDARY_FIELD_SHARE
    )
    assert record["named_shot_frame_pairs"] == 3
    assert record["route"]["all_frames_converged"] is True
    assert record["free_boundary_context"]["production_gate_passed"] is False
    assert record["free_boundary_context"]["displayed_share"] == pytest.approx(0.8499)
    assert all(
        frame["representability_ceiling"] == pytest.approx(0.0429)
        for frame in record["frames"]
    )


def test_figure_has_three_map_columns_and_qualified_caption(tmp_path, monkeypatch):
    frames = [_frame(f"shot_{index}.parquet", index, 0.01) for index in range(3)]
    captured: dict[str, object] = {}

    def capture(figure, *args, **kwargs):
        captured["axes"] = len(figure.axes)
        captured["text"] = " ".join(
            [*(text.get_text() for text in figure.texts)]
            + [axis.get_title() for axis in figure.axes]
        )
        Path(args[0]).touch()

    monkeypatch.setattr(plt.Figure, "savefig", capture)
    path = illustration.render(
        frames, tmp_path, illustration.FREE_BOUNDARY_BOUNDARY_FIELD_SHARE
    )
    assert path.is_file()
    assert captured["axes"] == 9
    assert "boundary-field share = 0.8499" in captured["text"]
    assert "not a passed production gate" in captured["text"]
    assert "frac RMS" in captured["text"]


def test_receipt_exposes_nonconvergence_if_a_frame_is_not_qualified():
    frames = [_frame(f"shot_{index}.parquet", index, 0.01) for index in range(3)]
    failed_metric = replace(frames[0].metrics, converged=False)
    frames[0] = replace(frames[0], metrics=failed_metric)
    record = illustration.receipt(
        frames, illustration.FREE_BOUNDARY_BOUNDARY_FIELD_SHARE
    )
    assert record["route"]["all_frames_converged"] is False
    assert record["frames"][0]["converged"] is False
