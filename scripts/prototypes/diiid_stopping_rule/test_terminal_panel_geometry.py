"""Pin the guarantees the terminal panels rest on.

A panel is only as durable as the two contracts behind it: the recorder writes
every field the renderer reads, and the renderer refuses a note that sits on
the field or collides with another text artist.  Neither is visible in a
finished PNG, so both are pinned here, on a synthetic lattice that needs no
machine description and no solve.

Run it by path:

    PYTHONPATH=scripts/prototypes/diiid_stopping_rule python -m pytest \
        scripts/prototypes/diiid_stopping_rule/test_terminal_panel_geometry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import record_machine_geometry  # noqa: E402
import render_terminal_panels  # noqa: E402

GEOMETRY_FIELDS = (
    "limiter",
    "limiter_source",
    "surface",
    "surface_found",
    "surface_class_margin",
    "surface_refusal",
    "efit_axis",
    "efit_x_point",
    "efit_boundary",
    "axis",
    "grid_r",
    "grid_z",
)


def _synthetic_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write one artifact and one record shaped like the probe's own."""

    radius = np.linspace(0.84, 2.54, 33)
    height = np.linspace(-1.6, 1.6, 33)
    grid_radius, grid_height = np.meshgrid(radius, height, indexing="ij")
    flux = -((grid_radius - 1.7) ** 1 + 0.4 * (grid_height - 0.0) ** 2)
    rho = np.hypot(grid_radius - 1.7, grid_height / 1.0)
    boundary = np.column_stack(
        (
            1.7 + 0.5 * np.cos(np.linspace(0, 2 * np.pi, 40, endpoint=False)),
            0.0 + 0.9 * np.sin(np.linspace(0, 2 * np.pi, 40, endpoint=False)),
        )
    )
    wall = np.column_stack(
        (
            1.7 + 1.2 * np.cos(np.linspace(0, 2 * np.pi, 40, endpoint=False)),
            0.0 + 1.3 * np.sin(np.linspace(0, 2 * np.pi, 40, endpoint=False)),
        )
    )
    artifact = tmp_path / "frame-1-limited-default.npz"
    np.savez_compressed(
        artifact,
        **{
            "radius": radius,
            "height": height,
            "wall": wall,
            "label": np.where(rho < 0.6, 3.0 - rho, rho - 3.0),
            "label_boundary": boundary,
            "frames.1.branches.limited.arms.default._flux_grid": flux,
        },
    )
    record = tmp_path / "probe-frame-1.json"
    arm = {
        "terminal_residual": 0.25,
        "qualified": False,
        "presence": {"axis_rz_m": [1.70, 0.00]},
        "x_point_rz_m": [1.60, 0.50],
    }
    record.write_text(
        json.dumps(
            {"frames": {"1": {"branches": {"limited": {"arms": {"default": arm}}}}}}
        )
    )
    output = tmp_path / "geometry-limiter-1-limited.npz"
    return artifact, record, output


def test_recorder_writes_every_field_a_panel_reads(tmp_path, monkeypatch):
    artifact, record, output = _synthetic_run(tmp_path)
    ring = np.column_stack(
        (
            1.7 + 1.1 * np.cos(np.linspace(0, 2 * np.pi, 40, endpoint=False)),
            0.0 + 1.2 * np.sin(np.linspace(0, 2 * np.pi, 40, endpoint=False)),
        )
    )
    monkeypatch.setattr(
        record_machine_geometry,
        "_governed_ring",
        lambda: (ring, '{"selector": "test"}'),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_machine_geometry.py",
            "--frame",
            "1",
            "--branch",
            "limited",
            "--artifact",
            str(artifact),
            "--record",
            str(record),
            "--output",
            str(output),
        ],
    )
    record_machine_geometry.main()

    geometry = np.load(output)
    missing = [field for field in GEOMETRY_FIELDS if field not in geometry.files]
    assert missing == []
    assert np.allclose(geometry["limiter"], ring)
    assert geometry["axis"].tolist() == [1.70, 0.00]
    axis = geometry["efit_axis"]
    assert axis[0] == pytest.approx(1.7, abs=0.12)
    assert axis[1] == pytest.approx(0.0, abs=0.11)


def test_renderer_refuses_a_note_inside_the_field(tmp_path):
    figure, axes = plt.subplots()
    axes.plot([0.0, 1.0], [0.0, 1.0])
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    inside = axes.text(0.5, 0.5, "axis R=1.70 m, Z=0.00 m", transform=axes.transAxes)
    with pytest.raises(SystemExit) as refusal:
        render_terminal_panels._refuse_bad_labels(
            figure, axes, [("axis_note", inside)], tmp_path / "x.png"
        )
    assert "inside field" in str(refusal.value)
    plt.close(figure)


def test_renderer_refuses_two_overlapping_notes(tmp_path):
    figure, axes = plt.subplots()
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    first = axes.text(0.0, -0.12, "notes one", transform=axes.transAxes, clip_on=False)
    second = axes.text(0.0, -0.13, "notes two", transform=axes.transAxes, clip_on=False)
    with pytest.raises(SystemExit) as refusal:
        render_terminal_panels._refuse_bad_labels(
            figure, axes, [("first", first), ("second", second)], tmp_path / "x.png"
        )
    assert "overlap" in str(refusal.value)
    plt.close(figure)


def test_a_pseudo_wall_is_refused():
    box = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    assert render_terminal_panels._on_bounding_box(box) == len(box)
    ring = np.vstack((box, [[0.5, 0.5]]))
    assert render_terminal_panels._on_bounding_box(ring) == len(box)
