"""Assert the exact-clip memory figures carry the values their receipt records.

The memory-scaling figure under
``docs/figures/cut-cell-current-attribution/exact-clip-memory`` is rebuilt from
the committed receipt by the driver's render entry point, which also writes
``render-receipt.json`` beside it and the solve panel.

Every mark on the scaling figure is plotted from a byte count in the source
receipt, and the render receipt records that coordinate beside the value it came
from. The checks below re-derive the coordinate from the receipt and require the
rendered point and its printed label to agree with it, so a figure whose marks
drift away from the receipt cannot pass.

Point ``NOVA_EXACT_CLIP_MEMORY_RENDER_RECEIPT`` at another render receipt to
audit it, and at a copy taken beside a perturbed source receipt to watch the
checks fail.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRECTORY = ROOT / "docs/figures/cut-cell-current-attribution/exact-clip-memory"
DEFAULT_RENDER_RECEIPT = FIGURE_DIRECTORY / "render-receipt.json"
SOURCE_RECEIPT_NAME = "base-memory.json"
SCALING_FIGURE = "base-memory-scaling.svg"
SCALING_PNG = "base-memory-scaling.png"
SOLVE_PANEL = (
    "solve-panels-final-receipted/"
    "weak-rotation-reactor-static-production-route-cells-1000.png"
)
RECEIPT_FIGURES = (SCALING_FIGURE, SOLVE_PANEL)


def _render_receipt_path() -> Path:
    return Path(
        os.environ.get(
            "NOVA_EXACT_CLIP_MEMORY_RENDER_RECEIPT", str(DEFAULT_RENDER_RECEIPT)
        )
    )


@pytest.fixture(scope="module")
def render_receipt() -> dict:
    return json.loads(_render_receipt_path().read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source_receipt() -> dict:
    return json.loads(
        (_render_receipt_path().parent / SOURCE_RECEIPT_NAME).read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def scaling_figure(render_receipt: dict) -> dict:
    for record in render_receipt["figures"]:
        if record["figure"] == SCALING_FIGURE:
            return record
    raise AssertionError(f"{SCALING_FIGURE} is absent from the render receipt")


@pytest.fixture(scope="module")
def solve_panel(render_receipt: dict) -> dict:
    for record in render_receipt["figures"]:
        if record["figure"] == SOLVE_PANEL:
            return record
    raise AssertionError(f"{SOLVE_PANEL} is absent from the render receipt")


def _source_gib(source_receipt: dict, realised_cells: int) -> float:
    matches = [
        row
        for row in source_receipt["rows"]
        if int(row["realised_cells"]) == int(realised_cells)
    ]
    assert len(matches) == 1, (
        f"the source receipt has {len(matches)} rows at {realised_cells}"
    )
    return matches[0]["memory_analysis"]["temp_size_in_bytes"] / 2**30


def test_the_render_entry_point_records_a_completed_figure_set(render_receipt):
    assert render_receipt["schema"] == "nova.exact-clip-memory-render-receipt"
    assert render_receipt["completed"] is True
    assert render_receipt["render_entry_point"] == (
        "benchmarks/exact_clip_memory_scaling.py render"
    )
    assert render_receipt["source_receipt"] == SOURCE_RECEIPT_NAME
    assert len(render_receipt["source_receipt_sha256"]) == 64
    names = {record["figure"] for record in render_receipt["figures"]}
    assert names == set(RECEIPT_FIGURES)


def test_every_plotted_point_equals_its_source_receipt_value(
    scaling_figure, source_receipt
):
    points = scaling_figure["points"]
    assert len(points) == len(source_receipt["rows"])
    for point in points:
        source_gib = _source_gib(source_receipt, point["realised_cells"])
        assert point["receipt_plotted_gib"] == pytest.approx(source_gib)
        assert point["plotted_y_gib"] == pytest.approx(source_gib)
        assert point["printed_label"] == f"{source_gib:.2f}"
        assert point["receipt_temporary_bytes"] == int(source_gib * 2**30)


def test_the_printed_labels_are_the_ones_the_figure_draws(scaling_figure):
    labels = {point["printed_label"] for point in scaling_figure["points"]}
    assert labels == {"25.47", "51.81", "70.41"}
    for point in scaling_figure["points"]:
        assert point["above_gate"] is (
            point["plotted_y_gib"] > scaling_figure["gate_gib"]
        )


def test_the_render_receipt_lists_the_title_lines_of_every_figure(render_receipt):
    for record in render_receipt["figures"]:
        assert record["title_lines"]
        assert all(line.strip() for line in record["title_lines"])


def test_the_solve_panel_title_prints_the_residual_and_the_convergence_flag(
    solve_panel,
):
    joined = " ".join(solve_panel["title_lines"])
    assert f"cells={solve_panel['realised_cells']}" in joined
    assert f"residual={solve_panel['terminal_residual']:.3e}" in joined
    assert "converged=yes" in joined
    assert solve_panel["converged"] is True


def test_each_panel_records_the_glyphs_of_every_null_set(solve_panel):
    panels = solve_panel["poloidal_panels"]
    assert panels
    for panel in panels:
        assert panel["null_sets"]
        for tally in panel["null_sets"].values():
            assert set(tally) == {"drawn", "dropped_outside_wall"}
            assert tally["drawn"] >= 0
            assert tally["dropped_outside_wall"] >= 0
        total = sum(tally["drawn"] for tally in panel["null_sets"].values())
        assert panel["null_glyph_total"] == total
        assert panel["wall_node_count"] > 0


def test_the_figure_files_the_receipt_names_are_on_disk(render_receipt):
    for record in render_receipt["figures"]:
        path = (_render_receipt_path().parent / record["figure"]).resolve()
        assert path.is_file(), f"{path} is missing"
        assert path.stat().st_size > 0
    companion = (_render_receipt_path().parent / SCALING_PNG).resolve()
    assert companion.is_file()
    assert companion.stat().st_size > 0
