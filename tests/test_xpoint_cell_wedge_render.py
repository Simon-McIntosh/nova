"""Audit the X-point-cell wedge panels against the null-glyph vocabulary.

Every panel draws two null sets over one another -- the analytic reference and
the production read -- so the drawing has to keep them separable and keep every
glyph a shape the ``draw_nulls`` vocabulary defines. The checks read the render
receipt the panels were written with, so a panel that was never regenerated
cannot pass them, and one check re-renders a panel live so the vocabulary is
guarded at the artist rather than only at the receipt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from benchmarks import xpoint_cell_wedge_oracle as oracle

DECLARED_NEGATIVE_CONTROL = (
    "draw the solved axis with the inverted triangle again and observe the "
    "marker check fail"
)
MUTATION_ENV = "NOVA_XPOINT_WEDGE_MUTATION"
MUTATION_INVERTED_AXIS = "inverted-axis-marker"


def receipt() -> dict[str, Any]:
    path = oracle.DEFAULT_OUTPUT / oracle.RENDER_RECEIPT_NAME
    if not path.exists():
        pytest.fail(
            f"no render receipt at {path}; run "
            "`python -m benchmarks.xpoint_cell_wedge_oracle --render-only` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def panel_rows() -> list[dict[str, Any]]:
    rows = receipt()["panels"]
    if not rows:
        pytest.fail("the render receipt lists no panels")
    return rows


def part_row(requested_cells: int) -> dict[str, Any]:
    path = oracle._part_path(oracle.DEFAULT_OUTPUT, requested_cells)
    return json.loads(path.read_text(encoding="utf-8"))


def test_render_receipt_audits_every_declared_panel() -> None:
    document = receipt()
    assert document["completed"] is True
    assert document["panels_rendered"] == list(oracle.RENDER_CELL_COUNTS)
    assert document["all_reference_axes_drawn"] is True
    assert document["all_reference_saddles_drawn"] is True
    assert document["no_unexpected_markers"] is True
    assert document["all_walls_drawn"] is True
    assert document["wall_node_count"] == 121


def test_every_panel_draws_both_null_sets_in_the_vocabulary() -> None:
    for panel in panel_rows():
        cells = panel["requested_cells"]
        glyphs = panel["glyphs"]
        assert glyphs["analytic_axis"] == 1, cells
        assert glyphs["analytic_saddle"] == 1, cells
        assert glyphs["solved_axis"] >= 1, (cells, glyphs)
        assert panel["unexpected_markers"] == [], (cells, panel["unexpected_markers"])
        if panel["saddle_admitted"] and panel["axis_admitted"]:
            assert glyphs["solved_saddle"] == 1, (cells, glyphs)


def test_no_downward_triangle_is_drawn_for_an_axis() -> None:
    for panel in panel_rows():
        drawn = set(panel["unexpected_markers"])
        assert "v" not in drawn, panel["requested_cells"]
        assert drawn == set(), panel["requested_cells"]


def test_every_panel_carries_its_wall_legend_and_regenerated_title() -> None:
    for panel in panel_rows():
        cells = panel["requested_cells"]
        assert panel["wall_drawn"] is True, cells
        assert panel["wall_node_count"] == 121, cells
        labels = panel["legend_labels"]
        assert any("solved" in label for label in labels), labels
        assert any("analytic" in label for label in labels), labels
        assert len(panel["title_lines"]) == 2, panel["title_lines"]
        assert panel["title_lines"][0].startswith(
            f"{cells} requested / {panel['realised_cells']} realised cells"
        ), panel["title_lines"][0]
        svg = Path(panel["figure_svg_path"]).read_text(encoding="utf-8")
        png = Path(panel["figure_png_path"])
        assert png.exists() and png.stat().st_size > 0, panel["figure_png_path"]
        assert "analytic nulls hollow blue" in svg, cells
        assert "analytic nulls blue;" not in svg, cells
        for label in labels:
            assert label in svg, (cells, label)


def test_the_marker_check_refuses_a_live_inverted_axis_marker() -> None:
    if os.environ.get(MUTATION_ENV) != MUTATION_INVERTED_AXIS:
        pytest.skip(
            f"declared negative control: set {MUTATION_ENV}={MUTATION_INVERTED_AXIS}"
        )
    print(DECLARED_NEGATIVE_CONTROL, flush=True)
    machine, exact = oracle._render_carrier(oracle.REFERENCE_CELLS)
    resolved = oracle._resolve_xpoint_cell(machine, exact)
    part = part_row(oracle.REFERENCE_CELLS)
    oracle.SOLVED_AXIS_MARKER = "v"
    panel = oracle._draw_panel(
        oracle.DEFAULT_OUTPUT,
        oracle.REFERENCE_CELLS,
        machine,
        exact,
        resolved["polygon"],
        resolved["wedges"],
        part["observed_nulls"],
        write=False,
    )
    assert panel["unexpected_markers"] == [], (
        "the vocabulary check admitted a marker outside the draw_nulls "
        f"vocabulary: {panel['unexpected_markers']}"
    )
