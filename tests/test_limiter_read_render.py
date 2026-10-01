"""Assert the limiter-read figures carry the labels their receipt records.

The figures under ``docs/figures/cut-cell-current-attribution/limiter-read``
are rebuilt from the committed receipt by the driver's render entry point,
which also writes ``render-receipt.json``.

That render receipt is the operand these checks read: it names every title
line drawn, every legend entry, the marker each glyph family uses and, per
poloidal panel, how many glyphs each null set contributed. A figure whose
title silently loses the fixture, the cell count, the residual or the
convergence flag cannot pass, and neither can a figure whose contact glyph is
the axis triangle redrawn.

Point ``NOVA_LIMITER_READ_RENDER_RECEIPT`` at another receipt to audit it, and
at a mutated copy to watch the checks fail.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRECTORY = ROOT / "docs/figures/cut-cell-current-attribution/limiter-read"
DEFAULT_RENDER_RECEIPT = FIGURE_DIRECTORY / "render-receipt.json"
POLOIDAL_FIGURE = "single-null-contact-shadow.png"
ERROR_FIGURE = "contact-error-vs-wall-resolution.svg"
GLYPH_FAMILIES = (
    "magnetic_axis",
    "admitted_x_point",
    "wall_contact",
    "excluded_private_wall_nodes",
)


def _receipt_path() -> Path:
    return Path(
        os.environ.get("NOVA_LIMITER_READ_RENDER_RECEIPT", str(DEFAULT_RENDER_RECEIPT))
    )


@pytest.fixture(scope="module")
def render_receipt() -> dict:
    return json.loads(_receipt_path().read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source_receipt() -> dict:
    return json.loads(
        (_receipt_path().parent / "receipt.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def poloidal_figure(render_receipt: dict) -> dict:
    for record in render_receipt["figures"]:
        if record["figure"] == POLOIDAL_FIGURE:
            return record
    raise AssertionError(f"{POLOIDAL_FIGURE} is absent from the render receipt")


def _source_row(source_receipt: dict, figure: dict) -> dict:
    key = figure["source_row_key"]
    matches = [
        row
        for row in source_receipt["rows"]
        if row["case"] == key["case"]
        and int(row["requested_cells"]) == int(key["requested_cells"])
        and int(row["wall_nodes"]) == int(key["wall_nodes"])
    ]
    assert len(matches) == 1, f"the source receipt has {len(matches)} rows for {key}"
    return matches[0]


def test_the_render_entry_point_records_a_completed_figure_set(render_receipt):
    assert render_receipt["schema"] == "nova.limiter-read-render-receipt"
    assert render_receipt["completed"] is True
    assert render_receipt["exit_marker"] == "LIMITER_READ_RENDER_EXIT=0"
    assert render_receipt["render_entry_point"] == (
        "benchmarks/limiter_read_resolution_audit.py render"
    )
    assert len(render_receipt["source_receipt_sha256"]) == 64
    names = {record["figure"] for record in render_receipt["figures"]}
    assert names == {POLOIDAL_FIGURE, ERROR_FIGURE}


def test_the_title_reports_the_fixture_cells_residual_and_convergence_flag(
    poloidal_figure, source_receipt
):
    joined = " ".join(poloidal_figure["title_lines"])
    fields = poloidal_figure["title_fields"]
    row = _source_row(source_receipt, poloidal_figure)

    residual = float(row["production_contact"]["level_error_in_span"])
    entered = bool(source_receipt["nonlinear_solve_entered"])
    assert fields["residual"]["source_value"] == pytest.approx(residual)
    assert fields["converged"]["source_value"] == entered
    assert fields["cells"]["source_value"] == int(row["realised_cells"])

    assert row["case"] in joined
    assert f"cells={int(row['realised_cells'])}" in joined
    assert f"residual={residual:.3e}" in joined
    assert f"converged={'yes' if entered else 'n/a'}" in joined


def test_every_glyph_family_the_figure_draws_has_a_legend_entry(poloidal_figure):
    entries = poloidal_figure["legend_entries"]
    for family in GLYPH_FAMILIES:
        assert family in entries


def test_the_contact_glyph_is_distinct_from_the_axis_and_the_saddle(poloidal_figure):
    markers = poloidal_figure["glyph_markers"]
    for family in GLYPH_FAMILIES:
        assert family in markers
    contact = markers["wall_contact"]
    assert contact != markers["magnetic_axis"]
    assert contact != markers["admitted_x_point"]


def test_the_wall_is_drawn_with_the_source_receipt_node_count(
    poloidal_figure, source_receipt
):
    assert poloidal_figure["wall_drawn"] is True
    row = _source_row(source_receipt, poloidal_figure)
    assert poloidal_figure["wall_node_count"] == len(row["render"]["wall"])


def test_each_poloidal_panel_records_the_null_glyphs_of_every_null_set(poloidal_figure):
    panels = poloidal_figure["poloidal_panels"]
    assert panels
    for panel in panels:
        for name, tally in panel["null_sets"].items():
            assert set(tally) == {"drawn", "dropped_outside_wall"}
            assert tally["drawn"] + tally["dropped_outside_wall"] >= 0
        assert panel["null_sets"]["analytic_magnetic_axis"]["drawn"] == 1
        assert panel["null_sets"]["analytic_admitted_x_points"]["drawn"] == 1
        assert panel["null_glyph_total"] == 2
        assert panel["wall_contact_glyphs"] == 1


def test_the_figure_files_the_receipt_names_are_on_disk(render_receipt):
    for record in render_receipt["figures"]:
        path = FIGURE_DIRECTORY / record["figure"]
        assert path.is_file(), f"{path} is missing"
        assert path.stat().st_size > 0
