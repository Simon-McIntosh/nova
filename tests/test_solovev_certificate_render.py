"""The certificate panel draws both null sets, unfilled, from a persisted part.

A certificate panel puts a solved terminal state beside the closed-form
reference, and the reader's whole reason for looking is the comparison. Two
failures defeat that: a panel carrying only one set of stationary points, so a
solved null draws alone and reads as the answer, and a scalar map painted as a
filled contour, where independent colour scales can make two maps agree that do
not. Both are pinned here on the persisted part the pin cites, so the guard runs
against the shipped input rather than against a fixture of its own making.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import pytest

from benchmarks import solovev_certificate as certificate

PART = (
    certificate.ROOT
    / "docs/figures/gs-absolute-accuracy/solovev/production-route-parts"
    / "diverted-single-null-production-route-cells-300.json"
)
OUTPUT_DIRECTORY = "docs/figures/gs-absolute-accuracy"
ANALYTIC = matplotlib.colors.to_hex(certificate.ANALYTIC_INK_COLOR)
SOLVED = matplotlib.colors.to_hex(certificate.SOLVED_INK_COLOR)


def _persisted_row() -> dict:
    assert PART.is_file(), f"the persisted part the pin cites is missing: {PART}"
    row = json.loads(PART.read_text(encoding="utf-8"))
    assert "render_data" in row, "the persisted part cannot rebuild its own panel"
    return row


def _draw(row: dict, path: Path):
    """Rebuild one panel in memory from the part, returning the open figure."""

    data = row["render_data"]
    errors = {
        name: np.asarray(data["error_fields"][name]) for name in certificate.NORM_FIELDS
    }
    return certificate._plot(
        np.asarray(data["coordinates_rz_m"]),
        np.asarray(data["terminal_flux_wb"]),
        np.asarray(data["analytic_flux_wb"]),
        np.asarray(data["derivative_coordinates_rz_m"]),
        errors,
        np.asarray(data["boundary_rz_m"]),
        np.asarray(data["wall_units_rz_m"][0]),
        data["terminal_topology"],
        data["analytic_topology"],
        path,
        f"{row['case']} · {certificate._slug(row['requested_cells'])}",
    )


def _stationary_markers(axes) -> dict[str, dict[str, list[float]]]:
    """Group the station-point markers on ``axes`` by colour and marker."""

    grouped: dict[str, dict[str, list[float]]] = {}
    for line in axes.lines:
        if str(line.get_linestyle()).lower() not in {"none", ""}:
            continue
        marker = line.get_marker()
        if marker is None or str(marker).lower() == "none":
            continue
        colour = matplotlib.colors.to_hex(line.get_color())
        grouped.setdefault(colour, {}).setdefault(str(marker), []).append(
            float(line.get_xdata()[0])
        )
    return grouped


def _with_analytic_saddle(row: dict, offset: float) -> dict:
    """Move the persisted part onto a reference that carries an X-point.

    The four pin cases read limited, and a limited closed-form solution has no
    saddle to draw; injecting one onto the reference's own axis is how the
    saddle vocabulary of the analytic set is exercised at all.
    """

    injected = copy.deepcopy(row)
    for name in ("analytic_topology", "terminal_topology"):
        topology = injected["render_data"][name]
        axis = topology["axis_rz_m"]
        topology["x_point_rz_m"] = [[axis[0] + offset, axis[1]]]
    return injected


@pytest.fixture(scope="module")
def persisted_row() -> dict:
    return _persisted_row()


def test_the_panel_draws_the_analytic_axis_beside_the_solved_axis(
    tmp_path, persisted_row
):
    figure = _draw(persisted_row, tmp_path / "panel.png")
    try:
        markers = _stationary_markers(figure.axes[0])
    finally:
        matplotlib.pyplot.close(figure)
    assert set(markers) >= {ANALYTIC, SOLVED}, (
        "the flux panel must carry both null sets in their own colours, "
        f"found {sorted(markers)}"
    )
    for colour in (ANALYTIC, SOLVED):
        assert "^" in markers[colour], "the magnetic axis is drawn as a triangle"


def test_the_panel_draws_a_saddle_for_both_sets(tmp_path, persisted_row):
    figure = _draw(_with_analytic_saddle(persisted_row, 0.05), tmp_path / "panel.png")
    try:
        markers = _stationary_markers(figure.axes[0])
        saddle = str(certificate.DEFAULT_INK.xpoint_marker)
        assert saddle in markers.get(ANALYTIC, {}), (
            "the analytic saddle must be drawn in the analytic colour, "
            f"found {sorted(markers)}"
        )
        assert saddle in markers.get(SOLVED, {})
        assert len(markers[ANALYTIC][saddle]) == 1
    finally:
        matplotlib.pyplot.close(figure)
    assert persisted_row["render_data"]["analytic_topology"]["x_point_rz_m"] is None, (
        "the injected saddle must not be written back into the persisted part"
    )


def test_the_scalar_maps_are_unfilled_contours_with_the_axis_off(
    tmp_path, persisted_row
):
    figure = _draw(persisted_row, tmp_path / "panel.png")
    try:
        for axes in figure.axes:
            assert axes.axison is False, "a poloidal panel draws no axis furniture"
            for collection in axes.collections:
                assert getattr(collection, "filled", False) is False
        assert figure.axes[0].collections, "the flux panel carries line contours"
        assert figure.axes[0].lines, "the flux panel carries its vessel and nulls"
    finally:
        matplotlib.pyplot.close(figure)


def test_the_rerender_receipts_the_bitmap_and_its_vector(
    tmp_path, monkeypatch, persisted_row
):
    """The receipt is read back off the two files the renderer wrote."""

    (tmp_path / OUTPUT_DIRECTORY).mkdir(parents=True)
    monkeypatch.setattr(certificate, "ROOT", tmp_path)
    row = copy.deepcopy(persisted_row)
    row["figure"] = {"filesystem_path": f"{OUTPUT_DIRECTORY}/panel.png"}
    row = certificate._render_persisted_row(row)
    figure = row["figure"]
    assert figure["render_source"] == "persisted_part_receipt"
    assert (
        figure["project_absolute_src"] == "/nova/figures/gs-absolute-accuracy/panel.png"
    )
    assert figure["vector_project_absolute_src"].endswith("panel.svg")
    bitmap = tmp_path / figure["filesystem_path"]
    vector = tmp_path / figure["vector_filesystem_path"]
    assert hashlib.sha256(bitmap.read_bytes()).hexdigest() == figure["sha256"]
    assert hashlib.sha256(vector.read_bytes()).hexdigest() == figure["vector_sha256"]
    assert row["stage_wall_seconds"]["figure_render"] >= 0.0
