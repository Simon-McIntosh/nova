"""The multi-unit wall figure obeys the plotting rules and keeps units distinct.

The figure is regenerated from a committed receipt, so these checks drive the
receipt-fed renderer rather than re-running a machine-description read.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from benchmarks import wall_units_render as renderer

SOLID = "-"


@pytest.fixture(scope="module")
def rendered():
    receipt = renderer.load_receipt()
    figure, axes = renderer.render(receipt)
    yield receipt, figure, axes
    plt.close(figure)


def _visible_gridlines(axes) -> list:
    return [
        line
        for line in list(axes.get_xgridlines()) + list(axes.get_ygridlines())
        if line.get_visible()
    ]


def _unit_lines(axes) -> list:
    """The per-unit polylines: the endpoint markers carry no line."""

    return [line for line in axes.lines if str(line.get_linestyle()).lower() != "none"]


def _is_solid(line) -> bool:
    return str(line.get_linestyle()) == SOLID


def test_both_panels_draw_no_axes_and_no_gridlines(rendered):
    _, _, axes = rendered
    assert len(axes) == 2
    for panel_axes in axes:
        assert panel_axes.axison is False
        # The gridline artists exist (six per axis), so this reads their
        # visibility rather than passing over an empty list.
        assert list(panel_axes.get_xgridlines())
        assert _visible_gridlines(panel_axes) == []


def test_left_panel_keeps_its_two_units(rendered):
    _, _, axes = rendered
    lines = _unit_lines(axes[0])
    assert len(lines) == 2
    assert sum(_is_solid(line) for line in lines) == 1
    assert sum(not _is_solid(line) for line in lines) == 1


def test_right_panel_draws_five_distinct_units_matching_the_first_wall(rendered):
    receipt, _, axes = rendered
    jt60sa = next(panel for panel in receipt["panels"] if panel["name"] == "jt60sa")
    units = jt60sa["units"]
    assert len(units) == 5
    assert [row["closed"] for row in units] == [True, False, False, False, False]

    lines = _unit_lines(axes[1])
    assert len(lines) == 5

    styles = {(line.get_color(), line.get_linestyle()) for line in lines}
    assert len(styles) == 5, "the five units must be distinguishable"

    for row, line in zip(units, lines, strict=True):
        assert _is_solid(line) is bool(row["closed"])


def test_jt60sa_units_are_one_closed_vessel_and_four_open_structures(rendered):
    receipt, _, _ = rendered
    jt60sa = next(panel for panel in receipt["panels"] if panel["name"] == "jt60sa")
    vessel = [row for row in jt60sa["units"] if row["closed"]]
    assert [row["kind"] for row in vessel] == ["vessel"]
    assert [row["index"] for row in jt60sa["units"]] == [0, 1, 2, 3, 4]
    assert all(not row["closed"] for row in jt60sa["units"][1:])
