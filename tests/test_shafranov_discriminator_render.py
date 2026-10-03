"""Render checks for the Shafranov discriminator panels and the fit figure.

The state panels must mark the magnetic axis as a solid triangle and the
admitted saddle as a filled cross in the ``draw_nulls`` vocabulary the sibling
Shafranov-row panels share, with the boundary flux among the drawn levels; the
projected flux-function fit figure must carry each freed-scale arm's own
verdict in its title, so a non-converged state is never drawn as a settled
flux.  The checks read the committed artifacts, so a painter that drops the
glyphs or a title that drops the residual fails them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

import numpy as np
import pytest

from benchmarks.shafranov_combination_discriminator import (
    LEADING_ARM_COMPONENT,
    NULL_GLYPH_STYLE,
    state_contour_levels,
)


ROOT = Path(__file__).resolve().parents[1]
#: Directory the figures are read from.  The default is the working tree; the
#: seam lets the same checks run against a scratch tree holding an earlier
#: revision's figures, so a reverted painter is shown to fail them.
FIGURE_ROOT = Path(os.environ.get("NOVA_SHAFRANOV_FIGURE_ROOT", str(ROOT)))
DISCRIMINATOR = (
    FIGURE_ROOT
    / "docs/figures/constraint-augmented-newton-krylov/shafranov-discriminator"
)
FLUX_FIT = (
    FIGURE_ROOT / "docs/figures/constraint-augmented-newton-krylov/flux-function-fit"
)
MANDATED_ARM_VERDICT = (
    "converged: false, terminal residual 5.07e-02, active_set_cycle_detected"
)

_PATH = re.compile(r'<path[^>]*d="([^"]+)"[^>]*style="([^"]*)"')
STATE_PANELS = sorted(DISCRIMINATOR.glob("row-*-state.svg"))
SHAFRANOV = FIGURE_ROOT / "docs/figures/constraint-augmented-newton-krylov/shafranov"
SHAFRANOV_ROWS = sorted(SHAFRANOV.glob("row-*.json"))
STRIP_SVG = DISCRIMINATOR / "shafranov-combination-discriminator.svg"
STRIP_RECEIPT = DISCRIMINATOR / "receipt.json"


def marker_shapes(svg_text: str, color: str) -> dict[str, int]:
    """Count the axis triangle and the saddle cross a panel draws."""
    shapes = {"triangle": 0, "cross": 0}
    for data, style in _PATH.findall(svg_text):
        if color not in style:
            continue
        vertices = data.count("L") + 1
        if vertices == 3:
            shapes["triangle"] += 1
        elif vertices == 12:
            shapes["cross"] += 1
    return shapes


@pytest.mark.parametrize("svg", STATE_PANELS, ids=lambda path: path.name)
def test_state_panel_marks_nulls_in_the_axis_and_saddle_glyphs(svg: Path) -> None:
    """Each panel draws one solid triangle and one filled cross."""
    shapes = marker_shapes(svg.read_text(encoding="utf-8"), NULL_GLYPH_STYLE.axis_color)
    assert shapes["triangle"] == 1
    assert shapes["cross"] == 1


@pytest.mark.parametrize("svg", STATE_PANELS, ids=lambda path: path.name)
def test_state_panel_draws_the_states_own_boundary_flux(svg: Path) -> None:
    """The recorded boundary flux is one of the drawn contour levels."""
    receipt = json.loads(
        svg.with_name(svg.name.replace("-state.svg", ".json")).read_text()
    )
    panel = receipt["state_panel"]
    boundary = panel["topology"]["boundary_flux_wb"]
    assert boundary is not None
    levels = panel["levels_wb"]
    nearest = min(abs(level - boundary) for level in levels)
    assert nearest < 1.0e-9


def test_state_levels_place_the_named_boundary_flux() -> None:
    """Naming the boundary flux to the level builder puts it among the levels."""
    field = np.linspace(-1.0, 100.0, 25).reshape(5, 5)
    topology = {
        "read_status": "qualified",
        "axis_flux_wb": 0.8,
        "boundary_flux_wb": -0.2,
    }
    levels = state_contour_levels(field, topology)
    assert float(np.min(levels)) == pytest.approx(-0.2)
    assert bool(np.isclose(levels, -0.2).any())


def test_flux_function_fit_title_carries_each_arms_verdict() -> None:
    """The fit figure's title states the arm's non-convergence and residual."""
    receipt = json.loads((FLUX_FIT / "receipt.json").read_text())
    entry = next(row for row in receipt["rows_receipt"] if row.get("figure"))
    arm = next(
        variant
        for variant in entry["variants"]
        if variant["component"] == LEADING_ARM_COMPONENT
    )
    needle = (
        f"converged: {str(bool(arm['converged'])).lower()}, "
        f"terminal residual {arm['terminal_residual']:.2e}, {arm['termination']}"
    )
    assert needle == MANDATED_ARM_VERDICT
    svg_text = re.sub(
        r"\s+",
        " ",
        (FLUX_FIT / f"row-{entry['identity'].replace('/', '-')}.svg").read_text(),
    )
    assert needle in svg_text
    assert needle in entry["figure"]["title"]


@pytest.mark.parametrize("receipt_path", SHAFRANOV_ROWS, ids=lambda path: path.name)
def test_shafranov_row_panel_subtitle_fits_the_canvas(receipt_path: Path) -> None:
    """The subtitle that keeps a refused row honest is wrapped to fit.

    An unbroken refusal sentence runs off both edges of the panel canvas and is
    unreadable at either end, so the receiver records the wrapped lines and the
    measured text extent, and each is required to sit inside the canvas.
    """
    figure = json.loads(receipt_path.read_text(encoding="utf-8"))["figure"]
    lines = figure["caption_lines"]
    assert len(lines) >= 2
    joined = " ".join(lines)
    assert "refused on this profile" in joined
    assert joined.endswith("shared levels")
    # The widest wrapped line is measured, not guessed: it must fit the canvas.
    assert figure["caption_widest_inches"] < figure["canvas_width_inches"]


@pytest.mark.parametrize("receipt_path", SHAFRANOV_ROWS, ids=lambda path: path.name)
def test_shafranov_row_panel_separatrix_reaches_the_marked_saddle(
    receipt_path: Path,
) -> None:
    """The drawn contours put a separatrix through the marked admitted saddle.

    The reference boundary flux is the admitted saddle's own flux, so naming it
    to the level builder makes one of the drawn lines the separatrix through
    that saddle. The panel records the gap in rendered pixels between the marked
    saddle and that contour, and it must sit within one pixel, because a level
    set measured at a null's own value can still sit far from the null where the
    map's gradient vanishes; a saddle no contour reaches reads as a marked
    X-point with no separatrix.
    """
    figure = json.loads(receipt_path.read_text(encoding="utf-8"))["figure"]
    topology = figure["reference_topology"]
    assert topology["read_status"] == "qualified"
    boundary = topology["boundary_flux_wb"]
    assert boundary is not None and np.isfinite(boundary)
    levels = figure["levels_wb"]
    nearest = min(abs(level - boundary) for level in levels)
    assert nearest < 1.0e-9
    distance = figure["saddle_contour_distance_px"]
    assert distance is not None
    assert distance < 1.0


def test_combination_strip_does_not_advertise_the_covered_series() -> None:
    """The strip draws the identical magnetics pair once and names the equality.

    The row-kernel reading equals the circular reading on every bank row, so the
    marker drawn for one sits exactly on the other and a legend entry for it
    advertises a series no reader can see.  The strip names the pair as one value.
    """
    svg_text = STRIP_SVG.read_text(encoding="utf-8")
    assert "magnetics (row kernel)" not in svg_text
    assert "magnetics (a) == row kernel" in svg_text
    assert "equals the circular reading, residual 0.0" in svg_text
    assert "read three ways" not in svg_text


def test_combination_strip_receipt_records_the_merged_series() -> None:
    """The receipt states the merge and the equality it rests on, on every row."""
    receipt = json.loads(STRIP_RECEIPT.read_text(encoding="utf-8"))
    policy = receipt["figure"]["series_policy"]
    assert policy["merged_series"] == ["magnetics_circular", "magnetics_discrete"]
    assert len(policy["drawn_series"]) == 4
    assert policy["row_kernel_versus_circular_residual_max"] == 0.0
    for row in receipt["rows_receipt"]:
        assert row["row_kernel_versus_circular_residual"] == 0.0
        assert (
            row["readings"]["magnetics_circular"]["combination"]
            == row["readings"]["magnetics_discrete"]["combination"]
        )


def test_combination_strip_suptitle_fits_the_canvas() -> None:
    """Every line of the strip's title lies inside the canvas.

    The title names both magnetics readings and the equality between them, and
    passed unwrapped its last words run off the right edge of the canvas. The
    strip records the wrapped lines and the measured extent of the widest, and
    that measurement must sit inside the canvas the strip is drawn on.
    """
    figure = json.loads(STRIP_RECEIPT.read_text(encoding="utf-8"))["figure"]
    lines = figure["suptitle_lines"]
    assert len(lines) >= 2
    joined = " ".join(lines)
    assert "read two ways" in joined
    assert "residual 0.0" in joined
    # The widest wrapped line is measured, not guessed: it must fit the canvas.
    assert figure["suptitle_widest_inches"] < figure["canvas_width_inches"]


def test_flux_function_fit_draws_the_reference_contours() -> None:
    """The fit's titled state is the reference flux, drawn as line contours.

    The receipt persists no terminal field, so the panel draws the reference
    state's flux and titles it as the reference; a title claiming a terminal
    flux with only the reference drawn is the defect this pins.
    """
    svg_text = re.sub(
        r"\s+",
        " ",
        (FLUX_FIT / "row-21978-35.svg").read_text(encoding="utf-8"),
    )
    assert "reference flux" in svg_text
    assert "terminal flux" not in svg_text
    assert "extracted" in svg_text and "projected" in svg_text
