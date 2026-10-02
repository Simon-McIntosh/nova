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
    FIGURE_ROOT / "docs/figures/constraint-augmented-newton-krylov/shafranov-discriminator"
)
FLUX_FIT = FIGURE_ROOT / "docs/figures/constraint-augmented-newton-krylov/flux-function-fit"
MANDATED_ARM_VERDICT = (
    "converged: false, terminal residual 5.07e-02, active_set_cycle_detected"
)

_PATH = re.compile(r'<path[^>]*d="([^"]+)"[^>]*style="([^"]*)"')
STATE_PANELS = sorted(DISCRIMINATOR.glob("row-*-state.svg"))


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