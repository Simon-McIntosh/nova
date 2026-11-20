"""Checks that the published panels draw the refused terminal iterate.

The panels exist to distinguish a converging arm from a refused one. A series
contoured on levels it never crosses draws no path at all, so a panel can show
a complete legend and no second field.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import panels  # noqa: E402

PUBLISHED = ROOT / "docs/figures/forward-solve-near-stored-states/stored-state-refusals"


def test_shared_levels_span_both_fields():
    """One level array, chosen from both fields, so neither series is dropped."""
    reference = [0.2, 0.6]
    solved = [-1.0, 0.1]
    levels = panels.shared_levels(reference, solved)
    assert levels.size == panels.LEVEL_COUNT
    assert levels[0] < reference[0] and levels[-1] > solved[-1]
    for low, high in (reference, solved):
        inside = int(np.sum((levels >= low) & (levels <= high)))
        assert inside >= 5, (low, high, inside)
    stored_only = np.linspace(*reference, 14)[1:-1]
    assert not np.any((stored_only >= solved[0]) & (stored_only <= solved[1]))


def test_drawn_series_reports_a_series_that_drew_nothing(tmp_path):
    """An empty path attribute is the difference between a drawn series and none."""
    svg = tmp_path / "panel.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="10cm" height="10cm">'
        '<g id="axes_1"><g id="QuadContourSet_1">'
        '<path style="fill: none; stroke: #929292" d="M 0 0 L 1 1"/>'
        '</g><g id="TriContourSet_1">'
        '<path style="fill: none; stroke: #3366cc" d=""/>'
        "</g></g></svg>"
    )
    assert panels.drawn_series(svg) == {"imas-efit": 1, "nova": 0}


def test_published_panels_draw_both_series():
    """Every cited panel carries a drawn contour from each series."""
    svgs = sorted(PUBLISHED.glob("*.svg"))
    assert len(svgs) == 6, [path.name for path in svgs]
    empty = {}
    for path in svgs:
        counts = panels.drawn_series(path)
        if counts["nova"] == 0 or counts["imas-efit"] == 0:
            empty[path.name] = counts
        descriptions = [
            element
            for element in ElementTree.parse(path).getroot().iter()
            if element.tag.rsplit("}", 1)[-1] == "desc"
        ]
        assert descriptions and (descriptions[0].text or "").strip(), path.name
    assert empty == {}


def test_published_previews_draw_the_blue_series():
    """A preview must not advertise a series its own pixels do not contain.

    The repaired SVGs carried the nova contours while the PNG beside them still
    showed the stored field alone: 127 blue pixels, all of them the legend
    swatch. The check asks the pixels below the legend band.
    """
    previews = {path.name: path for path in PUBLISHED.glob("*-preview.png")}
    assert {
        "iter-iter-stored-default-preview.png",
        "tcv-tcv-stored-default-preview.png",
    } <= set(previews)
    empty = {}
    for name, path in previews.items():
        counts = panels.preview_series(path)
        if counts["nova"] < 100:
            empty[name] = counts
    assert empty == {}
