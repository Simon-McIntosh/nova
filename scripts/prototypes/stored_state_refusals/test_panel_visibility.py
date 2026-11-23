"""Checks that every published panel draws the refused terminal iterate.

The panels exist to distinguish a converging arm from a refused one. A series
contoured on levels it never crosses draws no path at all, so a panel can show a
complete legend and no second field. Every directory this record publishes panels
into is checked, and the record fragment is checked to cite each of them, so a
panel that a driver stopped drawing cannot pass review unnoticed.
"""

from __future__ import annotations

import re
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

FIGURES = ROOT / "docs/figures/forward-solve-near-stored-states"
REFUSALS = FIGURES / "stored-state-refusals"
DRIFT = FIGURES / "default-policy-refusal-drift"
FRAGMENT = (
    ROOT
    / "docs/evidence/fragments/forward-solve-near-stored-states"
    / "default-policy-refusal-drift.html"
)
CITED = re.compile(
    r"/nova/figures/forward-solve-near-stored-states/"
    r"default-policy-refusal-drift/([A-Za-z0-9._-]+\.svg)"
)


def dropped_series(root: Path) -> dict:
    """Return every panel of ``root`` that promises a series it does not draw."""
    empty = {}
    for path in sorted(root.glob("*.svg")):
        counts = panels.drawn_series(path)
        if counts["nova"] == 0 or counts["imas-efit"] == 0:
            empty[path.name] = counts
    return empty


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
    assert dropped_series(tmp_path) == {"panel.svg": {"imas-efit": 1, "nova": 0}}


def test_the_publication_check_rejects_a_dropped_series(tmp_path):
    """A copy of a published panel with its blue paths emptied must be rejected.

    The negative control for this check: the guard above is only worth its noise
    if a panel that stopped drawing the solved field fails it.
    """
    source = sorted(DRIFT.glob("*.svg"))[0]
    text = source.read_text()
    pattern = re.compile(r'(<ns0:path )d="[^"]*"([^>]*stroke: #3366cc[^>]*>)')
    emptied, changed = pattern.subn(r'\1d=""\2', text)
    assert changed
    assert emptied != text
    (tmp_path / source.name).write_text(emptied)
    assert source.name in dropped_series(tmp_path)
    assert dropped_series(tmp_path)[source.name]["nova"] == 0


def test_published_panels_draw_both_series():
    """Every panel of every published directory carries a contour from each series."""
    for root in (REFUSALS, DRIFT):
        svgs = sorted(root.glob("*.svg"))
        assert len(svgs) == 6, (root, [path.name for path in svgs])
        for path in svgs:
            descriptions = [
                element
                for element in ElementTree.parse(path).getroot().iter()
                if element.tag.rsplit("}", 1)[-1] == "desc"
            ]
            assert descriptions and (descriptions[0].text or "").strip(), path.name
        assert dropped_series(root) == {}, root


def test_drift_fragment_cites_every_published_panel():
    """The record cites each panel this directory publishes, and cites no other."""
    cited = set(CITED.findall(FRAGMENT.read_text()))
    published = {path.name for path in DRIFT.glob("*.svg")}
    assert cited == published


def test_published_previews_draw_the_blue_series():
    """A preview must not advertise a series its own pixels do not contain.

    The repaired SVGs carried the nova contours while the PNG beside them still
    showed the stored field alone: 127 blue pixels, all of them the legend
    swatch. The check asks the pixels below the legend band. Only the refusal
    directory commits previews; the drift directory's previews are written into
    the run directory with the panels they were rendered from.
    """
    previews = {path.name: path for path in REFUSALS.glob("*-preview.png")}
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
