"""The exact-clip seed panels state their terminal state and draw both null sets.

The record these checks read is the render receipt the render-only entry point
writes: one title line per panel, and per poloidal panel the null glyphs drawn
for each of the two null sets.  ``check_render_receipt`` is the whole check, so
the negative control drives the same code against a mutated record rather than
against a second implementation of the rule.

The receipt's own counts are degenerate for these rows -- every case is a
limited equilibrium with one magnetic axis and no saddle -- so the vocabulary
the counts are drawn in is exercised separately, by the tally tests below.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

from benchmarks import exact_clip_seed_amplitude as driver


ROOT = Path(__file__).resolve().parents[1]
PANELS = ROOT / "docs/figures/cut-cell-current-attribution/exact-clip-seed"
RENDER_RECEIPT = PANELS / driver.RENDER_RECEIPT_NAME
SEED_RECEIPT = PANELS / "receipt.json"
TITLE_RESIDUAL = re.compile(r"residual=(\S+) ·")
TITLE_CONVERGED = re.compile(r"converged=(True|False)")
EXPECTED_FIGURES = 12
NULL_SETS = ("analytic", "solved")
NULL_KINDS = ("axis", "admitted_saddle", "other_qualified")


def _slug_cells(slug: str) -> int:
    return -110 if slug == "reduced" else -int(slug.split("-", 1)[1])


def _part_of(figure_path: str) -> tuple[Path, str, int]:
    """Return the part receipt, the case name and the requested cells."""

    name = Path(figure_path).stem
    if name.endswith("-comparison"):
        case, text = name[: -len("-comparison")].rsplit("-", 1)
        cells = -int(text)
        slug = "reduced" if cells == -110 else f"cells-{abs(cells)}"
    else:
        case, slug = name.split("-production-route-", 1)
        cells = _slug_cells(slug)
    return PANELS / "parts" / f"{case}-production-route-{slug}.json", case, cells


def check_render_receipt(receipt: dict) -> list[str]:
    """Return every way the render receipt fails the panel rules."""

    problems: list[str] = []
    figures = receipt.get("figures", [])
    if len(figures) != EXPECTED_FIGURES:
        problems.append(
            f"the receipt lists {len(figures)} figures, expected {EXPECTED_FIGURES}"
        )
    seed_rows = {
        (row["case"], int(row["requested_cells"])): row
        for row in json.loads(SEED_RECEIPT.read_text(encoding="utf-8"))["rows"]
    }
    for figure in figures:
        name = figure["figure"]
        title = figure["title"]
        part_path, case, cells = _part_of(name)
        part = json.loads(part_path.read_text(encoding="utf-8"))
        residual, converged = driver._row_terminal(part)
        stated = TITLE_RESIDUAL.search(title)
        flag = TITLE_CONVERGED.search(title)
        if stated is None:
            problems.append(f"{name}: the title states no residual: {title}")
        elif float(stated.group(1)) != float(residual):
            problems.append(
                f"{name}: the title states residual={stated.group(1)} while its "
                f"part receipt states {residual!r}"
            )
        if flag is None:
            problems.append(f"{name}: the title states no converged flag: {title}")
        elif (flag.group(1) == "True") != converged:
            problems.append(
                f"{name}: the title states converged={flag.group(1)} while its "
                f"part receipt states {converged}"
            )
        summary = seed_rows.get((case, cells))
        if summary is None:
            problems.append(
                f"{name}: the seed receipt carries no row for {case} {cells}"
            )
        else:
            solve = summary["solve"]
            if (
                solve["terminal_residual"] != residual
                or bool(solve["converged"]) != converged
            ):
                problems.append(
                    f"{name}: the seed receipt row states residual="
                    f"{solve['terminal_residual']!r} converged="
                    f"{solve['converged']!r} while the part receipt states "
                    f"{residual!r} {converged!r}"
                )
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if digest != figure["sha256"]:
            problems.append(f"{name}: the recorded digest does not match the panel")
        panels = figure["poloidal_panels"]
        if not panels:
            problems.append(f"{name}: the receipt records no poloidal panel")
        for panel in panels:
            for label in NULL_SETS:
                counts = panel["null_glyphs"].get(label)
                if counts is None:
                    problems.append(
                        f"{name} {panel['panel']}: no {label} null set recorded"
                    )
                    continue
                if set(counts) != set(NULL_KINDS):
                    problems.append(
                        f"{name} {panel['panel']}: the {label} null set records "
                        f"{sorted(counts)} rather than {sorted(NULL_KINDS)}"
                    )
                elif counts["axis"] < 1:
                    problems.append(
                        f"{name} {panel['panel']}: the {label} null set draws no "
                        f"axis glyph"
                    )
    return problems


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads(RENDER_RECEIPT.read_text(encoding="utf-8"))


def test_the_panels_state_their_terminal_state_and_draw_both_null_sets(receipt) -> None:
    assert RENDER_RECEIPT.is_file(), (
        "rebuild the panels with the render-only entry point before running this"
    )
    problems = check_render_receipt(receipt)
    assert not problems, "\n".join(problems)


def test_the_null_tally_tells_the_three_glyph_kinds_apart() -> None:
    """The receipt's own counts are 1/0/0, so the vocabulary needs its own case."""

    figure, axes = plt.subplots()
    try:
        tally = driver.null_glyph_tally(
            axes,
            {"axis_rz_m": [1.1, 0.0], "x_point_rz_m": np.asarray([[1.3, 0.4]])},
            driver.ANALYTIC_INK_COLOR,
            None,
            other_x_points=np.asarray([[2.4, 1.9]]),
        )
    finally:
        plt.close(figure)
    assert tally == {"axis": 1, "admitted_saddle": 1, "other_qualified": 1}


def test_the_tally_drops_a_saddle_outside_the_vessel() -> None:
    """Containment is what makes the count a measurement rather than an echo."""

    wall = np.asarray([[1.0, -0.5], [1.5, -0.5], [1.5, 0.5], [1.0, 0.5]])
    figure, axes = plt.subplots()
    try:
        tally = driver.null_glyph_tally(
            axes,
            {"axis_rz_m": [1.2, 0.3], "x_point_rz_m": np.asarray([[4.0, 4.0]])},
            driver.ANALYTIC_INK_COLOR,
            (wall,),
        )
    finally:
        plt.close(figure)
    assert tally["admitted_saddle"] == 0, tally
    assert tally["axis"] == 1, tally


def test_a_title_without_the_residual_is_refused(receipt) -> None:
    """Negative control: the declared mutation strips residual= from a title."""

    mutated = json.loads(json.dumps(receipt))
    mutated["figures"][0]["title"] = mutated["figures"][0]["title"].replace(
        " residual=", " terminal-residual "
    )
    problems = check_render_receipt(mutated)
    assert any("states no residual" in problem for problem in problems), problems