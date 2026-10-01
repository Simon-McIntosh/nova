"""Each positional-stiffness figure renders from the receipt it names.

A reader comparing two residual panels is entitled to know they are two
measurements rather than one picture drawn twice, and a reader looking at the
translated comparison is entitled to the residual it prints and the level array
the two fields share. Three failures defeat that: two figures sharing one
`source_receipt` and one pixel digest, so the pair is a duplicate; two figures
whose row selections are in fact the same selection; and a comparison panel that
names neither its residual nor its level array, or that carries one null set, so
a solved stationary point draws alone and reads as the answer.

The checks here read the render receipt and the PNGs on disk directly, with only
the standard library and Pillow, so the guard does not rest on the code path that
wrote them.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRECTORY = (
    ROOT / "docs/figures/cut-cell-current-attribution/positional-stiffness"
)
RENDER_RECEIPT = FIGURE_DIRECTORY / "render-receipt.json"
SOURCE_RECEIPT = FIGURE_DIRECTORY / "receipt.json"

# Set to make the receipt carry the alleged defect: one figure's pixel digest
# copied onto the figure it is compared against. The positive check then has to
# fail against it, which is how the guard is shown to discriminate.
IDENTICAL_FIGURES_ENV = "NOVA_POSITIONAL_STIFFNESS_IDENTICAL_FIGURES"

NULL_SETS = ("analytic", "translated")


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pixel_digest(path: Path) -> str:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def _receipt() -> dict:
    assert RENDER_RECEIPT.is_file(), (
        f"the render receipt the figures were drawn from is missing: {RENDER_RECEIPT}"
    )
    receipt = json.loads(RENDER_RECEIPT.read_text(encoding="utf-8"))
    if os.environ.get(IDENTICAL_FIGURES_ENV):
        figures = receipt["residual_figures"]
        figures[1]["pixel_sha256"] = figures[0]["pixel_sha256"]
        figures[1]["sha256"] = figures[0]["sha256"]
    return receipt


def _assert_residual_figures_are_distinct_measurements(receipt: dict) -> None:
    """Two residual figures, two row selections, two digests, one receipt."""

    figures = receipt["residual_figures"]
    assert len(figures) >= 2, "the receipt must carry the residual figure pair"

    selections = {
        (figure["case"], figure["requested_cells"], figure["realised_cells"])
        for figure in figures
    }
    assert len(selections) == len(figures), (
        "each residual figure draws its own disjoint row selection; "
        f"found {sorted(selections)}"
    )

    sources = {figure["source_receipt"] for figure in figures}
    assert len(sources) == 1, (
        "every residual figure is drawn from the one committed measurement "
        f"receipt; found {sorted(sources)}"
    )
    digests = {figure["source_receipt_sha256"] for figure in figures}
    assert digests == {_file_digest(SOURCE_RECEIPT)}, (
        "the receipt the figures name must be the committed measurement receipt "
        f"on disk; found {sorted(digests)}"
    )

    pixels = [figure["pixel_sha256"] for figure in figures]
    assert len(set(pixels)) == len(pixels), (
        "two residual figures sharing one pixel digest are the same picture "
        f"drawn twice; found {sorted(pixels)}"
    )
    bitmaps = [figure["sha256"] for figure in figures]
    assert len(set(bitmaps)) == len(bitmaps), (
        f"two residual figures sharing one file digest are one file; found {bitmaps}"
    )
    for figure in figures:
        assert figure["poloidal_panels"] == [], (
            "a residual figure is a line plot and carries no poloidal panel; "
            f"{figure['figure']} carries {len(figure['poloidal_panels'])}"
        )


def test_the_residual_figure_pair_are_two_distinct_measurements():
    _assert_residual_figures_are_distinct_measurements(_receipt())


def test_every_regenerated_png_matches_its_receipt_digests():
    """The receipt describes the file on disk, not an earlier render of it."""

    receipt = _receipt()
    named = [*receipt["residual_figures"], receipt["translated_figure"]]
    for figure in named:
        assert figure["filesystem_path"] == (
            "docs/figures/cut-cell-current-attribution/positional-stiffness"
            f"/figures/{figure['figure']}"
        ), (
            f"{figure['figure']} is named under a path it does not live at: "
            f"{figure['filesystem_path']}"
        )
        path = ROOT / figure["filesystem_path"]
        assert path.is_file(), f"the receipt names a figure that is missing: {path}"
        assert figure["sha256"] == _file_digest(path), (
            f"{figure['figure']} does not match the byte digest the receipt records"
        )
        assert figure["pixel_sha256"] == _pixel_digest(path), (
            f"{figure['figure']} does not match the pixel digest the receipt records"
        )
        assert figure["project_absolute_src"] == (
            f"/nova/figures/cut-cell-current-attribution/positional-stiffness"
            f"/figures/{figure['figure']}"
        ), f"{figure['figure']} must be addressed project-absolutely"


def test_the_translated_panel_prints_its_residual_and_flags_convergence():
    receipt = _receipt()
    titles = receipt["translated_figure"]["titles"]
    assert titles, "the translated figure must state its own title lines"
    residual_lines = [line for line in titles if line.startswith("residual=")]
    assert residual_lines, f"the translated panel prints no residual: {titles}"
    for line in residual_lines:
        assert " of span" in line, f"the residual line states no span: {line}"
        assert "converged=" in line, f"the residual line states no convergence: {line}"
        assert line.rsplit("converged=", 1)[1] in {"yes", "no"}, (
            f"the convergence flag is not a yes or a no: {line}"
        )
    level_lines = [line for line in titles if line.startswith("levels Wb")]
    assert level_lines, f"the panel states no level array: {titles}"
    for line in level_lines:
        assert "(shared): [" in line, (
            f"the levels are not stated as one shared Wb array: {line}"
        )
        stated = json.loads(line.split("(shared): ", 1)[1])
        assert len(stated) >= 3, f"a level array that carries no levels: {line}"


def test_every_comparison_panel_carries_both_null_sets():
    receipt = _receipt()
    translated = receipt["translated_figure"]
    panels = translated["poloidal_panels"]
    assert len(panels) == 6, (
        "the 40 mm comparison draws three states in two directions; found "
        f"{len(panels)} panels"
    )
    for panel in panels:
        counts = panel["null_glyphs"]
        assert set(counts) == set(NULL_SETS), (
            f"a comparison panel must carry both null sets, found {sorted(counts)}"
        )
        for set_name in NULL_SETS:
            assert counts[set_name]["axis_drawn"] == 1, (
                f"the {set_name} magnetic axis is not drawn on panel "
                f"{panel['panel']}: {counts[set_name]}"
            )
            assert "x_points_drawn" in counts[set_name], (
                f"the {set_name} set names no saddle count on panel "
                f"{panel['panel']}: {counts[set_name]}"
            )


def test_the_guard_rejects_two_figures_with_one_pixel_digest(monkeypatch):
    """The declared negative control: the check fails on the alleged defect."""

    monkeypatch.setenv(IDENTICAL_FIGURES_ENV, "1")
    receipt = _receipt()
    identical = receipt["residual_figures"]
    assert identical[0]["pixel_sha256"] == identical[1]["pixel_sha256"], (
        "the negative control did not manage to duplicate the pixel digest"
    )
    try:
        _assert_residual_figures_are_distinct_measurements(receipt)
    except AssertionError as failure:
        assert "one pixel digest" in str(failure), (
            f"the guard failed for the wrong reason: {failure}"
        )
    else:
        raise AssertionError(
            "the guard accepted two residual figures sharing one pixel digest"
        )
