"""Contract checks for the zero-residual render receipt.

The figure beside this receipt draws, per case, a terminal poloidal flux panel
and a relative-residual panel. The check below is deliberately receipt-driven:
it asserts what the run recorded it drew, so the render pass must state the
contour levels it used, the finite range of the raster it drew them on, and the
number of glyphs each null set contributed. The empty-panel defect this guards
against was not a missing raster -- the raster was finite throughout -- but a
level array computed over the union of both cases, which intersected neither.
A panel whose stated levels do not cross its own raster range is therefore the
failure this check exists to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "docs"
    / "figures"
    / "cut-cell-current-attribution"
    / "zero-residual"
    / "render-receipt.json"
)

# The declared negative control, stated verbatim in the log its run writes. The
# union-derived array is the one the shared computation produced before the
# per-raster repair: it spans the weak case and intersects none of the moderate
# case's raster, so the moderate terminal-flux panel drew no contour.
NEGATIVE_CONTROL = (
    "negative control: replace the moderate terminal-flux level array with the "
    "union-derived array spanning both cases "
    "[-12.57141928806472, -6.546858634450527, -0.5222979808363348, 5.502262672777858, "
    "11.526823327006053, 17.551383981234247, 23.57594463546244, 29.600505289690637, "
    "35.62506594391883, 41.649626594463015] Wb"
)
UNION_LEVELS_WB = [
    -12.57141928806472,
    -6.546858634450527,
    -0.5222979808363348,
    5.502262672777858,
    11.526823327006053,
    17.551383981234247,
    23.57594463546244,
    29.600505289690637,
    35.62506594391883,
    41.649626594463015,
]


def load_receipt() -> dict:
    return json.loads(RECEIPT.read_text())


def apply_negative_control(receipt: dict) -> dict:
    for figure in receipt["figures"].values():
        for panel in figure["panels"]:
            if panel["kind"] != "terminal_flux":
                continue
            if not panel["case"].startswith("moderate"):
                continue
            panel["levels_wb"] = list(UNION_LEVELS_WB)
            panel["level_count"] = len(UNION_LEVELS_WB)
    return receipt


def levels_inside(panel: dict) -> int:
    low = panel["raster_min_wb"]
    high = panel["raster_max_wb"]
    if low is None or high is None:
        return 0
    return sum(1 for level in panel["levels_wb"] if low < level < high)


def assert_render_contract(receipt: dict) -> None:
    assert receipt["schema"] == "nova.zero-residual-render-receipt"
    figures = receipt["figures"]
    assert figures, "no figure records in the render receipt"
    for stem, entry in figures.items():
        panels = entry["panels"]
        assert panels, f"{stem} records no panels"
        for panel in panels:
            assert "residual=" in panel["title"], panel["title"]
            assert "converged=" in panel["title"], panel["title"]

        flux_panels = [p for p in panels if p["kind"] == "terminal_flux"]
        residual_panels = [p for p in panels if p["kind"] == "relative_residual"]
        assert flux_panels and residual_panels

        for panel in flux_panels:
            assert panel["level_count"] == len(panel["levels_wb"])
            if panel["persisted_raster_present"]:
                assert panel["level_count"] > 0
                assert panel["raster_min_wb"] is not None
                assert panel["raster_max_wb"] is not None
                recomputed = levels_inside(panel)
                assert recomputed == panel["levels_inside_raster"], (
                    f"{panel['case']}: stated levels do not match the counted "
                    f"levels inside the raster ({recomputed} vs "
                    f"{panel['levels_inside_raster']})"
                )
                assert recomputed > 0, (
                    f"{panel['case']}: no stated contour level crosses the "
                    "raster range, so the terminal-flux panel draws nothing"
                )
                glyphs = panel["null_glyphs"]
                solved = (
                    glyphs["solved_axis"]
                    + glyphs["solved_x_points"]
                    + glyphs["solved_other_x_points"]
                )
                reference = glyphs["reference_axis"] + glyphs["reference_x_points"]
                assert solved > 0, f"{panel['case']}: solved null set unmarked"
                assert reference > 0, (
                    f"{panel['case']}: reference null set unmarked"
                )
                source = panel["reference_receipt"]
                assert source, f"{panel['case']}: no source receipt named"
                assert (ROOT / source).exists(), source
            else:
                assert "no persisted flux raster exists" in panel["title"]
                assert panel["reference_receipt"] in panel["title"]

        for panel in residual_panels:
            assert panel["level_count"] > 0
            assert panel["level_count"] == len(panel["levels_relative_span"])


def test_render_contract() -> None:
    assert_render_contract(load_receipt())


if __name__ == "__main__":
    print(NEGATIVE_CONTROL)
    assert_render_contract(apply_negative_control(load_receipt()))