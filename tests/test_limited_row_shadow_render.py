"""The limited-shadow chord panels report their terminal state and their floor.

Four panels under ``limited-shadow/solve-panels/chord`` share one painter. Three
faults are pinned here. The title carried no terminal residual and no converged
flag, uniquely among the template's siblings. The absolute-psi-error panel of
the single-null control contoured machine round-off as a structured map,
because that field's whole range sits below any significance floor. And the
poloidal panels drew one null set where a second set is handed to the painter.

The receipt check runs against the committed render receipt, which the driver
rebuilds from the persisted part receipts with no solve. A second check rebuilds
the control panel in memory and reads the returned panel records and the axes,
so a renderer regression is caught even when the committed receipt is stale.

Set ``NOVA_LIMITED_RENDER_MUTATION`` to a key of ``DECLARED_MUTATIONS`` to apply
the declared mutation to the receipt before checking it; the check must then
redden, which is how the negative control is applied. No mutation reddens the
live-rebuild check, which reads the renderer directly rather than the receipt.
"""

from __future__ import annotations

import copy
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from benchmarks import limited_row_shadow_census as census


RENDER_RECEIPT = census.DEFAULT_OUTPUT_ROOT / census.RENDER_RECEIPT_NAME
PART_ROOT = census.DEFAULT_OUTPUT_ROOT / "solve-parts" / census.SHADOW_RENDER_MODE
MUTATION_ENV = "NOVA_LIMITED_RENDER_MUTATION"
EXPECTED_FIGURES = 4
NULL_SETS = ("analytic", "solved")
CONTROL_CASE = "diverted-single-null"

DECLARED_MUTATIONS = {
    "sub-floor-level": (
        "restore the diverted psi error panel's unfiltered contour levels, so "
        "a level below the 1e-12 floor is contoured as signal"
    ),
    "drop-residual-title": "strip residual= and converged= from every title",
    "drop-solved-null-set": "zero the solved null-set count on every panel",
}


def _load_receipt() -> dict:
    assert RENDER_RECEIPT.is_file(), (
        f"the render receipt the pin cites is missing: {RENDER_RECEIPT}"
    )
    return json.loads(RENDER_RECEIPT.read_text(encoding="utf-8"))


def _load_parts() -> dict[str, dict]:
    """The persisted part receipts the rendered figures were rebuilt from."""

    assert PART_ROOT.is_dir(), f"the persisted parts are missing: {PART_ROOT}"
    parts: dict[str, dict] = {}
    for path in sorted(PART_ROOT.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        parts[payload["case"]] = payload
    return parts


def apply_declared_mutation(receipt: dict, mutation: str | None) -> dict:
    mutated = copy.deepcopy(receipt)
    if not mutation:
        return mutated
    if mutation not in DECLARED_MUTATIONS:
        raise ValueError(f"unknown declared mutation: {mutation}")
    if mutation == "sub-floor-level":
        for row in mutated["rows"]:
            for panel in row["panels"]:
                if panel["levels_suppressed_below_floor"]:
                    panel["levels"] = [4.72e-20, *panel["levels"]]
                    panel["levels_suppressed_below_floor"] -= 1
    elif mutation == "drop-residual-title":
        for row in mutated["rows"]:
            row["title"] = row["title"].split(" · residual=")[0]
    elif mutation == "drop-solved-null-set":
        for row in mutated["rows"]:
            for panel in row["panels"]:
                panel["null_glyphs"]["solved"] = 0
    return mutated


def render_receipt_failures(receipt: dict, parts: dict[str, dict]) -> list[str]:
    """Every way the receipt falls short of the panel contract, as sentences."""

    failures: list[str] = []
    rows = receipt.get("rows", [])
    if len(rows) != EXPECTED_FIGURES:
        failures.append(
            f"expected {EXPECTED_FIGURES} rendered figures, found {len(rows)}"
        )
    floor = float(receipt["roundoff_level_floor"])
    for row in rows:
        case = row["case"]
        title = row["title"]
        payload = parts.get(case)
        if payload is None:
            failures.append(f"{case}: no part receipt to check the title against")
            continue
        residual = float(payload["solver"]["terminal_fixed_point_residual"])
        converged = bool(payload["solver"]["converged"])
        if "residual=" not in title:
            failures.append(f"{case}: the title carries no residual=")
        elif f"residual={residual:.3e}" not in title:
            failures.append(
                f"{case}: the title residual {residual:.3e} is not printed on the title"
            )
        if "converged=" not in title:
            failures.append(f"{case}: the title carries no converged=")
        else:
            expected = "yes" if converged else "no"
            if f"converged={expected}" not in title:
                failures.append(f"{case}: the title does not read converged={expected}")
        if (
            float(row["receipt_residual"]) != residual
            or bool(row["receipt_converged"]) != converged
        ):
            failures.append(
                f"{case}: the receipt terminal fields disagree with the part"
            )
        for panel in row["panels"]:
            glyphs = panel["null_glyphs"]
            for set_name in NULL_SETS:
                if int(glyphs.get(set_name, 0)) < 1:
                    failures.append(
                        f"{case}/{panel['name']}: the {set_name} null set is not drawn"
                    )
            if panel["name"] == "flux":
                continue
            below = [float(level) for level in panel["levels"] if float(level) < floor]
            if below:
                failures.append(
                    f"{case}/{panel['name']}: contour levels below the round-off "
                    f"floor {floor:.0e}: {below}"
                )
            if "round-off floor" not in panel["caption"]:
                failures.append(
                    f"{case}/{panel['name']}: the caption does not name the "
                    "round-off floor"
                )
    return failures


def _marker_colours(axes) -> set[str]:
    """The colours of every stationary-point marker drawn on ``axes``."""

    colours: set[str] = set()
    for line in axes.lines:
        marker = line.get_marker()
        if marker is None or str(marker).lower() == "none":
            continue
        colours.add(matplotlib.colors.to_hex(line.get_color()))
    return colours


@pytest.fixture(scope="module")
def render_receipt() -> dict:
    return _load_receipt()


def test_the_receipt_reports_residual_converged_floor_and_both_null_sets(
    render_receipt,
):
    mutation = os.environ.get(MUTATION_ENV) or None
    checked = apply_declared_mutation(render_receipt, mutation)
    failures = render_receipt_failures(checked, _load_parts())
    assert failures == [], "\n".join(failures)


def test_every_chord_panel_title_names_the_terminal_state(render_receipt):
    for row in render_receipt["rows"]:
        assert "residual=" in row["title"], row["title"]
        assert "converged=" in row["title"], row["title"]


def test_a_sub_floor_panel_is_blank_and_says_why(tmp_path):
    """The control's psi error is machine round-off, not a structured map."""

    payload = _load_parts()[CONTROL_CASE]
    figure, panels = census._draw_shadow_panel(
        payload, tmp_path / "control.png", "control"
    )
    try:
        psi = next(panel for panel in panels if panel["name"] == "psi")
        assert psi["max_absolute"] < census.ROUNDOFF_LEVEL_FLOOR
        assert psi["levels"] == [], (
            "every level of a round-off field sits below the floor and must not "
            f"be contoured, found {psi['levels']}"
        )
        assert psi["levels_suppressed_below_floor"] > 0
        assert "round-off floor" in psi["caption"]
    finally:
        plt.close(figure)


def test_every_poloidal_panel_draws_both_null_sets(tmp_path):
    payload = _load_parts()[CONTROL_CASE]
    figure, panels = census._draw_shadow_panel(
        payload, tmp_path / "control.png", "control"
    )
    try:
        for panel in panels:
            for set_name in NULL_SETS:
                assert panel["null_glyphs"][set_name] >= 1, (
                    f"{panel['name']}: the {set_name} null set is not drawn"
                )
        for axes in figure.axes:
            colours = _marker_colours(axes)
            assert census.ANALYTIC_RENDER_COLOR in colours, sorted(colours)
            assert census.SOLVED_RENDER_COLOR in colours, sorted(colours)
    finally:
        plt.close(figure)
