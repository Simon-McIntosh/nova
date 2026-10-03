"""Guard the reversed-current diagnosis panels and the receipt behind them.

The three panels beside this test's figure directory are drawn from one driver
receipt, and each panel's title must name the terminal state that panel
actually shows -- the census cannot be titled "converged" on a state whose
relative residual never reached the convergence criterion.

Every number a title states is re-derived here from the receipt's own JSON: the
converged flag, the same terminal residual, the termination reason and the
admission verdict. Point ``REVERSED_GUARD_DIR`` at another directory to run the
same assertions against a different artefact set; the negative control runs
this file against the pre-repair figures, which carry no receipt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(
    os.environ.get(
        "REVERSED_GUARD_DIR",
        REPOSITORY / "docs/figures/playable-forward-solve/reversed-current-guard",
    )
)

RECEIPT_KEYS = {"shot", "revision", "host", "wall_rz_m", "rows"}
ROW_KEYS = {"identity", "row", "free"}
FREE_KEYS = {
    "free_converged",
    "free_terminal_residual",
    "free_termination",
    "free_trips",
    "free_residual_trace",
    "candidate_census",
    "requested_class_name",
    "convergence_criterion",
}
CENSUS_KEYS = {
    "o_candidates",
    "x_candidates",
    "candidate_slots",
    "o_retained_valid",
    "x_retained_valid",
}
EXPECTED_ROWS = {30, 50, 55}
EXPECTED_FIGURES = {"diagnosis-row30.png", "diagnosis.png", "diagnosis-row55.png"}


def _receipt() -> dict:
    path = ARTIFACTS / "diagnosis.json"
    assert path.is_file(), (
        f"the reversed-current diagnosis receipt is missing at {path}; the "
        "panels beside it are undated and cannot be trusted"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _render() -> dict:
    path = ARTIFACTS / "render-receipt.json"
    assert path.is_file(), f"the render receipt is missing at {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_receipt_carries_its_structure() -> None:
    receipt = _receipt()
    assert RECEIPT_KEYS <= set(receipt)
    wall = receipt["wall_rz_m"]
    assert len(wall) >= 3, "the first wall polygon is too short to draw"
    assert all(len(point) == 2 for point in wall)
    assert {int(row["row"]) for row in receipt["rows"]} == EXPECTED_ROWS


def test_every_row_persists_its_terminal_state() -> None:
    for row in _receipt()["rows"]:
        assert ROW_KEYS <= set(row)
        free = row["free"]
        assert FREE_KEYS <= set(free)
        assert CENSUS_KEYS <= set(free["candidate_census"])
        assert isinstance(free["free_residual_trace"], list)


def test_census_rows_are_admitted_candidates_not_padding() -> None:
    for row in _receipt()["rows"]:
        census = row["free"]["candidate_census"]
        for block in ("o_candidates", "x_candidates"):
            candidates = census[block]
            assert len(candidates) <= census["candidate_slots"]
            for candidate in candidates:
                assert (candidate["r_m"], candidate["z_m"]) != (0.0, 0.0), (
                    f"{row['identity']} {block} reports a padded slot on the "
                    "machine origin"
                )
                assert candidate["r_m"] > 0.0


def test_panels_are_titled_for_the_state_each_one_shows() -> None:
    from benchmarks.reversed_current_guard_diagnosis import census_title

    rows = {int(row["row"]): row for row in _receipt()["rows"]}
    render = _render()
    assert {entry["figure"] for entry in render["figures"]} == EXPECTED_FIGURES

    for entry in render["figures"]:
        path = ARTIFACTS / entry["figure"]
        assert path.is_file(), f"{entry['figure']} is absent"
        assert path.stat().st_size > 5_000, f"{entry['figure']} is empty"
        free = rows[int(entry["row"])]["free"]
        assert entry["title"] == census_title(free), (
            f"{entry['figure']} title does not re-derive from the receipt"
        )
        assert entry["converged"] is bool(free["free_converged"])
        assert entry["terminal_residual"] == free["free_terminal_residual"]
        assert entry["termination"] == free["free_termination"]


def test_unconverged_rows_never_read_as_converged() -> None:
    from benchmarks.reversed_current_guard_diagnosis import census_title

    rows = {int(row["row"]): row for row in _receipt()["rows"]}
    for index in (50, 55):
        free = rows[index]["free"]
        assert free["free_converged"] is False
        title = census_title(free)
        assert "unconverged" in title
        assert "on the converged" not in title

    converged = rows[30]["free"]
    assert converged["free_converged"] is True
    assert "on the converged" in census_title(converged)
