"""The centroid figures state the flags their measurement receipts record.

The two figures are drawn from committed measurement receipts rather than from
a solve, so each bar's label and each bar's qualified and converged flags are
a reading of that receipt. These tests re-read the receipt the renderer wrote
and require every bar to agree with it, because a bar that reads as an endorsed
solve while its receipt records otherwise is exactly the defect the figure
exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CENTROID = REPO / "docs/figures/constraint-augmented-newton-krylov/centroid"
RENDER_RECEIPT = CENTROID / "render-receipt.json"
RENDER_RECEIPT_ENV = "NOVA_CONSTRAINT_CENTROID_RENDER_RECEIPT"
FIGURE_NAMES = {"two-rows.svg", "converged-compensator.svg"}
SERIES_DERIVATION = {"seed": "seed_derivation", "convergence": "converged_derivation"}


def _load(path: Path) -> dict:
    assert path.is_file(), f"no such receipt: {path}"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def render_receipt() -> tuple[Path, dict]:
    path = Path(os.environ.get(RENDER_RECEIPT_ENV, RENDER_RECEIPT))
    return path, _load(path)


def _figure(payload: dict, name: str) -> dict:
    matches = [figure for figure in payload["figures"] if figure["figure"] == name]
    assert len(matches) == 1, f"expected one {name} record, found {len(matches)}"
    return matches[0]


def test_render_receipt_records_both_figures(render_receipt):
    path, payload = render_receipt
    payload["_dir"] = path.parent
    assert payload["schema"] == "nova.constraint-centroid-render-receipt"
    assert payload["completed"] is True
    assert payload["render_entry_point"].endswith("--render")
    assert {figure["figure"] for figure in payload["figures"]} == FIGURE_NAMES
    for name, record in payload["source_receipts"].items():
        assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), name
        source = path.parent / record["name"]
        assert source.is_file(), f"{name} source receipt missing: {source}"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == record["sha256"], f"{name} moved since the render"


def test_two_rows_bars_carry_their_rows_coil_family(render_receipt):
    path, payload = render_receipt
    payload["_dir"] = path.parent
    receipt = _load(path.parent / payload["source_receipts"]["two-rows"]["name"])
    family = {row["identity"]: row["actuator"]["definition"] for row in receipt["rows"]}
    for bar in _figure(payload, "two-rows.svg")["bars"]:
        assert bar["label"] == family[bar["row"]], bar


def test_two_rows_bars_carry_the_flags_the_receipt_records(render_receipt):
    path, payload = render_receipt
    payload["_dir"] = path.parent
    receipt = _load(path.parent / payload["source_receipts"]["two-rows"]["name"])
    arms = {
        (row["identity"], arm): row[arm]
        for row in receipt["rows"]
        for arm in ("free", "protocol", "prototype")
    }
    for bar in _figure(payload, "two-rows.svg")["bars"]:
        record = arms[(bar["row"], bar["arm"])]
        assert bar["qualified"] == record.get("qualified"), bar
        assert bar["converged"] == record.get("converged"), bar
        assert bar["termination"] == record.get("termination"), bar
    qualified_count = receipt["verdict"]["qualified_count"]
    assert _figure(payload, "two-rows.svg")["qualified_count"] == qualified_count
    if qualified_count == 0:
        assert not [
            bar for bar in _figure(payload, "two-rows.svg")["bars"] if bar["qualified"]
        ]


def test_compensator_bars_carry_a_family_named_in_their_rows_selection(render_receipt):
    path, payload = render_receipt
    payload["_dir"] = path.parent
    receipt = _load(
        path.parent / payload["source_receipts"]["converged-compensator"]["name"]
    )
    rows = {row["identity"]: row for row in receipt["rows"]}
    for bar in _figure(payload, "converged-compensator.svg")["bars"]:
        derivation = rows[bar["row"]][SERIES_DERIVATION[bar["series"]]]
        families = [
            entry["family"] for entry in derivation["selection"]["leading_circuits"]
        ]
        assert bar["label"] in families, (bar, families)


def test_compensator_bars_carry_the_flags_the_receipt_records(render_receipt):
    path, payload = render_receipt
    payload["_dir"] = path.parent
    receipt = _load(
        path.parent / payload["source_receipts"]["converged-compensator"]["name"]
    )
    rows = {row["identity"]: row for row in receipt["rows"]}
    for bar in _figure(payload, "converged-compensator.svg")["bars"]:
        solve = rows[bar["row"]][SERIES_DERIVATION[bar["series"]]]["solve"]
        assert bar["qualified"] is bool(solve["qualified"]), bar
        assert bar["converged"] is bool(solve["termination"] == "converged"), bar


def test_compensator_panels_state_their_own_rows_families(render_receipt):
    path, payload = render_receipt
    payload["_dir"] = path.parent
    receipt = _load(
        path.parent / payload["source_receipts"]["converged-compensator"]["name"]
    )
    rows = {row["identity"]: row for row in receipt["rows"]}
    panels = _figure(payload, "converged-compensator.svg")["panels"]
    assert [panel["row"] for panel in panels] == [
        row["identity"] for row in receipt["rows"]
    ]
    for panel in panels:
        derivation = rows[panel["row"]]["seed_derivation"]
        families = [
            entry["family"] for entry in derivation["selection"]["leading_circuits"]
        ]
        assert panel["category_labels"] == families, (panel, families)
        angle = rows[panel["row"]]["comparison"]["direction_angle_degrees"]
        assert panel["angle_degrees"] == angle
        assert panel["printed_angle"] == format(angle, ".2f")
