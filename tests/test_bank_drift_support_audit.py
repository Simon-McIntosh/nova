"""Refusals of the bank-drift support audit, each made to fire.

The audit's answer is a comparison, so its failure modes are all shapes of a
silent non-result. A receipt field the merge never populated, a receipt field
that is present but blank, an arm that only one tree emitted, and a pair that
carries stage evidence on only one side each render as a blank or a skipped
comparison, and every one of them reads as a finding about the data ("the
emission does not report a residual", "the support is the same") rather than as
a gap in the instrument. The program refuses each case by name instead.

These tests present each case on synthetic emissions and assert the emitted
form, because a passing suite never shows that a guard fires. Nothing here
builds geometry, opens IMAS data or compiles: the program under test reads JSON
and source text only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmarks import bank_drift_support_audit as audit

SUPPORT_PARTS = {
    "inside_material": "7406355ef5eb56a2",
    "physical_node_number": "e706f97a7d7d60e5",
}
SUPPORT_STAGES = {
    "profile_support": {"digest": "5683f88ca97c076e", "parts": SUPPORT_PARTS},
    "partition_structure": {
        "digest": "0fe0dcdf920a9b66",
        "summary": {
            "leaf_count": 76,
            "tree_node_count": 90,
            "is_pytree": True,
            "operator_type": "FluxLattice",
        },
    },
    "partition_values": {"digest": "35c0be67ae2abdd8"},
}
PRODUCER_STAGES = {
    "profile_support": {"digest": "5683f88ca97c076e", "parts": SUPPORT_PARTS},
    "partition_structure": {
        "digest": "236cbb2e1af2dded",
        "summary": {
            "leaf_count": 1,
            "tree_node_count": 1,
            "is_pytree": True,
            "operator_type": "FluxLattice",
        },
    },
    "partition_values": {"digest": ""},
}


def _row(identity: str, receipt: dict, stages: dict | None = None) -> dict:
    row = {"identity": identity, "arms": {"pure": receipt}}
    if stages is not None:
        row["stages"] = stages
    return row


def _write_reports(root: Path, files: dict[str, object]) -> Path:
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, str):
            path.write_text(payload)
        else:
            path.write_text(json.dumps(payload))
    return root


def _tree(root: Path, *, traced: bool) -> Path:
    """A stub operator source, enough for the source-only row inventory."""
    operator = root / audit.OPERATOR_REL
    operator.parent.mkdir(parents=True, exist_ok=True)
    lines = ["register_pytree_node_class"]
    if traced:
        lines.append("_dynamic_extra_names")
        lines.extend('    "%s"' % name for name in audit.SUPPORT_ROWS)
    operator.write_text("\n".join(lines) + "\n")
    return root


# --- the receipt-cell guard -------------------------------------------------


def test_a_field_the_emission_never_reported_is_refused_by_name() -> None:
    """A key the merge never populated says so where a blank would read as data.

    The bare cell formatter renders the same absent value as ``-``; the guard
    exists so that the two are not the same output.
    """
    entry = {"receipt_source": "bank-drift/emit-main.json"}
    assert audit.cell(entry.get("terminal_residual")) == "-"
    assert audit.receipt_cell(entry, "terminal_residual") == "NO-FIELD"


def test_a_blank_field_names_the_source_it_came_from() -> None:
    source = "bank-drift/emit-main.json"
    for blank in ("", None):
        entry = {"receipt_source": source, "terminal_residual": blank}
        assert audit.receipt_cell(entry, "terminal_residual") == "EMPTY@%s" % source


def test_a_blank_field_with_no_source_says_so() -> None:
    assert (
        audit.receipt_cell({"terminal_residual": None}, "terminal_residual")
        == "EMPTY@(no source)"
    )


def test_a_reported_value_passes_through_unchanged() -> None:
    """Negative control: the guard does not decorate a value that is present."""
    entry = {"receipt_source": "bank-drift/emit-main.json"}
    entry["terminal_residual"] = "3.85734e-16"
    assert audit.receipt_cell(entry, "terminal_residual") == "3.85734e-16"


# --- stage evidence on one side only ----------------------------------------


def _paired_records(old: dict, new: dict) -> dict:
    return {("21978/35", "pure"): {"old": old, "new": new}}


def test_a_pair_with_stage_evidence_on_one_side_is_not_a_pass(capsys) -> None:
    """One digest against a blank is refused, not counted as a comparison."""
    records = _paired_records(
        {
            "support_digest": "5683f88ca97c076e",
            "structure_digest": "236cbb2e1af2dded",
            "termination_reason": "converged",
            "terminal_residual": "3.85734e-16",
        },
        {
            "termination_reason": "active_set_cycle_detected",
            "terminal_residual": "0.00245065",
        },
    )
    tally = audit.print_support(records)
    out = capsys.readouterr().out
    assert "21978/35   pure   NO STAGE EVIDENCE on one side -- not a pass" in out
    assert tally["missing"] == 1
    assert tally["support_same"] == 0
    assert tally["support_differ"] == 0
    assert tally["structure_same"] == 0
    assert tally["structure_differ"] == 0


def test_a_pair_with_stage_evidence_on_both_sides_is_compared(capsys) -> None:
    """Negative control: the refusal fires only where the evidence is absent."""
    records = _paired_records(
        {
            "support_digest": "5683f88ca97c076e",
            "structure_digest": "236cbb2e1af2dded",
            "termination_reason": "converged",
            "terminal_residual": "3.85734e-16",
        },
        {
            "support_digest": "5683f88ca97c076e",
            "structure_digest": "0fe0dcdf920a9b66",
            "termination_reason": "active_set_cycle_detected",
            "terminal_residual": "0.00245065",
        },
    )
    tally = audit.print_support(records)
    out = capsys.readouterr().out
    assert "NO STAGE EVIDENCE" not in out
    assert "support=same" in out
    assert "structure=DIFFERS" in out
    assert tally == {
        "paired": 1,
        "support_same": 1,
        "support_differ": 0,
        "structure_same": 0,
        "structure_differ": 1,
        "missing": 0,
    }


def test_an_arm_present_on_one_side_only_is_reported_unpaired(capsys) -> None:
    records = {
        ("21978/35", "pure"): {
            "old": {
                "support_digest": "5683f88ca97c076e",
                "structure_digest": "236cbb2e1af2dded",
            }
        }
    }
    tally = audit.print_support(records)
    out = capsys.readouterr().out
    assert "UNPAIRED old=True new=False" in out
    assert tally["missing"] == 1
    assert tally["paired"] == 0


# --- what the program does with a report it cannot read ---------------------


def test_a_malformed_report_is_named_rather_than_dropped(tmp_path: Path) -> None:
    reports = _write_reports(
        tmp_path / "reports",
        {
            "bank-drift/emit-good.json": {
                "tree_label": "fae50f15",
                "rows": [_row("21978/35", {"converged": True}, SUPPORT_STAGES)],
            },
            "bank-drift/emit-broken.json": "{ not json",
        },
    )
    collected = audit.collect(reports)
    assert len(collected["records"]) == 1
    assert len(collected["skipped"]) == 1
    assert "bank-drift/emit-broken.json" in collected["skipped"][0]


# --- the refusals as the program emits them ---------------------------------


def test_the_program_emits_each_refusal_end_to_end(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    reports = _write_reports(
        tmp_path / "reports",
        {
            "bank-drift/emit-old.json": {
                "tree_label": "fae50f15",
                "rows": [
                    _row(
                        "21978/35",
                        {
                            "converged": True,
                            "termination_reason": "converged",
                            "terminal_residual": None,
                        },
                        PRODUCER_STAGES,
                    )
                ],
            },
            "bank-drift/emit-new.json": {
                "tree_label": "main",
                "rows": [
                    _row(
                        "21978/35",
                        {
                            "converged": False,
                            "termination_reason": "active_set_cycle_detected",
                            "terminal_residual": 2.4506488e-03,
                        },
                    )
                ],
            },
        },
    )
    old_tree = _tree(tmp_path / "old-tree", traced=False)
    new_tree = _tree(tmp_path / "new-tree", traced=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bank_drift_support_audit",
            "--reports-root",
            str(reports),
            "--old-tree",
            str(old_tree),
            "--new-tree",
            str(new_tree),
        ],
    )
    assert audit.main() == 0
    out = capsys.readouterr().out
    assert "EMPTY@bank-drift/emit-old.json" in out
    assert "21978/35   pure   NO STAGE EVIDENCE on one side -- not a pass" in out
    assert (
        "summary: paired=1 support_identical=0 support_differs=0 "
        "structure_identical=0 structure_differs=0 unpaired_or_missing=1" in out
    )
    (old_row,) = [line for line in out.splitlines() if line.startswith("old  pytree")]
    (new_row,) = [line for line in out.splitlines() if line.startswith("new  pytree")]
    assert "trace_hook=False" in old_row
    assert "traced_rows=[]" in old_row
    assert "trace_hook=True" in new_row
    assert (
        "traced_rows=['declared_axis_flux', 'declared_boundary_flux', "
        "'declared_support']" in new_row
    )
