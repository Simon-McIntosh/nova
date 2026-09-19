"""The certificate gate's two arms must come from two revisions.

A certificate identity comparison is only informative when its baseline arm was
produced at the recorded base revision and its candidate arm at head.  When both
arms are built in one process at head, the comparison reports a bit-identity
between a state and itself, which is the one outcome that cannot distinguish a
preserved program from a rewritten one.

These tests read the committed certificate receipt as the recorded base state
and assert the properties the gate now depends on: each arm carries its own
revision key, the baseline key is the recorded base revision, and a record made
at another revision is refused rather than compared.
"""

import json
from pathlib import Path

import pytest

from benchmarks.solve_program_size_gate import (
    CERTIFICATE_BASE_REVISION,
    CERTIFICATE_BASELINE_RECEIPT,
    CERTIFICATE_ROWS,
    CertificateBaselineRefusal,
    certificate_baseline_source,
    certificate_identity_arms,
    load_certificate_baseline,
)


def test_baseline_receipt_is_committed() -> None:
    assert CERTIFICATE_BASELINE_RECEIPT.is_file(), (
        f"the recorded base terminal state is missing at {CERTIFICATE_BASELINE_RECEIPT}"
    )


def test_baseline_key_equals_recorded_base_revision() -> None:
    baseline = load_certificate_baseline(CERTIFICATE_BASELINE_RECEIPT)
    assert baseline["revision"] == CERTIFICATE_BASE_REVISION


def test_arms_carry_different_revision_keys() -> None:
    baseline = load_certificate_baseline(CERTIFICATE_BASELINE_RECEIPT)
    for case_name, requested_cells in CERTIFICATE_ROWS:
        arms = certificate_identity_arms(
            "candidate-revision-sentinel", baseline, case_name, requested_cells
        )
        assert arms["baseline"]["revision"] != arms["candidate"]["revision"]
        assert arms["baseline"]["revision"] == CERTIFICATE_BASE_REVISION
        assert arms["candidate"]["revision"] == "candidate-revision-sentinel"
        assert arms["baseline"]["source"] != arms["candidate"]["source"]


def test_arm_selection_probe_reports_both_keys_on_the_smallest_row() -> None:
    selection = certificate_baseline_source(CERTIFICATE_BASELINE_RECEIPT)
    assert (selection["case"], selection["requested_cells"]) == CERTIFICATE_ROWS[0]
    assert selection["baseline_revision"] == CERTIFICATE_BASE_REVISION
    assert selection["baseline_revision"] != selection["candidate_revision"]
    assert selection["arms_differ"] is True
    assert (
        selection["baseline_state_sha256_binary64"]
        == load_certificate_baseline(CERTIFICATE_BASELINE_RECEIPT)["rows"][
            (selection["case"], selection["requested_cells"])
        ]["state_sha256_binary64"]
    )


def test_foreign_revision_is_refused(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.json"
    foreign.write_text(
        json.dumps(
            {
                "measurement_revision": "0" * 40,
                "rows": [
                    {
                        "case": CERTIFICATE_ROWS[0][0],
                        "requested_cells": CERTIFICATE_ROWS[0][1],
                        "baseline_state_sha256_binary64": "1" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CertificateBaselineRefusal, match="not the recorded base"):
        load_certificate_baseline(foreign)


def test_record_without_a_baseline_digest_is_refused(tmp_path: Path) -> None:
    undigested = tmp_path / "undigested.json"
    undigested.write_text(
        json.dumps(
            {
                "measurement_revision": CERTIFICATE_BASE_REVISION,
                "rows": [
                    {
                        "case": CERTIFICATE_ROWS[0][0],
                        "requested_cells": CERTIFICATE_ROWS[0][1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CertificateBaselineRefusal, match="no baseline state digest"):
        load_certificate_baseline(undigested)


def test_record_without_a_revision_is_refused(tmp_path: Path) -> None:
    unlabelled = tmp_path / "unlabelled.json"
    unlabelled.write_text(json.dumps({"rows": []}), encoding="utf-8")
    with pytest.raises(CertificateBaselineRefusal, match="records no revision"):
        load_certificate_baseline(unlabelled)
