"""The certificate gate's two arms must come from two revisions.

A certificate identity comparison is only informative when its baseline arm was
produced at the recorded base revision and its candidate arm at head.  When both
arms are built in one process at head, the comparison reports a bit-identity
between a state and itself, which is the one outcome that cannot distinguish a
preserved program from a rewritten one.

These tests read the committed certificate receipt as the recorded base state
and assert the properties the gate now depends on: each arm carries its own
revision key, the baseline key is the recorded base revision, and a record made
at another revision is refused rather than compared.  They also assert what the
comparison can say: a baseline row that carries its terminal state array yields
a finite element-wise maximum difference, a row that carries only a digest
reports the difference as unmeasured with its reason rather than as zero, and an
array that disagrees with the digest beside it is refused.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.solve_program_size_gate import (
    CERTIFICATE_BASE_REVISION,
    CERTIFICATE_BASELINE_RECEIPT,
    CERTIFICATE_ROWS,
    CertificateBaselineRefusal,
    _persist_state_array,
    certificate_baseline_source,
    certificate_identity_arms,
    certificate_state_difference,
    load_certificate_baseline,
)


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def _receipt_with_row(tmp_path: Path, row: dict) -> Path:
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "measurement_revision": CERTIFICATE_BASE_REVISION,
                "rows": [
                    {
                        "case": CERTIFICATE_ROWS[0][0],
                        "requested_cells": CERTIFICATE_ROWS[0][1],
                        **row,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


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


def test_row_compared_against_a_state_array_reports_a_finite_difference(
    tmp_path: Path,
) -> None:
    baseline_state = np.array([0.5, -1.25, 2.0], dtype=np.float64)
    receipt = _receipt_with_row(
        tmp_path,
        {
            "baseline_state_sha256_binary64": _digest(baseline_state),
            "baseline_state_array": baseline_state.tolist(),
            "realised_state_values": baseline_state.size,
        },
    )
    baseline = load_certificate_baseline(receipt)
    recorded = baseline["rows"][(CERTIFICATE_ROWS[0][0], CERTIFICATE_ROWS[0][1])]
    assert recorded["arm_kind"] == "state array"
    arms = certificate_identity_arms(
        "candidate-revision-sentinel",
        baseline,
        *CERTIFICATE_ROWS[0],
    )
    assert arms["baseline"]["source"] == "recorded base-revision terminal state array"
    assert arms["baseline"]["arm_kind"] == "state array"

    candidate = np.array([0.5, -1.5, 2.0], dtype=np.float64)
    difference = certificate_state_difference(
        candidate, recorded, f"solovev:{CERTIFICATE_ROWS[0][0]}"
    )
    assert difference["maximum_absolute_state_difference"] == pytest.approx(0.25)
    assert difference["difference_values"] == baseline_state.size
    assert (
        "3 binary64 values" in difference["maximum_absolute_state_difference_measure"]
    )

    identical = certificate_state_difference(baseline_state, recorded)
    assert identical["maximum_absolute_state_difference"] == 0.0


def test_row_compared_against_a_digest_reports_the_difference_as_unmeasured() -> None:
    baseline = load_certificate_baseline(CERTIFICATE_BASELINE_RECEIPT)
    key = (CERTIFICATE_ROWS[0][0], CERTIFICATE_ROWS[0][1])
    recorded = baseline["rows"][key]
    assert recorded["state_array"] is None
    assert recorded["arm_kind"] == "digest"
    arms = certificate_identity_arms(
        "candidate-revision-sentinel", baseline, *CERTIFICATE_ROWS[0]
    )
    assert arms["baseline"]["source"] == "recorded base-revision terminal state digest"

    difference = certificate_state_difference(
        np.zeros(recorded["realised_state_values"], dtype=np.float64), recorded
    )
    assert difference["maximum_absolute_state_difference"] is None
    assert difference["maximum_absolute_state_difference"] != 0.0
    assert "unmeasured" in difference["maximum_absolute_state_difference_measure"]
    assert "digest" in difference["maximum_absolute_state_difference_measure"]


def test_committed_row_carries_no_state_array_so_its_recorded_difference_is_not_adopted(
    tmp_path: Path,
) -> None:
    """The superseded in-process comparison recorded a difference beside the digest.

    That number came from two arms built at one revision, so it is not the
    cross-revision difference this gate reports; loading the receipt must not
    surface it as a measured difference against the array, which the row does not
    carry.
    """
    payload = json.loads(CERTIFICATE_BASELINE_RECEIPT.read_text(encoding="utf-8"))
    committed = payload["certificate"]["rows"][0]
    assert committed["maximum_absolute_state_difference"] is not None
    assert "baseline_state_array_path" not in committed
    assert "baseline_state_array" not in committed

    baseline = load_certificate_baseline(CERTIFICATE_BASELINE_RECEIPT)
    recorded = baseline["rows"][
        (str(committed["case"]), int(committed["requested_cells"]))
    ]
    difference = certificate_state_difference(
        np.zeros(int(committed["realised_state_values"]), dtype=np.float64), recorded
    )
    assert difference["maximum_absolute_state_difference"] is None
    assert (
        difference["maximum_absolute_state_difference"]
        != committed["maximum_absolute_state_difference"]
    )


def test_persisted_state_array_round_trips_its_digest(tmp_path: Path) -> None:
    state = np.array([1.0, 2.0, 3.0, 4.5], dtype=np.float64)
    persisted = _persist_state_array(tmp_path / "states", "case", -300, state)
    assert persisted is not None
    assert persisted["values"] == state.size
    assert persisted["sha256_binary64"] == _digest(state)

    receipt = _receipt_with_row(
        tmp_path,
        {
            "baseline_state_sha256_binary64": persisted["sha256_binary64"],
            "baseline_state_array_path": f"states/{Path(persisted['path']).name}",
        },
    )
    recorded = load_certificate_baseline(receipt)["rows"][
        (CERTIFICATE_ROWS[0][0], CERTIFICATE_ROWS[0][1])
    ]
    assert recorded["state_array_source"] == f"states/{Path(persisted['path']).name}"
    np.testing.assert_array_equal(recorded["state_array"], state)
    difference = certificate_state_difference(
        state * 2.0, recorded, f"solovev:{CERTIFICATE_ROWS[0][0]}"
    )
    assert difference["maximum_absolute_state_difference"] == pytest.approx(4.5)


def test_state_array_that_disagrees_with_its_digest_is_refused(tmp_path: Path) -> None:
    receipt = _receipt_with_row(
        tmp_path,
        {
            "baseline_state_sha256_binary64": "a" * 64,
            "baseline_state_array": [1.0, 2.0, 3.0],
        },
    )
    with pytest.raises(CertificateBaselineRefusal, match="does not match the digest"):
        load_certificate_baseline(receipt)


def test_state_array_of_another_shape_is_refused(tmp_path: Path) -> None:
    baseline_state = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    receipt = _receipt_with_row(
        tmp_path,
        {
            "baseline_state_sha256_binary64": _digest(baseline_state),
            "baseline_state_array": baseline_state.tolist(),
        },
    )
    recorded = load_certificate_baseline(receipt)["rows"][
        (CERTIFICATE_ROWS[0][0], CERTIFICATE_ROWS[0][1])
    ]
    with pytest.raises(CertificateBaselineRefusal, match="element-wise"):
        certificate_state_difference(np.array([1.0, 2.0], dtype=np.float64), recorded)


def test_named_state_array_that_is_absent_is_refused(tmp_path: Path) -> None:
    receipt = _receipt_with_row(
        tmp_path,
        {
            "baseline_state_sha256_binary64": "b" * 64,
            "baseline_state_array_path": "states/absent.npy",
        },
    )
    with pytest.raises(CertificateBaselineRefusal, match="does not exist"):
        load_certificate_baseline(receipt)
