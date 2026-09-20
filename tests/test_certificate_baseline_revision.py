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
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from benchmarks import solve_program_size_gate as gate_module
from benchmarks.solve_program_size_gate import (
    CERTIFICATE_BASE_REVISION,
    CERTIFICATE_BASELINE_RECEIPT,
    CERTIFICATE_EXECUTION_PROVENANCE_KEYS,
    CERTIFICATE_ROWS,
    CertificateBaselineRefusal,
    _persist_state_array,
    certificate_baseline_source,
    certificate_identity_arms,
    certificate_row_entry,
    certificate_state_difference,
    load_certificate_baseline,
    run_certificate_identity,
)


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def stub_certificate_row(
    case_name: str,
    requested_cells: int,
    *,
    baseline: dict,
    candidate_revision: str,
    panel_path: Path | None = None,
    state_directory: Path | None = None,
) -> dict:
    """Return one row's payload without solving, for the process-shape case.

    The row carries the fields a merged receipt is read for — both revision
    keys, the baseline digest, the bit-identity verdict and the terminal
    residual — so the merge is exercised over rows shaped like solved ones.
    The child processes load this function by path, so the parent and the
    children build their rows with the same code.
    """
    recorded = baseline["rows"][(str(case_name), int(requested_cells))]
    return {
        "case": case_name,
        "requested_cells": requested_cells,
        "baseline_revision": baseline["revision"],
        "candidate_revision": candidate_revision,
        "baseline_state_sha256_binary64": recorded["state_sha256_binary64"],
        "candidate_state_sha256_binary64": "0" * 64,
        "terminal_state_bit_identical": True,
        "baseline_arm_kind": recorded["arm_kind"],
        "maximum_absolute_state_difference": None,
        "candidate_terminal_residual": 1.0e-9,
        "panel": None,
    }


# Each child loads this module by path so the row it writes is built by the same
# stub the parent uses, then runs the gate's own single-row entry point.  The
# solve itself is not exercised here: a real row costs an allocation, and what
# this case measures is the process boundary and the merge.
_CHILD_ROW_SCRIPT = """
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("certificate_shape_case", sys.argv[3])
case_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case_module)

from benchmarks import solve_program_size_gate as gate

gate._certificate_identity_row = case_module.stub_certificate_row
gate.run_certificate_identity(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    case_module.CERTIFICATE_BASELINE_RECEIPT,
    row=sys.argv[4],
    process_per_row=False,
)
"""


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


def test_row_selector_names_a_committed_row_and_refuses_any_other() -> None:
    for case_name, requested_cells in CERTIFICATE_ROWS:
        assert certificate_row_entry(case_name) == (case_name, requested_cells)
    with pytest.raises(gate_module.CertificateRowRefusal, match="not one of"):
        certificate_row_entry("no-such-case")


def _child_command(test_file: Path, cache_root: Path):
    def build(row, row_receipt, samples, summary, **_kwargs):
        return [
            sys.executable,
            "-c",
            _CHILD_ROW_SCRIPT,
            str(row_receipt),
            str(cache_root),
            str(test_file),
            row[0],
        ]

    return build


def test_merged_receipt_from_four_child_rows_matches_the_single_process_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The merge must mint the same receipt one process solving every row mints.

    One process per row is what keeps a four-row run under the kernel's
    per-process mapping cap, and the merged receipt is the evidence it did. So
    the receipt the four children produce must carry the same shape, row for
    row, as the one a single process produces — otherwise the gate would be
    reading a receipt of a different kind — and it must name four processes of
    its own, because rows solved by the receipt's own process are exactly the
    configuration the merge exists to refuse.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "1274080")
    monkeypatch.setenv("TMPDIR", "/tmp")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    cache_root = tmp_path / "cache"
    # Both runs name the same state directory, so the comparison below is of the
    # receipt's shape rather than of the two output paths this case chose.
    state_directory = tmp_path / "states"
    single_receipt = tmp_path / "single.json"
    merged_receipt = tmp_path / "merged.json"

    monkeypatch.setattr(gate_module, "_certificate_identity_row", stub_certificate_row)
    single = run_certificate_identity(
        single_receipt,
        cache_root,
        CERTIFICATE_BASELINE_RECEIPT,
        state_dir=state_directory,
        process_per_row=False,
    )
    merged = run_certificate_identity(
        merged_receipt,
        cache_root,
        CERTIFICATE_BASELINE_RECEIPT,
        state_dir=state_directory,
        row_command=_child_command(Path(__file__), cache_root),
    )

    # Row for row, and key for key, apart from the provenance the two paths
    # differ in by construction and the capture time that advances between them.
    varying = (*CERTIFICATE_EXECUTION_PROVENANCE_KEYS, "captured_at")
    assert merged["rows"] == single["rows"]
    assert len(merged["rows"]) == len(CERTIFICATE_ROWS)
    assert {key: value for key, value in merged.items() if key not in varying} == {
        key: value for key, value in single.items() if key not in varying
    }

    assert single["execution_mode"] is None
    assert merged["execution_mode"] == "one-process-per-row"
    assert merged["parent_process_id"] == os.getpid()
    child_processes = [entry["process_id"] for entry in merged["row_processes"]]
    assert len(child_processes) == len(CERTIFICATE_ROWS)
    assert len(set(child_processes)) == len(CERTIFICATE_ROWS)
    assert os.getpid() not in child_processes
    assert all(entry["exit_code"] == 0 for entry in merged["row_processes"])


def test_rows_solved_by_the_receipts_own_process_are_refused() -> None:
    """A receipt whose rows all carry the parent's pid is not evidence of four loads."""
    parent_process_id = os.getpid()
    rows = [
        stub_certificate_row(
            name,
            cells,
            baseline=load_certificate_baseline(CERTIFICATE_BASELINE_RECEIPT),
            candidate_revision="candidate-revision-sentinel",
        )
        for name, cells in CERTIFICATE_ROWS
    ]
    row_processes = [
        {"case": row["case"], "process_id": parent_process_id} for row in rows
    ]
    with pytest.raises(gate_module.CertificateRowProcessRefusal, match="which owns"):
        gate_module.assemble_certificate_receipt(
            {"identity_row_count": len(rows)}, rows, row_processes, parent_process_id
        )
    with pytest.raises(gate_module.CertificateRowProcessRefusal, match="distinct"):
        gate_module.assemble_certificate_receipt(
            {"identity_row_count": len(rows)},
            rows,
            [
                {"case": row["case"], "process_id": 4000 + index // 2}
                for index, row in enumerate(rows)
            ],
            parent_process_id,
        )
    with pytest.raises(gate_module.CertificateRowProcessRefusal, match="no receipt"):
        gate_module.assemble_certificate_receipt(
            {"identity_row_count": len(rows)},
            rows,
            [
                {"case": row["case"], "process_id": 4000 + index}
                if index
                else {"case": row["case"], "process_id": None}
                for index, row in enumerate(rows)
            ],
            parent_process_id,
        )
