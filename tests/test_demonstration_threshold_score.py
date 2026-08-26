"""Fail-closed tests for the fixed-cohort demonstration scorer."""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.demonstration_threshold_score import (
    EXPECTED_DIIID_ROWS,
    EXPECTED_MAST_ARMS,
    ScoreRefusal,
    run,
    score_payloads,
)


def _mast_bank() -> dict:
    rows = []
    for index, (shot, slice_index, arm) in enumerate(EXPECTED_MAST_ARMS):
        agrees = index < 7
        rows.append(
            {
                "identity": f"{shot}/{slice_index}",
                "shot": shot,
                "slice_index": slice_index,
                "arm": arm,
                "converged": agrees,
                "efit_label": "diverted",
                "nova_achieved_class": "diverted" if agrees else "limited",
                "label_agreement": agrees,
                "comparison_failures": [],
                "binding_to_efit_lcfs_rms_m": 0.2 + index / 100,
                "selected_saddle_to_efit_x_point_m": 0.02 + index / 1000,
            }
        )
    return {"rows": rows}


def _diiid_bank() -> dict:
    return {
        "result": {
            "frame_records": [
                {
                    "shot": shot,
                    "frame": frame,
                    "finite": True,
                    "converged": True,
                    "metrics": {"polished_saddle_to_nearest_efit_x_m": 0.025},
                }
                for shot, frame in EXPECTED_DIIID_ROWS
            ]
        }
    }


def test_synthetic_complete_banks_pass_all_three_gates():
    result = score_payloads(_mast_bank(), _diiid_bank())

    assert result["verdict"] == "PASS"
    assert result["gates"]["mast_class_agreement"]["value"] == 7
    assert result["gates"]["mast_class_agreement"]["denominator"] == 12
    assert (
        result["gates"]["declared_row_saddle_distance"]["declared_row_denominator"]
        == 17
    )


def test_a_complete_bank_can_fail_a_locked_threshold_without_refusal():
    mast = _mast_bank()
    mast["rows"][6]["binding_to_efit_lcfs_rms_m"] = 0.551

    result = score_payloads(mast, _diiid_bank())

    assert result["verdict"] == "FAIL"
    assert not result["gates"]["mast_closed_boundary_symmetric_rms"]["passes"]


def test_missing_mast_identity_is_refused():
    mast = _mast_bank()
    mast["rows"].pop()

    with pytest.raises(ScoreRefusal, match="cohort mismatch"):
        score_payloads(mast, _diiid_bank())


def test_duplicate_mast_identity_is_refused():
    mast = _mast_bank()
    mast["rows"][-1] = copy.deepcopy(mast["rows"][0])

    with pytest.raises(ScoreRefusal, match="duplicate identities"):
        score_payloads(mast, _diiid_bank())


def test_substituted_diiid_identity_is_refused():
    diiid = _diiid_bank()
    diiid["result"]["frame_records"][0]["frame"] = 180

    with pytest.raises(ScoreRefusal, match="substituted"):
        score_payloads(_mast_bank(), diiid)


def test_nonconverged_required_row_is_refused():
    mast = _mast_bank()
    mast["rows"][0]["converged"] = False

    with pytest.raises(ScoreRefusal, match="cannot be removed from the RMS gate"):
        score_payloads(mast, _diiid_bank())


def test_nonconverged_declared_diiid_row_is_refused():
    diiid = _diiid_bank()
    diiid["result"]["frame_records"][3]["converged"] = False

    with pytest.raises(ScoreRefusal, match="cannot be removed from the saddle gate"):
        score_payloads(_mast_bank(), diiid)


def test_empty_bank_is_refused():
    with pytest.raises(ScoreRefusal, match="bank is empty"):
        score_payloads({"rows": []}, _diiid_bank())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_declared_row_metric_is_refused(value):
    diiid = _diiid_bank()
    diiid["result"]["frame_records"][2]["metrics"][
        "polished_saddle_to_nearest_efit_x_m"
    ] = value

    with pytest.raises(ScoreRefusal, match="non-finite"):
        score_payloads(_mast_bank(), diiid)


def test_run_writes_strict_json_with_input_digests_and_fail_verdict(tmp_path):
    mast = _mast_bank()
    mast["rows"][0]["selected_saddle_to_efit_x_point_m"] = 0.041
    mast_path = tmp_path / "mast.json"
    diiid_path = tmp_path / "diiid.json"
    output = tmp_path / "receipt.json"
    mast_path.write_text(json.dumps(mast))
    diiid_path.write_text(json.dumps(_diiid_bank()))

    receipt = run(mast_path, diiid_path, output)
    decoded = json.loads(
        output.read_text(), parse_constant=lambda value: pytest.fail(value)
    )

    assert receipt["verdict"] == "FAIL"
    assert decoded == receipt
    assert len(receipt["inputs"]["mast"]["sha256"]) == 64
    assert len(receipt["inputs"]["diiid"]["sha256"]) == 64
    assert len(receipt["inputs"]["mast"]["declared_identities"]) == 12
    assert len(receipt["inputs"]["diiid"]["declared_identities"]) == 5
