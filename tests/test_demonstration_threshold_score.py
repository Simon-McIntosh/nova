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
                    "metrics": {
                        "topology_class_agreement": True,
                        "boundary_comparison_failures": [],
                        "closed_boundary_symmetric_rms_distance_m": 0.3,
                        "polished_saddle_to_nearest_efit_x_m": 0.025,
                    },
                }
                for shot, frame in EXPECTED_DIIID_ROWS
            ]
        }
    }


def test_synthetic_complete_banks_pass_all_three_gates():
    result = score_payloads(_mast_bank(), _diiid_bank())

    assert result["verdict"] == "PASS"
    assert result["gates"]["mast_class_agreement"]["measured"] == {
        "agreement_count": 7,
        "declared_denominator": 12,
    }
    assert (
        result["gates"]["declared_row_saddle_distance"]["measured"][
            "declared_denominator"
        ]
        == 17
    )
    assert all(gate["passes"] for gate in result["gates"].values())
    assert all("threshold" in gate for gate in result["gates"].values())
    assert all("contributing_rows" in gate for gate in result["gates"].values())
    assert all("excluded_rows" in gate for gate in result["gates"].values())


def test_a_complete_bank_can_fail_a_locked_threshold_without_refusal():
    mast = _mast_bank()
    mast["rows"][6]["binding_to_efit_lcfs_rms_m"] = 0.551

    result = score_payloads(mast, _diiid_bank())

    assert result["verdict"] == "FAIL"
    assert not result["gates"]["closed_boundary_symmetric_rms"]["passes"]


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


def test_null_class_agreement_counts_against_fixed_denominator():
    mast = _mast_bank()
    mast["rows"][0]["nova_achieved_class"] = None
    mast["rows"][0]["label_agreement"] = None

    result = score_payloads(mast, _diiid_bank())
    gate = result["gates"]["mast_class_agreement"]

    assert result["verdict"] == "FAIL"
    assert gate["measured"] == {"agreement_count": 6, "declared_denominator": 12}
    assert gate["contributing_rows"][0]["reason"] == "class_agreement_unavailable"
    assert gate["contributing_rows"][0]["value"] is None


def test_null_rms_is_excluded_without_shrinking_declared_denominator():
    mast = _mast_bank()
    mast["rows"][0]["binding_to_efit_lcfs_rms_m"] = None

    result = score_payloads(mast, _diiid_bank())
    gate = result["gates"]["closed_boundary_symmetric_rms"]

    assert result["verdict"] == "FAIL"
    assert gate["measured"]["declared_denominator"] == 17
    assert gate["measured"]["eligible_row_count"] == 11
    assert gate["excluded_rows"][0]["identity"] == {
        "machine": "MAST",
        "shot": EXPECTED_MAST_ARMS[0][0],
        "slice_index": EXPECTED_MAST_ARMS[0][1],
        "arm": EXPECTED_MAST_ARMS[0][2],
    }
    assert gate["excluded_rows"][0]["reason"] == "closed_boundary_rms_unavailable"


def test_nonconverged_row_is_rms_ineligible_without_refusal():
    mast = _mast_bank()
    mast["rows"][0]["converged"] = False

    result = score_payloads(mast, _diiid_bank())
    gate = result["gates"]["closed_boundary_symmetric_rms"]

    assert result["verdict"] == "FAIL"
    assert gate["measured"]["declared_denominator"] == 17
    assert gate["excluded_rows"][0]["reason"] == "non_converged"


def test_null_saddle_fails_gate_and_names_declared_row():
    diiid = _diiid_bank()
    diiid["result"]["frame_records"][3]["metrics"][
        "polished_saddle_to_nearest_efit_x_m"
    ] = None

    result = score_payloads(_mast_bank(), diiid)
    gate = result["gates"]["declared_row_saddle_distance"]

    assert result["verdict"] == "FAIL"
    assert gate["measured"]["declared_denominator"] == 17
    assert gate["excluded_rows"] == [
        {
            "identity": {
                "machine": "DIII-D",
                "shot": EXPECTED_DIIID_ROWS[3][0],
                "frame": EXPECTED_DIIID_ROWS[3][1],
            },
            "value": None,
            "reason": "saddle_distance_unavailable",
        }
    ]


def test_empty_bank_is_refused():
    with pytest.raises(ScoreRefusal, match="bank is empty"):
        score_payloads({"rows": []}, _diiid_bank())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_declared_row_metric_fails_saddle_gate(value):
    diiid = _diiid_bank()
    diiid["result"]["frame_records"][2]["metrics"][
        "polished_saddle_to_nearest_efit_x_m"
    ] = value

    result = score_payloads(_mast_bank(), diiid)

    assert result["verdict"] == "FAIL"
    assert (
        result["gates"]["declared_row_saddle_distance"]["excluded_rows"][0]["reason"]
        == "saddle_distance_unavailable"
    )


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
