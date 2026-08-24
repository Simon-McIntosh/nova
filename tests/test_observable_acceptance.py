from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import observable_batch_acceptance as batch_acceptance
from nova.equilibrium.observable_acceptance import (
    evaluate_observable_bound_acceptance,
)


def _registration():
    return [
        {
            "observable": "integer_label",
            "criterion_kind": "exact_equality",
            "dtype": "int64",
            "shape": [],
        },
        {
            "observable": "floating_label",
            "criterion_kind": "banked_dual_envelope",
            "dtype": "float64",
            "shape": [2],
            "absolute_bound": 0.2,
            "relative_bound": 0.1,
        },
    ]


def test_acceptance_scores_every_case_and_batch_member():
    reference = {
        "integer_label": np.ones((2, 3), dtype=np.int64),
        "floating_label": np.ones((2, 3, 2), dtype=np.float64),
    }
    candidate = {name: value.copy() for name, value in reference.items()}
    candidate["floating_label"][1, 2, 0] += 0.15

    result = evaluate_observable_bound_acceptance(
        reference=reference,
        candidate=candidate,
        registration=_registration(),
        case_ids=("first", "second"),
        batch_size=3,
    )

    assert result["registered_bound_count"] == 2
    assert result["observable_pass_count"] == 1
    assert result["case_observable_evaluation_pass_count"] == 3
    assert result["member_observable_evaluation_pass_count"] == 11
    floating = {row["observable"]: row for row in result["per_observable"]}[
        "floating_label"
    ]
    assert floating["passes"] is False
    assert floating["case_pass_count"] == 1
    assert floating["maximum_absolute_difference"] == pytest.approx(0.15)
    assert floating["maximum_relative_difference"] == pytest.approx(0.15)


def test_exact_equality_retains_nan_equality_and_rejects_a_changed_value():
    registration = [
        {
            "observable": "label",
            "criterion_kind": "exact_equality",
            "dtype": "float64",
            "shape": [2],
        }
    ]
    reference = {"label": np.array([[[np.nan, 1.0]], [[np.nan, 2.0]]])}
    equal = {"label": reference["label"].copy()}
    changed = {"label": reference["label"].copy()}
    changed["label"][1, 0, 1] = 3.0

    accepted = evaluate_observable_bound_acceptance(
        reference=reference,
        candidate=equal,
        registration=registration,
        case_ids=("first", "second"),
        batch_size=1,
    )
    refused = evaluate_observable_bound_acceptance(
        reference=reference,
        candidate=changed,
        registration=registration,
        case_ids=("first", "second"),
        batch_size=1,
    )

    assert accepted["passes"] is True
    assert refused["passes"] is False
    assert refused["per_observable"][0]["maximum_bound_ratio"] is None


def test_acceptance_fails_closed_on_shape_dtype_and_missing_registration():
    reference = {
        "integer_label": np.ones((2, 1), dtype=np.int64),
        "floating_label": np.ones((2, 1, 2), dtype=np.float64),
    }
    candidate = {name: value.copy() for name, value in reference.items()}

    with pytest.raises(ValueError, match="registered observables are absent"):
        evaluate_observable_bound_acceptance(
            reference=reference,
            candidate={"integer_label": candidate["integer_label"]},
            registration=_registration(),
            case_ids=("first", "second"),
            batch_size=1,
        )
    changed_shape = {**candidate, "floating_label": np.ones((2, 2, 2))}
    with pytest.raises(ValueError, match="requires reference and candidate shape"):
        evaluate_observable_bound_acceptance(
            reference=reference,
            candidate=changed_shape,
            registration=_registration(),
            case_ids=("first", "second"),
            batch_size=1,
        )
    changed_dtype = {
        **candidate,
        "floating_label": candidate["floating_label"].astype(np.float32),
    }
    with pytest.raises(ValueError, match="changes dtype"):
        evaluate_observable_bound_acceptance(
            reference=reference,
            candidate=changed_dtype,
            registration=_registration(),
            case_ids=("first", "second"),
            batch_size=1,
        )


def test_complete_registered_family_is_accepted_at_two_batch_sizes():
    criterion = json.loads(
        Path(
            "docs/figures/forward-operator-refinement/criterion-family.json"
        ).read_text(encoding="utf-8")
    )
    registration = criterion["criterion_family"]["terminal_compiled_parity"][
        "terminal_observable_registration"
    ]["bounds"]
    case_ids = tuple(f"case-{index}" for index in range(6))

    for batch_size in (1, 4):
        reference = {}
        for row in registration:
            shape = (len(case_ids), batch_size, *row["shape"])
            reference[row["observable"]] = np.zeros(shape, dtype=row["dtype"])
        result = evaluate_observable_bound_acceptance(
            reference=reference,
            candidate={name: value.copy() for name, value in reference.items()},
            registration=registration,
            case_ids=case_ids,
            batch_size=batch_size,
        )

        assert result["registered_bound_count"] == 69
        assert result["observable_pass_count"] == 69
        assert result["case_observable_evaluation_pass_count"] == 414
        assert result["passes"] is True


def test_batch_dependence_is_explicit_for_every_observable():
    base = {
        "batch_size": 1,
        "per_observable": [
            {
                "observable": "stable",
                "passes": True,
                "case_pass_count": 6,
                "maximum_absolute_difference": 0.0,
                "cases": [{"case_id": "held-out", "passes": True}],
            },
            {
                "observable": "dependent",
                "passes": True,
                "case_pass_count": 6,
                "maximum_absolute_difference": 0.0,
                "cases": [{"case_id": "held-out", "passes": True}],
            },
            {
                "observable": "case-dependent",
                "passes": False,
                "case_pass_count": 5,
                "maximum_absolute_difference": 1.0,
                "cases": [{"case_id": "held-out", "passes": True}],
            },
        ],
    }
    wider = {
        "batch_size": 4,
        "per_observable": [
            {
                "observable": "stable",
                "passes": True,
                "case_pass_count": 6,
                "maximum_absolute_difference": 0.0,
                "cases": [{"case_id": "held-out", "passes": True}],
            },
            {
                "observable": "dependent",
                "passes": False,
                "case_pass_count": 5,
                "maximum_absolute_difference": 1.0,
                "cases": [{"case_id": "held-out", "passes": False}],
            },
            {
                "observable": "case-dependent",
                "passes": False,
                "case_pass_count": 5,
                "maximum_absolute_difference": 1.0,
                "cases": [{"case_id": "held-out", "passes": False}],
            },
        ],
    }

    rows = {
        row["observable"]: row
        for row in batch_acceptance._batch_dependence([base, wider])
    }

    assert rows["stable"]["pass_status_depends_on_batch_size"] is False
    assert rows["dependent"]["pass_status_depends_on_batch_size"] is True
    assert rows["dependent"]["aggregate_pass_status_depends_on_batch_size"] is True
    assert rows["dependent"]["case_pass_status_depends_on_batch_size"] is True
    assert (
        rows["case-dependent"]["aggregate_pass_status_depends_on_batch_size"] is False
    )
    assert rows["case-dependent"]["case_pass_status_depends_on_batch_size"] is True
    assert rows["case-dependent"]["pass_status_depends_on_batch_size"] is True
    assert rows["dependent"]["pass_status_by_batch_size"] == {
        "1": True,
        "4": False,
    }


def test_repetition_difference_counts_bit_patterns_against_first_run():
    baseline = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    signed_zero = baseline.copy()
    signed_zero[0] = -0.0
    next_float = baseline.copy()
    next_float[1] = np.nextafter(1.0, 2.0)

    result = batch_acceptance._repetition_difference(
        np.stack([baseline, signed_zero, next_float])
    )

    assert result["reference_repetition"] == 1
    assert result["comparison_count"] == 2
    assert result["maximum_bitwise_unequal_element_count"] == 1
    assert result["maximum_absolute_difference"] == np.nextafter(1.0, 2.0) - 1.0
    assert result["maximum_relative_difference"] == (np.nextafter(1.0, 2.0) - 1.0) / 2.0
    assert [row["bitwise_unequal_element_count"] for row in result["comparisons"]] == [
        1,
        1,
    ]


def test_lossless_repetition_arrays_put_repetition_before_case(tmp_path):
    cases = []
    for case_index in range(2):
        batches = {}
        for batch_size in (1, 4):
            values = np.arange(3 * batch_size * 2, dtype=np.float64).reshape(
                3, batch_size, 2
            )
            values += 100.0 * case_index + 10.0 * batch_size
            batches[batch_size] = {
                "flux": values.copy(),
                "observables": {"label": values.copy()},
            }
        cases.append(
            {
                "case_id": f"case-{case_index}",
                "reference": {"label": np.array([case_index, case_index + 1])},
                "batches": batches,
            }
        )
    output = tmp_path / "repetitions.npz"

    manifest = batch_acceptance._write_repetition_arrays(
        output, cases, {"label"}, (1, 4)
    )

    with np.load(output, allow_pickle=False) as arrays:
        flux = arrays[manifest["terminal_flux"]["4"]["key"]]
        labels = arrays[manifest["observables"]["label"]["4"]["key"]]
        assert flux.shape == (3, 2, 4, 2)
        assert np.array_equal(flux, labels)
        assert arrays["case_ids"].tolist() == ["case-0", "case-1"]
        assert arrays["repetitions"].tolist() == [1, 2, 3]
    assert manifest["terminal_flux"]["4"]["axis_order"][:2] == [
        "repetition",
        "case",
    ]


def test_committed_receipt_covers_two_real_batch_sizes_and_all_bounds():
    receipt = json.loads(batch_acceptance.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    results = receipt["batch_results"]
    dependence = receipt["per_observable_batch_dependence"]

    assert receipt["status"] == "complete"
    assert receipt["measurement_contract"]["acceptance_entry_point"] == (
        "nova.equilibrium.observable_acceptance.evaluate_observable_bound_acceptance"
    )
    assert [row["batch_size"] for row in results] == [1, 4]
    assert [row["registered_bound_count"] for row in results] == [69, 69]
    assert [row["case_count"] for row in results] == [6, 6]
    assert all(0 <= row["observable_pass_count"] <= 69 for row in results)
    assert all(
        0 <= row["case_observable_evaluation_pass_count"] <= 414 for row in results
    )
    assert len(dependence) == 69
    assert all(
        isinstance(row["pass_status_depends_on_batch_size"], bool) for row in dependence
    )
    dependent_count = sum(
        row["pass_status_depends_on_batch_size"] for row in dependence
    )
    assert receipt["batch_dependent_bound_count"] == dependent_count
    assert len(receipt["measurement_repetitions"]) >= 2
    assert receipt["repetition_stability"]["repetition_count"] == len(
        receipt["measurement_repetitions"]
    )
