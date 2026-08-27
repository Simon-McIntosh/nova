"""Focused contracts for pinned-branch terminal failure retention."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "benchmarks" / "pinned_branch_contrast.py"


def _driver():
    spec = spec_from_file_location("pinned_branch_contrast_driver", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observables():
    return {
        "achieved_class": "diverted",
        "topology_consistent": True,
        "class_margin": 0.25,
        "class_margin_nonfinite": None,
        "p_diverted": 0.9,
        "terminal_xpoint_diagnostics": {
            "selection_status": "selected_typed_saddle_with_connectivity_support",
            "selected_x_normalized_flux_operand": 1.0,
            "selected_x_normalized_flux_operand_nonfinite": None,
            "wall_operand": {
                "status": "selected",
                "normalized_flux_before_shadow": 1.1,
                "normalized_flux": 1.1,
                "normalized_flux_nonfinite": None,
            },
        },
    }


def _arm(observables, residual):
    return {
        "terminal_residual": residual,
        "converged": observables["failure_exception_class"] is None,
        "fitted_contraction": {"contracts": True},
        **observables,
    }


def test_axis_disqualification_retains_twelve_arm_receipt(monkeypatch):
    driver = _driver()

    def terminal_observables(_profile, state):
        if state == 5:
            raise driver.NoQualifiedAxisError("synthetic axis disqualification")
        return _observables()

    monkeypatch.setattr(driver, "_terminal_observables", terminal_observables)
    observed = [
        driver._terminal_observables_retaining_axis_failure(None, state)
        for state in range(12)
    ]
    records = [
        {
            "reference": {"shot": 21_000 + index, "slice_index": index},
            "pure_arm": _arm(observed[2 * index], 1.0e-8 + index * 1.0e-9),
            "mixed_arm": _arm(observed[2 * index + 1], 2.0e-8 + index * 1.0e-9),
        }
        for index in range(6)
    ]

    summary = driver._receipt_summary(records)
    regenerated = {"references": records}
    merged = driver._merge_terminal_diagnostics({"references": records}, regenerated)
    failed = records[2]["mixed_arm"]

    assert summary["reference_count"] == 6
    assert summary["arm_count"] == 12
    assert summary["retained_failure_arm_count"] == 1
    assert summary["retained_failure_arms"] == [
        {
            "shot": 21_002,
            "slice_index": 2,
            "arm": "mixed_arm",
            "exception_class": "NoQualifiedAxisError",
        }
    ]
    assert failed["converged"] is False
    assert failed["termination_reason"] == "NoQualifiedAxisError"
    assert failed["failure_exception_class"] == "NoQualifiedAxisError"
    assert failed["achieved_class"] is None
    assert failed["topology_consistent"] is None
    assert failed["class_margin"] is None
    assert failed["p_diverted"] is None
    assert failed["terminal_xpoint_diagnostics"] is None
    assert merged["semantic_rebaseline"] == {
        "status": "unavailable_due_to_retained_terminal_failure",
        "retained_failure_count": 1,
        "retained_failures": summary["retained_failure_arms"],
    }
    json.dumps(merged, allow_nan=False)


def test_unrelated_terminal_observation_exception_propagates(monkeypatch):
    driver = _driver()
    monkeypatch.setattr(
        driver,
        "_terminal_observables",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("unrelated failure")),
    )

    with pytest.raises(RuntimeError, match="unrelated failure"):
        driver._terminal_observables_retaining_axis_failure(None, 0)


def test_campaign_releases_compilation_state_before_single_publication(
    monkeypatch, tmp_path
):
    driver = _driver()
    selected = [
        ({"shot": 20_001 + index, "slice_index": 2 * index + 3}, f"row-{index}")
        for index in range(6)
    ]
    events = []

    def build_reference(
        store,
        response_cache,
        selected_row,
        qualification,
        *,
        include_parity,
    ):
        assert store == "store"
        assert response_cache == "responses"
        identity = f"{selected_row['shot']}/{selected_row['slice_index']}"
        events.append(("build", identity, qualification, include_parity))
        record = {"reference": identity}
        parity = {"passes": True} if include_parity else None
        return record, 0, parity

    monkeypatch.setattr(driver, "_build_reference_contrast", build_reference)
    monkeypatch.setattr(driver.jax, "clear_caches", lambda: events.append(("clear",)))
    monkeypatch.setattr(driver.gc, "collect", lambda: events.append(("collect",)))
    monkeypatch.setattr(driver, "configure_dtypes", lambda: None)
    monkeypatch.setattr(driver, "_source_revision", lambda: "revision")
    monkeypatch.setattr(
        driver,
        "_persisted_response_cache",
        lambda *_args: ("responses", {"carrier": "evidence"}),
    )
    monkeypatch.setattr(driver, "select_slices_by_shot", lambda _bank: selected)
    monkeypatch.setattr(
        driver,
        "_receipt_summary",
        lambda records: {"reference_count": len(records)},
    )
    monkeypatch.setattr(
        driver,
        "_publish_receipt",
        lambda output, receipt: events.append(
            ("write", output, list(receipt["references"]))
        ),
    )

    output = tmp_path / "receipt.json"
    receipt = driver.run(
        store="store",
        bank=tmp_path / "bank.json",
        output=output,
        carrier=tmp_path / "carrier.npz",
        carrier_receipt=tmp_path / "carrier-receipt.json",
    )

    records = [
        {"reference": f"{20_001 + index}/{2 * index + 3}"} for index in range(6)
    ]
    assert receipt["references"] == records
    assert receipt["direct_green_operator_builder_entries"] == 0
    assert receipt["jit_vmap_batch_two_parity"] == {"passes": True}
    expected_events = []
    for index in range(6):
        expected_events.extend(
            (
                (
                    "build",
                    f"{20_001 + index}/{2 * index + 3}",
                    f"row-{index}",
                    index == 0,
                ),
                ("clear",),
                ("collect",),
            )
        )
    expected_events.append(("write", output, records))
    assert events == expected_events
