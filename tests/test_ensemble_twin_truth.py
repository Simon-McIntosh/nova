"""Forward-only deterministic twin package checks."""

from __future__ import annotations

import csv
import json

from nova.transport import CouplingState
from scripts.ensemble_twin_truth import generate


def _rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_package_marches_six_member_batched_windows(tmp_path, monkeypatch):
    calls = []
    public_batch = generate.solve_window_batch

    def observe_batch(*args, **kwargs):
        receipt = public_batch(*args, **kwargs)
        calls.append(receipt.member_ids)
        return receipt

    monkeypatch.setattr(generate, "solve_window_batch", observe_batch)
    receipt = generate.generate_package(tmp_path, tree_sha="known-tree")
    trajectory = _rows(tmp_path / "trajectory.tsv")

    assert receipt["window_count"] == 6
    assert receipt["member_count"] == 4
    assert receipt["trajectory_rows"] == 24
    assert receipt["all_windows_converged"] is True
    assert len(calls) == 6
    assert all(call == generate.MEMBER_IDS for call in calls)
    assert {row["tree_sha"] for row in trajectory} == {"known-tree"}
    assert {row["window_index"] for row in trajectory} == {
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
    }
    assert {row["role"] for row in trajectory} == {"truth", "counterfactual"}
    assert all(float(row["gating_norm"]) <= 1.0e-10 for row in trajectory)


def test_observation_rows_carry_deterministic_receipts(tmp_path):
    receipt = generate.generate_package(tmp_path, tree_sha="observation-tree")
    observations = _rows(tmp_path / "observations.tsv")

    assert receipt["observation_rows"] == 144
    assert receipt["all_observations_supported"] is True
    assert len(observations) == 6 * 4 * 6
    assert {row["thomson_cocos"] for row in observations} == {"17"}
    assert {row["temperature_unit"] for row in observations} == {"eV"}
    assert {row["density_unit"] for row in observations} == {"m^-3"}
    assert {row["net_current_unit"] for row in observations} == {"A"}
    assert all(row["supported"] == "True" for row in observations)
    assert all(float(row["temperature_error_bound_ev"]) >= 0.0 for row in observations)
    assert all(float(row["density_error_bound_m-3"]) >= 0.0 for row in observations)


def test_coupling_state_handoffs_are_lossless_and_gate_is_ambix_owned(tmp_path):
    receipt = generate.generate_package(tmp_path, tree_sha="handoff-tree")
    coupling_rows = [
        json.loads(line)
        for line in (tmp_path / "coupling_states.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert receipt["coupling_state_rows"] == 24
    assert receipt["joint_recovery_gate_run"] is False
    assert receipt["joint_recovery_gate_owner"] == "ambix"
    assert "Not run in Nova" in receipt["joint_recovery_gate_note"]
    assert all(
        CouplingState.from_dict(row["coupling_state"]).to_dict()
        == row["coupling_state"]
        for row in coupling_rows
    )
    consumption = (tmp_path / "consumption.md").read_text(encoding="utf-8")
    assert "not run here" in consumption
    assert "Ambix-owned" in consumption
