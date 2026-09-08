"""Regression checks for the allocation-free strict-exit batch proof."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks import strict_exit_incidence as incidence


def test_batched_self_check_requires_explicit_batched_mode():
    with pytest.raises(ValueError, match="--self-check requires --batched"):
        incidence._validate_arguments(batched=False, self_check=True)


def test_select_member_keeps_scalar_receipt_leaves_unindexed():
    selected = incidence._select_member(
        {
            "per_member": incidence.jnp.asarray([3, 5]),
            "scalar": incidence.jnp.asarray(7),
        },
        1,
    )

    assert int(selected["per_member"]) == 5
    assert int(selected["scalar"]) == 7


def test_batch_member_result_selects_the_sequential_fallback_member():
    results = ("first", "second")

    assert incidence._batch_member_result(results, 1, stacked=False) == "second"


def test_cpu_self_check_requires_the_debug_allocation(monkeypatch):
    environment = {
        "SLURM_JOB_ID": "42",
        "SLURM_JOB_NAME": "cpu-proof",
        "SLURM_JOB_PARTITION": "all_debug",
        "SLURM_CPUS_PER_TASK": "4",
        "SLURM_MEM_PER_NODE": str(64 * 1024),
        "SLURM_TIMELIMIT": "00:55:00",
        "TMPDIR": "/tmp",
        "JAX_PLATFORMS": "cpu",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        incidence.jax,
        "devices",
        lambda: [SimpleNamespace(platform="cpu", device_kind="cpu")],
    )

    receipt = incidence._require_cpu_self_check()

    assert receipt["job_id"] == 42
    assert receipt["partition"] == "all_debug"
    assert receipt["cpu_count"] == 4
    assert receipt["requested_memory_mib"] == 64 * 1024
    assert receipt["execution_mode"] == "slurm_cpu_self_check"
