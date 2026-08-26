"""Contract tests for the tracked domain-partition performance receipt."""

from __future__ import annotations

import json
import math
from pathlib import Path

from benchmarks.domain_partition_performance import (
    MINIMUM_REPETITIONS,
    _synthetic_case,
    static_control_flow_record,
)


RECEIPT = (
    Path(__file__).parents[1] / "docs/figures/domain-partition-performance/receipt.json"
)


def _reject_nonfinite(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def test_receipt_records_fixed_trip_cpu_and_explicit_gpu_status():
    """The tracked receipt is complete, finite, and candid about accelerators."""

    receipt = json.loads(RECEIPT.read_text(), parse_constant=_reject_nonfinite)
    assert receipt["schema_version"] == 1
    assert receipt["policy"]["repetitions"] >= MINIMUM_REPETITIONS
    assert receipt["static_control_flow"]["passed"] is True
    assert receipt["static_control_flow"]["data_dependent_while_loop_count"] == 0

    platforms = {row["platform"]: row for row in receipt["platforms"]}
    assert set(platforms) == {"cpu", "gpu"}
    assert platforms["cpu"]["status"] == "measured"
    for name, platform in platforms.items():
        assert platform["status"] in {"measured", "skipped"}
        if platform["status"] == "skipped":
            assert name == "gpu"
            assert platform["skip_reason"]
            continue
        assert platform["jit_compile_success"] is True
        assert platform["vmap_compile_success"] is True
        assert len(platform["cases"]) == 5
        assert sum(row["kind"] == "committed_cache" for row in platform["cases"]) == 2
        assert sum(row["kind"] == "synthetic" for row in platform["cases"]) >= 3
        for row in platform["cases"]:
            assert row["eager"]["repetitions"] >= MINIMUM_REPETITIONS
            assert row["jit"]["repetitions"] >= MINIMUM_REPETITIONS
            assert math.isfinite(row["eager"]["median_ms"])
            assert math.isfinite(row["eager"]["p95_ms"])
            assert math.isfinite(row["jit"]["median_ms"])
            assert math.isfinite(row["jit"]["p95_ms"])
            assert row["eager_jit_equal"] is True
            assert row["expected_partition_match"] is True

    assert receipt["overall"]["strict_json"] is True
    assert receipt["overall"]["all_measured_jit_compile_success"] is True
    assert receipt["overall"]["all_measured_vmap_compile_success"] is True
    assert receipt["overall"]["all_measured_p95_within_interactive_budget"] is True


def test_partition_trace_uses_static_scan_not_data_dependent_while():
    """The partition trip count is the fixed number of cells in the input shape."""

    record = static_control_flow_record(_synthetic_case((9, 9)))
    assert record["data_dependent_while_loop_count"] == 0
    assert record["expected_partition_trip_count"] == 81
    assert 81 in record["scan_lengths"]
    assert record["fixed_trip_match"] is True
    assert record["passed"] is True
