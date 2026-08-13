"""Precision declarations emitted by the field-null benchmark family."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import re

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PRECISION_LINE = re.compile(
    r"^FIELD_NULL_PRECISION x64_enabled=(?:True|False) "
    r"working_dtype=(float32|float64)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class BenchmarkInvocation:
    """Smallest command that resolves one benchmark's null-fit dtype."""

    module: str
    working_dtypes: tuple[type[np.floating], ...]
    expected_dtypes: frozenset[str]

    @property
    def path(self) -> Path:
        """Return the benchmark source path."""
        return ROOT / (self.module.replace(".", "/") + ".py")


FIELD_NULL_BENCHMARKS = (
    BenchmarkInvocation(
        "benchmarks.stencil_null_route",
        (np.float64,),
        frozenset({"float64"}),
    ),
    BenchmarkInvocation(
        "benchmarks.null_stack_route",
        (np.float64,),
        frozenset({"float64"}),
    ),
    BenchmarkInvocation(
        "benchmarks.fieldnull_production_route",
        (np.float64,),
        frozenset({"float64"}),
    ),
    BenchmarkInvocation(
        "benchmarks.fieldnull_candidate_audit",
        (np.float64,),
        frozenset({"float64"}),
    ),
    BenchmarkInvocation(
        "benchmarks.fieldnull_precision_audit",
        (np.float32, np.float64),
        frozenset({"float32", "float64"}),
    ),
    BenchmarkInvocation(
        "benchmarks.select_precision_contract",
        (np.float64,),
        frozenset({"float64"}),
    ),
    BenchmarkInvocation(
        "benchmarks.fieldnull_gpu_throughput",
        (np.float32, np.float64),
        frozenset({"float32", "float64"}),
    ),
)


@pytest.mark.parametrize(
    "benchmark",
    FIELD_NULL_BENCHMARKS,
    ids=lambda benchmark: benchmark.path.stem,
)
def test_field_null_benchmark_declares_resolved_precision(
    benchmark: BenchmarkInvocation,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every null-fit benchmark states x64 capability and working dtype."""
    source = benchmark.path.read_text(encoding="utf-8")
    assert source.count("_print_field_null_precision(") >= 2

    module = importlib.import_module(benchmark.module)
    module._print_field_null_precision(*benchmark.working_dtypes)
    stdout = capsys.readouterr().out
    declared = frozenset(PRECISION_LINE.findall(stdout))
    assert declared == benchmark.expected_dtypes, stdout
