"""Measure fixed-trip domain-partition latency and accelerator compatibility."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.domain import PlasmaDomain, axis_connected_component
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).parents[1]
OPERANDS = (
    ROOT / "docs/figures/topology-visual-corroboration/mast-topology-operands.npz"
)
DEFAULT_OUTPUT = ROOT / "docs/figures/domain-partition-performance/receipt.json"
SYNTHETIC_SHAPES = ((9, 9), (17, 17), (33, 33))
INTERACTIVE_P95_BUDGET_MS = 125.0
MINIMUM_REPETITIONS = 50


@dataclass(frozen=True)
class PartitionCase:
    """One fixed-shape set of operands for the production partition kernel."""

    name: str
    kind: str
    confined: np.ndarray
    rings: np.ndarray
    link_admissible: np.ndarray
    axis_seed: np.ndarray
    expected_component_cells: int | None = None
    expected_private_cells: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the raster shape carrying the hex cells."""

        return self.confined.shape


def _committed_case(row: int, name: str, expected: tuple[int, int]) -> PartitionCase:
    """Load one corrected partition case from the committed operand bank."""

    with np.load(OPERANDS, allow_pickle=False) as stored:
        coordinate = stored[f"row_{row:02d}_cell_rz"]
        old_label = stored[f"row_{row:02d}_domain_labels"]
        axis = stored[f"row_{row:02d}_selected_o"][0]

    radius = np.unique(coordinate[:, 0])
    height = np.unique(coordinate[:, 1])
    shape = (height.size, radius.size)
    closed = np.isin(
        old_label,
        (int(PlasmaDomain.CORE), int(PlasmaDomain.PRIVATE_FLUX)),
    )
    inside = old_label != int(PlasmaDomain.EXCLUDED_MATERIAL)
    confined = (closed & inside).reshape((radius.size, height.size)).T
    rings = np.asarray(hex_stencil(shape), dtype=np.int32)
    links = np.ones(rings.shape, dtype=bool)
    seed_index = int(np.argmin(np.sum((coordinate - axis) ** 2, axis=1)))
    seed = np.zeros(coordinate.shape[0], dtype=bool)
    seed[seed_index] = True
    seed = seed.reshape((radius.size, height.size)).T
    return PartitionCase(
        name=name,
        kind="committed_cache",
        confined=confined,
        rings=rings,
        link_admissible=links,
        axis_seed=seed,
        expected_component_cells=expected[0],
        expected_private_cells=expected[1],
    )


def _synthetic_case(shape: tuple[int, int]) -> PartitionCase:
    """Construct a deterministic two-basin hex graph at one resolution."""

    height, width = shape
    row, column = np.indices(shape)
    radial = (column - (width - 1) / 2.0) / max(width - 1, 1)
    vertical = (row - (height - 1) / 2.0) / max(height - 1, 1)
    confined = radial**2 / 0.23 + vertical**2 / 0.22 <= 1.0
    rings = np.asarray(hex_stencil(shape), dtype=np.int32)
    links = np.ones(rings.shape, dtype=bool)

    flat_row = row.reshape(-1)
    centre_rows = flat_row[rings[:, :1]]
    neighbour_rows = flat_row[rings[:, 1:]]
    cut = height // 2
    crosses_cut = (centre_rows < cut) != (neighbour_rows < cut)
    links[:, 1:] &= ~crosses_cut

    seed = np.zeros(shape, dtype=bool)
    candidates = np.argwhere(confined & (row < cut))
    seed[tuple(candidates[len(candidates) // 2])] = True
    return PartitionCase(
        name=f"synthetic-{height}x{width}",
        kind="synthetic",
        confined=confined,
        rings=rings,
        link_admissible=links,
        axis_seed=seed,
    )


def build_cases() -> list[PartitionCase]:
    """Return both committed caches and the resolution sweep."""

    return [
        _committed_case(0, "mast-21978-35-pure", (277, 20)),
        _committed_case(1, "mast-21978-35-mixed", (244, 33)),
        *(_synthetic_case(shape) for shape in SYNTHETIC_SHAPES),
    ]


def _nested_jaxprs(value: Any):
    """Yield every JAX expression nested in one equation parameter."""

    if hasattr(value, "eqns"):
        yield value
        for equation in value.eqns:
            yield from _nested_jaxprs(equation.params)
    elif hasattr(value, "jaxpr"):
        yield from _nested_jaxprs(value.jaxpr)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _nested_jaxprs(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _nested_jaxprs(item)


def static_control_flow_record(case: PartitionCase) -> dict[str, Any]:
    """Prove the traced kernel has fixed trips and no data-dependent loop."""

    arguments = (
        jnp.asarray(case.confined),
        jnp.asarray(case.rings),
        jnp.asarray(case.link_admissible),
        jnp.asarray(case.axis_seed),
    )
    traced = jax.make_jaxpr(axis_connected_component.__wrapped__)(*arguments)
    expressions = list(_nested_jaxprs(traced))
    equations = [equation for expression in expressions for equation in expression.eqns]
    while_count = sum(equation.primitive.name == "while" for equation in equations)
    scan_lengths = [
        int(equation.params["length"])
        for equation in equations
        if equation.primitive.name == "scan"
    ]
    expected_trips = int(case.confined.size)
    fixed_trip_match = expected_trips in scan_lengths
    return {
        "case": case.name,
        "shape": list(case.shape),
        "cell_count": expected_trips,
        "labelling_cap": expected_trips,
        "data_dependent_while_loop_count": while_count,
        "scan_lengths": scan_lengths,
        "expected_partition_trip_count": expected_trips,
        "fixed_trip_match": fixed_trip_match,
        "passed": while_count == 0 and fixed_trip_match,
    }


def _block(value: jax.Array) -> np.ndarray:
    """Synchronize one result and return its host representation."""

    return np.asarray(jax.device_get(value.block_until_ready()))


def _latency(
    function, arguments, repetitions: int
) -> tuple[dict[str, Any], np.ndarray]:
    """Measure synchronized warm-call latency after compilation and allocation."""

    result = function(*arguments)
    expected = _block(result)
    _block(function(*arguments))
    samples = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        started = time.perf_counter_ns()
        result = function(*arguments)
        _block(result)
        samples[repetition] = (time.perf_counter_ns() - started) / 1.0e6
    return (
        {
            "repetitions": repetitions,
            "median_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95)),
            "minimum_ms": float(np.min(samples)),
            "maximum_ms": float(np.max(samples)),
        },
        expected,
    )


def _device_case(
    case: PartitionCase,
    device: jax.Device,
    repetitions: int,
    budget_ms: float,
) -> dict[str, Any]:
    """Measure eager orchestration, outer JIT, and batched compilation."""

    with jax.default_device(device):
        arguments = tuple(
            jax.device_put(value, device)
            for value in (
                case.confined,
                case.rings,
                case.link_admissible,
                case.axis_seed,
            )
        )
        eager = axis_connected_component.__wrapped__
        compiled = jax.jit(axis_connected_component.__wrapped__)
        eager_timing, eager_result = _latency(eager, arguments, repetitions)
        compiled_timing, compiled_result = _latency(compiled, arguments, repetitions)
        equal = bool(np.array_equal(eager_result, compiled_result))

        batched = jax.jit(jax.vmap(axis_connected_component.__wrapped__))
        batched_arguments = tuple(
            jnp.stack((argument, argument)) for argument in arguments
        )
        batched_result = _block(batched(*batched_arguments))
        vmap_success = bool(
            batched_result.shape == (2, *case.shape)
            and np.array_equal(batched_result[0], compiled_result)
            and np.array_equal(batched_result[1], compiled_result)
        )

    component_cells = int(np.sum(compiled_result))
    confined_cells = int(np.sum(case.confined))
    private_cells = confined_cells - component_cells
    expected_match = case.expected_component_cells is None or (
        component_cells == case.expected_component_cells
        and private_cells == case.expected_private_cells
    )
    return {
        "case": case.name,
        "kind": case.kind,
        "shape": list(case.shape),
        "cell_count": int(case.confined.size),
        "labelling_cap": int(case.confined.size),
        "component_cells": component_cells,
        "private_cells": private_cells,
        "expected_partition_match": expected_match,
        "eager": eager_timing,
        "jit": compiled_timing,
        "eager_jit_equal": equal,
        "jit_compile_success": equal,
        "vmap_compile_success": vmap_success,
        "within_interactive_p95_budget": (
            eager_timing["p95_ms"] <= budget_ms
            and compiled_timing["p95_ms"] <= budget_ms
        ),
    }


def _platform_record(
    backend: str,
    cases: list[PartitionCase],
    repetitions: int,
    budget_ms: float,
) -> dict[str, Any]:
    """Return one measured platform row or an explicit named skip."""

    try:
        devices = jax.devices(backend)
    except RuntimeError as error:
        return {
            "platform": backend,
            "status": "skipped",
            "skip_reason": f"{backend} backend unavailable: {error}",
            "jit_compile_success": False,
            "vmap_compile_success": False,
            "cases": [],
        }
    if not devices:
        return {
            "platform": backend,
            "status": "skipped",
            "skip_reason": f"{backend} backend reported no devices",
            "jit_compile_success": False,
            "vmap_compile_success": False,
            "cases": [],
        }

    device = devices[0]
    measured = [_device_case(case, device, repetitions, budget_ms) for case in cases]
    return {
        "platform": backend,
        "status": "measured",
        "device": str(device),
        "jit_compile_success": all(row["jit_compile_success"] for row in measured),
        "vmap_compile_success": all(row["vmap_compile_success"] for row in measured),
        "cases": measured,
    }


def _git_commit() -> str:
    """Return the source revision used for the measurement."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _strict_json(value: Any, path: str = "receipt") -> None:
    """Reject non-finite values before JSON serialization."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _strict_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _strict_json(item, f"{path}[{index}]")


def run(repetitions: int, budget_ms: float) -> dict[str, Any]:
    """Run the complete CPU/GPU receipt measurement."""

    if repetitions < MINIMUM_REPETITIONS:
        raise ValueError(f"repetitions must be at least {MINIMUM_REPETITIONS}")
    if not math.isfinite(budget_ms) or budget_ms <= 0.0:
        raise ValueError("the interactive budget must be positive and finite")
    configure_dtypes()
    cases = build_cases()
    static_rows = [static_control_flow_record(case) for case in cases]
    platforms = [
        _platform_record(backend, cases, repetitions, budget_ms)
        for backend in ("cpu", "gpu")
    ]
    measured_rows = [row for row in platforms if row["status"] == "measured"]
    receipt = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "generator": "benchmarks/domain_partition_performance.py",
            "production_callable": ("nova.equilibrium.domain.axis_connected_component"),
            "committed_operands": str(OPERANDS.relative_to(ROOT)),
            "git_commit": _git_commit(),
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
        },
        "policy": {
            "repetitions": repetitions,
            "minimum_repetitions": MINIMUM_REPETITIONS,
            "interactive_p95_budget_ms": budget_ms,
            "labelling_cap_rule": "confined.size",
            "eager_definition": (
                "Python-level production adapter with its existing inner device kernels"
            ),
            "jit_definition": "one outer jax.jit over the production adapter",
            "timing_scope": "warm synchronized calls; compilation excluded",
        },
        "static_control_flow": {
            "passed": all(row["passed"] for row in static_rows),
            "data_dependent_while_loop_count": sum(
                row["data_dependent_while_loop_count"] for row in static_rows
            ),
            "cases": static_rows,
        },
        "platforms": platforms,
        "overall": {
            "strict_json": True,
            "cpu_measured": platforms[0]["status"] == "measured",
            "gpu_measured": platforms[1]["status"] == "measured",
            "all_measured_jit_compile_success": all(
                row["jit_compile_success"] for row in measured_rows
            ),
            "all_measured_vmap_compile_success": all(
                row["vmap_compile_success"] for row in measured_rows
            ),
            "all_partition_counts_match": all(
                case["expected_partition_match"]
                for row in measured_rows
                for case in row["cases"]
            ),
            "all_measured_p95_within_interactive_budget": all(
                case["within_interactive_p95_budget"]
                for row in measured_rows
                for case in row["cases"]
            ),
        },
    }
    _strict_json(receipt)
    return receipt


def main() -> None:
    """Parse arguments, run the measurement, and write strict JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=MINIMUM_REPETITIONS)
    parser.add_argument(
        "--interactive-p95-budget-ms",
        type=float,
        default=INTERACTIVE_P95_BUDGET_MS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = run(arguments.repetitions, arguments.interactive_p95_budget_ms)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(receipt["overall"], sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
