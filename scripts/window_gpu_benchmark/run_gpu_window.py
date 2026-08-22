"""Diagnose the first CPU--GPU divergence in the gentle coupled window."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import dataclasses
import inspect
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ["JAX_PLATFORMS"] = "cuda,cpu"

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import numpy as np

from nova.equilibrium.forward import ForwardProfile
from nova.transport import coupled_window, torax_geometry


CPU_WINDOW_SECONDS = 423.03271608706564
CPU_CONTRACTION = 0.53710396334179378
CPU_MAXIMUM_RESIDUAL = 0.0049860186161842365
COMPARISON_TOLERANCE = 5.0e-3
GENTLE_WINDOW_SECONDS = 2.5e-3
GENTLE_SOURCE_MULTIPLIER = 0.5
GENTLE_ITERATION_CAP = 10
GENTLE_DAMPING = 0.5
TSV_FIELDS = (
    "category",
    "iteration",
    "side",
    "field",
    "value",
    "unit",
    "cpu_baseline",
    "absolute_deviation",
    "tolerance",
    "status",
)


def _format(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool | np.bool_):
        return str(bool(value)).lower()
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, str):
        return value
    return f"{float(np.asarray(value)):.17g}"


def _append(
    rows: list[dict[str, str]],
    category: str,
    field: str,
    value: Any,
    unit: str,
    *,
    iteration: int | str = "",
    side: str = "",
    cpu_baseline: Any = "",
    tolerance: Any = "",
    status: str = "",
) -> None:
    deviation = ""
    if cpu_baseline != "":
        deviation = abs(float(value) - float(cpu_baseline))
    rows.append(
        {
            "category": category,
            "iteration": str(iteration),
            "side": side,
            "field": field,
            "value": _format(value),
            "unit": unit,
            "cpu_baseline": _format(cpu_baseline) if cpu_baseline != "" else "",
            "absolute_deviation": _format(deviation) if deviation != "" else "",
            "tolerance": _format(tolerance) if tolerance != "" else "",
            "status": status,
        }
    )


def _write_tsv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=TSV_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            {**row, "status": row["status"] or "RECORDED"} for row in rows
        )


def _cpu_trace() -> dict[tuple[int, str], float]:
    path = Path("scripts/window_demonstration/receipts.tsv")
    with path.open(encoding="utf-8", newline="") as stream:
        return {
            (int(row["iteration"]), row["field"]): float(row["value"])
            for row in csv.DictReader(stream, delimiter="\t")
            if row["regime"] == "gentle"
            and row["candidate"] == "1"
            and row["kind"] == "residual_trace"
        }


def _array_records(tree: Any, prefix: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    visited: set[int] = set()

    def visit(value: Any, path: str) -> None:
        if isinstance(value, jax.Array):
            devices = sorted(str(device) for device in value.devices())
            backends = sorted(device.platform for device in value.devices())
            records.append(
                {
                    "path": path,
                    "dtype": str(value.dtype),
                    "backend": ",".join(backends),
                    "device": ",".join(devices),
                    "shape": str(tuple(value.shape)),
                }
            )
            return
        if isinstance(value, np.ndarray):
            records.append(
                {
                    "path": path,
                    "dtype": str(value.dtype),
                    "backend": "host",
                    "device": "host",
                    "shape": str(value.shape),
                }
            )
            return
        if isinstance(value, str | bytes | int | float | bool | type(None)):
            return
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name), f"{path}.{field.name}")
        elif isinstance(value, Mapping):
            for name, item in value.items():
                visit(item, f"{path}.{name}")
        elif isinstance(value, tuple | list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(tree, prefix)
    return records


def _require_cuda(records: Sequence[Mapping[str, str]], label: str) -> None:
    device_records = [row for row in records if row["backend"] != "host"]
    if not device_records:
        raise RuntimeError(f"{label} exposed no JAX arrays")
    refused = [row for row in device_records if row["backend"] != "gpu"]
    if refused:
        raise RuntimeError(f"{label} contains non-CUDA solve arrays: {refused[:3]}")


def _source_location(function: Any) -> str:
    path = inspect.getsourcefile(function)
    _source, line = inspect.getsourcelines(function)
    return f"{path}:{line}"


def _array_rows(
    rows: list[dict[str, str]], category: str, records: Sequence[Mapping[str, Any]]
) -> None:
    for record in records:
        _append(
            rows,
            category,
            record["path"],
            record["dtype"],
            record["shape"],
            iteration=record.get("iteration", ""),
            side=record.get("side", record["backend"]),
            status=f"{record['backend']}:{record['device']}",
        )


def _branch_rows(
    rows: list[dict[str, str]], branch_records: Sequence[Mapping[str, Any]]
) -> None:
    for branch in branch_records:
        for field in (
            "sample_time",
            "limited_core_cells",
            "diverted_core_cells",
            "selected_class",
            "previous_class",
            "switched",
            "reason",
            "limited_available",
            "diverted_available",
            "limited_residual",
            "diverted_residual",
        ):
            _append(
                rows,
                "branch_receipt",
                field,
                branch[field],
                "receipt",
                iteration=branch["exchange"],
                side=f"sample_{branch['sample']}",
            )


def _diagnostic_rows(
    *,
    outcome_type: str,
    outcome: str,
    convergence: Any,
    cpu_residuals: Mapping[tuple[int, str], float],
    precision: Mapping[str, Any],
    branch_records: Sequence[Mapping[str, Any]],
    guard_events: Sequence[Mapping[str, Any]],
    side_events: Sequence[Mapping[str, Any]],
    solve_arrays: Sequence[Mapping[str, Any]],
    exchange_arrays: Sequence[Mapping[str, Any]],
    probe_arrays: Sequence[Mapping[str, Any]],
    transfer_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for field, value, unit in (
        ("outcome_type", outcome_type, "text"),
        ("outcome", outcome, "text"),
        ("window_length", GENTLE_WINDOW_SECONDS, "s"),
        ("source_multiplier", GENTLE_SOURCE_MULTIPLIER, "fraction"),
        ("iteration_cap", GENTLE_ITERATION_CAP, "count"),
        ("tolerance", COMPARISON_TOLERANCE, "relative"),
        ("damping", GENTLE_DAMPING, "fraction"),
        ("cpu_window_baseline", CPU_WINDOW_SECONDS, "s"),
        ("cpu_baseline_lineage", "pre-band", "text"),
        ("boundary_band_commit", "32942ac3", "git_commit"),
    ):
        _append(rows, "metadata", field, value, unit)
    for side in ("cpu", "gpu"):
        _append(
            rows,
            "precision",
            "inner_fixed_point_residual",
            precision[f"{side}_residual"],
            "relative",
            side=side,
            status=precision[f"{side}_placement"],
        )
        _append(
            rows,
            "precision",
            "inner_fixed_point_residual_dtype",
            precision[f"{side}_dtype"],
            "dtype",
            side=side,
            status=precision[f"{side}_placement"],
        )
    _append(rows, "precision", "verdict", precision["verdict"], "text")
    _append(
        rows,
        "convergence",
        "iterations_used",
        convergence.iterations_used,
        "count",
    )
    _append(
        rows,
        "convergence",
        "contraction_estimate",
        convergence.contraction_estimate,
        "ratio",
        cpu_baseline=CPU_CONTRACTION,
    )
    _append(
        rows,
        "convergence",
        "maximum_exit_residual",
        convergence.maximum_residual,
        "relative",
        cpu_baseline=CPU_MAXIMUM_RESIDUAL,
        tolerance=COMPARISON_TOLERANCE,
    )
    _append(
        rows,
        "convergence",
        "damping_applied",
        convergence.damping_applied,
        "fraction",
    )
    for iteration, residuals in enumerate(convergence.residual_trace, start=1):
        for field, value in residuals.items():
            baseline = cpu_residuals.get((iteration, field), "")
            _append(
                rows,
                "residual_trace",
                field,
                value,
                "relative",
                iteration=iteration,
                cpu_baseline=baseline,
                status=(
                    "EXACT_MATCH" if baseline != "" and value == baseline else "DIFF"
                ),
            )
    for field, value in convergence.exit_residual.items():
        baseline = cpu_residuals.get((convergence.iterations_used, field), "")
        _append(
            rows,
            "exit_residual",
            field,
            value,
            "relative",
            cpu_baseline=baseline,
        )
    _branch_rows(rows, branch_records)
    for event in guard_events:
        _append(
            rows,
            "guard_callback",
            "uniform_grid_roundtrip_wall_time",
            event["seconds"],
            "s",
            iteration=event["iteration"],
        )
    for event in side_events:
        _append(
            rows,
            "sweep_timing",
            "wall_time",
            event["seconds"],
            "s",
            iteration=event["iteration"],
            side=event["side"],
        )
    for event in transfer_events:
        _append(
            rows,
            "host_roundtrip",
            event["operation"],
            event["gpu_fields"],
            "field_count",
            iteration=event["call"],
        )
    _array_rows(rows, "solve_array_dtype", solve_arrays)
    _array_rows(rows, "exchange_array_dtype", exchange_arrays)
    _array_rows(rows, "precision_probe_array", probe_arrays)
    return rows


def _first_divergence(
    convergence: Any, cpu_residuals: Mapping[tuple[int, str], float]
) -> dict[str, Any] | None:
    for iteration, residuals in enumerate(convergence.residual_trace, start=1):
        for field, gpu_value in residuals.items():
            cpu_value = cpu_residuals.get((iteration, field))
            if cpu_value is not None and float(gpu_value) != cpu_value:
                return {
                    "iteration": iteration,
                    "field": field,
                    "cpu": cpu_value,
                    "gpu": float(gpu_value),
                    "absolute": abs(float(gpu_value) - cpu_value),
                }
    return None


def _render_report(
    *,
    job_id: str,
    backend: str,
    device_kind: str,
    result: Any,
    precision: Mapping[str, Any],
    first: Mapping[str, Any] | None,
    cpu_residuals: Mapping[tuple[int, str], float],
    guard_events: Sequence[Mapping[str, Any]],
    side_events: Sequence[Mapping[str, Any]],
    solve_arrays: Sequence[Mapping[str, Any]],
    exchange_arrays: Sequence[Mapping[str, Any]],
    probe_arrays: Sequence[Mapping[str, Any]],
    preparation_seconds: float,
    window_seconds: float,
    conservation: Mapping[str, Any] | None,
) -> str:
    convergence = result.convergence
    lines = [
        "# CPU--H200 coupled-window divergence diagnosis",
        "",
        (
            f"SLURM job `{job_id}` ran on JAX backend `{backend}` and device "
            f"`{device_kind}`. Configuration: window `{GENTLE_WINDOW_SECONDS}` s, "
            f"auxiliary source multiplier `{GENTLE_SOURCE_MULTIPLIER}`, cap "
            f"`{GENTLE_ITERATION_CAP}`, tolerance `{COMPARISON_TOLERANCE}`, "
            f"damping `{GENTLE_DAMPING}`."
        ),
        "",
        "## Precision discriminator",
        "",
        "| backend | inner fixed-point residual | dtype | placement |",
        "|---|---:|---|---|",
        (
            f"| CPU | `{_format(precision['cpu_residual'])}` | "
            f"`{precision['cpu_dtype']}` | `{precision['cpu_placement']}` |"
        ),
        (
            f"| H200 | `{_format(precision['gpu_residual'])}` | "
            f"`{precision['gpu_dtype']}` | `{precision['gpu_placement']}` |"
        ),
        "",
        f"Verdict: **{precision['verdict']}**",
        "",
        (
            f"The CUDA sweep probe recorded `{len(probe_arrays)}` input/output "
            "arrays for the first equilibrium and TORAX sweeps. The full TSV names "
            "each array's path, dtype, backend, device and shape."
        ),
        "",
        "## First trajectory divergence",
        "",
    ]
    if first is None:
        lines.append("No residual-trace quantity differed from the landed CPU trace.")
    else:
        lines.extend(
            [
                (
                    f"The first non-identical quantity is iteration "
                    f"`{first['iteration']}`, `{first['field']}`: CPU "
                    f"`{_format(first['cpu'])}`, H200 `{_format(first['gpu'])}`, "
                    f"absolute difference `{_format(first['absolute'])}`."
                ),
                "",
                (
                    "| iteration | CPU maximum residual | H200 maximum residual | "
                    "difference |"
                ),
                "|---:|---:|---:|---:|",
            ]
        )
        for iteration, residuals in enumerate(convergence.residual_trace, start=1):
            gpu_max = max(residuals.values())
            cpu_max = max(
                value
                for (sample, _field), value in cpu_residuals.items()
                if sample == iteration
            )
            lines.append(
                f"| {iteration} | `{_format(cpu_max)}` | `{_format(gpu_max)}` | "
                f"`{_format(abs(gpu_max - cpu_max))}` |"
            )
    lines.extend(
        [
            "",
            "## Exhausted or converged receipt",
            "",
            f"Typed outcome: `{result.outcome_type}` — `{result.outcome}`.",
            (
                f"Iterations `{convergence.iterations_used}`; measured contraction "
                f"`{_format(convergence.contraction_estimate)}`; exit residual "
                f"`{_format(convergence.maximum_residual)}`; damping "
                f"`{_format(convergence.damping_applied)}`. The tolerance remains "
                f"`{COMPARISON_TOLERANCE}`."
            ),
            "",
            (
                "Failure-path serialization retained "
                f"`{len(convergence.residual_trace)}` "
                f"residual rows, `{len(result.branches)}` branch receipts, "
                f"`{len(guard_events)}` guard timings, `{len(side_events)}` side "
                f"timings, `{len(solve_arrays)}` solve-array dtype records and "
                f"`{len(exchange_arrays)}` exchange-array dtype records before the "
                "typed exhaustion crossed the caller boundary."
            ),
            "",
            "## Wall-time structure",
            "",
            (
                f"Fixture and precision-probe preparation took "
                f"`{_format(preparation_seconds)}` s. The window took "
                f"`{_format(window_seconds)}` s; the landed CPU window figure is "
                f"`{_format(CPU_WINDOW_SECONDS)}` s. That CPU figure is pre-band: "
                "boundary-band sparsification landed later at `32942ac3`, whose "
                "warm CPU assembly figure is 24.6 s; the CPU window was not rerun."
            ),
            "",
            "| iteration | side | wall time (s) |",
            "|---:|---|---:|",
        ]
    )
    lines.extend(
        f"| {event['iteration']} | {event['side']} | `{_format(event['seconds'])}` |"
        for event in side_events
    )
    lines.extend(["", "## Guard callback round trips", ""])
    if guard_events:
        total = sum(float(event["seconds"]) for event in guard_events)
        lines.append(
            f"`{len(guard_events)}` calls cost `{_format(total)}` s total and "
            f"`{_format(np.median([event['seconds'] for event in guard_events]))}` "
            "s median per adapter construction."
        )
    else:
        lines.append("No guard callback completed.")
    if conservation is not None:
        lines.extend(
            [
                "",
                "## Latest transport ledgers",
                "",
                (
                    "Flux closure absolute/relative: "
                    f"`{_format(conservation['flux_closure_error'])}` / "
                    f"`{_format(conservation['flux_closure_residual'])}`."
                ),
                (
                    "Current continuity absolute/relative: "
                    f"`{_format(conservation['current_continuity_error'])}` / "
                    f"`{_format(conservation['current_continuity_residual'])}`."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Solver identity",
            "",
            (
                "`equilibrium_sweep` at "
                f"`{_source_location(coupled_window.equilibrium_sweep)}` "
                "routes through the plasma-current-bearing cold portfolio and "
                f"`ForwardProfile.solve_portfolio` at "
                f"`{_source_location(ForwardProfile.solve_portfolio)}`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    arguments = parser.parse_args()

    gpu_devices = jax.devices("gpu")
    cpu_devices = jax.devices("cpu")
    if jax.default_backend() != "gpu" or not gpu_devices or not cpu_devices:
        raise RuntimeError("one CUDA device and the CPU callback platform are required")
    gpu_device = gpu_devices[0]
    cpu_device = cpu_devices[0]

    from scripts.window_demonstration import run_window as demonstration

    original_equilibrium_sweep = demonstration.equilibrium_sweep
    original_transport_sweep = demonstration.transport_sweep
    original_solve_window = demonstration.solve_window
    original_block_and_copy = demonstration._block_and_copy
    original_grid_callback = torax_geometry._validated_grid_callback
    guard_events: list[dict[str, Any]] = []
    side_events: list[dict[str, Any]] = []
    solve_arrays: list[dict[str, Any]] = []
    exchange_arrays: list[dict[str, Any]] = []
    probe_arrays: list[dict[str, Any]] = []
    transfer_events: list[dict[str, Any]] = []
    branch_records: list[dict[str, Any]] = []
    equilibrium_iteration = 0
    transport_iteration = 0
    callback_iteration = 0
    cpu_residuals = _cpu_trace()
    precision: dict[str, Any] = {}

    def observed_equilibrium_sweep(*args, **kwargs):
        nonlocal equilibrium_iteration
        equilibrium_iteration += 1
        if equilibrium_iteration == 1:
            probe_arrays.extend(_array_records((args, kwargs), "equilibrium.input"))
        started = time.perf_counter()
        receipt = original_equilibrium_sweep(*args, **kwargs)
        jax.block_until_ready(receipt.equilibria)
        side_events.append(
            {
                "iteration": equilibrium_iteration,
                "side": "equilibrium_sweep",
                "seconds": time.perf_counter() - started,
            }
        )
        for sample, equilibrium in enumerate(receipt.equilibria):
            records = _array_records(
                equilibrium, f"equilibrium[{equilibrium_iteration},{sample}]"
            )
            _require_cuda(records, "equilibrium sweep output")
            solve_arrays.extend(
                {**row, "iteration": equilibrium_iteration, "side": f"sample_{sample}"}
                for row in records
            )
        branch_records.extend(
            demonstration._branch_measurement(branch, equilibrium_iteration)
            for branch in receipt.branch_receipts
        )
        if equilibrium_iteration == 1:
            probe_arrays.extend(_array_records(receipt, "equilibrium.output"))
        return receipt

    def observed_transport_sweep(*args, **kwargs):
        nonlocal transport_iteration, callback_iteration
        transport_iteration += 1
        callback_iteration = transport_iteration
        if transport_iteration == 1:
            probe_arrays.extend(_array_records((args, kwargs), "transport.input"))
        started = time.perf_counter()
        receipt = original_transport_sweep(*args, **kwargs)
        jax.block_until_ready(receipt.receipts)
        side_events.append(
            {
                "iteration": transport_iteration,
                "side": "transport",
                "seconds": time.perf_counter() - started,
            }
        )
        records = _array_records(receipt, f"transport[{transport_iteration}]")
        _require_cuda(records, "transport sweep output")
        exchange_arrays.extend(
            {**row, "iteration": transport_iteration, "side": "transport"}
            for row in records
        )
        if transport_iteration == 1:
            probe_arrays.extend(_array_records(receipt, "transport.output"))
        return receipt

    def observed_grid_callback(rho_face):
        started = time.perf_counter()
        validated = original_grid_callback(rho_face)
        validated.block_until_ready()
        guard_events.append(
            {
                "iteration": callback_iteration,
                "seconds": time.perf_counter() - started,
            }
        )
        return validated

    def observed_block_and_copy(tree):
        transfer_events.append(
            {
                "call": len(transfer_events) + 1,
                "operation": "FSA device record materialisation",
                "gpu_fields": sum(
                    isinstance(value, jax.Array)
                    and all(device.platform == "gpu" for device in value.devices())
                    for value in tree.values()
                ),
            }
        )
        return original_block_and_copy(tree)

    def observed_solve_window(*args, **kwargs):
        def serialize_failure(error):
            exchange_records = [
                {
                    **row,
                    "iteration": error.convergence.iterations_used,
                    "side": "exhausted_waveforms",
                }
                for row in _array_records(
                    (error.geometry_waveform, error.source_waveform),
                    "failure.exchange",
                )
            ]
            exchange_arrays.extend(exchange_records)
            rows = _diagnostic_rows(
                outcome_type=type(error).__name__,
                outcome=str(error),
                convergence=error.convergence,
                cpu_residuals=cpu_residuals,
                precision=precision,
                branch_records=branch_records,
                guard_events=guard_events,
                side_events=side_events,
                solve_arrays=solve_arrays,
                exchange_arrays=exchange_arrays,
                probe_arrays=probe_arrays,
                transfer_events=transfer_events,
            )
            _write_tsv(arguments.results, rows)
            print(f"failure_snapshot={arguments.results}", flush=True)

        kwargs["failure_serializer"] = serialize_failure
        return original_solve_window(*args, **kwargs)

    demonstration.equilibrium_sweep = observed_equilibrium_sweep
    demonstration.transport_sweep = observed_transport_sweep
    demonstration.solve_window = observed_solve_window
    demonstration._block_and_copy = observed_block_and_copy
    torax_geometry._validated_grid_callback = observed_grid_callback

    preparation_started = time.perf_counter()
    with jax.default_device(cpu_device):
        cpu_profile, cpu_seed, _cpu_vacuum = demonstration._fixture_machine()
        cpu_equilibrium = cpu_profile.solve(
            cpu_seed, route="anderson", evaluations=demonstration.EVALUATIONS
        )
        jax.block_until_ready(cpu_equilibrium)
    cpu_records = _array_records(cpu_equilibrium, "precision.cpu")
    cpu_residual = cpu_equilibrium.fixed_point.residual
    precision.update(
        cpu_residual=float(np.asarray(cpu_residual)),
        cpu_dtype=str(cpu_residual.dtype),
        cpu_placement=",".join(
            sorted({row["device"] for row in cpu_records if row["backend"] != "host"})
        ),
    )
    del cpu_profile, cpu_seed, cpu_equilibrium

    with jax.default_device(gpu_device):
        profile, seed, _vacuum = demonstration._fixture_machine()
        baseline_equilibrium = profile.solve(
            seed, route="anderson", evaluations=demonstration.EVALUATIONS
        )
        jax.block_until_ready(baseline_equilibrium)
    gpu_records = _array_records(baseline_equilibrium, "precision.gpu")
    _require_cuda(gpu_records, "precision GPU equilibrium")
    gpu_residual = baseline_equilibrium.fixed_point.residual
    precision.update(
        gpu_residual=float(np.asarray(gpu_residual)),
        gpu_dtype=str(gpu_residual.dtype),
        gpu_placement=",".join(
            sorted({row["device"] for row in gpu_records if row["backend"] != "host"})
        ),
    )
    gpu_float_dtypes = {
        row["dtype"] for row in gpu_records if row["dtype"].startswith("float")
    }
    if "float32" in gpu_float_dtypes:
        precision["verdict"] = "FLOAT32 PATH PRESENT"
    elif gpu_float_dtypes == {"float64"}:
        precision["verdict"] = "FLOAT64 CONFIRMED; PRECISION ACQUITTED"
    else:
        precision["verdict"] = f"MIXED OR UNKNOWN DTYPES: {sorted(gpu_float_dtypes)}"

    extraction_lattice = demonstration._extraction_lattice(profile)
    fixture_sources = demonstration._fixture_sources(profile)
    baseline_geometry, baseline_extraction = demonstration._geometry_from_equilibrium(
        baseline_equilibrium, profile.source, extraction_lattice, fixture_sources
    )
    baseline_extraction.update(iteration=0, sample=0)
    preparation_seconds = time.perf_counter() - preparation_started

    config = demonstration.RegimeConfig(
        "gentle", GENTLE_WINDOW_SECONDS, GENTLE_SOURCE_MULTIPLIER, 1
    )
    window_started = time.perf_counter()
    result = demonstration._run_regime(
        config,
        profile=profile,
        baseline_equilibrium=baseline_equilibrium,
        baseline_geometry=baseline_geometry,
        baseline_extraction=baseline_extraction,
        extraction_lattice=extraction_lattice,
        fixture_sources=fixture_sources,
    )
    window_seconds = time.perf_counter() - window_started
    if result.convergence is None:
        raise RuntimeError(f"window returned no convergence trace: {result.outcome}")

    rows = _diagnostic_rows(
        outcome_type=result.outcome_type,
        outcome=result.outcome,
        convergence=result.convergence,
        cpu_residuals=cpu_residuals,
        precision=precision,
        branch_records=branch_records,
        guard_events=guard_events,
        side_events=side_events,
        solve_arrays=solve_arrays,
        exchange_arrays=exchange_arrays,
        probe_arrays=probe_arrays,
        transfer_events=transfer_events,
    )
    first = _first_divergence(result.convergence, cpu_residuals)
    if first is not None:
        _append(
            rows,
            "finding",
            first["field"],
            first["gpu"],
            "relative",
            iteration=first["iteration"],
            side="first_cpu_gpu_divergence",
            cpu_baseline=first["cpu"],
            status="FIRST_NON_IDENTICAL_QUANTITY",
        )
    conservation = (
        demonstration._transport_conservation(result.transport_receipt)
        if result.transport_receipt is not None
        else None
    )
    if conservation is not None:
        for field, value in conservation.items():
            _append(rows, "conservation", field, value, "receipt")
    for field, value, unit in (
        ("preparation_wall_time", preparation_seconds, "s"),
        ("window_wall_time", window_seconds, "s"),
    ):
        _append(rows, "timing", field, value, unit)
    _write_tsv(arguments.results, rows)
    arguments.report.write_text(
        _render_report(
            job_id=os.environ.get("SLURM_JOB_ID", "unknown"),
            backend=jax.default_backend(),
            device_kind=gpu_device.device_kind,
            result=result,
            precision=precision,
            first=first,
            cpu_residuals=cpu_residuals,
            guard_events=guard_events,
            side_events=side_events,
            solve_arrays=solve_arrays,
            exchange_arrays=exchange_arrays,
            probe_arrays=probe_arrays,
            preparation_seconds=preparation_seconds,
            window_seconds=window_seconds,
            conservation=conservation,
        ),
        encoding="utf-8",
    )
    print(f"precision_verdict={precision['verdict']}")
    print(f"cpu_inner_residual={precision['cpu_residual']:.17g}")
    print(f"gpu_inner_residual={precision['gpu_residual']:.17g}")
    print(f"outcome={result.outcome_type}")
    print(f"window_seconds={window_seconds:.17g}")
    if first is not None:
        print(
            "first_divergence="
            f"iteration:{first['iteration']},field:{first['field']},"
            f"cpu:{first['cpu']:.17g},gpu:{first['gpu']:.17g}"
        )
    print(f"report={arguments.report}")
    print(f"results={arguments.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
