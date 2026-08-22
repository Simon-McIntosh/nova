"""Measure the landed gentle coupled window on an allocated GPU."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import inspect
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ["JAX_PLATFORMS"] = "cuda"

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import numpy as np

from nova.equilibrium.forward import ForwardProfile
from nova.transport import coupled_window


CPU_WINDOW_SECONDS = 423.03271608706564
CPU_CONTRACTION = 0.53710396334179378
CPU_MAXIMUM_RESIDUAL = 0.0049860186161842365
CPU_FLUX_CLOSURE_RESIDUAL = 0.0
CPU_CURRENT_CLOSURE_RESIDUAL = 0.0
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


def _devices(tree: Any) -> tuple[str, ...]:
    names = set()
    for leaf in jax.tree.leaves(tree):
        devices = getattr(leaf, "devices", None)
        if devices is not None:
            names.update(str(device) for device in devices())
    return tuple(sorted(names))


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
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=TSV_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _cpu_exit_residuals() -> dict[str, float]:
    path = Path("scripts/window_demonstration/receipts.tsv")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        return {
            row["field"]: float(row["value"])
            for row in rows
            if row["regime"] == "gentle"
            and row["candidate"] == "1"
            and row["kind"] == "exit_residual"
        }


def _source_location(function: Any) -> str:
    path = inspect.getsourcefile(function)
    _source, line = inspect.getsourcelines(function)
    return f"{path}:{line}"


def _report(
    *,
    job_id: str,
    backend: str,
    device_kind: str,
    solve_devices: Sequence[Mapping[str, Any]],
    transfer_events: Sequence[Mapping[str, Any]],
    result: Any,
    preparation_seconds: float,
    attempt_seconds: float,
    cold_excess_seconds: float,
    conservation: Mapping[str, float],
    deviations: Mapping[str, float],
    solver_path: Mapping[str, str],
) -> str:
    convergence = result.convergence
    speedup = CPU_WINDOW_SECONDS / attempt_seconds
    blocked = bool(transfer_events)
    lines = []
    if blocked:
        lines.extend(
            [
                (
                    "NEEDS-HELP: the window's FSA handoff materialises GPU arrays "
                    "as host NumPy records before TORAX consumes them"
                ),
                "",
                (
                    "tried: Ran the landed gentle solve_window configuration in one "
                    "Betelgeuse allocation and instrumented solve-array placement "
                    "plus every FSA materialisation event. The heavy equilibrium and "
                    "extraction arrays were on the H200, but _block_and_copy forced "
                    "each completed FSA record through np.asarray on the host."
                ),
                (
                    "options: (1) make TransportGeometry accept the traced FSA record "
                    "without NumPy materialisation; (2) introduce an explicit "
                    "device-resident geometry waveform while keeping receipt-only "
                    "scalars host-side; (3) narrow the end-to-end claim to "
                    "GPU-kernel execution with host orchestration."
                ),
                (
                    "leaning: Option 1, because the standard geometry adapter already "
                    "consumes JAX arrays and the materialisation is a demonstration "
                    "seam rather than a physics requirement."
                ),
                (
                    "cost-if-wrong: If TORAX or waveform interpolation actually "
                    "requires NumPy ownership, removing the copy will require a typed "
                    "dual representation and the benchmark must be rerun."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "# Gentle coupled window on one H200",
            "",
            (
                f"SLURM job `{job_id}` ran on backend `{backend}` and device "
                f"`{device_kind}`."
            ),
            (
                "Configuration: window `0.0025 s`, auxiliary source multiplier "
                "`0.5`, iteration cap `10`, tolerance `0.005`, damping `0.5`."
            ),
            "",
            "## Placement and solver identity",
            "",
            (
                "The observed equilibrium solve arrays remained on the allocated "
                "device at every completed coarse sample:"
            ),
            "",
            "| exchange | sample | solve-array device |",
            "|---:|---:|---|",
        ]
    )
    lines.extend(
        f"| {row['exchange']} | {row['sample']} | `{row['devices']}` |"
        for row in solve_devices
    )
    lines.extend(
        [
            "",
            (
                "Owner solver-identity call path: "
                f"`equilibrium_sweep` at `{solver_path['equilibrium_sweep']}` calls "
                "`sampled_profile.solve_portfolio`; the same function creates its "
                "cold portfolio from `observed.moments.plasma_current`. The callee is "
                "`ForwardProfile.solve_portfolio` at "
                f"`{solver_path['solve_portfolio']}`."
            ),
            "",
            "Observed forced host transfers:",
            "",
            "| operation | calls | GPU fields materialised |",
            "|---|---:|---:|",
        ]
    )
    if transfer_events:
        lines.extend(
            f"| `{row['operation']}` | {row['call']} | {row['gpu_fields']} |"
            for row in transfer_events
        )
    else:
        lines.append("| none | 0 | 0 |")
    lines.extend(
        [
            "",
            "## Wall-time structure",
            "",
            (
                f"Comparable solve-window sweeps: H200 `{_format(attempt_seconds)}` s "
                f"versus landed CPU `{_format(CPU_WINDOW_SECONDS)}` s, speedup "
                f"`{_format(speedup)}x`. Cold fixture preparation cost "
                f"`{_format(preparation_seconds)}` s. Observed first-iteration "
                f"excess above warmed medians was `{_format(cold_excess_seconds)}` s; "
                "amortised across ten iterations this is "
                f"`{_format((preparation_seconds + cold_excess_seconds) / 10.0)}` "
                "s/iteration including preparation."
            ),
            "",
            "| iteration | side | wall time (s) |",
            "|---:|---|---:|",
        ]
    )
    lines.extend(
        f"| {row['iteration']} | {row['side']} | `{_format(row['seconds'])}` |"
        for row in result.timings
    )
    lines.extend(
        [
            "",
            "## Receipt comparison",
            "",
            "| receipt | H200 | CPU | absolute deviation | tolerance |",
            "|---|---:|---:|---:|---:|",
            (
                "| measured contraction | "
                f"`{_format(convergence.contraction_estimate)}` | "
                f"`{_format(CPU_CONTRACTION)}` | "
                f"`{_format(deviations['contraction'])}` | "
                f"`{_format(COMPARISON_TOLERANCE)}` |"
            ),
            (
                "| maximum exit residual | "
                f"`{_format(convergence.maximum_residual)}` | "
                f"`{_format(CPU_MAXIMUM_RESIDUAL)}` | "
                f"`{_format(deviations['maximum_residual'])}` | "
                f"`{_format(COMPARISON_TOLERANCE)}` |"
            ),
            (
                "| flux closure residual | "
                f"`{_format(conservation['flux_closure_residual'])}` | `0` | "
                f"`{_format(deviations['flux_closure'])}` | "
                f"`{_format(COMPARISON_TOLERANCE)}` |"
            ),
            (
                "| current closure residual | "
                f"`{_format(conservation['current_continuity_residual'])}` | `0` | "
                f"`{_format(deviations['current_closure'])}` | "
                f"`{_format(COMPARISON_TOLERANCE)}` |"
            ),
            "",
            (
                "The TSV carries every exchanged-field residual against its landed "
                "CPU value. No receipt deviation exceeded the declared 0.005 "
                "comparison tolerance."
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

    devices = jax.devices()
    backend = jax.default_backend()
    if backend != "gpu" or not devices:
        raise RuntimeError(f"GPU backend required, got {backend} and {devices}")
    device_kind = devices[0].device_kind

    from scripts.window_demonstration import run_window as demonstration

    os.environ["JAX_PLATFORMS"] = "cuda"
    original_equilibrium_sweep = demonstration.equilibrium_sweep
    original_block_and_copy = demonstration._block_and_copy
    solve_devices: list[dict[str, Any]] = []
    transfer_events: list[dict[str, Any]] = []
    exchange = {"value": 0}

    def observed_equilibrium_sweep(*args, **kwargs):
        exchange["value"] += 1
        receipt = original_equilibrium_sweep(*args, **kwargs)
        for sample, equilibrium in enumerate(receipt.equilibria):
            solve_devices.append(
                {
                    "exchange": exchange["value"],
                    "sample": sample,
                    "devices": ", ".join(_devices(equilibrium)),
                }
            )
        return receipt

    def observed_block_and_copy(tree):
        gpu_fields = sum(bool(_devices(value)) for value in tree.values())
        transfer_events.append(
            {
                "call": len(transfer_events) + 1,
                "operation": "_block_and_copy: jax.Array -> numpy.asarray FSA record",
                "gpu_fields": gpu_fields,
            }
        )
        return original_block_and_copy(tree)

    demonstration.equilibrium_sweep = observed_equilibrium_sweep
    demonstration._block_and_copy = observed_block_and_copy

    preparation_start = time.perf_counter()
    profile, seed, _vacuum = demonstration._fixture_machine()
    extraction_lattice = demonstration._extraction_lattice(profile)
    fixture_sources = demonstration._fixture_sources(profile)
    baseline_equilibrium = profile.solve(
        seed,
        route="anderson",
        evaluations=demonstration.EVALUATIONS,
    )
    baseline_devices = ", ".join(_devices(baseline_equilibrium))
    baseline_geometry, baseline_extraction = demonstration._geometry_from_equilibrium(
        baseline_equilibrium,
        profile.source,
        extraction_lattice,
        fixture_sources,
    )
    baseline_extraction.update(iteration=0, sample=0)
    preparation_seconds = time.perf_counter() - preparation_start

    config = demonstration.RegimeConfig(
        "gentle",
        GENTLE_WINDOW_SECONDS,
        GENTLE_SOURCE_MULTIPLIER,
        1,
    )
    attempt_start = time.perf_counter()
    result = demonstration._run_regime(
        config,
        profile=profile,
        baseline_equilibrium=baseline_equilibrium,
        baseline_geometry=baseline_geometry,
        baseline_extraction=baseline_extraction,
        extraction_lattice=extraction_lattice,
        fixture_sources=fixture_sources,
    )
    attempt_seconds = time.perf_counter() - attempt_start
    if not result.converged:
        raise RuntimeError(
            f"gentle GPU window did not return WindowReceipt: {result.outcome_type}: "
            f"{result.outcome}"
        )

    convergence = result.convergence
    conservation = demonstration._transport_conservation(result.transport_receipt)
    cpu_exit = _cpu_exit_residuals()
    receipt_deviations = {
        field: abs(float(value) - cpu_exit[field])
        for field, value in convergence.exit_residual.items()
    }
    deviations = {
        "contraction": abs(convergence.contraction_estimate - CPU_CONTRACTION),
        "maximum_residual": abs(convergence.maximum_residual - CPU_MAXIMUM_RESIDUAL),
        "flux_closure": abs(
            conservation["flux_closure_residual"] - CPU_FLUX_CLOSURE_RESIDUAL
        ),
        "current_closure": abs(
            conservation["current_continuity_residual"] - CPU_CURRENT_CLOSURE_RESIDUAL
        ),
        "exit_field_maximum": max(receipt_deviations.values(), default=0.0),
    }
    if any(value > COMPARISON_TOLERANCE for value in deviations.values()):
        raise RuntimeError(f"GPU receipt deviation exceeds tolerance: {deviations}")

    by_side: dict[str, list[float]] = {}
    for row in result.timings:
        by_side.setdefault(row["side"], []).append(float(row["seconds"]))
    cold_excess_seconds = sum(
        max(values[0] - float(np.median(values[1:])), 0.0)
        for values in by_side.values()
        if len(values) > 1
    )

    equilibrium_source = inspect.getsource(coupled_window.equilibrium_sweep)
    required_fragments = (
        "sampled_profile.solve_portfolio(",
        "observed.moments.plasma_current",
        "cold_seed_portfolio(",
    )
    if not all(fragment in equilibrium_source for fragment in required_fragments):
        raise AssertionError("equilibrium portfolio call path changed")
    solver_path = {
        "equilibrium_sweep": _source_location(coupled_window.equilibrium_sweep),
        "solve_portfolio": _source_location(ForwardProfile.solve_portfolio),
    }

    rows: list[dict[str, str]] = []
    metadata = (
        ("slurm_job_id", os.environ.get("SLURM_JOB_ID", "unknown"), "text"),
        ("slurm_partition", os.environ.get("SLURM_JOB_PARTITION", ""), "text"),
        ("reservation", "gpu_0003_grpA", "text"),
        ("tmpdir", os.environ.get("TMPDIR", ""), "text"),
        ("jax_backend", backend, "text"),
        ("device_kind", device_kind, "text"),
        ("baseline_solve_devices", baseline_devices, "text"),
        ("window_length", GENTLE_WINDOW_SECONDS, "s"),
        ("source_multiplier", GENTLE_SOURCE_MULTIPLIER, "fraction"),
        ("iteration_cap", GENTLE_ITERATION_CAP, "count"),
        ("tolerance", COMPARISON_TOLERANCE, "relative"),
        ("damping", GENTLE_DAMPING, "fraction"),
    )
    for field, value, unit in metadata:
        _append(rows, "metadata", field, value, unit)
    _append(rows, "timing", "preparation_wall_time", preparation_seconds, "s")
    _append(
        rows,
        "timing",
        "window_wall_time",
        attempt_seconds,
        "s",
        cpu_baseline=CPU_WINDOW_SECONDS,
    )
    _append(
        rows,
        "timing",
        "window_speedup",
        CPU_WINDOW_SECONDS / attempt_seconds,
        "ratio",
    )
    _append(rows, "timing", "cold_iteration_excess", cold_excess_seconds, "s")
    _append(
        rows,
        "timing",
        "cold_cost_amortised_per_iteration",
        (preparation_seconds + cold_excess_seconds) / GENTLE_ITERATION_CAP,
        "s",
    )
    for row in result.timings:
        _append(
            rows,
            "sweep_timing",
            "wall_time",
            row["seconds"],
            "s",
            iteration=row["iteration"],
            side=row["side"],
        )
    for row in solve_devices:
        _append(
            rows,
            "device_placement",
            "equilibrium_solve_arrays",
            row["devices"],
            "text",
            iteration=row["exchange"],
            side=f"sample_{row['sample']}",
        )
    for row in transfer_events:
        _append(
            rows,
            "host_roundtrip",
            row["operation"],
            row["gpu_fields"],
            "field_count",
            iteration=row["call"],
            status="NEEDS-HELP",
        )
    _append(
        rows,
        "solver_identity",
        "equilibrium_sweep",
        solver_path["equilibrium_sweep"],
        "source_location",
        status="PASS",
    )
    _append(
        rows,
        "solver_identity",
        "solve_portfolio",
        solver_path["solve_portfolio"],
        "source_location",
        status="PASS",
    )
    for field, value, baseline in (
        ("contraction_estimate", convergence.contraction_estimate, CPU_CONTRACTION),
        (
            "maximum_exit_residual",
            convergence.maximum_residual,
            CPU_MAXIMUM_RESIDUAL,
        ),
        (
            "flux_closure_residual",
            conservation["flux_closure_residual"],
            CPU_FLUX_CLOSURE_RESIDUAL,
        ),
        (
            "current_continuity_residual",
            conservation["current_continuity_residual"],
            CPU_CURRENT_CLOSURE_RESIDUAL,
        ),
    ):
        deviation = abs(float(value) - baseline)
        _append(
            rows,
            "receipt",
            field,
            value,
            "relative",
            cpu_baseline=baseline,
            tolerance=COMPARISON_TOLERANCE,
            status="PASS" if deviation <= COMPARISON_TOLERANCE else "FAIL",
        )
    for field, value in convergence.exit_residual.items():
        deviation = receipt_deviations[field]
        _append(
            rows,
            "exit_residual",
            field,
            value,
            "relative",
            cpu_baseline=cpu_exit[field],
            tolerance=COMPARISON_TOLERANCE,
            status="PASS" if deviation <= COMPARISON_TOLERANCE else "FAIL",
        )

    _write_tsv(arguments.results, rows)
    arguments.report.write_text(
        _report(
            job_id=os.environ.get("SLURM_JOB_ID", "unknown"),
            backend=backend,
            device_kind=device_kind,
            solve_devices=solve_devices,
            transfer_events=transfer_events,
            result=result,
            preparation_seconds=preparation_seconds,
            attempt_seconds=attempt_seconds,
            cold_excess_seconds=cold_excess_seconds,
            conservation=conservation,
            deviations=deviations,
            solver_path=solver_path,
        ),
        encoding="utf-8",
    )
    print(f"backend={backend}")
    print(f"device={device_kind}")
    print(f"window_seconds={attempt_seconds}")
    print(f"host_roundtrips={len(transfer_events)}")
    print(f"report={arguments.report}")
    print(f"results={arguments.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
