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

os.environ["JAX_PLATFORMS"] = "cuda,cpu"

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import numpy as np

from nova.equilibrium.forward import ForwardProfile
from nova.transport import coupled_window
from nova.transport import torax_geometry


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


def _require_cuda_arrays(tree: Any, label: str) -> dict[str, Any]:
    arrays = [leaf for leaf in jax.tree.leaves(tree) if isinstance(leaf, jax.Array)]
    if not arrays:
        raise RuntimeError(f"{label} exposed no JAX arrays for device verification")
    devices = sorted({str(device) for array in arrays for device in array.devices()})
    non_cuda = sorted(
        {
            str(device)
            for array in arrays
            for device in array.devices()
            if device.platform != "gpu"
        }
    )
    if non_cuda:
        raise RuntimeError(f"{label} contains non-CUDA arrays on {non_cuda}")
    jax.block_until_ready(tree)
    return {"array_count": len(arrays), "devices": ", ".join(devices)}


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
    state_devices: Sequence[Mapping[str, Any]],
    transfer_events: Sequence[Mapping[str, Any]],
    guard_events: Sequence[Mapping[str, Any]],
    result: Any,
    preparation_seconds: float,
    attempt_seconds: float,
    cold_excess_seconds: float,
    cold_excess_by_side: Mapping[str, float],
    conservation: Mapping[str, float],
    deviations: Mapping[str, float],
    solver_path: Mapping[str, str],
) -> str:
    convergence = result.convergence
    speedup = CPU_WINDOW_SECONDS / attempt_seconds
    guard_total = sum(float(row["seconds"]) for row in guard_events)
    guard_median = float(np.median([row["seconds"] for row in guard_events]))
    lines = [
        "# Gentle coupled window on one H200",
        "",
        (
            f"SLURM job `{job_id}` ran the complete window on JAX backend `{backend}` "
            f"and device `{device_kind}`. The CPU platform was also registered solely "
            "for JAX host callbacks."
        ),
        (
            "Configuration: window `0.0025 s`, auxiliary source multiplier `0.5`, "
            "iteration cap `10`, tolerance `0.005`, damping `0.5`."
        ),
        "",
        "## Device placement and solver identity",
        "",
        (
            "Every JAX array in every returned equilibrium solve and every channel "
            "of every returned TORAX state was asserted to have only CUDA devices. "
            "A non-CUDA array would have terminated the run."
        ),
        "",
        "| exchange | sample | solve arrays | CUDA device |",
        "|---:|---:|---:|---|",
    ]
    lines.extend(
        f"| {row['exchange']} | {row['sample']} | {row['array_count']} | "
        f"`{row['devices']}` |"
        for row in solve_devices
    )
    lines.extend(
        [
            "",
            "| transport exchange | interval | state arrays | CUDA device |",
            "|---:|---:|---:|---|",
        ]
    )
    lines.extend(
        f"| {row['exchange']} | {row['interval']} | {row['array_count']} | "
        f"`{row['devices']}` |"
        for row in state_devices
    )
    lines.extend(
        [
            "",
            (
                "Owner solver-identity call path: "
                f"`equilibrium_sweep` at `{solver_path['equilibrium_sweep']}` creates "
                "the cold portfolio from `observed.moments.plasma_current` and calls "
                "`sampled_profile.solve_portfolio`; the callee is "
                "`ForwardProfile.solve_portfolio` at "
                f"`{solver_path['solve_portfolio']}`."
            ),
            "",
            "## Wall-time structure",
            "",
            (
                f"The converged H200 window took `{_format(attempt_seconds)}` s versus "
                f"the landed pre-band CPU window's `{_format(CPU_WINDOW_SECONDS)}` s: "
                f"a measured `{_format(speedup)}x` speedup on the same pre-band "
                "extraction lineage. Fixture preparation before the measured window "
                f"took `{_format(preparation_seconds)}` s and is reported separately."
            ),
            (
                "The empirical cold compile/startup cost is defined as the first "
                "iteration's excess above that side's warmed median. It measured "
                f"`{_format(cold_excess_seconds)}` s total "
                f"(`{_format(cold_excess_by_side.get('equilibrium_plus_fsa', 0.0))}` s "
                "equilibrium plus FSA and "
                f"`{_format(cold_excess_by_side.get('transport', 0.0))}` s TORAX), "
                "or "
                f"`{_format(cold_excess_seconds / GENTLE_ITERATION_CAP)}` s per "
                "iteration amortised over the ten exchanges."
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
            "## Uniform-grid guard callback cost",
            "",
            (
                "The guard fired once per TORAX adapter construction and therefore "
                "once per transport exchange. Each measurement synchronously blocks "
                "the callback result, so it includes dispatch, the CUDA-to-host grid "
                "transfer, CPU validation, and the returned-array handoff."
            ),
            "",
            "| iteration | callback count | round-trip wall time (s) |",
            "|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['iteration']} | 1 | `{_format(row['seconds'])}` |"
        for row in guard_events
    )
    lines.extend(
        [
            "",
            (
                f"Across `{len(guard_events)}` calls the guard cost "
                f"`{_format(guard_total)}` s total, median "
                f"`{_format(guard_median)}` s/call, or "
                f"`{_format(guard_total / attempt_seconds)}` of total window wall time."
            ),
            "",
            "The FSA record materialisation boundary was also observed:",
            "",
            "| operation | call | CUDA fields materialised |",
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
            "## CPU lineage qualification",
            "",
            (
                "The `423.032716 s` CPU window is the landed, directly comparable "
                "pre-band measurement. Boundary-band sparsification landed on main "
                "after this worktree was cut (`32942ac3`); its warm ITER CPU assembly "
                "is `24.6 s`. The complete CPU window has not been remeasured after "
                "that change, so no post-band CPU-window time or speedup is inferred."
            ),
            "",
            "## Receipt comparison",
            "",
            "| receipt | H200 | landed CPU | absolute deviation | tolerance |",
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
                "CPU value. No deviation exceeded the declared 0.005 comparison "
                "tolerance."
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

    devices = jax.devices("gpu")
    cpu_devices = jax.devices("cpu")
    backend = jax.default_backend()
    if backend != "gpu" or not devices or not cpu_devices:
        raise RuntimeError(
            "GPU computation and a CPU callback device are required, got "
            f"backend={backend}, gpu={devices}, cpu={cpu_devices}"
        )
    device_kind = devices[0].device_kind

    from scripts.window_demonstration import run_window as demonstration

    os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    original_equilibrium_sweep = demonstration.equilibrium_sweep
    original_transport_sweep = demonstration.transport_sweep
    original_block_and_copy = demonstration._block_and_copy
    original_grid_callback = torax_geometry._validated_grid_callback
    solve_devices: list[dict[str, Any]] = []
    state_devices: list[dict[str, Any]] = []
    transfer_events: list[dict[str, Any]] = []
    guard_events: list[dict[str, Any]] = []
    equilibrium_exchange = {"value": 0}
    transport_exchange = {"value": 0}
    callback_context = {"iteration": 0}

    def observed_equilibrium_sweep(*args, **kwargs):
        equilibrium_exchange["value"] += 1
        receipt = original_equilibrium_sweep(*args, **kwargs)
        for sample, equilibrium in enumerate(receipt.equilibria):
            placement = _require_cuda_arrays(
                equilibrium,
                (
                    "equilibrium exchange "
                    f"{equilibrium_exchange['value']} sample {sample}"
                ),
            )
            solve_devices.append(
                {
                    "exchange": equilibrium_exchange["value"],
                    "sample": sample,
                    **placement,
                }
            )
        return receipt

    def observed_transport_sweep(*args, **kwargs):
        transport_exchange["value"] += 1
        callback_context["iteration"] = transport_exchange["value"]
        receipt = original_transport_sweep(*args, **kwargs)
        for interval, item in enumerate(receipt.receipts):
            channels = {
                name: getattr(item.state, name)
                for name in (
                    "rho",
                    "psi",
                    "ion_temperature",
                    "electron_temperature",
                    "electron_density",
                )
            }
            placement = _require_cuda_arrays(
                channels,
                f"transport exchange {transport_exchange['value']} interval {interval}",
            )
            state_devices.append(
                {
                    "exchange": transport_exchange["value"],
                    "interval": interval,
                    **placement,
                }
            )
        return receipt

    def observed_grid_callback(rho_face):
        started = time.perf_counter()
        validated = original_grid_callback(rho_face)
        validated.block_until_ready()
        guard_events.append(
            {
                "iteration": callback_context["iteration"],
                "seconds": time.perf_counter() - started,
            }
        )
        return validated

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
    demonstration.transport_sweep = observed_transport_sweep
    demonstration._block_and_copy = observed_block_and_copy
    torax_geometry._validated_grid_callback = observed_grid_callback

    preparation_start = time.perf_counter()
    profile, seed, _vacuum = demonstration._fixture_machine()
    extraction_lattice = demonstration._extraction_lattice(profile)
    fixture_sources = demonstration._fixture_sources(profile)
    baseline_equilibrium = profile.solve(
        seed,
        route="anderson",
        evaluations=demonstration.EVALUATIONS,
    )
    baseline_placement = _require_cuda_arrays(
        baseline_equilibrium, "baseline equilibrium solve"
    )
    baseline_devices = baseline_placement["devices"]
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

    expected_iterations = int(result.convergence.iterations_used)
    callbacks_by_iteration = {
        iteration: sum(row["iteration"] == iteration for row in guard_events)
        for iteration in range(1, expected_iterations + 1)
    }
    if expected_iterations != GENTLE_ITERATION_CAP:
        raise RuntimeError(
            f"expected {GENTLE_ITERATION_CAP} converged exchanges, got "
            f"{expected_iterations}"
        )
    if callbacks_by_iteration != {
        iteration: 1 for iteration in range(1, expected_iterations + 1)
    }:
        raise RuntimeError(
            "uniform-grid callback did not fire exactly once per transport exchange: "
            f"{callbacks_by_iteration}"
        )
    if len(solve_devices) != 2 * expected_iterations:
        raise RuntimeError(
            "equilibrium device audit did not observe both coarse samples in every "
            f"exchange: {len(solve_devices)} arrays receipts"
        )
    if len(state_devices) != expected_iterations:
        raise RuntimeError(
            "transport device audit did not observe one returned state per exchange: "
            f"{len(state_devices)} state receipts"
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
    cold_excess_by_side = {
        side: max(values[0] - float(np.median(values[1:])), 0.0)
        for side, values in by_side.items()
        if len(values) > 1
    }
    cold_excess_seconds = sum(cold_excess_by_side.values())

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
        ("cpu_callback_device", str(cpu_devices[0]), "text"),
        ("baseline_solve_devices", baseline_devices, "text"),
        (
            "baseline_solve_array_count",
            baseline_placement["array_count"],
            "count",
        ),
        ("cpu_baseline_lineage", "pre-band", "text"),
        ("boundary_band_commit", "32942ac3", "git_commit"),
        ("post_band_warm_iter_cpu_assembly", 24.6, "s"),
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
    for side, seconds in cold_excess_by_side.items():
        _append(
            rows,
            "timing",
            "cold_iteration_excess",
            seconds,
            "s",
            side=side,
        )
    _append(
        rows,
        "timing",
        "cold_cost_amortised_per_iteration",
        cold_excess_seconds / GENTLE_ITERATION_CAP,
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
            status=f"PASS:{row['array_count']}_CUDA_ARRAYS",
        )
    for row in state_devices:
        _append(
            rows,
            "device_placement",
            "transport_state_arrays",
            row["devices"],
            "text",
            iteration=row["exchange"],
            side=f"interval_{row['interval']}",
            status=f"PASS:{row['array_count']}_CUDA_ARRAYS",
        )
    for row in guard_events:
        _append(
            rows,
            "guard_callback",
            "uniform_grid_roundtrip_wall_time",
            row["seconds"],
            "s",
            iteration=row["iteration"],
            status="MEASURED",
        )
    for row in transfer_events:
        _append(
            rows,
            "host_roundtrip",
            row["operation"],
            row["gpu_fields"],
            "field_count",
            iteration=row["call"],
            status="RECORDED_HOST_BOUNDARY",
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
            state_devices=state_devices,
            transfer_events=transfer_events,
            guard_events=guard_events,
            result=result,
            preparation_seconds=preparation_seconds,
            attempt_seconds=attempt_seconds,
            cold_excess_seconds=cold_excess_seconds,
            cold_excess_by_side=cold_excess_by_side,
            conservation=conservation,
            deviations=deviations,
            solver_path=solver_path,
        ),
        encoding="utf-8",
    )
    print(f"backend={backend}")
    print(f"device={device_kind}")
    print(f"window_seconds={attempt_seconds}")
    print(f"cuda_equilibrium_receipts={len(solve_devices)}")
    print(f"cuda_transport_states={len(state_devices)}")
    print(f"guard_callbacks={len(guard_events)}")
    print(f"host_roundtrips={len(transfer_events)}")
    print(f"report={arguments.report}")
    print(f"results={arguments.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
