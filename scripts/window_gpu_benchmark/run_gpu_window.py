"""Measure one coupled-window configuration on CPU and CUDA from one tree."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import dataclasses
import inspect
import os
import statistics
import subprocess
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


REPETITIONS = 3
WARM_PAIR_COUNT = 9
TSV_FIELDS = (
    "tree_sha",
    "backend",
    "repetition",
    "category",
    "iteration",
    "side",
    "field",
    "value",
    "unit",
    "cpu_reference",
    "absolute_deviation",
    "status",
)


@dataclasses.dataclass(frozen=True)
class ArmMeasurement:
    """One backend's repeated timing sample and first typed receipt."""

    backend: str
    device: str
    outcome_type: str
    outcome: str
    convergence: Any
    conservation: Any
    preparation_seconds: float
    window_seconds: tuple[float, ...]
    pairs: tuple[Mapping[str, Any], ...]
    warm_pairs: tuple[Mapping[str, Any], ...]
    array_count: int
    solve_location: str


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


def _tree_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _source_location(function: Any) -> str:
    path = inspect.getsourcefile(function)
    _source, line = inspect.getsourcelines(function)
    return f"{path}:{line}"


def _array_records(tree: Any, prefix: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    visited: set[int] = set()

    def visit(value: Any, path: str) -> None:
        if isinstance(value, jax.Array):
            records.append(
                {
                    "path": path,
                    "dtype": str(value.dtype),
                    "platform": ",".join(
                        sorted({device.platform for device in value.devices()})
                    ),
                    "device": ",".join(
                        sorted(str(device) for device in value.devices())
                    ),
                }
            )
            return
        if isinstance(
            value, np.ndarray | str | bytes | int | float | bool | type(None)
        ):
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


def _require_platform(
    tree: Any, expected: str, label: str, inventory: list[dict[str, str]]
) -> None:
    records = _array_records(tree, label)
    if not records:
        raise RuntimeError(f"{label} exposed no JAX solve or state arrays")
    refused = [record for record in records if record["platform"] != expected]
    if refused:
        raise RuntimeError(f"{label} contains arrays off {expected}: {refused[:3]}")
    inventory.extend(records)


def _paired_timings(result: Any, repetition: int) -> list[dict[str, Any]]:
    by_iteration: dict[int, dict[str, float]] = {}
    for event in result.timings:
        iteration = int(event["iteration"])
        side = str(event["side"])
        if side == "equilibrium":
            continue
        by_iteration.setdefault(iteration, {})[side] = float(event["seconds"])
    pairs: list[dict[str, Any]] = []
    for iteration, sides in sorted(by_iteration.items()):
        if "equilibrium_plus_fsa" not in sides or "transport" not in sides:
            raise RuntimeError(
                f"iteration {iteration} has incomplete side timings: {sides}"
            )
        pairs.append(
            {
                "repetition": repetition,
                "iteration": iteration,
                "equilibrium_plus_fsa": sides["equilibrium_plus_fsa"],
                "transport": sides["transport"],
                "combined": sides["equilibrium_plus_fsa"] + sides["transport"],
            }
        )
    return pairs


def _receipt_signature(result: Any) -> tuple[Any, ...]:
    convergence = result.convergence
    conservation = result.conservation_receipt
    return (
        result.outcome_type,
        convergence.iterations_used,
        convergence.contraction_estimate,
        convergence.gating_norm,
        convergence.all_field_norm,
        conservation.flux_closure_error,
        conservation.flux_closure_residual,
        conservation.current_continuity_error,
        conservation.current_continuity_residual,
    )


def _measure_arm(demonstration: Any, device: Any) -> ArmMeasurement:
    platform = device.platform
    original_equilibrium_sweep = demonstration.equilibrium_sweep
    original_transport_sweep = demonstration.transport_sweep
    inventory: list[dict[str, str]] = []

    def observed_equilibrium_sweep(*args, **kwargs):
        receipt = original_equilibrium_sweep(*args, **kwargs)
        jax.block_until_ready(receipt.equilibria)
        _require_platform(receipt.equilibria, platform, "equilibrium", inventory)
        return receipt

    def observed_transport_sweep(*args, **kwargs):
        receipt = original_transport_sweep(*args, **kwargs)
        jax.block_until_ready(receipt.receipts)
        _require_platform(receipt.receipts, platform, "transport", inventory)
        return receipt

    demonstration.equilibrium_sweep = observed_equilibrium_sweep
    demonstration.transport_sweep = observed_transport_sweep
    try:
        preparation_started = time.perf_counter()
        with jax.default_device(device):
            profile, seed, _vacuum = demonstration._fixture_machine()
            baseline_equilibrium = profile.solve(
                seed, route="anderson", evaluations=demonstration.EVALUATIONS
            )
            jax.block_until_ready(baseline_equilibrium)
            _require_platform(
                baseline_equilibrium, platform, "baseline_equilibrium", inventory
            )
            extraction_lattice = demonstration._extraction_lattice(profile)
            fixture_sources = demonstration._fixture_sources(profile)
            baseline_geometry, baseline_extraction = (
                demonstration._geometry_from_equilibrium(
                    baseline_equilibrium,
                    profile.source,
                    extraction_lattice,
                    fixture_sources,
                )
            )
        preparation_seconds = time.perf_counter() - preparation_started

        results = []
        window_seconds = []
        pairs: list[dict[str, Any]] = []
        config = demonstration.RegimeConfig("gentle", 0.0025, 0.5, 1)
        for repetition in range(1, REPETITIONS + 1):
            started = time.perf_counter()
            with jax.default_device(device):
                result = demonstration._run_regime(
                    config,
                    profile=profile,
                    baseline_equilibrium=baseline_equilibrium,
                    baseline_geometry=baseline_geometry,
                    baseline_extraction=baseline_extraction,
                    extraction_lattice=extraction_lattice,
                    fixture_sources=fixture_sources,
                )
            window_seconds.append(time.perf_counter() - started)
            if not result.converged or result.convergence is None:
                raise RuntimeError(
                    f"{platform} gentle window did not converge: "
                    f"{result.outcome_type}: {result.outcome}"
                )
            results.append(result)
            pairs.extend(_paired_timings(result, repetition))

        reference_signature = np.asarray(
            _receipt_signature(results[0])[1:], dtype=float
        )
        for repetition, result in enumerate(results[1:], start=2):
            signature = np.asarray(_receipt_signature(result)[1:], dtype=float)
            if not np.allclose(signature, reference_signature, rtol=1.0e-12, atol=0.0):
                raise RuntimeError(
                    f"{platform} receipt changed in repetition {repetition}: "
                    f"{signature} != {reference_signature}"
                )

        warm = tuple(pair for pair in pairs if pair["iteration"] > 1)
        if len(warm) < WARM_PAIR_COUNT:
            raise RuntimeError(
                f"{platform} produced {len(warm)} warm pairs; "
                f"{WARM_PAIR_COUNT} are required"
            )
        warm = warm[:WARM_PAIR_COUNT]
        first = results[0]
        return ArmMeasurement(
            backend=platform,
            device=str(device),
            outcome_type=first.outcome_type,
            outcome=first.outcome,
            convergence=first.convergence,
            conservation=first.conservation_receipt,
            preparation_seconds=preparation_seconds,
            window_seconds=tuple(window_seconds),
            pairs=tuple(pairs),
            warm_pairs=warm,
            array_count=len(inventory),
            solve_location=_source_location(ForwardProfile.solve_portfolio),
        )
    finally:
        demonstration.equilibrium_sweep = original_equilibrium_sweep
        demonstration.transport_sweep = original_transport_sweep


def _median(arm: ArmMeasurement, field: str) -> float:
    return statistics.median(float(pair[field]) for pair in arm.warm_pairs)


def _append(
    rows: list[dict[str, str]],
    tree_sha: str,
    backend: str,
    category: str,
    field: str,
    value: Any,
    unit: str,
    *,
    repetition: int | str = "",
    iteration: int | str = "",
    side: str = "",
    cpu_reference: Any = "",
    status: str = "RECORDED",
) -> None:
    deviation = ""
    if cpu_reference != "":
        deviation = abs(float(value) - float(cpu_reference))
    rows.append(
        {
            "tree_sha": tree_sha,
            "backend": backend,
            "repetition": str(repetition),
            "category": category,
            "iteration": str(iteration),
            "side": side,
            "field": field,
            "value": _format(value),
            "unit": unit,
            "cpu_reference": _format(cpu_reference) if cpu_reference != "" else "",
            "absolute_deviation": _format(deviation) if deviation != "" else "",
            "status": status,
        }
    )


def _arm_rows(
    rows: list[dict[str, str]], tree_sha: str, arm: ArmMeasurement, cpu: ArmMeasurement
) -> None:
    convergence = arm.convergence
    conservation = arm.conservation
    comparison = arm.backend != "cpu"
    cpu_convergence = cpu.convergence
    cpu_conservation = cpu.conservation
    for field, value, unit, reference in (
        (
            "preparation_wall_time",
            arm.preparation_seconds,
            "s",
            cpu.preparation_seconds,
        ),
        ("first_window_wall_time", arm.window_seconds[0], "s", cpu.window_seconds[0]),
        (
            "iterations_used",
            convergence.iterations_used,
            "count",
            cpu_convergence.iterations_used,
        ),
        (
            "contraction_estimate",
            convergence.contraction_estimate,
            "ratio",
            cpu_convergence.contraction_estimate,
        ),
        (
            "gating_exit_norm",
            convergence.gating_norm,
            "relative",
            cpu_convergence.gating_norm,
        ),
        (
            "all_field_exit_norm",
            convergence.all_field_norm,
            "relative",
            cpu_convergence.all_field_norm,
        ),
        (
            "damping_applied",
            convergence.damping_applied,
            "fraction",
            cpu_convergence.damping_applied,
        ),
        (
            "flux_closure_error",
            conservation.flux_closure_error,
            "Wb",
            cpu_conservation.flux_closure_error,
        ),
        (
            "flux_closure_residual",
            conservation.flux_closure_residual,
            "relative",
            cpu_conservation.flux_closure_residual,
        ),
        (
            "current_continuity_error",
            conservation.current_continuity_error,
            "A",
            cpu_conservation.current_continuity_error,
        ),
        (
            "current_continuity_residual",
            conservation.current_continuity_residual,
            "relative",
            cpu_conservation.current_continuity_residual,
        ),
        (
            "warm_equilibrium_plus_fsa_median",
            _median(arm, "equilibrium_plus_fsa"),
            "s",
            _median(cpu, "equilibrium_plus_fsa"),
        ),
        (
            "warm_transport_median",
            _median(arm, "transport"),
            "s",
            _median(cpu, "transport"),
        ),
        ("warm_pair_median", _median(arm, "combined"), "s", _median(cpu, "combined")),
        ("warm_pair_count", len(arm.warm_pairs), "count", len(cpu.warm_pairs)),
        ("solve_and_state_array_count", arm.array_count, "count", cpu.array_count),
    ):
        _append(
            rows,
            tree_sha,
            arm.backend,
            "summary",
            field,
            value,
            unit,
            cpu_reference=reference if comparison else "",
            status="SAME_TREE_COMPARISON" if comparison else "CPU_REFERENCE",
        )
    _append(
        rows,
        tree_sha,
        arm.backend,
        "summary",
        "outcome_type",
        arm.outcome_type,
        "text",
    )
    _append(
        rows,
        tree_sha,
        arm.backend,
        "summary",
        "device",
        arm.device,
        "text",
        status="ALL_OBSERVED_JAX_ARRAYS_ON_DECLARED_BACKEND",
    )
    for repetition, seconds in enumerate(arm.window_seconds, start=1):
        _append(
            rows,
            tree_sha,
            arm.backend,
            "window_timing",
            "wall_time",
            seconds,
            "s",
            repetition=repetition,
        )
    for pair in arm.pairs:
        for side in ("equilibrium_plus_fsa", "transport", "combined"):
            _append(
                rows,
                tree_sha,
                arm.backend,
                "iteration_pair_timing",
                "wall_time",
                pair[side],
                "s",
                repetition=pair["repetition"],
                iteration=pair["iteration"],
                side=side,
                status=(
                    "WARM_MEDIAN_MEMBER"
                    if pair in arm.warm_pairs
                    else "COLD_OR_EXCESS_SAMPLE"
                ),
            )
    for iteration, (gating, all_field) in enumerate(
        zip(
            convergence.gating_norm_trace,
            convergence.all_field_norm_trace,
            strict=True,
        ),
        start=1,
    ):
        _append(
            rows,
            tree_sha,
            arm.backend,
            "convergence_trace",
            "gating_norm",
            gating,
            "relative",
            iteration=iteration,
        )
        _append(
            rows,
            tree_sha,
            arm.backend,
            "convergence_trace",
            "all_field_norm",
            all_field,
            "relative",
            iteration=iteration,
        )
    for field, value in convergence.exit_residual.items():
        _append(rows, tree_sha, arm.backend, "exit_residual", field, value, "relative")


def _write_tsv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=TSV_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _render_report(
    tree_sha: str, job_id: str, cpu: ArmMeasurement, gpu: ArmMeasurement
) -> str:
    speedup = _median(cpu, "combined") / _median(gpu, "combined")
    cpu_contraction = _format(cpu.convergence.contraction_estimate)
    gpu_contraction = _format(gpu.convergence.contraction_estimate)
    cpu_gating = _format(cpu.convergence.gating_norm)
    cpu_all_field = _format(cpu.convergence.all_field_norm)
    gpu_gating = _format(gpu.convergence.gating_norm)
    gpu_all_field = _format(gpu.convergence.all_field_norm)
    cpu_flux_error = _format(cpu.conservation.flux_closure_error)
    cpu_flux_residual = _format(cpu.conservation.flux_closure_residual)
    gpu_flux_error = _format(gpu.conservation.flux_closure_error)
    gpu_flux_residual = _format(gpu.conservation.flux_closure_residual)
    cpu_current_error = _format(cpu.conservation.current_continuity_error)
    cpu_current_residual = _format(cpu.conservation.current_continuity_residual)
    gpu_current_error = _format(gpu.conservation.current_continuity_error)
    gpu_current_residual = _format(gpu.conservation.current_continuity_residual)
    rows = []
    for arm in (cpu, gpu):
        convergence = arm.convergence
        conservation = arm.conservation
        rows.append(
            "| "
            + " | ".join(
                (
                    arm.backend,
                    arm.device,
                    str(convergence.iterations_used),
                    f"{arm.window_seconds[0]:.6f}",
                    f"{_median(arm, 'equilibrium_plus_fsa'):.6f}",
                    f"{_median(arm, 'transport'):.6f}",
                    f"{_median(arm, 'combined'):.6f}",
                    f"{convergence.gating_norm:.9g}",
                    f"{convergence.all_field_norm:.9g}",
                    f"{conservation.flux_closure_residual:.9g}",
                    f"{conservation.current_continuity_residual:.9g}",
                )
            )
            + " |"
        )
    return "\n".join(
        [
            "# Current-tree H200 coupled-window measurement",
            "",
            f"Tree: `{tree_sha}`. SLURM job: `{job_id}`.",
            "",
            (
                "One job measured the identical gentle window on the CPU and one H200. "
                "The first run on each backend supplies the typed receipt and total "
                "window wall. Three deterministic repetitions supply nine warm "
                "iteration-pair samples by omitting each repetition's first pair."
            ),
            "",
            "## Declared window",
            "",
            "- Length: `0.0025 s`",
            "- Auxiliary-source multiplier: `0.5`",
            "- Ordinary iteration cap: `10`; hard ceiling: `20`",
            "- Convergence tolerance: `0.005`; contraction threshold: `0.8`",
            "- Initial damping: `1.0`; damping floor: `0.125`",
            "- Platforms: `cuda,cpu`; temporary directory: `/tmp`",
            "",
            "## Direct same-tree result",
            "",
            (
                "| backend | device | iterations | first window wall (s) | "
                "warm equilibrium + FSA median (s) | warm TORAX median (s) | "
                "warm pair median (s), n=9 | gating exit norm | "
                "all-field exit norm | flux closure relative | "
                "current closure relative |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"The measured warm iteration-pair speedup is `{speedup:.6f}x`.",
            "",
            "The prior cross-tree wall projection is retired by these direct, "
            "tree-identical measurements. Every TSV row carries the full tree SHA.",
            "",
            "## Receipt comparison",
            "",
            f"CPU contraction: `{cpu_contraction}`; H200: `{gpu_contraction}`.",
            f"CPU gating/all-field exit: `{cpu_gating}` / `{cpu_all_field}`.",
            f"H200 gating/all-field exit: `{gpu_gating}` / `{gpu_all_field}`.",
            (
                "CPU flux closure absolute/relative: "
                f"`{cpu_flux_error}` / `{cpu_flux_residual}`."
            ),
            (
                "H200 flux closure absolute/relative: "
                f"`{gpu_flux_error}` / `{gpu_flux_residual}`."
            ),
            (
                "CPU current closure absolute/relative: "
                f"`{cpu_current_error}` / `{cpu_current_residual}`."
            ),
            (
                "H200 current closure absolute/relative: "
                f"`{gpu_current_error}` / `{gpu_current_residual}`."
            ),
            "",
            "## Solver and placement checks",
            "",
            (
                "The equilibrium leg routes through `ForwardProfile.solve_portfolio` "
                f"at `{gpu.solve_location}`. The run inspected `{cpu.array_count}` CPU "
                f"and `{gpu.array_count}` H200 solve/state array observations and "
                "failed closed if any JAX array occupied the wrong backend."
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    arguments = parser.parse_args()

    gpu_devices = jax.devices("gpu")
    cpu_devices = jax.devices("cpu")
    if not gpu_devices or not cpu_devices or jax.default_backend() != "gpu":
        raise RuntimeError("one CUDA device and the CPU platform are required")

    from scripts.window_demonstration import run_window as demonstration

    tree_sha = _tree_sha()
    cpu = _measure_arm(demonstration, cpu_devices[0])
    gpu = _measure_arm(demonstration, gpu_devices[0])

    rows: list[dict[str, str]] = []
    _arm_rows(rows, tree_sha, cpu, cpu)
    _arm_rows(rows, tree_sha, gpu, cpu)
    _write_tsv(arguments.results, rows)
    arguments.report.write_text(
        _render_report(
            tree_sha,
            os.environ.get("SLURM_JOB_ID", "unknown"),
            cpu,
            gpu,
        ),
        encoding="utf-8",
    )
    print(f"tree_sha={tree_sha}")
    print(f"cpu_iterations={cpu.convergence.iterations_used}")
    print(f"gpu_iterations={gpu.convergence.iterations_used}")
    print(f"cpu_window_seconds={cpu.window_seconds[0]:.17g}")
    print(f"gpu_window_seconds={gpu.window_seconds[0]:.17g}")
    print(f"cpu_warm_pair_median={_median(cpu, 'combined'):.17g}")
    print(f"gpu_warm_pair_median={_median(gpu, 'combined'):.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
