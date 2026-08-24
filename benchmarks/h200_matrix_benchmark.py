"""Benchmark the four fixed-point matrix arms on reserved H200 devices.

The receipt uses the output-data arm labels from the matrix experiment:
``A`` is exact flux with first-order support, ``B`` exact flux with second-order
support, ``C`` coefficient-carried plasma flux with first-order support, and
``D`` coefficient-carried plasma flux with second-order support.  Those labels
remain data only; implementation names describe the route they execute.

The top-level command must run inside one SLURM allocation.  It launches fresh
processes for every arm, batch width, and card rung so peak device-memory
statistics are isolated.  Static interactions are replicated per card and
independent slices are sharded over cards with one saturated local batch per
device; there are no cross-device collectives in the timed region.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/coefficient-space-newton/h200-matrix-benchmark.json"
)
ROOT_BANK = ROOT / "scripts/oracle_rebaseline/root-coarse.npz"
CONVERGENCE_RECEIPT = (
    ROOT / "docs/figures/coefficient-space-newton/support-order-arms.json"
)
CARRIER_RECEIPT = (
    ROOT / "docs/figures/coefficient-space-newton/plasma-only-carrier.json"
)
BANKED_SLICES_PER_SECOND_PER_DEVICE = 46.577604167
KNOTS_PER_AXIS = 6
TIMING_REPEATS = 5
SATURATION_FRACTION = 0.95
BATCH_WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
VALID_ARM_LABELS = ("A", "B", "C", "D")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _device_memory() -> dict[str, int]:
    import jax

    stats = jax.devices()[0].memory_stats() or {}
    return {
        "bytes_in_use": int(stats.get("bytes_in_use", 0)),
        "peak_bytes_in_use": int(
            stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0))
        ),
        "bytes_limit": int(stats.get("bytes_limit", 0)),
    }


def _device_inventory() -> list[dict[str, Any]]:
    import jax

    return [
        {
            "id": int(device.id),
            "platform": device.platform,
            "kind": device.device_kind,
            "description": str(device),
        }
        for device in jax.devices()
    ]


def _construct_workload(
    arm_label: str,
) -> tuple[Callable, Any, dict[str, Any], dict[str, Any]]:
    import jax.numpy as jnp

    from benchmarks import carrier_arms
    from nova.equilibrium.coefficient_carrier import (
        CoefficientCarrier,
        coefficient_fixed_point_map,
    )

    fixture = carrier_arms._load_module(  # noqa: SLF001 - benchmark reuse seam
        carrier_arms.FIXTURE_MODULE, "h200_matrix_fixture"
    )
    case = fixture.analytic_case()

    machine_started = time.perf_counter()
    machine = fixture.cached_machine(
        case,
        fixture.FIXTURE_REQUESTS["coarse"],
        wall_nodes=fixture.WALL_POINT_COUNT,
    )
    cached_machine_seconds = time.perf_counter() - machine_started
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic_state = fixture.exact_state(case, coordinates)
    empty_operator = fixture.forward_operator(case, machine)
    physical_moments = fixture.exact_current_moments(
        case, empty_operator, analytic_state
    )
    moment_coefficients = empty_operator.coupling_current_moments(physical_moments)
    internal_flux = fixture._internal_flux_image(  # noqa: SLF001 - fixture contract
        empty_operator, moment_coefficients
    )
    operator = fixture.forward_operator(case, machine, analytic_state - internal_flux)
    first_support_map = operator.flux_map()
    known_external = np.asarray(operator.external(), dtype=np.float64)

    support_seconds = 0.0
    if arm_label in ("B", "D"):
        support_started = time.perf_counter()
        support_geometry = carrier_arms._support_geometry(  # noqa: SLF001
            machine, coordinates
        )
        map_function = carrier_arms._quadratic_support_map(  # noqa: SLF001
            operator, support_geometry
        )
        support_seconds = time.perf_counter() - support_started
    else:
        map_function = first_support_map

    with np.load(ROOT_BANK, allow_pickle=False) as bank:
        exact_state = np.asarray(bank["root_state"], dtype=np.float64)

    carrier_seconds = 0.0
    if arm_label in ("C", "D"):
        carrier_started = time.perf_counter()
        carrier = CoefficientCarrier.from_coordinates(
            coordinates,
            radial_knots=KNOTS_PER_AXIS,
            vertical_knots=KNOTS_PER_AXIS,
        )
        carrier_seconds = time.perf_counter() - carrier_started
        map_function = coefficient_fixed_point_map(
            map_function, carrier, external=known_external
        )[0]
        state = carrier.project(exact_state - known_external)
    else:
        state = jnp.asarray(exact_state)

    state = jnp.asarray(state, dtype=jnp.float64)
    build = {
        "cached_machine_and_primary_interaction_load_seconds": cached_machine_seconds,
        "primary_interaction_cache_hit": bool(machine.cache["hit"]),
        "primary_interaction_semantic_key": machine.cache["semantic_key"],
        "additional_quadratic_interaction_build_seconds": support_seconds,
        "coefficient_carrier_build_seconds": carrier_seconds,
        "interaction_build_seconds": cached_machine_seconds + support_seconds,
        "interaction_build_definition": (
            "warm semantic-cache acquisition of the shipped first-order interaction "
            "plus any live quadratic weighted-response construction; coefficient-basis "
            "construction and executable compilation are reported separately"
        ),
    }
    shape = {
        "realised_plasma_cells": int(len(machine.node)),
        "exact_state_values": int(exact_state.size),
        "iterated_state_values": int(state.size),
        "iterate": (
            "exact total-flux values"
            if arm_label in ("A", "B")
            else "six-by-six plasma-flux spline knot values"
        ),
        "support": (
            "monopole plus two first moments"
            if arm_label in ("A", "C")
            else "adds three quadratic moments"
        ),
    }
    return map_function, state, build, shape


def _measure_once(arm_label: str, batch_width: int, card_count: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    devices = jax.devices()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU backend required, got {jax.default_backend()!r}")
    if not jax.config.x64_enabled:
        raise RuntimeError("JAX float64 support is disabled")
    if len(devices) != card_count:
        raise RuntimeError(
            f"requested {card_count} visible cards but JAX discovered {len(devices)}"
        )

    map_function, state, build, shape = _construct_workload(arm_label)
    total_slices = card_count * batch_width
    host_state = np.broadcast_to(np.asarray(state), (total_slices, state.shape[0]))
    mesh = Mesh(np.asarray(devices), ("device",))
    slice_sharding = NamedSharding(mesh, PartitionSpec("device", None))
    sharded_state = jax.device_put(host_state, slice_sharding)
    execute = jax.jit(
        jax.vmap(map_function),
        in_shardings=slice_sharding,
        out_shardings=slice_sharding,
    )
    compile_started = time.perf_counter()
    compiled = execute.lower(sharded_state).compile()
    compile_seconds = time.perf_counter() - compile_started

    result = compiled(sharded_state)
    jax.block_until_ready(result)
    if result.dtype != jnp.float64:
        raise RuntimeError(f"expected float64 output, got {result.dtype}")
    samples = []
    for _ in range(TIMING_REPEATS):
        started = time.perf_counter()
        result = compiled(sharded_state)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - started)
    samples.sort()
    median_seconds = float(samples[len(samples) // 2])
    throughput = total_slices / median_seconds
    memory = _device_memory()
    per_device_peak = memory["peak_bytes_in_use"]
    build["executable_compile_seconds"] = compile_seconds
    build["total_one_time_cost_seconds"] = (
        build["interaction_build_seconds"]
        + build["coefficient_carrier_build_seconds"]
        + compile_seconds
    )
    return {
        "arm": arm_label,
        "batch_width_per_device": batch_width,
        "card_count": card_count,
        "total_slices": total_slices,
        "wall_seconds": {
            "median": median_seconds,
            "minimum": float(samples[0]),
            "maximum": float(samples[-1]),
            "repeats": TIMING_REPEATS,
        },
        "execute_seconds_per_slice": median_seconds / total_slices,
        "aggregate_slices_per_second": throughput,
        "slices_per_second_per_device": throughput / card_count,
        "banked_requirement_slices_per_second_per_device": (
            BANKED_SLICES_PER_SECOND_PER_DEVICE
        ),
        "requirement_multiple_per_device": (
            throughput / card_count / BANKED_SLICES_PER_SECOND_PER_DEVICE
        ),
        "meets_banked_requirement": (
            throughput / card_count >= BANKED_SLICES_PER_SECOND_PER_DEVICE
        ),
        "peak_device_memory_bytes_per_device": per_device_peak,
        "peak_device_memory_gib_per_device": per_device_peak / 2**30,
        "device_memory": memory,
        "build": build,
        "shape": shape,
        "dtype": str(result.dtype),
        "devices": _device_inventory(),
        "sharding": {
            "arrangement": "independent_slice_batches",
            "local_batch_per_device": batch_width,
            "static_interactions": "replicated once per device",
            "dynamic_slice_axis": "evenly sharded over visible devices",
            "input_residency": (
                "one local batch preplaced on every target device before warmup "
                "and timing"
            ),
            "cross_device_collectives_in_timed_region": False,
        },
    }


def _visible_device_tokens() -> list[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not value:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is absent; run inside a SLURM GPU job")
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise RuntimeError("CUDA_VISIBLE_DEVICES names no granted cards")
    return tokens


def _child_measurement(
    arm_label: str,
    batch_width: int,
    device_tokens: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(device_tokens)
    environment["JAX_PLATFORMS"] = "cuda"
    environment["JAX_ENABLE_X64"] = "true"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "measure-one",
        "--arm",
        arm_label,
        "--batch-width",
        str(batch_width),
        "--card-count",
        str(len(device_tokens)),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if completed.returncode != 0:
        failure = {
            "arm": arm_label,
            "batch_width_per_device": batch_width,
            "card_count": len(device_tokens),
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
            "stdout_tail": completed.stdout[-1000:],
        }
        return None, failure
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        return json.loads(lines[-1]), None
    except (IndexError, json.JSONDecodeError) as error:
        output_tail = completed.stdout[-2000:]
        raise RuntimeError(
            f"child measurement returned no parseable payload: {output_tail}"
        ) from error


def _saturation(
    sweep: list[dict[str, Any]], failed: dict[str, Any] | None
) -> dict[str, Any]:
    peak_throughput = max(row["slices_per_second_per_device"] for row in sweep)
    threshold = SATURATION_FRACTION * peak_throughput
    selected = next(
        row for row in sweep if row["slices_per_second_per_device"] >= threshold
    )
    selected_index = sweep.index(selected)
    plateau_observed = selected_index < len(sweep) - 1 or failed is not None
    if not plateau_observed:
        raise RuntimeError(
            "batch sweep ended at its first saturation candidate without a wider "
            "measurement or an exhaustion boundary"
        )
    return {
        "criterion": (
            "smallest measured width reaching at least 95 percent of the maximum "
            "successful one-card throughput, with a wider width or exhaustion "
            "boundary observed"
        ),
        "fraction_of_observed_peak": SATURATION_FRACTION,
        "observed_peak_slices_per_second": peak_throughput,
        "batch_width": selected["batch_width_per_device"],
        "slices_per_second_per_device": selected["slices_per_second_per_device"],
        "execute_seconds_per_slice": selected["execute_seconds_per_slice"],
        "peak_device_memory_bytes": selected["peak_device_memory_bytes_per_device"],
        "peak_device_memory_gib": selected["peak_device_memory_gib_per_device"],
        "wider_width_or_exhaustion_observed": plateau_observed,
        "first_failed_width": failed,
    }


def _convergence_context() -> dict[str, Any]:
    support = json.loads(CONVERGENCE_RECEIPT.read_text(encoding="utf-8"))
    carrier = json.loads(CARRIER_RECEIPT.read_text(encoding="utf-8"))
    exact_residuals = {
        label: [
            row["convergence"]["terminal_exact_field_relative_residual"]
            for row in support["arms"][label]
        ]
        for label in ("A", "B")
    }
    return {
        "receipts": {
            "exact_value_support_order": {
                "path": str(CONVERGENCE_RECEIPT.relative_to(ROOT)),
                "sha256": _sha256(CONVERGENCE_RECEIPT),
            },
            "coefficient_carrier": {
                "path": str(CARRIER_RECEIPT.relative_to(ROOT)),
                "sha256": _sha256(CARRIER_RECEIPT),
            },
        },
        "single_frame_convergence": {
            "A_terminal_residuals": exact_residuals["A"],
            "B_terminal_residuals": exact_residuals["B"],
            "C_terminal_residual": carrier["arms"]["C"][
                "terminal_exact_field_residual"
            ],
            "D_terminal_residual": carrier["arms"]["D"][
                "terminal_exact_field_residual"
            ],
            "classification": (
                "A and B tie at float64 convergence; C and D are excluded because "
                "each stalls after one admitted advance near 3.85e-2"
            ),
            "preferred_arms": ["A", "B"],
        },
    }


def _run(output: Path) -> dict[str, Any]:
    granted_tokens = _visible_device_tokens()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("a SLURM allocation is required")

    arm_records: dict[str, Any] = {}
    for arm_label in VALID_ARM_LABELS:
        sweep = []
        failed = None
        for width in BATCH_WIDTHS:
            measurement, failure = _child_measurement(
                arm_label, width, granted_tokens[:1]
            )
            if failure is not None:
                failed = failure
                break
            if measurement is None:
                raise RuntimeError("child returned neither a measurement nor a failure")
            sweep.append(measurement)
        saturation = _saturation(sweep, failed)

        ladder = []
        for card_count in range(1, len(granted_tokens) + 1):
            measurement, failure = _child_measurement(
                arm_label,
                saturation["batch_width"],
                granted_tokens[:card_count],
            )
            if failure is not None or measurement is None:
                raise RuntimeError(
                    f"card ladder failed for arm {arm_label} at {card_count} cards: "
                    f"{failure}"
                )
            ladder.append(measurement)
        one_card_throughput = ladder[0]["aggregate_slices_per_second"]
        for row in ladder:
            ideal = one_card_throughput * row["card_count"]
            row["scaling_efficiency"] = row["aggregate_slices_per_second"] / ideal
            row["ideal_from_measured_one_card_slices_per_second"] = ideal

        arm_records[arm_label] = {
            "one_card_batch_sweep": sweep,
            "saturation": saturation,
            "card_ladder": ladder,
            "one_time_costs": sweep[0]["build"],
            "single_frame_execute": {
                "seconds_per_slice": sweep[0]["execute_seconds_per_slice"],
                "slices_per_second_per_device": sweep[0][
                    "slices_per_second_per_device"
                ],
            },
            "saturated_execute": {
                "seconds_per_slice": saturation["execute_seconds_per_slice"],
                "slices_per_second_per_device": saturation[
                    "slices_per_second_per_device"
                ],
                "requirement_multiple": (
                    saturation["slices_per_second_per_device"]
                    / BANKED_SLICES_PER_SECOND_PER_DEVICE
                ),
            },
        }

    common_widths = sorted(
        set(
            row["batch_width_per_device"]
            for row in arm_records["A"]["one_card_batch_sweep"]
        )
        & set(
            row["batch_width_per_device"]
            for row in arm_records["B"]["one_card_batch_sweep"]
        )
    )
    memory_ratios = []
    for width in common_widths:
        exact_first = next(
            row
            for row in arm_records["A"]["one_card_batch_sweep"]
            if row["batch_width_per_device"] == width
        )
        exact_second = next(
            row
            for row in arm_records["B"]["one_card_batch_sweep"]
            if row["batch_width_per_device"] == width
        )
        ratio = (
            exact_second["peak_device_memory_bytes_per_device"]
            / exact_first["peak_device_memory_bytes_per_device"]
        )
        memory_ratios.append(
            {
                "batch_width": width,
                "A_peak_device_memory_bytes": exact_first[
                    "peak_device_memory_bytes_per_device"
                ],
                "B_peak_device_memory_bytes": exact_second[
                    "peak_device_memory_bytes_per_device"
                ],
                "B_over_A_ratio": ratio,
                "B_percent_more_than_A": 100.0 * (ratio - 1.0),
            }
        )

    convergence = _convergence_context()
    eligible = ("A", "B")
    preferred = max(
        eligible,
        key=lambda label: arm_records[label]["saturated_execute"][
            "slices_per_second_per_device"
        ],
    )
    campaign_verdict = {
        "preferred_arm": preferred,
        "eligibility": (
            "A and B only: both converge to float64 precision; C and D are not "
            "campaign candidates because the banked single-frame runs stall after "
            "one admitted advance"
        ),
        "basis": (
            "highest measured saturated per-device throughput among the arms that "
            "retain float64 single-frame convergence, with isolated peak device "
            "memory and the full granted-card ladder reported"
        ),
        "single_frame_convergence_preference": ["A", "B"],
        "differs_from_single_frame_convergence_preference": False,
        "difference_statement": (
            f"campaign scale selects {preferred} from the A/B convergence tie; it "
            "does not overturn that tie in favour of a stalled carrier arm"
        ),
    }

    receipt = {
        "schema": "nova.h200-matrix-benchmark",
        "source_revision": _source_revision(),
        "command": sys.argv,
        "measurement_scope": {
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "cuda_visible_devices_at_allocation": granted_tokens,
            "granted_card_count": len(granted_tokens),
            "nominal_node_card_count": 8,
            "largest_measured_card_rung": len(granted_tokens),
            "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "hostname": platform.node(),
            "production_dtype": "float64",
            "frame_set": ["closed-form-oracle-coarse"],
            "banked_root": str(ROOT_BANK.relative_to(ROOT)),
            "banked_root_sha256": _sha256(ROOT_BANK),
            "batch_saturation_definition": (
                "smallest width reaching 95 percent of the observed successful "
                "one-card throughput maximum, qualified by a wider measurement or "
                "the first failed width"
            ),
        },
        "banked_requirement": {
            "slices_per_second_per_device": BANKED_SLICES_PER_SECOND_PER_DEVICE,
            "source": "docs/figures/mast-catalog-gpu-solve/slice-census.json",
        },
        "arms": arm_records,
        "arm_B_device_memory_remeasurement": {
            "statement": (
                "Every ratio below is an isolated fresh-process H200 peak at the "
                "same batch width; none is inherited from the shared CPU allocation"
            ),
            "ratios": memory_ratios,
        },
        "convergence_context": convergence,
        "campaign_verdict": campaign_verdict,
    }
    _write_json(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    one_parser = commands.add_parser("measure-one")
    one_parser.add_argument("--arm", choices=VALID_ARM_LABELS, required=True)
    one_parser.add_argument("--batch-width", type=int, required=True)
    one_parser.add_argument("--card-count", type=int, required=True)
    arguments = parser.parse_args()

    if arguments.command == "measure-one":
        payload = _measure_once(
            arguments.arm, arguments.batch_width, arguments.card_count
        )
        print(json.dumps(_strict(payload), sort_keys=True, allow_nan=False))
        return

    payload = _run(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "granted_card_count": payload["measurement_scope"][
                    "granted_card_count"
                ],
                "campaign_verdict": payload["campaign_verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
