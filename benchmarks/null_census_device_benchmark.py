"""Measure staggered null-census cost and frozen-field candidate identity.

The timed route is an explicit ``vmap`` over independent float64 slices.  The
correctness route reconstructs the twelve banked MAST terminal arms and compares
the resolved candidate records returned by the rectangular and combined orbit
families.  The ITER scenario EQDSK is intentionally absent: it is a fixed-boundary
field without a divertor saddle and therefore cannot test null-census topology.
"""

from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from benchmarks.out_of_vessel_saddle_selection import _terminal_states
from nova.equilibrium.stencil_nulls import critical_point_candidates_batch
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path(
    "docs/figures/primary-xpoint-evidence/dual-sweep-device-lock.json"
)
GRID_SIZES = (33, 129)
BATCH_WIDTHS = (1, 4, 16)
EXECUTION_REPETITIONS = 21
K_SLOTS = 8
RATIO_BOUND = 2.0
COARSE_PER_SLICE_BOUND_MS = 1.0
DUAL_SWEEP_IMPLEMENTATION_COMMIT = "46e5d9b8"


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _representative_fields(size: int, width: int) -> tuple[np.ndarray, ...]:
    """Return deterministic smooth fields and their all-eligible tensor grid."""
    radius = np.linspace(0.2, 1.8, size, dtype=np.float64)
    height = np.linspace(-1.2, 1.2, size, dtype=np.float64)
    radial_offset = radius[None, :] - 0.93
    vertical_offset = height[:, None] + 0.07
    base = (
        radial_offset**2
        - 1.17 * vertical_offset**2
        + 0.31 * radial_offset * vertical_offset
        + 0.025 * radial_offset**3
        - 0.018 * vertical_offset**3
    )
    fields = np.stack([base + np.float64(index) * 1.0e-5 for index in range(width)])
    inside = np.ones((size, size), dtype=bool)
    return fields, radius, height, inside


def _one_slice_census(
    field: jax.Array,
    radius: jax.Array,
    height: jax.Array,
    inside: jax.Array,
    *,
    dual_sweep: bool,
) -> dict[str, jax.Array]:
    result = critical_point_candidates_batch(
        field[None],
        radius,
        height,
        inside,
        k_slots=K_SLOTS,
        material_dilate=0,
        target_index=-1,
        noise_sigma=0.0,
        dual_sweep=dual_sweep,
    )
    return jax.tree.map(lambda value: value[0], result)


def _block_tree(tree: Any) -> None:
    for leaf in jax.tree.leaves(tree):
        leaf.block_until_ready()


def _quartiles(samples: list[float]) -> tuple[float, float, float]:
    lower, median, upper = np.percentile(np.asarray(samples), [25.0, 50.0, 75.0])
    return float(lower), float(median), float(upper)


def _measure_configuration(size: int, width: int, dual_sweep: bool) -> dict[str, Any]:
    fields, radius, height, inside = _representative_fields(size, width)
    mapped = jax.jit(
        jax.vmap(
            partial(
                _one_slice_census,
                radius=jnp.asarray(radius),
                height=jnp.asarray(height),
                inside=jnp.asarray(inside),
                dual_sweep=dual_sweep,
            )
        )
    )
    device_fields = jax.device_put(fields)
    compile_started = time.perf_counter()
    executable = mapped.lower(device_fields).compile()
    compile_seconds = time.perf_counter() - compile_started
    _block_tree(executable(device_fields))

    samples_ms = []
    for _ in range(EXECUTION_REPETITIONS):
        started = time.perf_counter()
        result = executable(device_fields)
        _block_tree(result)
        samples_ms.append((time.perf_counter() - started) * 1.0e3)
    lower, median, upper = _quartiles(samples_ms)
    return {
        "grid_size": [size, size],
        "batch_width": width,
        "orbit_families": "rectangular_and_centroid" if dual_sweep else "rectangular",
        "compile_seconds": compile_seconds,
        "execute_repetitions": EXECUTION_REPETITIONS,
        "warmed_execute_batch_ms": {
            "median": median,
            "q1": lower,
            "q3": upper,
            "iqr": upper - lower,
        },
        "warmed_execute_per_slice_ms": {
            "median": median / width,
            "q1": lower / width,
            "q3": upper / width,
            "iqr": (upper - lower) / width,
        },
    }


def _resolved_records(
    result: dict[str, Any], batch_index: int = 0
) -> list[dict[str, Any]]:
    host = jax.device_get(result)
    present = np.asarray(host["present"])[batch_index]
    resolved = np.asarray(host["resolved"])[batch_index]
    records = []
    for slot in np.flatnonzero(present & resolved):
        records.append(
            {
                "r_m": float(host["r"][batch_index, slot]),
                "z_m": float(host["z"][batch_index, slot]),
                "psi_wb": float(host["psi"][batch_index, slot]),
                "signed_index": int(host["native_signed_index"][batch_index, slot]),
            }
        )
    return sorted(
        records,
        key=lambda row: (row["signed_index"], row["r_m"], row["z_m"], row["psi_wb"]),
    )


def _census_terminal_state(
    profile: Any, state: jax.Array, dual_sweep: bool
) -> list[dict[str, Any]]:
    operator = profile.operator
    physical = jnp.asarray(state)[: operator.physical_node_number]
    grid_flux, _wall_flux = operator.topology.split_flux_map(physical)
    radius, height, shape = operator.connectivity_grid_axes()
    radial_count, vertical_count = shape
    field = grid_flux.reshape((radial_count, vertical_count)).T
    inside = jnp.ones((vertical_count, radial_count), dtype=bool)
    result = critical_point_candidates_batch(
        field[None],
        radius,
        height,
        inside,
        k_slots=K_SLOTS,
        material_dilate=0,
        target_index=-1,
        noise_sigma=0.0,
        dual_sweep=dual_sweep,
    )
    _block_tree(result)
    return _resolved_records(result)


def _mast_correctness() -> dict[str, Any]:
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    comparisons = []
    direct_builder_entries = 0
    for selected_row, qualification in select_slices_by_shot(DECOMPOSITION_BANK):
        case, context = _mast_case_from_selection(
            SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        direct_builder_entries += int(policy["section_kernel_evaluations_this_shot"])
        reference = passive_case["reference"]
        seed = jnp.asarray(passive_case["state"])
        target_current = abs(float(reference["plasma_current_a"]))
        states = _terminal_states(profile, seed, target_current)
        for arm_name, state in states.items():
            single = _census_terminal_state(profile, state, False)
            dual = _census_terminal_state(profile, state, True)
            comparisons.append(
                {
                    "shot": int(reference["shot"]),
                    "slice_index": int(reference["slice_index"]),
                    "arm": arm_name,
                    "single_resolved_candidates": single,
                    "dual_resolved_candidates": dual,
                    "exact_match": single == dual,
                }
            )
    if direct_builder_entries != 0:
        raise RuntimeError("frozen-arm reconstruction entered the response builder")
    disagreements = sum(not row["exact_match"] for row in comparisons)
    return {
        "cohort": "six frozen MAST references with pure and mixed terminal arms",
        "arm_count": len(comparisons),
        "disagreement_count": disagreements,
        "all_exactly_equal": disagreements == 0 and len(comparisons) == 12,
        "response_carrier_direct_builder_entries": direct_builder_entries,
        "response_carrier": carrier_evidence,
        "comparisons": comparisons,
    }


def _lock_rule(
    timings: list[dict[str, Any]], correctness: dict[str, Any]
) -> dict[str, Any]:
    by_key = {
        (tuple(row["grid_size"]), row["batch_width"], row["orbit_families"]): row
        for row in timings
    }
    comparisons = []
    for size in GRID_SIZES:
        for width in BATCH_WIDTHS:
            single = by_key[((size, size), width, "rectangular")]
            dual = by_key[((size, size), width, "rectangular_and_centroid")]
            single_ms = single["warmed_execute_per_slice_ms"]["median"]
            dual_ms = dual["warmed_execute_per_slice_ms"]["median"]
            comparisons.append(
                {
                    "grid_size": [size, size],
                    "batch_width": width,
                    "single_median_per_slice_ms": single_ms,
                    "dual_median_per_slice_ms": dual_ms,
                    "dual_over_single": dual_ms / single_ms,
                }
            )
    clause_a = all(row["dual_over_single"] <= RATIO_BOUND for row in comparisons)
    coarse = [row for row in comparisons if row["grid_size"] == [33, 33]]
    clause_b = all(
        row["dual_median_per_slice_ms"] <= COARSE_PER_SLICE_BOUND_MS for row in coarse
    )
    return {
        "clause_a": {
            "description": (
                "dual median is at most twice single at both grids and every "
                "measured width"
            ),
            "bound": RATIO_BOUND,
            "passes": clause_a,
            "maximum_ratio": max(row["dual_over_single"] for row in comparisons),
        },
        "clause_b": {
            "description": (
                "dual 33x33 median is at most 1.0 ms per slice at every measured width"
            ),
            "bound_ms_per_slice": COARSE_PER_SLICE_BOUND_MS,
            "passes": clause_b,
            "maximum_dual_median_ms_per_slice": max(
                row["dual_median_per_slice_ms"] for row in coarse
            ),
        },
        "clause_c": {
            "description": (
                "the six named topology files pass with the production default "
                "temporarily set to dual_sweep=True"
            ),
            "passes": None,
            "results": "recorded after the external pytest validation pass",
        },
        "correctness_guard": {
            "description": (
                "dual and single return exactly the same merged resolved candidates "
                "on all twelve frozen MAST arms"
            ),
            "passes": correctness["all_exactly_equal"],
            "disagreement_count": correctness["disagreement_count"],
        },
        "timing_comparisons": comparisons,
        "production_decision": "pending external clause-C validation",
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    configure_dtypes()
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("the benchmark requires JAX float64")
    devices = jax.devices()
    if jax.default_backend() != "gpu" or not devices:
        raise RuntimeError("the benchmark must run on a GPU device")
    timings = [
        _measure_configuration(size, width, dual_sweep)
        for size in GRID_SIZES
        for width in BATCH_WIDTHS
        for dual_sweep in (False, True)
    ]
    correctness = _mast_correctness()
    payload = {
        "artifact": "device null-census dual-family production lock",
        "source_commit": _source_commit(),
        "runtime": {
            "platform": platform.platform(),
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in devices],
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
        "measurement_contract": {
            "device": "reserved H200 on the betelgeuse partition",
            "dual_sweep_implementation_commit": DUAL_SWEEP_IMPLEMENTATION_COMMIT,
            "dtype": "float64",
            "timing": (
                "ahead-of-time compile reported separately; one warm execution; "
                "median and IQR over execute-only synchronized calls"
            ),
            "batching": "explicit jax.vmap over independent slices",
            "grid_sizes": [[size, size] for size in GRID_SIZES],
            "batch_widths": list(BATCH_WIDTHS),
            "iter_eqdsk_exclusion": (
                "iterhybrid_cocos17.eqdsk is excluded entirely because it is a "
                "fixed-boundary scenario field carrying no divertor saddle and is "
                "not a valid topology test case"
            ),
        },
        "timings": timings,
        "mast_correctness": correctness,
    }
    payload["lock_rule"] = _lock_rule(timings, correctness)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = run(arguments.output)
    print(json.dumps(payload["lock_rule"], indent=2))


if __name__ == "__main__":
    main()
