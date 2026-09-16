"""Gate whole-solve executable size, instructions, and compilation time.

The input receipts are emitted by :mod:`benchmarks.program_scope_census`.
Sentinel copy counts remain diagnostic metrics because optimized HLO inlines
called programs.  The gate still requires each map control to see the known
operator paths, which keeps an empty or stale instrument from reporting an
improvement.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any


REQUIRED_CELLS = (300, 1000)
BASELINE_300_EXECUTABLE_BYTES = 461_724_765
MAX_300_EXECUTABLE_BYTES = 50_000_000
MAX_300_SOLVE_INSTRUCTIONS = 210_000
REPLICATION_PATHS = ("current-moment path", "topology read")
CERTIFICATE_ROWS = (
    ("weak-rotation-reactor-static", -300),
    ("moderate-rotation-conventional-static", -300),
    ("strong-rotation-compact-static", -300),
    ("diverted-single-null", -500),
)
BANKED_BOUNDARY_MS_PER_TRIP = 46.1


def _load_rungs(directory: Path) -> dict[int, dict[str, Any]]:
    """Load complete census receipts for the required cell counts."""
    rows: dict[int, dict[str, Any]] = {}
    for cells in REQUIRED_CELLS:
        path = directory / f"{cells}.json"
        if not path.exists():
            raise ValueError(f"missing census receipt {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if int(row.get("requested_cells", -1)) != cells:
            raise ValueError(f"receipt {path} does not describe {cells} cells")
        rows[cells] = row
    return rows


def _copy_count(program: dict[str, Any], path: str) -> int:
    replication = program.get("replication", {})
    if path not in replication:
        raise ValueError(f"missing replication sentinel {path!r}")
    return int(replication[path]["copy_count"])


def _effective_executable_bytes(program: dict[str, Any]) -> tuple[int, str]:
    """Prefer serialized bytes and retain generated code as a conservative fallback."""
    executable = program.get("executable")
    if not isinstance(executable, dict):
        raise ValueError("missing executable-size measurement")
    serialized = executable.get("serialized_bytes")
    if serialized is not None:
        return int(serialized), "serialized executable"
    generated = executable.get("generated_code_bytes")
    if generated is not None and int(generated) > 0:
        return int(generated), "generated code fallback"
    error = executable.get("serialization_error")
    suffix = f": {error}" if error else ""
    raise ValueError(f"executable size is absent{suffix}")


def _compile_seconds(row: dict[str, Any], cells: int, label: str) -> float:
    value = float(row.get("compile_seconds", 0.0))
    if value <= 0.0:
        raise ValueError(f"{cells}-cell {label} compile time is absent")
    return value


def evaluate_gate(
    baseline: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
    *,
    baseline_300_executable_bytes: int = BASELINE_300_EXECUTABLE_BYTES,
) -> dict[str, Any]:
    """Compare before and after receipts and return a quantitative verdict."""
    rows = []
    failures: list[str] = []
    findings: list[str] = []
    for cells in REQUIRED_CELLS:
        before = baseline[cells]
        after = candidate[cells]
        map_counts = {
            path: _copy_count(after["map"], path) for path in REPLICATION_PATHS
        }
        for path, count in map_counts.items():
            if count <= 0:
                failures.append(
                    f"{cells}-cell map sentinel {path!r} saw no known-present path"
                )
        solve_counts = {
            path: _copy_count(after["solve"], path) for path in REPLICATION_PATHS
        }
        before_instructions = int(before["solve"]["total_instructions"])
        after_instructions = int(after["solve"]["total_instructions"])
        if after_instructions >= before_instructions:
            failures.append(
                f"{cells}-cell solve instructions did not shrink: "
                f"{before_instructions} before, {after_instructions} after"
            )
        try:
            before_compile_seconds = _compile_seconds(before, cells, "baseline")
            after_compile_seconds = _compile_seconds(after, cells, "candidate")
        except ValueError as error:
            before_compile_seconds = before.get("compile_seconds")
            after_compile_seconds = after.get("compile_seconds")
            failures.append(str(error))
        else:
            if after_compile_seconds >= before_compile_seconds:
                failures.append(
                    f"{cells}-cell compile time did not shrink: "
                    f"{before_compile_seconds:.3f}s before, "
                    f"{after_compile_seconds:.3f}s after"
                )
        try:
            executable_bytes, executable_measure = _effective_executable_bytes(
                after["solve"]
            )
        except ValueError as error:
            executable_bytes = None
            executable_measure = "unavailable"
            if cells == 300:
                failures.append(f"{cells}-cell solve {error}")
            else:
                findings.append(f"{cells}-cell solve {error}")
        before_executable_bytes = (
            baseline_300_executable_bytes if cells == 300 else None
        )
        if cells == 300 and executable_bytes is not None:
            if executable_bytes >= baseline_300_executable_bytes:
                failures.append(
                    "300-cell solve executable did not shrink: "
                    f"{baseline_300_executable_bytes} bytes before, "
                    f"{executable_bytes} after"
                )
            if executable_bytes > MAX_300_EXECUTABLE_BYTES:
                failures.append(
                    f"300-cell solve executable is {executable_bytes} bytes, "
                    f"limit {MAX_300_EXECUTABLE_BYTES}"
                )
            if after_instructions > MAX_300_SOLVE_INSTRUCTIONS:
                failures.append(
                    f"300-cell solve has {after_instructions} instructions, "
                    f"limit {MAX_300_SOLVE_INSTRUCTIONS}"
                )
        rows.append(
            {
                "requested_cells": cells,
                "before": {
                    "solve_instructions": before_instructions,
                    "map_instructions": int(before["map"]["total_instructions"]),
                    "compile_seconds": before_compile_seconds,
                    "executable_bytes": before_executable_bytes,
                    "operator_copies": {
                        path: _copy_count(before["solve"], path)
                        for path in REPLICATION_PATHS
                    },
                },
                "after": {
                    "solve_instructions": after_instructions,
                    "map_instructions": int(after["map"]["total_instructions"]),
                    "compile_seconds": after_compile_seconds,
                    "instruction_ratio": (
                        float(after["solve"]["total_instructions"])
                        / float(after["map"]["total_instructions"])
                    ),
                    "operator_copies": solve_counts,
                    "map_operator_copies": map_counts,
                    "executable_bytes": executable_bytes,
                    "executable_measure": executable_measure,
                },
            }
        )
    return {
        "passed": not failures,
        "failures": failures,
        "findings": findings,
        "limits": {
            "300_cell_solve_instructions": MAX_300_SOLVE_INSTRUCTIONS,
            "300_cell_executable_bytes": MAX_300_EXECUTABLE_BYTES,
        },
        "rows": rows,
    }


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# Solve program size gate",
        "",
        "| cells | solve instructions before / after | map floor before / after | "
        "compile seconds before / after | executable bytes before / after | "
        "solve copies moment / topology | map copies moment / topology |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        before = row["before"]
        after = row["after"]
        executable_bytes = after["executable_bytes"]
        executable_display = (
            f"{executable_bytes:,}" if executable_bytes is not None else "unavailable"
        )
        before_executable = before["executable_bytes"]
        before_executable_display = (
            f"{before_executable:,}" if before_executable is not None else "unavailable"
        )
        before_compile = before["compile_seconds"]
        after_compile = after["compile_seconds"]
        compile_display = (
            f"{float(before_compile):.3f} / {float(after_compile):.3f}"
            if before_compile is not None and after_compile is not None
            else "unavailable"
        )
        lines.append(
            f"| {row['requested_cells']} | "
            f"{before['solve_instructions']:,} / {after['solve_instructions']:,} | "
            f"{before['map_instructions']:,} / {after['map_instructions']:,} | "
            f"{compile_display} | "
            f"{before_executable_display} / {executable_display} | "
            f"{after['operator_copies']['current-moment path']} / "
            f"{after['operator_copies']['topology read']} | "
            f"{after['map_operator_copies']['current-moment path']} / "
            f"{after['map_operator_copies']['topology read']} |"
        )
    lines.extend(["", f"Verdict: **{'PASS' if result['passed'] else 'FAIL'}**."])
    if result["failures"]:
        lines.extend(["", "Refusals:"])
        lines.extend(f"- {failure}" for failure in result["failures"])
    if result["findings"]:
        lines.extend(["", "Measured findings:"])
        lines.extend(f"- {finding}" for finding in result["findings"])
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def strict(value):
        if isinstance(value, dict):
            return {key: strict(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [strict(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _certificate_operands(case_name: str, requested_cells: int):
    """Build the exact production certificate operands for one committed row."""
    import numpy as np

    from benchmarks import solovev_certificate as certificate
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.stencil_mesh import StencilMesh

    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(case_name, exact, coordinates)
    empty_operator = certificate.oracle_fixture.forward_operator(source_case, machine)
    exact_physical, fixture_exterior, _fixture_cache = (
        certificate.oracle_fixture.cached_fixture_exterior(
            source_case, exact, machine, empty_operator, oracle_state
        )
    )
    operator = certificate.oracle_fixture.forward_operator(
        source_case, machine, fixture_exterior
    )
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=certificate.recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        case_name, source_case, operator, exact_physical
    )
    seed, requested_class, _seed_receipt = certificate._production_seed(
        profile, case_name, target_current, centroid, current_receipt
    )
    request = certificate._certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"solovev:{case_name}:{requested_cells}",
    )
    return profile, seed, requested_class, target_current, request


class EmptyIdentitySetError(ValueError):
    """A certificate identity comparison carries no rows to compare.

    An empty identity set makes the comparison vacuous: ``array_equal`` over two
    empty arrays is true and a maximum over an empty difference defaults to zero,
    so a row that compared nothing reads as a machine-precision match.
    """


def _require_identity_rows(identity_row_count: int, label: str) -> int:
    """Refuse a certificate identity comparison that carries no rows.

    A row whose identity set is empty reports a bit-identical state and a zero
    difference over nothing, which is indistinguishable from a genuine
    machine-precision match, so the refusal fires before either is read as a
    within-drift result.  The returned count is the denominator the comparison
    states.
    """
    count = int(identity_row_count)
    if count <= 0:
        raise EmptyIdentitySetError(
            f"identity set for {label} carries {count} rows; a comparison over "
            "no rows cannot read as within drift"
        )
    return count


def _certificate_identity_row(case_name: str, requested_cells: int) -> dict[str, Any]:
    """Compare the pre-wrapper and frozen-partition terminal states exactly."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from nova.equilibrium import fixed_point

    profile, seed, requested_class, target_current, request = _certificate_operands(
        case_name, requested_cells
    )
    state = jnp.asarray(seed)
    external = profile.operator.external()
    mapped = profile.operator.traced_flux_map(requested_class, target_current)
    shadowed = profile.operator.traced_flux_map_with_shadow(
        requested_class, target_current
    )

    def shadow_mask(value, operator):
        return operator.residual_shadow_mask(value, requested_class)

    def promoted_shadow_mask(value, previous, operator):
        return operator.residual_shadow_mask(
            value, requested_class, previous_shadow=previous
        )

    options = request.policy.kernel_options()

    def baseline_solve(initial, exterior):
        return fixed_point.newton_krylov(
            mapped,
            initial,
            shadow_mask_fn=shadow_mask,
            promoted_shadow_mask_fn=promoted_shadow_mask,
            shadowed_map_fn=shadowed,
            map_arguments=(exterior, profile.operator),
            callback_arguments=(profile.operator,),
            **options,
        )

    baseline_program = jax.jit(baseline_solve)
    baseline_started = time.perf_counter()
    baseline = baseline_program(state, external)
    jax.block_until_ready(baseline.state)
    baseline_seconds = time.perf_counter() - baseline_started
    candidate_program = profile._accelerated_history_program(
        "newton_krylov",
        requested_class=requested_class,
        target_current=target_current,
        **options,
    )
    candidate_started = time.perf_counter()
    candidate = candidate_program(state, external, profile.operator)
    jax.block_until_ready(candidate.state)
    candidate_seconds = time.perf_counter() - candidate_started
    baseline_state = np.asarray(baseline.state, dtype=np.float64)
    candidate_state = np.asarray(candidate.state, dtype=np.float64)
    identity_row_count = _require_identity_rows(
        baseline_state.size, f"solovev:{case_name}:{requested_cells}"
    )
    baseline_hash = hashlib.sha256(baseline_state.tobytes()).hexdigest()
    candidate_hash = hashlib.sha256(candidate_state.tobytes()).hexdigest()
    baseline_state_finite = bool(np.all(np.isfinite(baseline_state)))
    candidate_state_finite = bool(np.all(np.isfinite(candidate_state)))
    difference = np.abs(candidate_state - baseline_state)
    max_absolute_difference = (
        float(np.max(difference, initial=0.0))
        if bool(np.all(np.isfinite(difference)))
        else None
    )
    baseline_residual = float(baseline.residual)
    candidate_residual = float(candidate.residual)
    return {
        "case": case_name,
        "requested_cells": requested_cells,
        "identity_row_count": identity_row_count,
        "realised_state_values": identity_row_count,
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds,
        "baseline_state_sha256_binary64": baseline_hash,
        "candidate_state_sha256_binary64": candidate_hash,
        "baseline_state_finite": baseline_state_finite,
        "candidate_state_finite": candidate_state_finite,
        "terminal_state_bit_identical": bool(
            np.array_equal(candidate_state, baseline_state)
        ),
        "maximum_absolute_state_difference": max_absolute_difference,
        "baseline_terminal_residual": (
            baseline_residual if math.isfinite(baseline_residual) else None
        ),
        "candidate_terminal_residual": (
            candidate_residual if math.isfinite(candidate_residual) else None
        ),
        "baseline_terminal_residual_finite": math.isfinite(baseline_residual),
        "candidate_terminal_residual_finite": math.isfinite(candidate_residual),
        "converged_equal": bool(
            np.asarray(candidate.converged).item()
            == np.asarray(baseline.converged).item()
        ),
    }


def run_certificate_identity(output: Path, cache_root: Path | None) -> dict[str, Any]:
    """Persist the four certificate identity rows as each comparison lands."""
    identity_row_count = _require_identity_rows(
        len(CERTIFICATE_ROWS), "certificate identity rows"
    )
    import jax

    from benchmarks.trip_quantum_width_one import _require_revision
    from nova.jax.config import (
        configure_dtypes,
        configure_persistent_compilation_cache,
        default_persistent_compilation_cache_root,
    )

    configure_dtypes()
    if os.environ.get("SLURM_JOB_ID") is None:
        raise RuntimeError("certificate identity requires a SLURM allocation")
    receipt: dict[str, Any] = {
        "schema": "nova.solve-program-certificate-identity",
        "identity_row_count": identity_row_count,
        "measurement_revision": _require_revision(),
        "captured_at": datetime.now(UTC).isoformat(),
        "assignment": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node": os.environ.get("SLURMD_NODENAME")
            or os.environ.get("SLURM_JOB_NODELIST"),
            "platform": jax.default_backend(),
        },
        "persistent_compilation_cache": configure_persistent_compilation_cache(
            cache_root or default_persistent_compilation_cache_root(),
            minimum_compile_seconds=0.0,
        ).receipt(),
        "comparison": (
            "pre-wrapper traced map against the frozen-partition accelerated program"
        ),
        "rows": [],
        "passed": None,
    }
    _write_json(output, receipt)
    for case_name, requested_cells in CERTIFICATE_ROWS:
        print(f"CERTIFICATE_START case={case_name} cells={requested_cells}", flush=True)
        row = _certificate_identity_row(case_name, requested_cells)
        receipt["rows"].append(row)
        _write_json(output, receipt)
        print(
            f"CERTIFICATE_DONE case={case_name} cells={requested_cells} "
            f"bit_identical={int(row['terminal_state_bit_identical'])}",
            flush=True,
        )
    receipt["passed"] = len(receipt["rows"]) == identity_row_count and all(
        row["terminal_state_bit_identical"] and row["converged_equal"]
        for row in receipt["rows"]
    )
    _write_json(output, receipt)
    return receipt


def measure_300_program(output: Path, cache_root: Path | None) -> dict[str, Any]:
    """Compile the explicit-operator certificate program and gate its byte size."""
    import jax
    import jax.numpy as jnp

    from benchmarks.trip_quantum_width_one import _require_revision
    from nova.jax.config import (
        configure_dtypes,
        configure_persistent_compilation_cache,
        default_persistent_compilation_cache_root,
    )

    configure_dtypes()
    if os.environ.get("SLURM_JOB_ID") is None:
        raise RuntimeError("program-size measurement requires a SLURM allocation")
    profile, seed, requested_class, target_current, request = _certificate_operands(
        CERTIFICATE_ROWS[0][0], -300
    )
    external = profile.operator.external(request.current, request.prescribed_current)
    program = profile._accelerated_history_program(
        request.route,
        requested_class=requested_class,
        target_current=target_current,
        **request.policy.kernel_options(),
    )
    receipt: dict[str, Any] = {
        "schema": "nova.solve-program-size",
        "measurement_revision": _require_revision(),
        "captured_at": datetime.now(UTC).isoformat(),
        "assignment": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node": os.environ.get("SLURMD_NODENAME")
            or os.environ.get("SLURM_JOB_NODELIST"),
            "platform": jax.default_backend(),
        },
        "requested_cells": 300,
        "baseline_executable_bytes": BASELINE_300_EXECUTABLE_BYTES,
        "limit_executable_bytes": MAX_300_EXECUTABLE_BYTES,
        "completed": False,
        "checkpoints": [],
    }
    _write_json(output, receipt)
    cache = configure_persistent_compilation_cache(
        cache_root or default_persistent_compilation_cache_root(),
        minimum_compile_seconds=0.0,
    )
    started = time.perf_counter()
    lowered = program.lower(
        jnp.asarray(seed, dtype=jnp.float64), external, profile.operator
    )
    receipt["checkpoints"].append(
        {
            "name": "lowered",
            "seconds": time.perf_counter() - started,
            "stablehlo_sha256": hashlib.sha256(
                lowered.as_text(dialect="stablehlo").encode()
            ).hexdigest(),
        }
    )
    receipt["persistent_compilation_cache"] = cache.receipt()
    _write_json(output, receipt)
    compile_started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started
    runtime = compiled.runtime_executable()
    serialized_bytes = None
    serialization_error = None
    try:
        serialized_bytes = len(runtime.serialize())
    except (MemoryError, RuntimeError, ValueError) as error:
        serialization_error = f"{type(error).__name__}: {error}"
    generated = getattr(runtime, "size_of_generated_code_in_bytes", None)
    generated_code_bytes = generated() if callable(generated) else generated
    generated_code_bytes = (
        None if generated_code_bytes is None else int(generated_code_bytes)
    )
    effective_bytes = (
        serialized_bytes if serialized_bytes is not None else generated_code_bytes
    )
    receipt.update(
        {
            "compile_seconds": compile_seconds,
            "serialized_executable_bytes": serialized_bytes,
            "generated_code_bytes": generated_code_bytes,
            "serialization_error": serialization_error,
            "effective_executable_bytes": effective_bytes,
            "completed": True,
            "passed": effective_bytes is not None
            and effective_bytes < MAX_300_EXECUTABLE_BYTES,
        }
    )
    _write_json(output, receipt)
    print(
        "SOLVE_PROGRAM_SIZE_300 "
        f"effective_bytes={effective_bytes} "
        f"generated_code_bytes={generated_code_bytes} "
        f"compile_seconds={compile_seconds:.3f} "
        f"verdict={'PASS' if receipt['passed'] else 'FAIL'}",
        flush=True,
    )
    return receipt


def run_mast_identity(
    output: Path, dispatch_path: Path, cache_root: Path | None
) -> dict[str, Any]:
    """Run the twelve MAST members through the current compiled-slice signature."""
    import jax
    import jax.numpy as jnp

    from benchmarks.compiled_slice_cache_receipt import _host, _solve, _ulp_distance
    from benchmarks.trip_quantum_width_one import (
        _build_members,
        _require_allocation,
        _require_revision,
    )
    from nova.equilibrium import reduced_newton
    from nova.equilibrium.topology import TopologyClass
    from nova.jax.config import (
        configure_dtypes,
        configure_persistent_compilation_cache,
        default_persistent_compilation_cache_root,
    )

    configure_dtypes()
    dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
    references = {
        str(row["identity"]): float(
            row["compiled"]["program_dispatch_wall_per_solve_s"]
        )
        for row in dispatch["width_one"]["members"]
    }
    receipt: dict[str, Any] = {
        "schema": "nova.solve-program-mast-identity",
        "measurement_revision": _require_revision(),
        "captured_at": datetime.now(UTC).isoformat(),
        "assignment": _require_allocation(),
        "persistent_compilation_cache": configure_persistent_compilation_cache(
            cache_root or default_persistent_compilation_cache_root(),
            minimum_compile_seconds=0.0,
        ).receipt(),
        "dispatch_reference": str(dispatch_path),
        "members": [],
        "verdict": None,
    }
    _write_json(output, receipt)
    members, inputs = _build_members()
    receipt["inputs"] = inputs
    _write_json(output, receipt)
    reduced_newton._compiled_program_cache.clear()
    for number, member in enumerate(members, start=1):
        print(f"MAST_START member={number} identity={member.identity}", flush=True)
        cold_started = time.perf_counter()
        first = _solve(member)
        cold_seconds = time.perf_counter() - cold_started
        state = jnp.asarray(member.state)
        requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
        shadow = jnp.ravel(
            jnp.asarray(
                member.operator.residual_shadow_mask(state, requested), dtype=bool
            )
        )
        external = member.operator.external()
        jax.block_until_ready(
            first.program.slice_solver(state, shadow, external, member.operator)
        )
        second = _solve(member)
        direct_started = time.perf_counter()
        direct = first.program.slice_solver(state, shadow, external, member.operator)
        jax.block_until_ready(direct)
        direct_seconds = time.perf_counter() - direct_started
        warm_started = time.perf_counter()
        warm = _solve(member)
        warm_seconds = time.perf_counter() - warm_started
        host = _host(member)
        trips = int(jax.device_get(direct)[7])
        host_ulp = _ulp_distance(warm.state, host.state)
        direct_ulp = _ulp_distance(warm.state, direct[0])
        row = {
            "identity": member.identity,
            "trips": trips,
            "cold_public_wall_s": cold_seconds,
            "warm_public_wall_s": warm_seconds,
            "dispatch_reference_wall_s": references[member.identity],
            "same_job_direct_wall_s": direct_seconds,
            "same_cached_program": second.program is first.program
            and warm.program is first.program,
            "compiled_host_terminal_flux_ulp": host_ulp,
            "cached_direct_terminal_flux_ulp": direct_ulp,
            "terminal_flux_bit_identical": host_ulp == 0 and direct_ulp == 0,
        }
        receipt["members"].append(row)
        _write_json(output, receipt)
        print(
            f"MAST_DONE member={number} identity={member.identity} "
            f"ulp={host_ulp} direct_s={direct_seconds:.6f}",
            flush=True,
        )
    receipt["verdict"] = {
        "member_count": len(receipt["members"]),
        "cached_program_reuse": all(
            row["same_cached_program"] for row in receipt["members"]
        ),
        "terminal_flux_bit_identical": all(
            row["terminal_flux_bit_identical"] for row in receipt["members"]
        ),
    }
    _write_json(output, receipt)
    return receipt


def write_semantic_report(
    certificate_path: Path,
    mast_path: Path,
    dispatch_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Join certificate identity and MAST timing into one reviewable receipt."""
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    mast = json.loads(mast_path.read_text(encoding="utf-8"))
    dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
    reference_trip_counts = {
        str(row["identity"]): int(row["compiled"]["program_dispatch_trips"])
        for row in dispatch["width_one"]["members"]
    }
    timing_rows = []
    for row in mast["members"]:
        identity = str(row["identity"])
        trips = int(row.get("trips", reference_trip_counts[identity]))
        timing_rows.append(
            {
                "identity": identity,
                "trips": trips,
                "banked_boundary_ms_per_trip": BANKED_BOUNDARY_MS_PER_TRIP,
                "before_compiled_boundary_ms_per_trip": (
                    1.0e3 * float(row["dispatch_reference_wall_s"]) / trips
                ),
                "after_compiled_boundary_ms_per_trip": (
                    1.0e3 * float(row["same_job_direct_wall_s"]) / trips
                ),
                "compiled_host_terminal_flux_ulp": int(
                    row["compiled_host_terminal_flux_ulp"]
                ),
                "terminal_flux_bit_identical": int(
                    row["compiled_host_terminal_flux_ulp"]
                )
                == 0,
            }
        )
    result = {
        "schema": "nova.solve-program-semantic-gate",
        "certificate": certificate,
        "mast_assignment": mast["assignment"],
        "mast_measurement_revision": mast["measurement_revision"],
        "mast_rows": timing_rows,
        "passed": bool(certificate["passed"])
        and len(timing_rows) == 12
        and all(row["terminal_flux_bit_identical"] for row in timing_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "receipt.json", result)
    lines = [
        "# Solve program semantic and boundary gate",
        "",
        "## Certificate terminal-state identity",
        "",
        "| case | cells | state values | baseline / candidate seconds | "
        "bit identical |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in certificate["rows"]:
        lines.append(
            f"| {row['case']} | {abs(int(row['requested_cells']))} | "
            f"{int(row['realised_state_values']):,} | "
            f"{float(row['baseline_seconds']):.3f} / "
            f"{float(row['candidate_seconds']):.3f} | "
            f"{'yes' if row['terminal_state_bit_identical'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## MAST compiled-slice boundary and terminal identity",
            "",
            "| member | trips | banked boundary [ms/trip] | before compiled "
            "[ms/trip] | after compiled [ms/trip] | host difference [ULP] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in timing_rows:
        lines.append(
            f"| {row['identity']} | {row['trips']} | "
            f"{row['banked_boundary_ms_per_trip']:.1f} | "
            f"{row['before_compiled_boundary_ms_per_trip']:.3f} | "
            f"{row['after_compiled_boundary_ms_per_trip']:.3f} | "
            f"{row['compiled_host_terminal_flux_ulp']} |"
        )
    lines.extend(["", f"Verdict: **{'PASS' if result['passed'] else 'FAIL'}**."])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--certificate-output", type=Path)
    parser.add_argument("--measure-300-output", type=Path)
    parser.add_argument("--mast-output", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--semantic-report", action="store_true")
    parser.add_argument("--certificate-receipt", type=Path)
    parser.add_argument("--mast-receipt", type=Path)
    parser.add_argument("--dispatch-receipt", type=Path)
    parser.add_argument(
        "--baseline-300-executable-bytes",
        type=int,
        default=BASELINE_300_EXECUTABLE_BYTES,
        help="recorded baseline executable bytes for the 300-cell comparison",
    )
    args = parser.parse_args()
    if args.measure_300_output is not None:
        result = measure_300_program(args.measure_300_output, args.cache_root)
        return 0 if result["passed"] else 1
    if args.certificate_output is not None:
        result = run_certificate_identity(args.certificate_output, args.cache_root)
        print(
            f"CERTIFICATE_IDENTITY_GATE={'PASS' if result['passed'] else 'FAIL'}",
            flush=True,
        )
        return 0 if result["passed"] else 1
    if args.mast_output is not None:
        if args.dispatch_receipt is None:
            parser.error("MAST identity gate requires dispatch-receipt")
        result = run_mast_identity(
            args.mast_output, args.dispatch_receipt, args.cache_root
        )
        passed = (
            result["verdict"]["member_count"] == 12
            and result["verdict"]["cached_program_reuse"]
            and result["verdict"]["terminal_flux_bit_identical"]
        )
        print(f"MAST_IDENTITY_GATE={'PASS' if passed else 'FAIL'}", flush=True)
        return 0 if passed else 1
    if args.semantic_report:
        required = (
            args.certificate_receipt,
            args.mast_receipt,
            args.dispatch_receipt,
            args.output_dir,
        )
        if any(path is None for path in required):
            parser.error("semantic report requires all three receipts and output-dir")
        result = write_semantic_report(
            args.certificate_receipt,
            args.mast_receipt,
            args.dispatch_receipt,
            args.output_dir,
        )
        print(f"SEMANTIC_GATE={'PASS' if result['passed'] else 'FAIL'}")
        return 0 if result["passed"] else 1
    if (
        args.baseline_dir is None
        or args.candidate_dir is None
        or args.output_dir is None
    ):
        parser.error("size gate requires baseline-dir, candidate-dir, and output-dir")
    result = evaluate_gate(
        _load_rungs(args.baseline_dir),
        _load_rungs(args.candidate_dir),
        baseline_300_executable_bytes=args.baseline_300_executable_bytes,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(_report(result), encoding="utf-8")
    print(f"SOLVE_PROGRAM_SIZE_GATE={'PASS' if result['passed'] else 'FAIL'}")
    for failure in result["failures"]:
        print(f"REFUSAL {failure}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
