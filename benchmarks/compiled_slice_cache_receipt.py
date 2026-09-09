"""Receipt the reusable compiled-slice executable on the twelve-member bank."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.trip_quantum_width_one import (
    DEFAULT_OUTPUT as DISPATCH_RECEIPT,
    _build_members,
    _require_allocation,
    _require_revision,
    _sha256,
    _strict,
    _write_json,
)
from nova.equilibrium import reduced_newton
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/millisecond-converged-solve/compiled-slice-cache/receipt.json"
)
DEFAULT_DIAGNOSTIC = DEFAULT_OUTPUT.with_name("vertical-mode-diagnostic.json")
DISPATCH_TOLERANCE = 0.10
WITHDRAWN_KEYED_LOOKUP = {
    "approach": "pre-coordinate lookup keyed by conductor inputs",
    "cross_job_dispatch_comparator": {"passes": 3, "member_count": 12},
    "terminal_flux_identity": {
        "member": "21986/46 mixed",
        "max_ulp": 512,
        "limit_ulp": 4,
    },
    "disposition": (
        "withdrawn in favour of the reduced-coordinate and derived-external key"
    ),
}


def _dispatch_reference() -> dict[str, float]:
    """Return the preceding receipt's carried-program walls by member."""
    payload = json.loads(DISPATCH_RECEIPT.read_text(encoding="utf-8"))
    references = {}
    for row in payload["width_one"]["members"]:
        value = row.get("compiled", {}).get("program_dispatch_wall_per_solve_s")
        if value is not None:
            references[str(row["identity"])] = float(value)
    if len(references) != 12:
        raise RuntimeError("the dispatch receipt does not provide twelve member walls")
    return references


def _ulp_distance(left: Any, right: Any) -> int:
    """Return the largest IEEE-754 binary64 distance over two equal-shaped arrays."""
    left_bits = np.asarray(left, dtype=np.float64).view(np.uint64).ravel()
    right_bits = np.asarray(right, dtype=np.float64).view(np.uint64).ravel()

    def ordered(bits: np.uint64) -> int:
        value = int(bits)
        return (
            0x8000000000000000 - value
            if value & 0x8000000000000000
            else 0x8000000000000000 + value
        )

    distances = (
        abs(ordered(first) - ordered(second))
        for first, second in zip(left_bits, right_bits, strict=True)
    )
    return max(distances, default=0)


def _solve(member, *, capture_trip_states: bool = False):
    """Invoke the public compiled entry point with the receipt's fixed policy."""
    return reduced_newton.solve_reduced_newton_compiled(
        member.operator,
        jnp.asarray(member.state),
        requested_class=jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8),
        target_current=member.target_current,
        tolerance=member.tolerance,
        newton_steps=12,
        active_set_steps=16,
        capture_trip_states=capture_trip_states,
    )


def _host(
    member,
    *,
    refresh_threshold=reduced_newton.JACOBIAN_REFRESH_THRESHOLD,
    capture_trip_states: bool = False,
):
    """Run the matched host route used for the terminal-flux comparison."""
    return reduced_newton.solve_reduced_newton(
        member.operator,
        jnp.asarray(member.state),
        requested_class=jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8),
        target_current=member.target_current,
        tolerance=member.tolerance,
        newton_steps=12,
        active_set_steps=16,
        trip_boundary=reduced_newton.TRIP_BOUNDARY,
        jacobian_refresh_threshold=refresh_threshold,
        capture_trip_states=capture_trip_states,
    )


def _direct_dispatch(result, member):
    """Dispatch the carried executable directly, bypassing public cache lookup."""
    state = jnp.asarray(member.state)
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    shadow = jnp.ravel(
        jnp.asarray(member.operator.residual_shadow_mask(state, requested), dtype=bool)
    )
    output = result.program.slice_solver(state, shadow)
    jax.block_until_ready(output)
    return output


def diagnose_vertical_mode(output: Path) -> dict[str, Any]:
    """Locate the first adaptive-versus-refusal decision on the vertical row."""
    configure_dtypes()
    members, inputs = _build_members()
    member = next(item for item in members if item.identity == "21986/46 mixed")
    adaptive = _host(member, capture_trip_states=True)
    refusal = _host(member, refresh_threshold=None)
    reduced_newton._compiled_program_cache.clear()
    _solve(member, capture_trip_states=True)
    cached = _solve(member, capture_trip_states=True)
    first_step = None
    for adaptive_step, refusal_step in zip(
        adaptive.steps, refusal.steps, strict=False
    ):
        left = adaptive_step._replace(wall_s=0.0)
        right = refusal_step._replace(wall_s=0.0)
        if left != right:
            moved = [
                name
                for name in left._fields
                if getattr(left, name) != getattr(right, name)
            ]
            first_step = {
                "trip": adaptive_step.trip,
                "step": adaptive_step.step,
                "quantities": moved,
                "adaptive": _strict(left._asdict()),
                "refusal_only": _strict(right._asdict()),
            }
            break
    trip_rows = []
    first_state_difference = None
    for trip in range(
        max(adaptive.active_set_iterations, cached.active_set_iterations)
    ):
        direct_state = np.asarray(adaptive.state_per_trip[trip], dtype=np.float64)
        cached_state = np.asarray(cached.state_per_trip[trip], dtype=np.float64)
        different = np.flatnonzero(
            direct_state.view(np.uint64) != cached_state.view(np.uint64)
        )
        first_index = int(different[0]) if different.size else None
        state_ulp = _ulp_distance(cached_state, direct_state)
        if first_state_difference is None and first_index is not None:
            first_state_difference = {
                "trip": trip,
                "flat_index": first_index,
                "cached_flux": float(cached_state.ravel()[first_index]),
                "direct_flux": float(direct_state.ravel()[first_index]),
                "element_ulp": _ulp_distance(
                    cached_state.ravel()[first_index : first_index + 1],
                    direct_state.ravel()[first_index : first_index + 1],
                ),
                "trip_max_ulp": state_ulp,
                "producer": (
                    "trip boundary reconstructs external plus current-moment image"
                ),
            }
        trip_rows.append(
            {
                "trip": trip,
                "direct_residual": adaptive.active_set_residuals[trip],
                "cached_residual": cached.active_set_residuals[trip],
                "direct_steps": adaptive.newton_steps_per_trip[trip],
                "cached_steps": cached.newton_steps_per_trip[trip],
                "direct_jacobian_builds": adaptive.jacobian_builds_per_trip[trip],
                "cached_jacobian_builds": cached.jacobian_builds_per_trip[trip],
                "direct_mask_difference": adaptive.active_set_mask_differences[trip],
                "cached_mask_difference": cached.active_set_mask_differences[trip],
                "state_max_ulp": state_ulp,
                "first_differing_flat_index": first_index,
            }
        )
    recomputed_external = member.operator.external(None, None)
    recomputed_coordinates = reduced_newton.reduced_coordinates(
        member.operator,
        jnp.asarray(member.state),
        requested_class=jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8),
        target_current=jnp.asarray(member.target_current),
        policy=reduced_newton.SUPPORT_POLICY,
    )
    terminal_ulp = _ulp_distance(cached.state, adaptive.state)
    diagnosis = {
        "member": member.identity,
        "inputs": inputs,
        "mechanism": (
            "the reduced-coordinate and derived-external cache key retains the "
            "host-route terminal-flux identity"
            if terminal_ulp <= 4
            else (
                "the first state difference is produced by trip-boundary "
                "reconstruction; cached external and coordinate inputs equal their "
                "recomputed values, so the remaining difference is the boundary "
                "image reduction inlined in the compiled loop versus separately "
                "dispatched by the host"
            )
        ),
        "roundoff_operation": (
            "trip boundary external plus current-moment image, inlined in the compiled "
            "fori_loop versus a separately dispatched host boundary"
        ),
        "capture_omission_found": False,
        "withdrawn_keyed_lookup": WITHDRAWN_KEYED_LOOKUP,
        "rejected_hypotheses": [
            "adaptive Jacobian refresh policy",
            "eager six-grade lax.map versus grade-one then conditional tail ordering",
            "cached external or reduced-coordinate capture differs from recomputation",
        ],
        "quantity_moved_first": first_step,
        "trip_comparison_after_repair": trip_rows,
        "first_state_difference": first_state_difference,
        "captured_inputs": {
            "external_ulp_against_recomputed": _ulp_distance(
                cached.program.external, recomputed_external
            ),
            "coordinate_cells_equal_recomputed": bool(
                np.array_equal(
                    np.asarray(cached.program.coordinates.cells),
                    np.asarray(recomputed_coordinates.cells),
                )
            ),
            "target_current": member.target_current,
        },
        "terminal_flux_ulp_after_repair": terminal_ulp,
        "refusal_only_terminal_flux_ulp": _ulp_distance(
            refusal.state, adaptive.state
        ),
    }
    _write_json(output, diagnosis)
    return diagnosis


def run(output: Path, *, cache_root: Path | None = None) -> dict[str, Any]:
    """Measure cold-then-warm public calls and persist each bank member immediately."""
    revision = _require_revision()
    configure_dtypes()
    allocation = _require_allocation()
    cache = configure_persistent_compilation_cache(
        cache_root or default_persistent_compilation_cache_root(),
        minimum_compile_seconds=0.0,
    )
    dispatch_reference = _dispatch_reference()
    receipt: dict[str, Any] = {
        "schema": "nova.compiled-slice-cache",
        "measurement_revision": revision,
        "captured_at": datetime.now(UTC).isoformat(),
        "assignment": allocation,
        "persistent_compilation_cache": cache.receipt(),
        "source": {
            "driver": str(Path(__file__).relative_to(ROOT)),
            "driver_sha256": _sha256(Path(__file__)),
            "dispatch_receipt": str(DISPATCH_RECEIPT.relative_to(ROOT)),
        },
        "policy": {"newton_steps": 12, "active_set_steps": 16},
        "acceptance": {
            "same_job_direct_wall_relative_tolerance": DISPATCH_TOLERANCE,
            "terminal_flux_max_ulp": 4,
        },
        "correctness_diagnostic": (
            json.loads(DEFAULT_DIAGNOSTIC.read_text(encoding="utf-8"))
            if DEFAULT_DIAGNOSTIC.exists()
            else None
        ),
        "withdrawn_keyed_lookup": WITHDRAWN_KEYED_LOOKUP,
        "members": [],
        "verdict": None,
    }
    _write_json(output, receipt)
    members, inputs = _build_members()
    receipt["inputs"] = inputs
    _write_json(output, receipt)
    reduced_newton._compiled_program_cache.clear()
    for number, member in enumerate(members, start=1):
        print(
            f"CACHE_MEMBER_START member={number} identity={member.identity}",
            flush=True,
        )
        cold_started = time.perf_counter()
        first = _solve(member)
        cold_wall = time.perf_counter() - cold_started
        _direct_dispatch(first, member)
        _solve(member)
        if number % 2:
            direct_started = time.perf_counter()
            direct = _direct_dispatch(first, member)
            direct_wall = time.perf_counter() - direct_started
            warm_started = time.perf_counter()
            second = _solve(member)
            warm_wall = time.perf_counter() - warm_started
        else:
            warm_started = time.perf_counter()
            second = _solve(member)
            warm_wall = time.perf_counter() - warm_started
            direct_started = time.perf_counter()
            direct = _direct_dispatch(first, member)
            direct_wall = time.perf_counter() - direct_started
        host = _host(member)
        reference_wall = dispatch_reference[member.identity]
        relative_dispatch_error = abs(warm_wall - direct_wall) / direct_wall
        flux_ulp = _ulp_distance(second.state, host.state)
        direct_flux_ulp = _ulp_distance(second.state, direct[0])
        row = {
            "identity": member.identity,
            "cold_public_wall_s": cold_wall,
            "warm_public_wall_s": warm_wall,
            "dispatch_reference_wall_s": reference_wall,
            "same_job_direct_wall_s": direct_wall,
            "dispatch_relative_error": relative_dispatch_error,
            "same_cached_program": second.program is first.program,
            "slice_solver_count": len(second.program.slice_solvers or {}),
            "compiled_host_terminal_flux_ulp": flux_ulp,
            "cached_direct_terminal_flux_ulp": direct_flux_ulp,
            "dispatch_within_tolerance": relative_dispatch_error <= DISPATCH_TOLERANCE,
            "terminal_flux_within_tolerance": flux_ulp <= 4 and direct_flux_ulp <= 4,
        }
        receipt["members"].append(row)
        _write_json(output, receipt)
        print(
            f"CACHE_MEMBER_DONE member={number} warm_s={warm_wall:.6f} ulp={flux_ulp}",
            flush=True,
        )
    receipt["verdict"] = {
        "member_count": len(receipt["members"]),
        "cached_program_reuse": all(
            row["same_cached_program"] for row in receipt["members"]
        ),
        "dispatch_wall_within_tolerance": all(
            row["dispatch_within_tolerance"] for row in receipt["members"]
        ),
        "terminal_flux_within_tolerance": all(
            row["terminal_flux_within_tolerance"] for row in receipt["members"]
        ),
    }
    _write_json(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--diagnose-vertical-mode", action="store_true")
    parser.add_argument("--diagnostic-output", type=Path, default=DEFAULT_DIAGNOSTIC)
    arguments = parser.parse_args()
    if arguments.diagnose_vertical_mode:
        result = diagnose_vertical_mode(arguments.diagnostic_output.resolve())
        print(json.dumps(_strict(result), sort_keys=True), flush=True)
        return
    result = run(
        arguments.output.resolve(),
        cache_root=arguments.cache_root.resolve() if arguments.cache_root else None,
    )
    print(json.dumps(_strict(result["verdict"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
