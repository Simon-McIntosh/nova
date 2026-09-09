"""Receipt the reusable compiled-slice executable on the twelve-member bank."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

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
DISPATCH_TOLERANCE = 0.10


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


def _solve(member):
    """Invoke the public compiled entry point with the receipt's fixed policy."""
    return reduced_newton.solve_reduced_newton_compiled(
        member.operator,
        jnp.asarray(member.state),
        requested_class=jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8),
        target_current=member.target_current,
        tolerance=member.tolerance,
        newton_steps=12,
        active_set_steps=16,
    )


def _host(member):
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
    )


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
            "dispatch_wall_relative_tolerance": DISPATCH_TOLERANCE,
            "terminal_flux_max_ulp": 4,
        },
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
        warm_started = time.perf_counter()
        second = _solve(member)
        warm_wall = time.perf_counter() - warm_started
        host = _host(member)
        reference_wall = dispatch_reference[member.identity]
        relative_dispatch_error = abs(warm_wall - reference_wall) / reference_wall
        flux_ulp = _ulp_distance(second.state, host.state)
        row = {
            "identity": member.identity,
            "cold_public_wall_s": cold_wall,
            "warm_public_wall_s": warm_wall,
            "dispatch_reference_wall_s": reference_wall,
            "dispatch_relative_error": relative_dispatch_error,
            "same_cached_program": second.program is first.program,
            "slice_solver_count": len(second.program.slice_solvers or {}),
            "compiled_host_terminal_flux_ulp": flux_ulp,
            "dispatch_within_tolerance": relative_dispatch_error <= DISPATCH_TOLERANCE,
            "terminal_flux_within_tolerance": flux_ulp <= 4,
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
    arguments = parser.parse_args()
    result = run(
        arguments.output.resolve(),
        cache_root=arguments.cache_root.resolve() if arguments.cache_root else None,
    )
    print(json.dumps(_strict(result["verdict"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
