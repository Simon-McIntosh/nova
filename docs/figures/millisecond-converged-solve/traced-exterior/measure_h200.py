"""Measure semantic identity and compile reuse for traced exterior fields."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.forward_operator import set_support_clip_mode, support_clip_mode
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import configure_dtypes


CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
)
REQUESTED_CELLS = -300
RELATIVE_IDENTITY_TOLERANCE = 1.0e-14


def _certificate_row(case_name: str):
    """Return the production profile and typed request for one analytic row."""
    print(f"STAGE {case_name} load cached mesh", flush=True)
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, REQUESTED_CELLS)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(case_name, exact, coordinates)
    print(f"STAGE {case_name} assemble operator", flush=True)
    empty_operator = certificate.oracle_fixture.forward_operator(source_case, machine)
    exact_moments = certificate.oracle_fixture.exact_current_moments(
        source_case, empty_operator, oracle_state
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_moments)
    exact_internal = certificate.oracle_fixture._internal_flux_image(
        empty_operator, exact_coefficients
    )
    operator = certificate.oracle_fixture.forward_operator(
        source_case, machine, oracle_state - exact_internal
    )
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=certificate.recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        case_name, source_case, operator, exact_moments
    )
    seed, _requested_class, _seed_receipt = certificate._production_seed(
        profile,
        case_name,
        target_current,
        centroid,
        current_receipt,
    )
    request = certificate._certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"solovev:{case_name}:{REQUESTED_CELLS}",
    )
    return profile, request


def _configure() -> None:
    print("STAGE configure H200 extended precision and whole-cell clip", flush=True)
    configure_dtypes()
    set_support_clip_mode("chord")
    if support_clip_mode() != "chord":
        raise RuntimeError("production whole-cell clip mode was not selected")


def _solve(case_name: str, output: Path) -> int:
    _configure()
    profile, request = _certificate_row(case_name)
    print(f"STAGE {case_name} solve start", flush=True)
    started = perf_counter()
    receipt = profile.solve(request)
    state = np.asarray(jax.block_until_ready(receipt.equilibrium.flux))
    residual = np.asarray(
        jax.block_until_ready(receipt.equilibrium.fixed_point.residual)
    )
    wall_seconds = perf_counter() - started
    np.savez(output, terminal_flux=state, residual=residual)
    metadata = {
        "case": case_name,
        "clip_mode": support_clip_mode(),
        "requested_cells": REQUESTED_CELLS,
        "wall_seconds": wall_seconds,
        "terminal_residual": float(residual),
        "converged": bool(np.asarray(receipt.equilibrium.fixed_point.converged)),
        "backend": jax.default_backend(),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"STAGE {case_name} solve finished {json.dumps(metadata, sort_keys=True)}",
        flush=True,
    )
    return 0


def _compare(base: Path, after: Path, output: Path) -> int:
    print("STAGE compare base and candidate terminal arrays", flush=True)
    with np.load(base) as base_data, np.load(after) as after_data:
        base_flux = base_data["terminal_flux"]
        after_flux = after_data["terminal_flux"]
        base_residual = float(base_data["residual"])
        after_residual = float(after_data["residual"])
        max_absolute_flux_difference = float(np.max(np.abs(after_flux - base_flux)))
        flux_scale = float(np.max(np.abs(base_flux)))
        max_relative_flux_difference = max_absolute_flux_difference / flux_scale
        residual_difference = abs(after_residual - base_residual)
        relative_residual_difference = residual_difference / abs(base_residual)
    receipt = {
        "clip_mode": "chord",
        "flux_bit_identical": np.array_equal(base_flux, after_flux),
        "residual_bit_identical": base_residual == after_residual,
        "max_absolute_flux_difference": max_absolute_flux_difference,
        "max_relative_flux_difference": max_relative_flux_difference,
        "residual_difference": residual_difference,
        "relative_residual_difference": relative_residual_difference,
        "relative_tolerance": RELATIVE_IDENTITY_TOLERANCE,
        "within_tolerance": (
            max_relative_flux_difference <= RELATIVE_IDENTITY_TOLERANCE
            and relative_residual_difference <= RELATIVE_IDENTITY_TOLERANCE
        ),
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        f"STAGE comparison finished {json.dumps(receipt, sort_keys=True)}", flush=True
    )
    return 0 if receipt["within_tolerance"] else 1


def _cache_monitor():
    """Return persistent-cache counts and compilation-duration counters."""
    import jax.monitoring as monitoring

    counters = {"hits": 0, "misses": 0}
    durations: dict[str, float] = defaultdict(float)

    def event(name: str, **_kwargs) -> None:
        if name == "/jax/compilation_cache/cache_hits":
            counters["hits"] += 1
        elif name == "/jax/compilation_cache/cache_misses":
            counters["misses"] += 1

    def duration(name: str, duration_secs: float, **_kwargs) -> None:
        if "compil" in name:
            durations[name] += duration_secs

    monitoring.register_event_listener(event)
    monitoring.register_event_duration_secs_listener(duration)
    return counters, durations


def _snapshot(counters, durations) -> dict[str, object]:
    return {
        "hits": counters["hits"],
        "misses": counters["misses"],
        "durations": dict(durations),
    }


def _delta(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    before_durations = before["durations"]
    after_durations = after["durations"]
    duration_names = set(before_durations) | set(after_durations)
    return {
        "persistent_cache_hits": after["hits"] - before["hits"],
        "persistent_cache_misses": after["misses"] - before["misses"],
        "compilation_duration_seconds": {
            name: after_durations.get(name, 0.0) - before_durations.get(name, 0.0)
            for name in sorted(duration_names)
        },
    }


def _timing(output: Path) -> int:
    _configure()
    counters, durations = _cache_monitor()
    case_name = CASES[0]
    profile, request = _certificate_row(case_name)
    fixture_current = np.asarray(profile.operator.external_current)
    variants = (
        ("fixture", fixture_current),
        ("scaled-0.9", fixture_current * 0.9),
    )
    rows = []
    for exterior_name, current in variants:
        variant_request = replace(request, current=current)
        before = _snapshot(counters, durations)
        print(f"STAGE {exterior_name} timed solve start", flush=True)
        started = perf_counter()
        receipt = profile.solve(variant_request)
        jax.block_until_ready(receipt.equilibrium.flux)
        wall_seconds = perf_counter() - started
        after = _snapshot(counters, durations)
        compilation = _delta(before, after)
        backend_compile_seconds = compilation["compilation_duration_seconds"].get(
            "/jax/core/compile/backend_compile_duration", 0.0
        )
        row = {
            "case": case_name,
            "exterior": exterior_name,
            "wall_seconds": wall_seconds,
            "terminal_residual": float(receipt.equilibrium.fixed_point.residual),
            "backend_compile_seconds": backend_compile_seconds,
            **compilation,
        }
        rows.append(row)
        message = json.dumps(row, sort_keys=True)
        print(
            f"STAGE {exterior_name} timed solve finished {message}",
            flush=True,
        )
        output.write_text(
            json.dumps(
                {
                    "clip_mode": support_clip_mode(),
                    "requested_cells": REQUESTED_CELLS,
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    second = rows[1]
    return (
        0
        if second["persistent_cache_misses"] == 0
        and second["backend_compile_seconds"] == 0.0
        else 1
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    solve = subparsers.add_parser("solve")
    solve.add_argument("--case", choices=CASES, required=True)
    solve.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--base", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    timing = subparsers.add_parser("timing")
    timing.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "solve":
        return _solve(arguments.case, arguments.output)
    if arguments.command == "compare":
        return _compare(arguments.base, arguments.after, arguments.output)
    return _timing(arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
