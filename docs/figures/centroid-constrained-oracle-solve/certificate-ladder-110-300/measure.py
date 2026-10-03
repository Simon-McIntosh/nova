"""Measure constrained Solovev rows against exact and whole-cell current support."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax._src import compiler
import numpy as np

from benchmarks import centroid_constrained_fixture_receipt as fixture
from nova.equilibrium import ForwardProfile
from nova.equilibrium.solve_request import ForwardSolveRequest
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache
from scripts.analytic_oracle_fixtures import measure as oracle_fixture


CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
)
CELLS = (110, 300)


def _require_support_mode(requested: str, actual: str, stage: str) -> None:
    if actual != requested:
        raise RuntimeError(
            f"{stage} support mode {actual!r} differs from {requested!r}"
        )


def _require_distinct_states(exact_digest: str | None, control_digest: str) -> None:
    if exact_digest is not None and exact_digest == control_digest:
        raise RuntimeError(
            "exact row and whole-cell control share a terminal state digest"
        )


def _span(context: dict, result: dict) -> dict:
    topology = result["topology"]
    reading = oracle_fixture.gauge_free_flux_read(
        context["exact"],
        np.asarray(topology["axis_rz_m"], dtype=np.float64),
        np.asarray(topology["boundary_rz_m"], dtype=np.float64),
        float(topology["axis_flux_wb"]),
        float(topology["boundary_flux_wb"]),
        np.asarray(result["compensating_field_t"], dtype=np.float64),
    )
    net = float(reading["gauge_free_flux_offset_wb"]) - float(
        reading["compensator_span_contribution_wb"]
    )
    return {
        **reading,
        "net_offset_wb": net,
        "net_offset_of_solved_span": net / abs(float(reading["solved_span_wb"])),
    }


def _whole_cell_context(exact: dict) -> dict:
    """Change only support; retain the exact fixture exterior and production seed."""
    context = dict(exact)
    operator = exact["profile"].operator.with_clip_mode("chord")
    profile = exact["profile"]
    context["operator"] = operator
    context["profile"] = ForwardProfile(
        operator,
        profile.lattice,
        evaluations=profile.evaluations,
        relaxation=profile.relaxation,
        newton_steps=profile.newton_steps,
    )
    return context


def _measure(
    context: dict,
    support: str,
    initial_level: float,
    exact_tolerance: np.ndarray,
    exact_state_digest: str | None,
    output: Path,
    *,
    trips: int | None = None,
    preflight: bool = False,
) -> dict:
    requested_mode = "exact" if support == "exact" else "chord"
    _require_support_mode(
        requested_mode, context["profile"].operator.clip_mode, "input operator"
    )
    request_factory = fixture.certificate._certificate_solve_request
    pair_factory = fixture._certificate_pairs
    solve = ForwardProfile.solve
    request_modes: list[str] = []
    realised_modes: list[str] = []

    def explicit_request(*args, **kwargs):
        supplied_mode = kwargs.pop("clip_mode", None)
        if supplied_mode not in (None, requested_mode):
            raise RuntimeError("conflicting typed request support mode")
        request = request_factory(*args, **kwargs, clip_mode=requested_mode)
        _require_support_mode(requested_mode, request.clip_mode, "typed request")
        request_modes.append(request.clip_mode)
        return request

    def shared_row_tolerance(*args, **kwargs):
        pairs = pair_factory(*args, **kwargs)
        centroid = replace(
            pairs[0],
            binding=replace(pairs[0].binding, tolerance=jnp.asarray(exact_tolerance)),
        )
        return (centroid, *pairs[1:])

    def checked_solve(self, initial_flux, *args, **kwargs):
        if not isinstance(initial_flux, ForwardSolveRequest):
            return solve(self, initial_flux, *args, **kwargs)
        request = initial_flux
        _require_support_mode(requested_mode, request.clip_mode, "typed solve request")
        receipt = solve(self, initial_flux, *args, **kwargs)
        realised_mode = self._with_source(
            request.source_profile, clip_mode=request.clip_mode
        ).operator.clip_mode
        _require_support_mode(requested_mode, receipt.clip_mode, "solve receipt")
        _require_support_mode(requested_mode, realised_mode, "realised operator")
        realised_modes.append(realised_mode)
        return receipt

    compilation = {"seconds": 0.0, "calls": 0}
    original = compiler.compile_or_get_cached

    def timed_compile(*args, **kwargs):
        started = perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            compilation["seconds"] += perf_counter() - started
            compilation["calls"] += 1

    compiler.compile_or_get_cached = timed_compile
    try:
        with (
            patch.object(
                fixture.certificate, "_certificate_solve_request", explicit_request
            ),
            patch.object(fixture, "_certificate_pairs", shared_row_tolerance),
            patch.object(ForwardProfile, "solve", checked_solve),
        ):
            result, state = fixture._solve(
                context,
                context["seed"],
                constrained=True,
                trips=trips,
                field_scale_t=fixture.DEFAULT_FIELD_BOUND_T,
                initial_level_wb=initial_level,
            )
    finally:
        compiler.compile_or_get_cached = original
    if request_modes != [requested_mode] or realised_modes != [requested_mode]:
        raise RuntimeError("typed request and realised support mode were not observed")
    if preflight:
        return {
            "requested_clip_mode": request_modes[0],
            "clip_mode": realised_modes[0],
            "solve": result,
        }
    _require_distinct_states(exact_state_digest, result["state_sha256_binary64"])

    pair = fixture._certificate_pairs(
        context, level=True, field_scale_t=fixture.DEFAULT_FIELD_BOUND_T
    )[0]
    pitch = context["pitch"]
    span = _span(context, result)
    analytic_span = abs(float(span["analytic_span_wb"]))
    state_path = output.with_suffix(".npy")
    np.save(state_path, np.asarray(state, dtype=np.float64))
    receipt = {
        "schema": "nova.constrained-certificate-row",
        "source_revision": fixture._revision(),
        "case": context["case_name"],
        "requested_cells": context["requested_cells"],
        "realised_cells": len(context["machine"].node),
        "support": support,
        "clip_mode": realised_modes[0],
        "requested_clip_mode": request_modes[0],
        "exact_exterior_retained": True,
        "production_seed_sha256_binary64": fixture._digest(context["seed"]),
        "characteristic_pitch_m": pitch,
        "row_tolerance_pitches": (exact_tolerance / pitch).tolist(),
        "analytic_row_observation_m": np.asarray(
            pair.binding.payload, dtype=np.float64
        ).tolist(),
        "analytic_span_wb": analytic_span,
        "maximum_difference_of_span": float(
            np.max(np.abs(np.asarray(state) - context["analytic"])) / analytic_span
        ),
        "axis_error_pitches": float(
            np.linalg.norm(
                np.asarray(result["topology"]["axis_rz_m"])
                - np.asarray(context["exact"].magnetic_axis)
            )
            / pitch
        ),
        "boundary_level_offset_of_span": span["net_offset_of_solved_span"],
        "span": span,
        "terminal_global_residual": result["terminal_residual"],
        "terminal_row_residual": result["row_scaled_residual_sup"],
        "terminal_row_residual_pitches": result["centroid_error_pitches"],
        "compensating_field_t": result["compensating_field_t"],
        "compensating_field_t_abs_sup": result["compensating_field_t_abs_sup"],
        "level_amplitude_wb": result["level_amplitude_wb"],
        "backend_compile_or_cache_seconds": compilation["seconds"],
        "backend_compile_calls": compilation["calls"],
        "solve_other_seconds": max(
            0.0, result["wall_seconds"] - compilation["seconds"]
        ),
        "solve_wall_seconds": result["wall_seconds"],
        "qualified": result["qualified"],
        "row_qualified": result["row_qualified"],
        "converged": result["newton_history"]["converged"],
        "accepted_newton_promotions": result["newton_history"][
            "accepted_newton_promotions"
        ],
        "newton_termination_reason": result["newton_history"]["termination_reason"],
        "solve": result,
        "terminal_state_path": str(state_path),
    }
    fixture._write_json(output, receipt)
    print(
        "RECEIPT "
        + str(output)
        + " "
        + json.dumps(
            {
                "global": receipt["terminal_global_residual"],
                "row": receipt["terminal_row_residual"],
                "map": receipt["maximum_difference_of_span"],
                "boundary": receipt["boundary_level_offset_of_span"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt


def _preflight(output: Path) -> None:
    case = "weak-rotation-reactor-static"
    exact = fixture._context(case, -110, clip_mode="exact")
    level = fixture._seed_level_offset_wb(exact, exact["seed"])
    exact_tolerance = np.asarray(
        fixture._certificate_pairs(
            exact, level=True, field_scale_t=fixture.DEFAULT_FIELD_BOUND_T
        )[0].binding.tolerance,
        dtype=np.float64,
    )
    whole = _whole_cell_context(exact)
    readings = []
    for support, context in (("exact", exact), ("whole-cell", whole)):
        reading = _measure(
            context,
            support,
            level["initial_level_wb"],
            exact_tolerance,
            None,
            output / f"{case}-110-{support}.json",
            trips=1,
            preflight=True,
        )
        print(
            f"PREFLIGHT_ARM {support} requested={reading['requested_clip_mode']} "
            f"realised={reading['clip_mode']} "
            f"accepted={reading['solve']['newton_history']['accepted_newton_promotions']}",
            flush=True,
        )
        readings.append(reading)
    if readings[0]["clip_mode"] == readings[1]["clip_mode"]:
        raise RuntimeError("preflight arms realised the same operator support mode")
    print("PREFLIGHT_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"REVISION {fixture._revision()} TREE {fixture.ROOT} COMMAND {__file__} "
        f"--output {output}{' --preflight' if args.preflight else ''}",
        flush=True,
    )
    configure_dtypes()
    assert jax.config.jax_enable_x64
    if args.preflight:
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            raise RuntimeError("CPU preflight requires JAX_PLATFORMS=cpu")
        _preflight(output)
        return
    configure_persistent_compilation_cache(
        fixture.default_forward_compilation_cache_root()
    )
    lane = fixture._lane(
        "h200" if os.environ["SLURM_JOB_PARTITION"] == "betelgeuse" else "titan"
    )
    fixture._write_json(output / "lane.json", lane)
    for case in CASES:
        for cells in CELLS:
            started = perf_counter()
            exact = fixture._context(case, -cells, clip_mode="exact")
            assert exact["profile"].operator.clip_mode == "exact"
            level = fixture._seed_level_offset_wb(exact, exact["seed"])
            key = f"{case}-{cells}"
            print(
                f"ROW_BEGIN {key} setup_seconds={perf_counter() - started:.3f}",
                flush=True,
            )
            exact_tolerance = np.asarray(
                fixture._certificate_pairs(
                    exact, level=True, field_scale_t=fixture.DEFAULT_FIELD_BOUND_T
                )[0].binding.tolerance,
                dtype=np.float64,
            )
            exact_receipt = _measure(
                exact,
                "exact",
                level["initial_level_wb"],
                exact_tolerance,
                None,
                output / f"{key}-exact.json",
            )
            whole = _whole_cell_context(exact)
            assert whole["profile"].operator.clip_mode == "chord"
            assert np.array_equal(whole["baseline"], exact["baseline"])
            assert np.array_equal(whole["seed"], exact["seed"])
            _measure(
                whole,
                "whole-cell",
                level["initial_level_wb"],
                exact_tolerance,
                exact_receipt["solve"]["state_sha256_binary64"],
                output / f"{key}-whole-cell.json",
            )
    print("MEASUREMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
