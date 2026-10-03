"""Measure constrained Solovev rows against exact and whole-cell current support."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import jax
from jax._src import compiler
import numpy as np

from benchmarks import centroid_constrained_fixture_receipt as fixture
from nova.equilibrium import ForwardProfile
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache
from scripts.analytic_oracle_fixtures import measure as oracle_fixture


CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
)
CELLS = (110, 300)


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


def _measure(context: dict, support: str, initial_level: float, output: Path) -> dict:
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
        result, state = fixture._solve(
            context,
            context["seed"],
            constrained=True,
            field_scale_t=fixture.DEFAULT_FIELD_BOUND_T,
            initial_level_wb=initial_level,
        )
    finally:
        compiler.compile_or_get_cached = original

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
        "clip_mode": context["profile"].operator.clip_mode,
        "exact_exterior_retained": True,
        "production_seed_sha256_binary64": fixture._digest(context["seed"]),
        "characteristic_pitch_m": pitch,
        "row_tolerance_pitches": (
            np.asarray(pair.binding.tolerance, dtype=np.float64) / pitch
        ).tolist(),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"REVISION {fixture._revision()} TREE {fixture.ROOT} COMMAND {__file__} "
        f"--output {output}",
        flush=True,
    )
    configure_dtypes()
    assert jax.config.jax_enable_x64
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
            _measure(
                exact, "exact", level["initial_level_wb"], output / f"{key}-exact.json"
            )
            whole = _whole_cell_context(exact)
            assert whole["profile"].operator.clip_mode == "chord"
            assert np.array_equal(whole["baseline"], exact["baseline"])
            assert np.array_equal(whole["seed"], exact["seed"])
            _measure(
                whole,
                "whole-cell",
                level["initial_level_wb"],
                output / f"{key}-whole-cell.json",
            )
    print("MEASUREMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
