"""Bounded exact-clip certificate measurement for the larger fixture meshes."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax._src import compiler

from benchmarks import centroid_constrained_fixture_receipt as fixture
from benchmarks import oracle_start_newton_probe as oracle_probe
from benchmarks import solovev_certificate as certificate
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture


FIGURES = Path(__file__).resolve().parent
OUTPUT = Path(os.environ["CERTIFICATE_PROBE_OUTPUT"])
ROWS = (
    ("weak-rotation-reactor-static", -1000),
    ("moderate-rotation-conventional-static", -1000),
    ("strong-rotation-compact-static", -1000),
    ("diverted-single-null", -500),
)


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture._strict(value), indent=2, sort_keys=True) + "\n")


def compile_meter() -> dict:
    meter = {"seconds": 0.0, "program_sizes_bytes": [], "calls": 0}
    original = compiler.compile_or_get_cached

    def measured(*args, **kwargs):
        started = time.perf_counter()
        try:
            executable = original(*args, **kwargs)
        finally:
            meter["seconds"] += time.perf_counter() - started
            meter["calls"] += 1
        size = getattr(executable, "size_of_generated_code_in_bytes", None)
        if size is not None:
            try:
                meter["program_sizes_bytes"].append(int(size()))
            except TypeError, ValueError, RuntimeError:
                pass
        return executable

    compiler.compile_or_get_cached = measured
    return meter


def span(context: dict, result: dict) -> dict:
    topology = result["topology"]
    points = np.vstack(
        (
            np.asarray(topology["axis_rz_m"], dtype=np.float64),
            np.asarray(topology["boundary_rz_m"], dtype=np.float64),
        )
    )
    analytic = certificate._exact_state(context["case_name"], context["exact"], points)
    field = np.asarray(result["compensating_field_t"], dtype=np.float64)
    compensator = oracle_fixture.uniform_exterior_field_flux(
        context["exact"], points, field
    )
    solved_span = float(topology["axis_flux_wb"]) - float(topology["boundary_flux_wb"])
    analytic_span = float(analytic[0] - analytic[1])
    compensator_span = float(compensator[0] - compensator[1])
    offset = solved_span - analytic_span
    net = offset - compensator_span
    return {
        "solved_span_wb": solved_span,
        "analytic_span_wb": analytic_span,
        "compensator_span_contribution_wb": compensator_span,
        "gauge_free_flux_offset_wb": offset,
        "net_offset_wb": net,
        "net_offset_of_span": net / abs(analytic_span),
    }


def panel(context: dict, state: np.ndarray, result: dict, stem: str) -> str:
    wall = np.asarray(context["machine"].wall_node, dtype=np.float64)
    radial, height, analytic = certificate._raster_field(
        context["coordinates"], context["analytic"], wall
    )
    _, _, solved = certificate._raster_field(context["coordinates"], state, wall)
    levels = poloidal.contour_levels(analytic, count=10)
    analytic_nulls = oracle_probe._topology(
        context["profile"].operator, context["analytic"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis in axes:
        poloidal.draw_wall(axis, units=(wall,))
        poloidal_axes(axis)
    for field, color in (
        (analytic, fixture.ANALYTIC_INK),
        (solved, fixture.TERMINAL_INK),
    ):
        poloidal.draw_flux_contours(
            axes[0],
            radial,
            height,
            field,
            levels,
            color=color,
            linewidth=2.6,
            wall=wall,
        )
    for nulls, color in (
        (analytic_nulls, fixture.ANALYTIC_INK),
        (result["topology"], fixture.TERMINAL_INK),
    ):
        poloidal.draw_nulls(
            axes[0],
            magnetic_axis=nulls["axis_rz_m"],
            x_points=nulls["x_point_rz_m"],
            style=DEFAULT_INK.variant(
                axis_marker="^", axis_color=color, xpoint_color=color
            ),
            contain=(wall,),
        )
    difference = solved - analytic
    magnitude = float(np.nanmax(np.abs(difference)))
    if magnitude > 0:
        difference_levels = np.linspace(-magnitude, magnitude, 11)
        poloidal.draw_flux_contours(
            axes[1],
            radial,
            height,
            difference,
            difference_levels,
            color=fixture.TERMINAL_INK,
            linewidth=2.6,
            wall=wall,
        )
    axes[0].text(0.03, 0.95, "solved / analytic", transform=axes[0].transAxes, va="top")
    axes[1].text(
        0.03, 0.95, "solved − analytic [Wb]", transform=axes[1].transAxes, va="top"
    )
    path = FIGURES / f"{stem}.png"
    fig.savefig(path, dpi=85)
    plt.close(fig)
    return (
        "/nova/figures/centroid-constrained-oracle-solve/"
        f"certificate-ladder-500-1000-probe/{stem}.png"
    )


def measure(case: str, requested_cells: int) -> None:
    stem = f"{case}-{abs(requested_cells)}"
    receipt_path = OUTPUT / f"{stem}.json"
    partial = {
        "case": case,
        "requested_cells": requested_cells,
        "clip_mode_requested": "exact",
        "status": "in-progress",
        "source_revision": fixture._revision(),
    }
    write(receipt_path, partial)
    phase = "context"
    meter = {"seconds": 0.0, "program_sizes_bytes": [], "calls": 0}
    started = time.perf_counter()
    try:
        configure_dtypes()
        assert jax.config.jax_enable_x64
        configure_persistent_compilation_cache(
            fixture.default_forward_compilation_cache_root()
        )
        lane = fixture._lane(os.environ.get("CERTIFICATE_PROBE_LANE", "h200"))
        context = fixture._context(case, requested_cells, clip_mode="exact")
        context["operator"] = context["profile"].operator
        assert context["operator"].clip_mode == "exact"
        partial.update(
            lane=lane,
            realised_cells=len(context["machine"].node),
            seed_sha256_binary64=fixture._digest(context["seed"]),
        )
        write(receipt_path, partial)
        phase = "solve"
        meter = compile_meter()
        jax.jit(lambda value: value + 1.0)(jax.numpy.asarray(1.0)).block_until_ready()
        assert meter["calls"] > 0, "compile timer missed a known JIT compilation"
        partial["compile_timer_control_calls"] = meter["calls"]
        meter.update(seconds=0.0, program_sizes_bytes=[], calls=0)
        solve_started = time.perf_counter()
        result, state = fixture._solve(
            context,
            context["seed"],
            constrained=True,
            field_scale_t=fixture.DEFAULT_FIELD_BOUND_T,
        )
        solve_wall = time.perf_counter() - solve_started
        partial.update(
            status="solved",
            solve=result,
            compiled_program_size_bytes=(
                max(meter["program_sizes_bytes"])
                if meter["program_sizes_bytes"]
                else None
            ),
            compile_calls=meter["calls"],
            compile_seconds=meter["seconds"],
            solve_seconds=max(0.0, solve_wall - meter["seconds"]),
            total_seconds=time.perf_counter() - started,
        )
        write(receipt_path, partial)
        state_path = OUTPUT / f"{stem}-state.npy"
        np.save(state_path, np.asarray(state, dtype=np.float64))
        partial["terminal_state_path"] = str(state_path)
        write(receipt_path, partial)
        phase = "readout"
        reading = span(context, result)
        analytic_span = abs(float(reading["analytic_span_wb"]))
        max_difference = float(np.max(np.abs(state - context["analytic"])))
        axis_pitch_error = float(
            np.linalg.norm(
                np.asarray(result["topology"]["axis_rz_m"])
                - np.asarray(context["exact"].magnetic_axis)
            )
            / context["pitch"]
        )
        pair = fixture._certificate_pairs(
            context, level=True, field_scale_t=fixture.DEFAULT_FIELD_BOUND_T
        )[0]
        row_tolerance = (np.asarray(pair.binding.tolerance) / context["pitch"]).tolist()
        clauses = {
            "max_difference_over_span": max_difference / analytic_span,
            "max_difference_limit": 10 * 1.1e-4,
            "axis_error_pitches": axis_pitch_error,
            "axis_limit_pitches": 0.1,
            "net_boundary_level_over_span": reading["net_offset_of_span"],
            "boundary_limit_over_span": 1e-3,
            "terminal_global_residual": result["terminal_residual"],
            "row_residual_sup": result["row_scaled_residual_sup"],
            "residual_limit": 1e-12,
            "row_tolerance_pitches": row_tolerance,
            "compensating_field_t": result["compensating_field_t"],
            "level_amplitude_wb": result["level_amplitude_wb"],
            "converged": result["qualified"],
            "row_qualified": result["row_qualified"],
            "bound_refusal": result["bound_refusal"],
        }
        partial.update(
            span=reading, clauses=clauses, total_seconds=time.perf_counter() - started
        )
        write(receipt_path, partial)
        phase = "panel"
        partial["panel_src"] = panel(context, state, result, stem)
        write(receipt_path, partial)
    except Exception as error:
        partial.update(
            status=(
                partial["status"]
                if partial["status"] == "solved"
                else "refused"
                if meter["calls"] == 0
                else "compile-failed"
            ),
            failure_phase=phase,
            failure_text=str(error),
            traceback=traceback.format_exc(),
            compile_seconds=meter["seconds"],
            compile_calls=meter["calls"],
            total_seconds=time.perf_counter() - started,
        )
        write(receipt_path, partial)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=[row[0] for row in ROWS])
    parser.add_argument("cells", type=int)
    args = parser.parse_args()
    assert (args.case, args.cells) in ROWS
    measure(args.case, args.cells)
