"""Measure the exact-clipped displaced fixture on one allocated H200."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import centroid_constrained_fixture_receipt as fixture
from benchmarks import oracle_start_newton_probe as oracle_probe
from benchmarks import solovev_certificate as certificate
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture


OUT = Path(__file__).resolve().parent


def record(name: str, payload: dict) -> None:
    fixture._write_json(OUT / name, payload)
    print(f"RECEIPT {name}", flush=True)


def span(context: dict, result: dict) -> dict:
    topology = result["topology"]
    reading = oracle_fixture.gauge_free_flux_read(
        context["exact"],
        np.asarray(topology["axis_rz_m"], dtype=np.float64),
        np.asarray(topology["boundary_rz_m"], dtype=np.float64),
        float(topology["axis_flux_wb"]),
        float(topology["boundary_flux_wb"]),
        np.asarray(result["compensating_field_t"], dtype=np.float64),
    )
    solved_span = abs(float(reading["solved_span_wb"]))
    net = float(reading["gauge_free_flux_offset_wb"]) - float(
        reading["compensator_span_contribution_wb"]
    )
    return {
        **reading,
        "net_offset_wb": net,
        "net_offset_of_solved_span": net / solved_span,
        "net_clause": abs(net / solved_span) <= 1.0e-3,
    }


def draw_panels(context: dict, state: np.ndarray, result: dict) -> dict:
    wall = np.asarray(context["machine"].wall_node, dtype=np.float64)
    radial, height, analytic = certificate._raster_field(
        context["coordinates"], context["analytic"], wall
    )
    _, _, solved = certificate._raster_field(context["coordinates"], state, wall)
    levels = poloidal.contour_levels(analytic, count=12)
    analytic_nulls = oracle_probe._topology(
        context["profile"].operator, context["analytic"]
    )
    solved_nulls = result["topology"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, field, label in zip(
        axes, (analytic, solved), ("analytic", "constrained"), strict=True
    ):
        poloidal.draw_flux_contours(
            axis,
            radial,
            height,
            field,
            levels,
            color=fixture.ANALYTIC_INK if label == "analytic" else fixture.TERMINAL_INK,
            linewidth=2.6,
        )
        poloidal.draw_wall(axis, units=(wall,))
        for nulls, color in (
            (analytic_nulls, fixture.ANALYTIC_INK),
            (solved_nulls, fixture.TERMINAL_INK),
        ):
            poloidal.draw_nulls(
                axis,
                magnetic_axis=nulls["axis_rz_m"],
                x_points=nulls["x_point_rz_m"],
                style=DEFAULT_INK.variant(
                    axis_marker="^", axis_color=color, xpoint_color=color
                ),
                contain=(wall,),
            )
        poloidal_axes(axis)
        axis.text(0.03, 0.95, label, transform=axis.transAxes, va="top")
    path = OUT / "flux-panels.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return {
        "path": str(path),
        "src": (
            "/nova/figures/centroid-constrained-oracle-solve/"
            "repaired-remeasure-receipt/flux-panels.png"
        ),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "shared_levels_wb": levels.tolist(),
    }


def arm(
    context: dict, seed: np.ndarray, lane: dict, constrained: bool, level: float
) -> tuple[dict, np.ndarray]:
    result, state = fixture._solve(
        context,
        seed,
        constrained=constrained,
        field_scale_t=fixture.DEFAULT_FIELD_BOUND_T,
        initial_level_wb=level,
    )
    name = "positive" if constrained else "negative"
    state_path = OUT / f"{name}-state.npy"
    np.save(state_path, np.asarray(state, dtype=np.float64))
    receipt = fixture.control_receipt(
        context,
        arm=name,
        constrained=constrained,
        lane=lane,
        displaced=seed,
        result=result,
        state=state,
        figure=None,
    )
    receipt["field_identity"]["field_scale_t"] = fixture.DEFAULT_FIELD_BOUND_T
    receipt["terminal_state_path"] = str(state_path)
    receipt["requested_clip_mode"] = "exact"
    pair = fixture._certificate_pairs(
        context, level=True, field_scale_t=fixture.DEFAULT_FIELD_BOUND_T
    )[0]
    receipt["row_tolerance_pitches"] = (
        np.asarray(pair.binding.tolerance, dtype=np.float64) / context["pitch"]
    ).tolist()
    receipt["analytic_row_observation_m"] = np.asarray(
        pair.binding.payload, dtype=np.float64
    ).tolist()
    assert receipt["row_tolerance_pitches"][0] > 0.0
    receipt["span"] = None if not constrained else span(context, result)
    difference = np.asarray(state) - np.asarray(context["analytic"])
    cell_count = len(context["machine"].node)
    wall_count = len(context["machine"].wall_node)
    wall_difference = difference[cell_count : cell_count + wall_count]
    receipt["map_difference"] = {
        "sample_count": int(difference.size),
        "rms_wb": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_wb": float(np.max(np.abs(difference))),
        "wall_min_wb": float(np.min(wall_difference)),
        "wall_max_wb": float(np.max(wall_difference)),
        "wall_sample_count": int(wall_count),
    }
    record(f"{name}.json", receipt)
    return receipt, state


def replay(context: dict, seed: np.ndarray, level: float) -> None:
    """Record a one-step continuation separately from the primary solve."""
    state = seed
    field = None
    rows = []
    for step in range(certificate.recovery.NEWTON_STEPS):
        result, state = fixture._solve(
            context,
            state,
            constrained=True,
            trips=1,
            initial_field_t=field,
            initial_level_wb=level,
            field_scale_t=fixture.DEFAULT_FIELD_BOUND_T,
        )
        field = np.asarray(result["compensating_field_t"], dtype=np.float64).tolist()
        level = float(result["level_amplitude_wb"])
        row = {
            "step": step + 1,
            "global_residual": result["terminal_residual"],
            "row_residual_pitches": (
                np.abs(np.asarray(result["centroid_error_m"])) / context["pitch"]
            ).tolist(),
            "row_qualified": result["row_qualified"],
            "compensating_field_t": field,
            "level_amplitude_wb": level,
            "net_span": span(context, result),
        }
        rows.append(row)
        record(
            "step-replay.json",
            {
                "method": (
                    "one-step continuation; separate from primary full-budget solve"
                ),
                "rows": rows,
            },
        )
        print("STEP", json.dumps(fixture._strict(row), sort_keys=True), flush=True)
        if result["qualified"]:
            break


def main() -> None:
    print(
        f"REVISION {fixture._revision()} TREE {fixture.ROOT} COMMAND {__file__}",
        flush=True,
    )
    configure_dtypes()
    assert jax.config.jax_enable_x64
    configure_persistent_compilation_cache(
        fixture.default_forward_compilation_cache_root()
    )
    lane = fixture._lane("h200")
    context = fixture._context("weak-rotation-reactor-static", -110, clip_mode="exact")
    context["operator"] = context["profile"].operator
    assert context["profile"].operator.clip_mode == "exact"
    assert len(context["machine"].node) == 135
    seed = fixture._translated_state(context)
    assert fixture._digest(seed).startswith(fixture.UNIT_LEVERAGE_SEED_DIGEST)
    preflight = fixture.control_receipt(
        context,
        arm="preflight",
        constrained=False,
        lane=lane,
        displaced=seed,
        result={},
        state=seed,
        figure=None,
    )
    assert preflight["clip_mode"] == "exact"
    print("RECEIPT_PREFLIGHT exact", flush=True)
    level = fixture._seed_level_offset_wb(context, seed)["initial_level_wb"]
    request = json.loads((OUT / "request.json").read_text())
    assert request["clip_mode_requested"] == "exact"
    print(f"LANE {json.dumps(lane, sort_keys=True)}", flush=True)
    positive, state = arm(context, seed, lane, True, level)
    positive["figure"] = draw_panels(context, state, positive["solve"])
    record("positive.json", positive)
    negative, _ = arm(context, seed, lane, False, level)
    test = {
        "same_seed": positive["displaced_seed_sha256_binary64"]
        == negative["displaced_seed_sha256_binary64"],
        "no_row_qualifies": bool(
            np.all(
                np.abs(np.asarray(negative["solve"]["centroid_error_m"]))
                / context["pitch"]
                <= np.asarray(positive["row_tolerance_pitches"])
            )
        ),
    }
    assert test["same_seed"]
    assert not test["no_row_qualifies"], "the no-row arm met the row tolerance"
    record("control.json", test)
    replay(context, seed, level)
    print("MEASUREMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
