"""Read the first-moment row after each accepted bounded Newton solve."""

from __future__ import annotations

from dataclasses import replace
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
from nova.equilibrium.solve_request import default_forward_compilation_cache_root


OUT = Path(__file__).resolve().parent
BANK = OUT.parent / "repaired-remeasure-receipt"


def write(payload: dict) -> None:
    fixture._write_json(OUT / "row-history.json", payload)
    print(
        f"ROW_HISTORY rows={len(payload['rows'])} status={payload['status']}",
        flush=True,
    )


def draw_panel(context: dict, state: np.ndarray, topology: dict) -> dict:
    wall = np.asarray(context["machine"].wall_node, dtype=np.float64)
    radial, height, analytic = certificate._raster_field(
        context["coordinates"], context["analytic"], wall
    )
    _, _, solved = certificate._raster_field(context["coordinates"], state, wall)
    levels = poloidal.contour_levels(analytic, count=12)
    analytic_nulls = oracle_probe._topology(
        context["profile"].operator, context["analytic"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, field, label in zip(
        axes, (analytic, solved), ("analytic", "solved"), strict=True
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
        poloidal.draw_nulls(
            axis,
            magnetic_axis=analytic_nulls["axis_rz_m"],
            x_points=analytic_nulls["x_point_rz_m"],
            style=DEFAULT_INK.variant(
                axis_marker="o",
                axis_markersize=12,
                axis_color=fixture.ANALYTIC_INK,
            ),
            contain=(wall,),
        )
        analytic_axis = analytic_nulls["axis_rz_m"]
        axis.plot(
            analytic_axis[0],
            analytic_axis[1],
            marker="o",
            markersize=12,
            markerfacecolor="white",
            markeredgecolor=fixture.ANALYTIC_INK,
            markeredgewidth=2,
            linestyle="none",
            zorder=DEFAULT_INK.zorder_markers + 1,
        )
        poloidal.draw_nulls(
            axis,
            magnetic_axis=topology["axis_rz_m"],
            x_points=topology["x_point_rz_m"],
            style=DEFAULT_INK.variant(
                axis_marker="^",
                axis_markersize=8.5,
                axis_color=fixture.TERMINAL_INK,
                zorder_markers=DEFAULT_INK.zorder_markers + 2,
            ),
            contain=(wall,),
        )
        poloidal_axes(axis)
        axis.text(0.03, 0.95, label, transform=axis.transAxes, va="top")
    path = OUT / "flux-panels.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return {
        "src": (
            "/nova/figures/centroid-constrained-oracle-solve/"
            "row-history-receipt/flux-panels.png"
        ),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "shared_levels_wb": levels.tolist(),
        "analytic_axis_marker": "large hollow circle",
        "solved_axis_marker": "filled triangle",
    }


def main() -> None:
    print(
        f"REVISION {fixture._revision()} TREE {fixture.ROOT} "
        f"COMMAND /home/ITER/mcintos/Code/nova/.venv/bin/python {__file__}",
        flush=True,
    )
    configure_dtypes()
    assert jax.config.jax_enable_x64
    configure_persistent_compilation_cache(default_forward_compilation_cache_root())
    lane = fixture._lane("h200")
    assert lane["allocated_cpus"] == 8 and lane["memory_mb"] == 131072
    context = fixture._context("weak-rotation-reactor-static", -110, clip_mode="exact")
    assert context["profile"].operator.clip_mode == "exact"
    assert len(context["machine"].node) == 135
    seed = fixture._translated_state(context)
    bank = json.loads((BANK / "positive.json").read_text())
    assert fixture._digest(seed) == bank["displaced_seed_sha256_binary64"]
    assert bank["clip_mode"] == "exact"
    history = bank["solve"]["newton_history"]
    count = history["accepted_newton_promotions"]
    assert count == 4
    terminal = np.load(BANK / "positive-state.npy")
    assert fixture._digest(terminal) == bank["solve"]["state_sha256_binary64"]
    panel = draw_panel(context, terminal, bank["solve"]["topology"])
    level = fixture._seed_level_offset_wb(context, seed)["initial_level_wb"]
    pairs = fixture._certificate_pairs(
        context,
        level=True,
        initial_level_wb=level,
        field_scale_t=fixture.DEFAULT_FIELD_BOUND_T,
    )
    tolerance = (
        np.asarray(pairs[0].binding.tolerance, dtype=np.float64) / context["pitch"]
    )
    assert np.array_equal(tolerance, np.asarray(bank["row_tolerance_pitches"]))
    payload = {
        "schema": "nova.centroid-row-promotion-history",
        "source_revision": fixture._revision(),
        "source_receipt": "repaired-remeasure-receipt/positive.json",
        "case": "weak-rotation-reactor-static",
        "requested_cells": -110,
        "realised_cells": 135,
        "clip_mode": "exact",
        "seed_sha256_binary64": fixture._digest(seed),
        "lane": lane,
        "row_tolerance_pitches": tolerance.tolist(),
        "terminal_state_sha256_binary64": fixture._digest(terminal),
        "figure": panel,
        "rows": [],
        "status": "in_progress",
    }
    write(payload)
    for promotions in range(1, count + 1):
        request = certificate._certificate_solve_request(
            context["profile"],
            seed,
            context["target_current"],
            carrier_identity=f"centroid-row-history-{promotions}",
            clip_mode="exact",
        )
        stopping_tolerance = (
            float(history["relative_residual_after"][promotions - 1]) * 2.0
            if promotions < count
            else request.policy.kernel_tolerance
        )
        request = replace(
            request,
            constraint_pairs=pairs,
            policy=replace(request.policy, kernel_tolerance=stopping_tolerance),
        )
        receipt = context["profile"].solve(request)
        equilibrium = receipt.equilibrium
        state = np.asarray(jax.block_until_ready(equilibrium.flux), dtype=np.float64)
        current = fixture._newton_history(equilibrium.fixed_point)
        centroid, flux_level = equilibrium.constraints
        observed = np.asarray(centroid.observed, dtype=np.float64)
        amplitudes = np.concatenate(
            (
                np.asarray(centroid.physical_unknown, dtype=np.float64),
                np.asarray(flux_level.physical_unknown, dtype=np.float64),
            )
        )
        promotion_global = current["relative_residual_after"][promotions - 1]
        expected_global = history["relative_residual_after"][promotions - 1]
        prefix_matches = bool(
            np.array_equal(
                np.asarray(current["relative_residual_after"][:promotions]),
                np.asarray(history["relative_residual_after"][:promotions]),
            )
        )
        row = {
            "promotion": promotions,
            "stopping_tolerance": stopping_tolerance,
            "accepted_newton_promotions": current["accepted_newton_promotions"],
            "global_residual": current["terminal_relative_residual"],
            "promotion_global_residual": promotion_global,
            "committed_promotion_global_residual": expected_global,
            "committed_history_prefix_matches": prefix_matches,
            "first_moment_row_residual_pitches": float(
                np.linalg.norm(observed - context["centroid"]) / context["pitch"]
            ),
            "first_moment_row_residual_components_pitches": (
                (observed - context["centroid"]) / context["pitch"]
            ),
            "row_tolerance_pitches": tolerance,
            "compensating_amplitudes": amplitudes,
            "state_sha256_binary64": fixture._digest(state),
        }
        payload["rows"].append(row)
        write(payload)
        if current["accepted_newton_promotions"] != promotions or not prefix_matches:
            payload["status"] = "history_mismatch"
            write(payload)
            raise RuntimeError(
                f"bounded solve at promotion {promotions} differs "
                "from committed history"
            )
    last = payload["rows"][-1]
    expected = bank["solve"]
    terminal_matches = (
        last["global_residual"] == expected["terminal_residual"]
        and last["state_sha256_binary64"] == expected["state_sha256_binary64"]
        and last["first_moment_row_residual_pitches"]
        == expected["centroid_error_pitches"]
    )
    payload["terminal_matches"] = terminal_matches
    payload["row_residual_monotonic_decrease"] = all(
        later["first_moment_row_residual_pitches"]
        < earlier["first_moment_row_residual_pitches"]
        for earlier, later in zip(payload["rows"], payload["rows"][1:], strict=False)
    )
    payload["status"] = "matched" if terminal_matches else "terminal_mismatch"
    write(payload)
    if not terminal_matches:
        raise RuntimeError("bounded terminal state does not match committed receipt")
    print("MEASUREMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
