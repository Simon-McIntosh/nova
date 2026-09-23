"""Capture active-set boundaries and discriminate support from census loss."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


NEGATIVE_CONTROL = (
    "the reading-rule function is fed a synthetic arm pair with non-finite moments "
    "at the saddle-loss trip in the pre-repair arm and a retained saddle in the "
    "repaired arm, and must return support-overflow; fed a pair where both arms "
    "lose the saddle with all-finite moments it must return census-read; swapping "
    "the two inputs must change the verdict"
)
POSITION_BOUND_M = 0.15


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reading_rule(before, after):
    """Apply the ordered support-versus-census discriminator to complete arms."""
    if any(arm.get("status") != "measured" for arm in (before, after)):
        return "neither"
    loss = before["saddle_loss_trip"]
    repaired_loss = after["saddle_loss_trip"]
    if loss is not None:
        lost = next(row for row in before["trips"] if row["trip"] == loss)
        if lost["nonfinite_moment_count"] > 0 and repaired_loss is None:
            return "support-overflow"
    if loss is not None and repaired_loss is not None:
        if all(
            row["nonfinite_moment_count"] == 0
            for arm in (before, after)
            for row in arm["trips"]
        ):
            return "census-read"
    return "neither"


def negative_control(path):
    def arm(loss, count):
        return {
            "status": "measured",
            "saddle_loss_trip": loss,
            "trips": [{"trip": 1, "nonfinite_moment_count": count}],
        }

    before, retained = arm(1, 3), arm(None, 0)
    checks = [
        (
            "nonfinite loss then retained",
            reading_rule(before, retained),
            "support-overflow",
        ),
        (
            "finite losses in both arms",
            reading_rule(arm(1, 0), arm(1, 0)),
            "census-read",
        ),
        (
            "reverse support pair refuses attribution",
            reading_rule(retained, before),
            "neither",
        ),
        ("unmeasured arm refuses attribution", reading_rule({}, retained), "neither"),
    ]
    with path.open("w") as log:
        log.write(NEGATIVE_CONTROL + "\n")
        for name, actual, expected in checks:
            log.write(f"{name}: observed={actual} expected={expected}\n")
            assert actual == expected, name
        log.write(
            "PASS: reversing the asymmetric support pair changes its verdict; "
            "the finite/finite census rule is symmetric.\n"
        )
    return checks


def measure_arm(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import nova
    from benchmarks import plasma_cell_terminal_state as driver
    from benchmarks.plasma_cell_fixed_point_attribution import build_construction
    from nova.equilibrium import fixed_point
    from nova.jax.config import (
        configure_persistent_compilation_cache,
        default_persistent_compilation_cache_root,
    )

    driver.configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    assert Path(nova.__file__).resolve().is_relative_to(Path.cwd())
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())
    driver.set_support_clip_mode(args.clip)
    row = {
        "revision": args.revision,
        "nova_file": nova.__file__,
        "clip_mode": args.clip,
        "status": "not-measured",
        "trips": [],
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "devices": [str(device) for device in jax.devices()],
        "x64": True,
        "position_bound_m": POSITION_BOUND_M,
    }
    path = args.output / f"{args.revision[:9]}-{args.clip}.json"
    write_json(path, row)
    print(f"BUILD {args.revision} {args.clip}", flush=True)
    built = build_construction(driver, "diverted-single-null", "production")
    operator, request = built["operator"], built["request"]
    if args.clip == "chord":
        request = replace(request, policy=replace(request.policy, active_set_steps=1))
    reference, _ = driver._nulls(operator, built["analytic"])
    assert reference["x_point_rz_m"] is not None, (
        "analytic saddle positive control missing"
    )
    row.update(
        reference_nulls=reference,
        construction=built["snapshot"],
        provenance=built["provenance"],
        realised_cells=len(built["machine"].node),
    )
    assert request.policy.route == "newton_krylov", request.policy.route
    write_json(path, row)
    print(f"SOLVE route={request.policy.route}", flush=True)
    seed = np.asarray(request.seed_policy.state)
    np.savez(args.output / f"{args.revision[:9]}-{args.clip}-seed.npz", state=seed)
    row["analytic_shape"] = shape_diagnostic(driver, built, built["analytic"])
    assert row["analytic_shape"]["is_plasma"], "analytic closed-contour control failed"
    ramp = built["coordinates"][:, 0]
    ramp_shape = shape_diagnostic(driver, built, ramp, candidates=[])
    assert not ramp_shape["has_closed_contour"], "linear-ramp control failed"
    row["shape_controls"] = {"analytic": True, "linear_ramp": False}
    write_json(path, row)
    captures = []
    execution = {"total": 0, "last_boundary": 0, "trip_counts": {}}
    original = fixed_point._ActiveSetIterationState
    operator_type = type(operator)
    original_scale = operator_type.scaled_current_moments

    def count_scale(amplitude, net):
        execution["total"] += 1

    def observe_scale(moments, amplitude):
        result = original_scale(moments, amplitude)
        jax.debug.callback(
            count_scale, amplitude, jnp.sum(result.cell_current), ordered=True
        )
        return result

    def capture(state, residual, iterations):
        count = int(iterations)
        if count == 0:
            execution["last_boundary"] = execution["total"]
            return
        assert count == len(captures) + 1
        captures.append((np.asarray(state).copy(), float(residual)))
        execution["trip_counts"][count] = (
            execution["total"] - execution["last_boundary"]
        )
        execution["last_boundary"] = execution["total"]
        np.savez(
            args.output / f"{args.revision[:9]}-{args.clip}-boundary-{count}.npz",
            state=captures[-1][0],
            residual=captures[-1][1],
            lambda_scaling_execution_count=execution["trip_counts"][count],
        )
        print(
            f"CAPTURE trip={count} residual={float(residual):.14g} "
            f"lambda_calls={execution['trip_counts'][count]}",
            flush=True,
        )

    def observe_carry(*values, **keywords):
        result = original(*values, **keywords)
        jax.debug.callback(
            capture,
            result.state,
            result.live_residual,
            result.iterations,
            ordered=True,
        )
        return result

    operator_type.scaled_current_moments = staticmethod(observe_scale)
    try:
        jax.block_until_ready(
            operator.internal(jnp.asarray(seed), target_current=request.target_current)
        )
        jax.effects_barrier()
        seed_executions = execution["total"]
        assert seed_executions > 0, "seed lambda path did not execute"
        execution["total"] = 0
        fixed_point._ActiveSetIterationState = observe_carry
        solved = built["profile"].solve(request)
        jax.block_until_ready(solved.equilibrium.flux)
        jax.effects_barrier()
    finally:
        fixed_point._ActiveSetIterationState = original
        operator_type.scaled_current_moments = staticmethod(original_scale)
    history = solved.equilibrium.fixed_point
    assert len(captures) == int(history.active_set_iterations) > 0
    np.testing.assert_array_equal(captures[-1][0], np.asarray(solved.equilibrium.flux))
    np.testing.assert_array_equal(
        [x[1] for x in captures],
        np.asarray(history.active_set_residuals)[: len(captures)],
    )
    row.update(
        terminal_residual=float(history.residual),
        converged=bool(history.converged),
        trip_count=len(captures),
        certificate_residual_bound=driver.certificate.TERMINAL_RESIDUAL_BOUND,
        resolved_defaults=solved.resolved_defaults.to_dict(),
        capture_terminal_and_history_equal=True,
    )
    write_json(path, row)
    for trip, (state, residual) in enumerate([(seed, float("nan")), *captures]):
        nulls, _ = driver._nulls(operator, state)
        masks, _, sample_flux, support = operator._support_partition(jnp.asarray(state))
        moment_masks = operator._moment_support_masks(masks, support)
        raw = operator.source.current_moments(
            moment_masks,
            operator.support_current_moments,
            support,
            sample_flux=sample_flux,
        )
        values = np.asarray(jax.block_until_ready(raw))
        assert values.size > 0 and values.shape[-1] == row["realised_cells"]
        # A deliberately non-finite element checks the count instrument itself.
        injected = values.copy()
        injected.flat[0] = np.nan
        assert np.count_nonzero(~np.isfinite(injected)) >= 1
        labels = np.asarray(masks.label)
        admitted = nulls["x_point_rz_m"]
        distance = (
            None
            if admitted is None
            else float(np.linalg.norm(np.asarray(admitted) - reference["x_point_rz_m"]))
        )
        record = {
            "trip": trip,
            "residual": finite_number(residual),
            "converged": bool(
                np.isfinite(residual)
                and residual <= driver.certificate.TERMINAL_RESIDUAL_BOUND
            ),
            "nulls": nulls,
            "saddle_distance_m": distance,
            "saddle_outside_bound": distance is None or distance > POSITION_BOUND_M,
            "nonfinite_moment_count": int(np.count_nonzero(~np.isfinite(values))),
            "nonfinite_cell_count": int(
                np.count_nonzero(np.any(~np.isfinite(values), axis=0))
            ),
            "moment_element_count": int(values.size),
            "finite_nonzero_moment_count": int(
                np.count_nonzero(np.isfinite(values) & (values != 0))
            ),
            "nonfinite_counter_positive_control": int(
                np.count_nonzero(~np.isfinite(injected))
            ),
            "cell_current_a": [float(x) if np.isfinite(x) else None for x in values[0]],
            "flood_labels": labels.tolist(),
            "flood_label_counts": {
                str(k): int(v)
                for k, v in zip(*np.unique(labels, return_counts=True), strict=True)
            },
        }
        record["shape"] = shape_diagnostic(driver, built, state)
        target = float(request.target_current)
        scaled, amplitude = jax.jit(
            lambda flux: operator.normalised_current_moments(flux, target)
        )(jnp.asarray(state))
        carried = float(jnp.sum(scaled.cell_current))
        unscaled = float(
            jnp.sum(operator.cell_current_moments(jnp.asarray(state)).cell_current)
        )
        calls = seed_executions if trip == 0 else execution["trip_counts"][trip]
        record["lambda"] = {
            "amplitude": finite_number(amplitude),
            "target_current_a": target,
            "unscaled_net_support_current_a": finite_number(unscaled),
            "scaled_net_support_current_a": finite_number(carried),
            "net_current_error_a": finite_number(carried - target),
            "scaling_armed": bool(request.policy.current_pin and target != 0),
            "plasma_cells_enabled": bool(
                operator.use_linear_moments and operator.grid.node_number > 0
            ),
            "scaling_execution_count": calls,
            "execution_context": "seed map replay"
            if trip == 0
            else "production solve between native active-set carries",
            "counts_include_trial_maps_and_primal_linearisation_evaluations": True,
        }
        record["closed_contour_invariant_violation"] = bool(
            record["lambda"]["scaling_armed"]
            and calls > 0
            and not record["shape"]["has_closed_contour"]
        )
        if trip == 0:
            row["seed"] = record
        else:
            row["trips"].append(record)
        write_json(path, row)
        render_panel(driver, built, row, record, state, args.output)
        write_json(path, row)
    row["saddle_loss_trip"] = next(
        (trip["trip"] for trip in row["trips"] if trip["saddle_outside_bound"]), None
    )
    states = [row["seed"], *row["trips"]]
    row["first_non_plasma_state"] = next(
        (r["trip"] for r in states if not r["shape"]["is_plasma"]), None
    )
    row["first_closed_contour_invariant_violation"] = next(
        (r["trip"] for r in states if r["closed_contour_invariant_violation"]), None
    )
    row["status"] = "measured"
    write_json(path, row)
    print(
        "ARM_COMPLETE "
        + json.dumps(
            {
                key: row[key]
                for key in (
                    "revision",
                    "clip_mode",
                    "terminal_residual",
                    "trip_count",
                    "saddle_loss_trip",
                )
            }
        ),
        flush=True,
    )


def finite_number(value):
    import numpy as np

    number = float(value)
    return number if np.isfinite(number) else None


def shape_diagnostic(driver, built, state, candidates=None):
    """Census closed contours of the nodal interpolant independently of admission."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as PlotPath
    from scipy.interpolate import LinearNDInterpolator
    import numpy as np
    from nova.media.sources.frame import inside_wall_units

    operator = built["operator"]
    coordinates = built["coordinates"]
    wall = built["machine"].wall_node
    units = wall_units(driver, operator, wall)
    if candidates is None:
        census = operator._fixed_design_topology.grid.candidate_table_status(
            operator.null_flux_pool(driver.jnp.asarray(state))
        )
        points = np.asarray(census["retained_candidate"])[0]
        valid = np.asarray(census["retained_valid"])[0]
        candidates = points[valid, :2].tolist()
    candidates = np.asarray(candidates).reshape(-1, 2)
    if len(candidates):
        in_cells = np.zeros(len(candidates), dtype=bool)
        for polygon in built["machine"].cell_polygons:
            in_cells |= PlotPath(polygon).contains_points(candidates, radius=1e-10)
        candidates = candidates[in_cells & inside_wall_units(candidates, units)]
    values = np.asarray(state)
    inside = inside_wall_units(coordinates, units)
    domain_values = values[inside]
    low, high = float(np.min(domain_values)), float(np.max(domain_values))
    levels = np.unique(
        np.r_[
            np.linspace(low, high, 129)[1:-1],
            np.quantile(domain_values, np.linspace(0.001, 0.999, 129)),
        ]
    )
    if len(candidates):
        roots = LinearNDInterpolator(coordinates, values)(candidates)
        extra = [
            roots + fraction * (high - low)
            for fraction in (-0.01, -0.001, -0.0001, 0.0001, 0.001, 0.01)
        ]
        levels = np.unique(np.r_[levels, np.asarray(extra).ravel()])
        levels = levels[np.isfinite(levels) & (levels > low) & (levels < high)]
    fig, ax = plt.subplots()
    contours = ax.tricontour(
        coordinates[:, 0], coordinates[:, 1], values, levels=levels
    )
    closed = []
    for level, group in zip(levels, contours.allsegs, strict=True):
        for curve in group:
            if len(curve) < 4 or not np.allclose(
                curve[0], curve[-1], rtol=0, atol=1e-9
            ):
                continue
            if not np.all(inside_wall_units(curve, units)):
                continue
            area = (
                abs(np.sum(curve[:-1, 0] * curve[1:, 1] - curve[1:, 0] * curve[:-1, 1]))
                / 2
            )
            if area <= 1e-12:
                continue
            enclosed = PlotPath(curve).contains_points(candidates).tolist()
            closed.append(
                {
                    "level_wb": float(level),
                    "area_m2": float(area),
                    "encloses_o_points": enclosed,
                }
            )
    plt.close(fig)
    _, gradient, _ = driver.certificate._quadratic_derivatives(
        built["profile"].lattice, values[: len(built["machine"].node)]
    )
    around_axis = any(any(curve["encloses_o_points"]) for curve in closed)
    return {
        "flux_min_wb": low,
        "flux_max_wb": high,
        "all_state_flux_min_wb": float(np.min(values)),
        "all_state_flux_max_wb": float(np.max(values)),
        "rms_gradient_wb_per_m": float(np.sqrt(np.mean(np.sum(gradient**2, axis=1)))),
        "o_points_inside_plasma_cells_rz_m": candidates.tolist(),
        "has_o_point_inside_plasma_cells": bool(len(candidates)),
        "has_closed_contour": bool(closed),
        "closed_contour_about_o_point": around_axis,
        "is_plasma": bool(len(candidates) and around_axis),
        "closed_contour_count": len(closed),
        "closed_contours": closed,
        "tested_levels_wb": levels.tolist(),
        "method": (
            "unfilled contours of the piecewise-linear nodal interpolant; "
            "loops must close inside wall units and enclose an independently "
            "qualified O-point in the cell mesh"
        ),
        "qualification": (
            "finite-resolution level census, not a continuum absence theorem"
        ),
    }


def wall_units(driver, operator, wall):
    return tuple(
        driver.WallUnit(wall[a:b, 0], wall[a:b, 1], closed=bool(closed), kind=kind)
        for a, b, closed, kind in zip(
            operator.wall_unit_offsets[:-1],
            operator.wall_unit_offsets[1:],
            operator.wall_unit_closed,
            operator.wall_unit_kinds,
            strict=True,
        )
    )


def render_panel(driver, built, row, trip, state, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import numpy as np

    reference = row["reference_nulls"]
    shared = driver.poloidal.contour_levels(
        built["analytic"],
        count=15,
        axis=reference["axis_flux_wb"],
        boundary=reference["boundary_flux_wb"],
    )
    own = np.linspace(trip["shape"]["flux_min_wb"], trip["shape"]["flux_max_wb"], 17)[
        1:-1
    ]
    wall = built["machine"].wall_node
    units = wall_units(driver, built["operator"], wall)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    colors = {0: "#b8b8b8", 1: "#00856a", 2: "#b87900", 3: "#a246a8"}
    panels = []
    for ax, levels, title in zip(
        axes,
        (shared, own),
        ("Shared analytic levels", "State's own levels"),
        strict=True,
    ):
        ax.add_collection(
            PolyCollection(
                built["machine"].cell_polygons,
                facecolors="none",
                edgecolors=[colors[label] for label in trip["flood_labels"]],
                linewidths=0.5,
            )
        )
        counts = []
        for values, color in ((built["analytic"], "#84a7dc"), (state, "#303030")):
            r, z, field = driver.certificate._raster_field(
                built["coordinates"], values, wall
            )
            contour = driver.poloidal.draw_flux_contours(
                ax, r, z, field, levels, color=color
            )
            counts.append(
                sum(len(part) > 1 for group in contour.allsegs for part in group)
            )
        assert counts[1] > 0, "state contours missing"
        driver.poloidal.draw_wall(ax, units=units)
        tallies = []
        for nulls, color, size in (
            (reference, "#3366cc", 9),
            (trip["nulls"], "#d52d28", 5),
        ):
            others = np.asarray(nulls["qualified_saddles_rz_m"]).reshape(-1, 2)
            admitted = nulls["x_point_rz_m"]
            if admitted is not None:
                others = others[np.linalg.norm(others - admitted, axis=1) > 1e-10]
            tallies.append(
                driver.poloidal.draw_nulls(
                    ax,
                    magnetic_axis=nulls["axis_rz_m"],
                    x_points=admitted,
                    other_x_points=others,
                    style=driver.DEFAULT_INK.variant(
                        axis_color=color,
                        xpoint_color=color,
                        axis_markersize=size,
                        xpoint_markersize=size + 2,
                    ),
                )
            )
        driver.poloidal_axes(ax)
        ax.set_title(title)
        panels.append(
            {
                "levels_wb": levels.tolist(),
                "contour_segments": counts,
                "null_markers": tallies,
                "axis_off": not ax.axison,
            }
        )
    state_name = "seed" if trip["trip"] == 0 else f"trip {trip['trip']}"
    caption = (
        f"{row['revision'][:9]} | {row['clip_mode']} | {state_name} | "
        f"residual={trip['residual']}; converged={trip['converged']}\n"
        f"plasma={trip['shape']['is_plasma']}; "
        f"closed contour={trip['shape']['has_closed_contour']}; "
        f"non-finite moments={trip['nonfinite_moment_count']}; "
        f"lambda={trip['lambda']['amplitude']}\n"
        f"net current={trip['lambda']['scaled_net_support_current_a']} A; "
        f"target={trip['lambda']['target_current_a']} A; "
        f"scaling calls={trip['lambda']['scaling_execution_count']}"
    )
    fig.suptitle(caption, fontsize=10)
    fig.supxlabel(
        "Blue contours/large nulls: analytic. Gray contours/red nulls: state. "
        "Triangles: axes; filled crosses: admitted saddles; "
        "hollow crosses: other saddles.\n"
        "Cell outlines: gray excluded, green core, "
        "ochre common SOL, purple private flux.",
        fontsize=9,
    )
    stem = f"{row['revision'][:9]}-{row['clip_mode']}-trip-{trip['trip']}"
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=140)
    plt.close(fig)
    trip["panel"] = {
        "png": stem + ".png",
        "svg": stem + ".svg",
        "caption": caption,
        "views": panels,
        **panels[0],
        "shared_levels_wb": shared.tolist(),
    }


def gpu_callback_control(output):
    import jax
    import jax.numpy as jnp

    captured = []
    assert jax.default_backend() == "gpu"
    assert jax.devices("cpu"), "CPU backend required for ordered callbacks"

    def program(value):
        jax.debug.callback(lambda x: captured.append(float(x)), value, ordered=True)
        return value + 1

    result = jax.jit(program)(jnp.asarray(7.0))
    result.block_until_ready()
    jax.effects_barrier()
    assert captured == [7.0] and float(result) == 8.0
    first_capture = list(captured)
    captured.clear()
    transformed = jax.jit(
        jax.vmap(lambda value: jax.jvp(program, (value,), (jnp.ones_like(value),))[0])
    )(jnp.asarray([2.0, 3.0]))
    transformed.block_until_ready()
    jax.effects_barrier()
    assert captured == [2.0, 3.0]
    control = {
        "captured": first_capture,
        "jvp_vmap_captured": captured,
        "result": float(result),
        "passed": True,
        "gpu_devices": [
            {"device": str(x), "kind": x.device_kind} for x in jax.devices()
        ],
        "cpu_devices": [str(x) for x in jax.devices("cpu")],
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
    }
    write_json(output / "gpu-callback-control.json", control)
    print("GPU_CALLBACK_CONTROL_PASS captured=7 before any solve arm", flush=True)


def measure(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    control = output / "negative-control.log"
    print("MEASUREMENT_DRIVER " + str(Path(__file__).resolve()), flush=True)
    negative_control(control)
    gpu_callback_control(output)
    script = Path(__file__).resolve()
    receipt = {
        "status": "in-progress",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "driver_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "dispatch_base": args.base,
        "pre_repair_revision": args.before,
        "negative_control_log": str(control),
        "arms": [],
        "lambda_static_trace": json.loads(
            (output / "static-lambda-trace.json").read_text()
        ),
        "gpu_callback_control": json.loads(
            (output / "gpu-callback-control.json").read_text()
        ),
    }
    path = output / "trip-panels.json"
    write_json(path, receipt)
    for revision, clip in (
        (args.before, "exact"),
        (args.base, "exact"),
        (args.base, "chord"),
    ):
        tree = output / ("source-" + revision[:9])
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(tree), revision], check=True
        )
        try:
            env = dict(os.environ, PYTHONPATH=str(tree), PYTHONDONTWRITEBYTECODE="1")
            command = [
                sys.executable,
                str(script),
                "arm",
                "--revision",
                revision,
                "--clip",
                clip,
                "--output",
                str(output),
            ]
            logpath = output / f"{revision[:9]}-{clip}.log"
            with logpath.open("w") as log:
                log.write(f"revision={revision} tree={tree} command={command!r}\n")
                log.flush()
                result = subprocess.run(
                    command, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            arm_path = output / f"{revision[:9]}-{clip}.json"
            arm = (
                json.loads(arm_path.read_text())
                if arm_path.exists()
                else {"revision": revision, "clip_mode": clip, "status": "not-measured"}
            )
            arm.update(process_exit_status=result.returncode, log=str(logpath))
            receipt["arms"].append(arm)
            write_json(path, receipt)
            if result.returncode:
                raise RuntimeError(f"Arm failed; inspect {logpath}")
        finally:
            subprocess.run(["git", "worktree", "remove", str(tree)], check=True)
    receipt["verdict"] = reading_rule(receipt["arms"][0], receipt["arms"][1])
    receipt["combination"] = [
        {
            "revision": arm["revision"],
            "clip_mode": arm["clip_mode"],
            "saddle_loss_trip": arm["saddle_loss_trip"],
            "nonfinite_moments_per_trip": [
                row["nonfinite_moment_count"] for row in arm["trips"]
            ],
        }
        for arm in receipt["arms"]
    ]
    receipt["plasma_loss"] = [
        {
            "revision": arm["revision"],
            "clip_mode": arm["clip_mode"],
            "first_non_plasma_state": arm["first_non_plasma_state"],
            "first_closed_contour_invariant_violation": arm[
                "first_closed_contour_invariant_violation"
            ],
        }
        for arm in receipt["arms"]
    ]
    receipt["verdict_detail"] = {
        "mechanism": receipt["verdict"],
        "plasma_loss": receipt["plasma_loss"],
        "closed_contour_invariant_violated": any(
            a["first_closed_contour_invariant_violation"] is not None
            for a in receipt["arms"]
        ),
    }
    receipt["status"] = "complete"
    write_json(path, receipt)
    print("MEASUREMENT_COMPLETE " + receipt["verdict"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("measure", "arm", "negative-control"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base")
    parser.add_argument("--before")
    parser.add_argument("--revision")
    parser.add_argument("--clip", choices=("exact", "chord"))
    args = parser.parse_args()
    if args.mode == "negative-control":
        negative_control(args.output)
    elif args.mode == "arm":
        try:
            measure_arm(args)
        except Exception:
            traceback.print_exc()
            return 1
    else:
        measure(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
