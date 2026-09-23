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
    from nova.equilibrium import reduced_newton
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
    captures = []
    original = reduced_newton._drive_trips

    def observe_drive(kernels, state, reduced, shadow, **kwargs):
        copied = dict(kernels)
        boundary = (
            kernels["boundary"] if kwargs["fused"] else kwargs["dispatched_boundary"]
        )

        def observe_boundary(*operands):
            result = boundary(*operands)
            jax.block_until_ready(result)
            captures.append((np.asarray(result[0]).copy(), float(result[3])))
            np.savez(
                args.output
                / f"{args.revision[:9]}-{args.clip}-boundary-{len(captures)}.npz",
                state=captures[-1][0],
                residual=captures[-1][1],
            )
            print(
                f"CAPTURE trip={len(captures)} residual={captures[-1][1]:.14g}",
                flush=True,
            )
            return result

        if kwargs["fused"]:
            copied["boundary"] = observe_boundary
        else:
            kwargs["dispatched_boundary"] = observe_boundary
        return original(copied, state, reduced, shadow, **kwargs)

    reduced_newton._drive_trips = observe_drive
    try:
        solved = built["profile"].solve(request)
    finally:
        reduced_newton._drive_trips = original
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
    for trip, (state, residual) in enumerate(captures, 1):
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
            "residual": residual,
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
        row["trips"].append(record)
        write_json(path, row)
        render_panel(driver, built, row, record, state, args.output)
        write_json(path, row)
    row["saddle_loss_trip"] = next(
        (trip["trip"] for trip in row["trips"] if trip["saddle_outside_bound"]), None
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


def render_panel(driver, built, row, trip, state, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    import numpy as np

    reference = row["reference_nulls"]
    levels = driver.poloidal.contour_levels(
        built["analytic"],
        count=15,
        axis=reference["axis_flux_wb"],
        boundary=reference["boundary_flux_wb"],
    )
    operator = built["operator"]
    wall = built["machine"].wall_node
    units = tuple(
        driver.WallUnit(wall[a:b, 0], wall[a:b, 1], closed=bool(closed), kind=kind)
        for a, b, closed, kind in zip(
            operator.wall_unit_offsets[:-1],
            operator.wall_unit_offsets[1:],
            operator.wall_unit_closed,
            operator.wall_unit_kinds,
            strict=True,
        )
    )
    fig, axis = plt.subplots(figsize=(7.2, 7.8), constrained_layout=True)
    colors = {0: "#b8b8b8", 1: "#00856a", 2: "#b87900", 3: "#a246a8"}
    labels = trip["flood_labels"]
    assert set(labels) <= set(colors)
    axis.add_collection(
        PolyCollection(
            built["machine"].cell_polygons,
            facecolors="none",
            edgecolors=[colors[label] for label in labels],
            linewidths=0.65,
        )
    )
    counts = []
    for values, color in ((built["analytic"], "#84a7dc"), (state, "#303030")):
        r, z, field = driver.certificate._raster_field(
            built["coordinates"], values, wall
        )
        contour = driver.poloidal.draw_flux_contours(
            axis, r, z, field, levels, color=color
        )
        count = sum(len(part) > 1 for group in contour.allsegs for part in group)
        assert count > 0, "contour positive control missing"
        counts.append(count)
    driver.poloidal.draw_wall(axis, units=units)
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
                axis,
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
    driver.poloidal_axes(axis)
    distance = trip["saddle_distance_m"]
    caption = (
        f"{row['revision'][:9]} | {row['clip_mode']} | trip {trip['trip']}\n"
        f"residual={trip['residual']:.7g}; converged={trip['converged']}\n"
        f"saddle offset={distance if distance is None else round(distance, 5)} m; "
        "bound=0.15 m\n"
        f"non-finite clipped moments={trip['nonfinite_moment_count']}/"
        f"{trip['moment_element_count']}"
    )
    axis.set_title(caption, fontsize=11)
    axis.legend(
        handles=[
            Line2D([], [], color=color, label=f"flood label {label}")
            for label, color in colors.items()
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    fig.supxlabel(
        "Blue: analytic contours and nulls; gray contours/red nulls: trip state.\n"
        "Triangle: axis; filled cross: admitted saddle; "
        "hollow cross: other qualified saddle.\n"
        "All nulls are shown, including admitted points outside the wall.",
        fontsize=9,
    )
    stem = f"{row['revision'][:9]}-{row['clip_mode']}-trip-{trip['trip']}"
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=150)
    plt.close(fig)
    trip["panel"] = {
        "png": stem + ".png",
        "svg": stem + ".svg",
        "caption": caption,
        "axis_off": not axis.axison,
        "contour_segments": counts,
        "null_markers": tallies,
        "shared_levels_wb": levels.tolist(),
    }


def measure(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    control = output / "negative-control.log"
    negative_control(control)
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
