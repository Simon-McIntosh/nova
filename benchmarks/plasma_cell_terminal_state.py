"""Measure exact-support terminal states under truncated and default trip budgets."""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.forward_operator import set_support_clip_mode, support_clip_mode
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.sources.frame import WallUnit
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery


CASES = ("diverted-single-null", "weak-rotation-reactor-static")
EXPECTED_ONE_TRIP = (0.152, 0.0471)
NEGATIVE_CONTROL = (
    "run the default-policy arm with the active-set budget forced to one trip "
    "and observe the receipt assembly refuse the converged assertion for that arm"
)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _point(value):
    point = np.asarray(value, dtype=np.float64)
    return point.tolist() if np.all(np.isfinite(point)) else None


def _nulls(operator, state):
    _, topology = operator.read(jnp.asarray(state, dtype=jnp.float64))
    grid = operator._fixed_design_topology.grid
    census = grid.candidate_table_status(operator.null_flux_pool(jnp.asarray(state)))
    assert int(census["retained_count"][0]) > 0, "axis positive control is empty"
    saddles = np.asarray(census["retained_candidate"])[1]
    valid = np.asarray(census["retained_valid"])[1]
    return {
        "axis_rz_m": _point(topology.axis),
        "x_point_rz_m": _point(topology.x_point),
        "qualified_saddles_rz_m": saddles[valid, :2].tolist(),
        "retained_count": np.asarray(census["retained_count"]).tolist(),
        "axis_flux_wb": float(topology.axis_flux),
        "boundary_flux_wb": float(topology.boundary_flux),
    }, census


def _cluster(operator, analytic, reference, census):
    """Measure the local quadratic root uncertainty and the active merge rule."""
    grid = operator._fixed_design_topology.grid
    pool = operator.null_flux_pool(jnp.asarray(analytic, dtype=jnp.float64))
    positions = np.asarray(census["candidate"])[:, :2]
    selected = np.asarray(census["representative_mask"])[1].copy()
    if reference["x_point_rz_m"] is None:
        assert int(census["retained_count"][1]) == 0
        return {"status": "no_admitted_reference_saddle", "candidates": [], "pairs": []}
    selected &= np.linalg.norm(positions - reference["x_point_rz_m"], axis=1) <= 0.15
    indices = np.flatnonzero(selected)
    assert len(indices) == 3, f"expected populated three-saddle control, got {indices}"
    assert not bool(census["spline_authored"]), "measurement requires quadratic carrier"
    samples = pool[grid.fit_locator.stencil]
    coefficient = jnp.sum(grid.fit_weight * samples[:, None, :], axis=-1)
    scale = grid.fit_locator.physical_scale
    local = (jnp.asarray(positions) - grid.fit_locator.physical_origin) / scale
    radial, vertical = local[:, 0], local[:, 1]
    gradient = (
        jnp.stack(
            (
                2 * coefficient[:, 0] * radial
                + coefficient[:, 4] * vertical
                + coefficient[:, 2],
                2 * coefficient[:, 1] * vertical
                + coefficient[:, 4] * radial
                + coefficient[:, 3],
            ),
            axis=-1,
        )
        / scale
    )
    hessian = jnp.stack(
        (
            jnp.stack((2 * coefficient[:, 0], coefficient[:, 4]), axis=-1),
            jnp.stack((coefficient[:, 4], 2 * coefficient[:, 1]), axis=-1),
        ),
        axis=-2,
    ) / (scale[..., :, None] * scale[..., None, :])
    coordinates = np.asarray(grid.fit_locator.coordinate)
    domain_scale = float(np.linalg.norm(np.ptp(coordinates, axis=0)))
    widths = jnp.asarray(grid.source_pitch)
    uncertainty = np.asarray(
        grid._root_uncertainty(
            {
                "hessian": hessian,
                "gradient_norm": jnp.linalg.norm(gradient, axis=-1),
                "position_rz": jnp.asarray(positions),
            },
            widths,
            domain_scale,
        )
    )
    merge_uncertainty = (
        256 * np.finfo(np.dtype(grid.fit_dtype)).eps * np.asarray(widths)
    )
    candidates = [
        {
            "source_index": int(index),
            "position_rz_m": positions[index].tolist(),
            "source_pitch_m": float(widths[index]),
            "root_uncertainty_radius_m": float(uncertainty[index]),
            "active_merge_radius_m": float(merge_uncertainty[index]),
            "gradient_norm": float(jnp.linalg.norm(gradient[index])),
        }
        for index in indices
    ]
    pairs = []
    for first, second in combinations(range(len(indices)), 2):
        left, right = indices[first], indices[second]
        distance = float(np.linalg.norm(positions[left] - positions[right]))
        radius_sum = float(uncertainty[left] + uncertainty[right])
        active_sum = float(merge_uncertainty[left] + merge_uncertainty[right])
        pairs.append(
            {
                "candidates": [first, second],
                "distance_m": distance,
                "uncertainty_sum_m": radius_sum,
                "merge_criterion_holds": distance <= radius_sum,
                "active_merge_radius_sum_m": active_sum,
                "active_merge_criterion_holds": distance <= active_sum,
            }
        )
    return {
        "status": "measured",
        "candidates": candidates,
        "pairs": pairs,
        "uncertainty_method": "_root_uncertainty evaluated on each local quadratic",
        "active_merge_method": "quadratic carrier uses 256 * eps * source_pitch",
        "domain_scale_m": domain_scale,
    }


def _arm(profile, request, operator, reference, pitch, name):
    started = perf_counter()
    receipt = profile.solve(request)
    equilibrium = receipt.equilibrium
    state = np.asarray(jax.block_until_ready(equilibrium.flux), dtype=np.float64)
    assert np.all(np.isfinite(state))
    history = equilibrium.fixed_point
    trips = int(history.active_set_iterations)
    residuals = np.asarray(history.active_set_residuals)[:trips]
    assert len(residuals) == trips and np.all(np.isfinite(residuals))
    nulls, _ = _nulls(operator, state)
    reference_x = reference["x_point_rz_m"]
    terminal_x = nulls["x_point_rz_m"]
    distance = (
        float(np.linalg.norm(np.asarray(terminal_x) - reference_x))
        if reference_x is not None and terminal_x is not None
        else None
    )
    row = {
        "arm": name,
        "active_set_budget": request.policy.active_set_steps,
        "residual": float(history.residual),
        "converged": bool(history.converged),
        "qualified": bool(receipt.qualified),
        "trip_count": trips,
        "per_trip_residual_history": [
            {"trip": index + 1, "residual": float(value)}
            for index, value in enumerate(residuals)
        ],
        "terminal_axis_rz_m": nulls["axis_rz_m"],
        "terminal_x_point_rz_m": terminal_x,
        "reference_axis_rz_m": reference["axis_rz_m"],
        "reference_x_point_rz_m": reference_x,
        "x_point_distance_m": distance,
        "x_point_distance_status": (
            "measured"
            if distance is not None
            else "no_admitted_reference_saddle"
            if reference_x is None
            else "terminal_saddle_absent"
        ),
        "certificate_position_bound_m": certificate.TOPOLOGY_POSITION_BOUND_PITCHES
        * pitch,
        "certificate_residual_bound": certificate.TERMINAL_RESIDUAL_BOUND,
        "jax_platform": jax.default_backend(),
        "nulls": nulls,
        "termination_reason": int(receipt.termination_reason),
        "resolved_defaults": receipt.resolved_defaults.to_dict(),
        "wall_seconds": perf_counter() - started,
        "compilation_cache_hit": bool(receipt.compilation_cache_hit),
    }
    print("ARM " + json.dumps(row, allow_nan=False), flush=True)
    return row, state


def require_converged(row):
    """Refuse a convergence claim unsupported by the solve receipt."""
    if (
        not row["converged"]
        or not row["qualified"]
        or not np.isfinite(row["residual"])
        or row["residual"] > row["certificate_residual_bound"]
    ):
        raise ValueError(
            f"converged assertion refused: {row['arm']} "
            f"residual={row['residual']:.12g} "
            f"converged={row['converged']} trips={row['trip_count']}"
        )
    if row["reference_x_point_rz_m"] is not None:
        distance = row["x_point_distance_m"]
        if distance is None or distance > row["certificate_position_bound_m"]:
            raise ValueError(
                "converged assertion refused: terminal saddle exceeds position bound"
            )


def _draw_nulls(axis, nulls, wall_units, style):
    admitted = nulls["x_point_rz_m"]
    others = np.asarray(nulls["qualified_saddles_rz_m"]).reshape(-1, 2)
    if admitted is not None:
        others = others[np.linalg.norm(others - admitted, axis=1) > 1e-10]
    return poloidal.draw_nulls(
        axis,
        magnetic_axis=nulls["axis_rz_m"],
        x_points=admitted,
        other_x_points=others,
        contain=wall_units,
        style=style,
    )


def _cluster_pixel_census(figure, axes, receipt):
    """Count style-coloured pixels at the analytic saddle cluster on each panel.

    The census is the instrument for the marker draw order: the reference
    markers are large and the panel-state markers small, so whichever style is
    drawn last is the one visible where the two coincide. Counting pixels of
    each style inside a box around the analytic cluster shows the covering
    directly rather than inferring it from the artist list.
    """
    import matplotlib.colors as mcolors

    def rgb(value):
        return np.asarray(mcolors.to_rgb(value)) * 255.0

    reference_rgb = rgb("#3366cc")
    panel_rgb = rgb(DEFAULT_INK.xpoint_color)
    buffer = np.asarray(figure.canvas.buffer_rgba())[..., :3].astype(np.float64)
    height = buffer.shape[0]
    census = []
    for row_index, row in enumerate(receipt["cases"]):
        candidates = row["analytic_saddle_cluster"]["candidates"]
        if not candidates:
            census.append({"status": "no_admitted_reference_saddle"})
            continue
        positions = np.asarray([entry["position_rz_m"] for entry in candidates])
        panels = []
        for column in range(3):
            axis = axes[row_index, column]
            display = axis.transData.transform(positions)
            low = display.min(axis=0) - 16.0
            high = display.max(axis=0) + 16.0
            x0, x1 = int(max(low[0], 0)), int(min(high[0], buffer.shape[1]))
            y0, y1 = int(max(height - high[1], 0)), int(min(height - low[1], height))
            block = buffer[y0:y1, x0:x1]
            reference_count = int(
                np.sum(np.all(np.abs(block - reference_rgb) <= 40.0, axis=-1))
            )
            panel_count = int(
                np.sum(np.all(np.abs(block - panel_rgb) <= 40.0, axis=-1))
            )
            panels.append(
                {
                    "reference_pixels": reference_count,
                    "panel_pixels": panel_count,
                }
            )
        census.append(
            {
                "status": "measured",
                "box_halfwidth_px": 16.0,
                "panels": panels,
            }
        )
    return census


DRAW_ORDERS = ("reference-last", "panel-last")


def render(receipt, output, draw_order="reference-last"):
    """Draw all measured fields on a shared physical contour array per case.

    ``draw_order`` names which null set is drawn last, and therefore which one
    stays visible where the two coincide. ``panel-last`` is the original
    ordering, kept reachable as the declared negative control: the small
    panel-state markers are then drawn over the large reference markers, and the
    analytic panel draws its own null set twice.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(12, 10), constrained_layout=True)
    figure.set_dpi(160)
    reference_style = DEFAULT_INK.variant(
        axis_color="#3366cc",
        xpoint_color="#3366cc",
        axis_markersize=9,
        xpoint_markersize=11,
    )
    panel_style = DEFAULT_INK.variant(axis_markersize=5, xpoint_markersize=6)
    for row_index, row in enumerate(receipt["cases"]):
        with np.load(output / row["state_file"]) as data:
            coordinates, wall = data["coordinates"], data["wall"]
            states = [data["analytic"], data["one_trip"], data["default_policy"]]
            units = tuple(
                WallUnit(
                    wall[start:stop, 0],
                    wall[start:stop, 1],
                    closed=bool(closed),
                    kind=kind,
                )
                for start, stop, closed, kind in row["wall_units"]
            )
            reference = row["reference_nulls"]
            levels = poloidal.contour_levels(
                states[0],
                count=15,
                axis=reference["axis_flux_wb"],
                boundary=reference["boundary_flux_wb"],
            )
            row["contour_levels_wb"] = levels.tolist()
            row["panels"] = []
            for column, state in enumerate(states):
                axis = axes[row_index, column]
                radial, height, raster = certificate._raster_field(
                    coordinates, state, wall
                )
                contour = poloidal.draw_flux_contours(
                    axis, radial, height, raster, levels, color="#444444"
                )
                segments = sum(
                    len(part) > 1 for group in contour.allsegs for part in group
                )
                assert segments > 0, "contour positive control is empty"
                poloidal.draw_wall(axis, units=units)
                if draw_order == "panel-last":
                    # The original ordering: the panel-state markers are drawn
                    # over the reference markers, and the
                    # analytic panel draws its own null set a second time.
                    current = (
                        reference if column == 0 else row["arms"][column - 1]["nulls"]
                    )
                    reference_tally = _draw_nulls(
                        axis, reference, units, reference_style
                    )
                    panel_tally = _draw_nulls(axis, current, units, panel_style)
                elif column == 0:
                    # The analytic state is both the reference and the panel
                    # state, so it is drawn once, in the reference style.
                    reference_tally = _draw_nulls(
                        axis, reference, units, reference_style
                    )
                    panel_tally = reference_tally
                else:
                    panel_tally = _draw_nulls(
                        axis, row["arms"][column - 1]["nulls"], units, panel_style
                    )
                    reference_tally = _draw_nulls(
                        axis, reference, units, reference_style
                    )
                poloidal_axes(axis)
                if column == 0:
                    caption = "Analytic input\nresidual=n/a; converged=n/a; trips=0"
                else:
                    arm = row["arms"][column - 1]
                    title = "One active-set trip" if column == 1 else "Default policy"
                    caption = (
                        f"{title}\nresidual={arm['residual']:.6g}; "
                        f"converged={arm['converged']}; trips={arm['trip_count']}"
                    )
                axis.set_title(caption, fontsize=10)
                if column == 0:
                    axis.text(
                        -0.04,
                        0.5,
                        f"{row['case']}\n{row['realised_cells']} cells",
                        transform=axis.transAxes,
                        rotation=90,
                        va="center",
                        ha="right",
                        fontsize=10,
                    )
                row["panels"].append(
                    {
                        "caption": caption,
                        "contour_segments": segments,
                        "reference_markers": reference_tally,
                        "panel_markers": panel_tally,
                        "axis_off": not axis.axison,
                    }
                )
    figure.suptitle(
        "Exact support: first trip and default-policy terminal state", fontsize=15
    )
    figure.supxlabel(
        "Blue large markers: analytic-state read; "
        "red small markers: panel-state read.\n"
        "Filled triangle: axis; filled cross: admitted saddle; "
        "hollow crosses: other qualified saddles. "
        "Reference markers are drawn last and stay visible where the sets coincide.",
        fontsize=10,
    )
    figure.canvas.draw()
    receipt["marker_pixel_census"] = _cluster_pixel_census(figure, axes, receipt)
    figure.savefig(output / "terminal-state-trip-arms.png", dpi=160)
    figure.savefig(output / "terminal-state-trip-arms.svg")
    plt.close(figure)


def measure(output, negative_log):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu", "measurement requires the GPU lane"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    receipt = {
        "revision": revision,
        "worktree": str(Path.cwd()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "devices": [str(device) for device in jax.devices()],
        "jax_platform": jax.default_backend(),
        "x64": True,
        "support_clip_mode": "exact",
        "requested_cells": -110,
        "cases": [],
        "negative_control": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    negative_log.write_text(NEGATIVE_CONTROL + "\n")
    previous = support_clip_mode()
    set_support_clip_mode("exact")
    try:
        for case_name, expected in zip(CASES, EXPECTED_ONE_TRIP, strict=True):
            print(f"BUILD case={case_name}", flush=True)
            carrier_case, source_case, exact = certificate._case(case_name)
            machine = certificate._case_machine(case_name, carrier_case, exact, -110)
            coordinates = np.vstack(
                (machine.node, machine.wall_node, machine.sample_coordinates)
            )
            analytic = certificate._exact_state(case_name, exact, coordinates)
            empty = oracle_fixture.forward_operator(source_case, machine)
            moments, exterior, _cache = oracle_fixture.cached_fixture_exterior(
                source_case, exact, machine, empty, analytic
            )
            operator = oracle_fixture.forward_operator(source_case, machine, exterior)
            profile = ForwardProfile(
                operator,
                StencilMesh(machine.node, machine.stencil, machine.area),
                newton_steps=recovery.NEWTON_STEPS,
            )
            target_current, _centroid, _current_receipt = (
                certificate._closed_form_current_target(
                    case_name, source_case, operator, moments
                )
            )
            request = certificate._certificate_solve_request(
                profile,
                jnp.asarray(analytic, dtype=jnp.float64),
                float(target_current),
                carrier_identity=f"analytic-hex:{case_name}:-110",
            )
            one_request = replace(
                request, policy=replace(request.policy, active_set_steps=1)
            )
            reference, census = _nulls(operator, analytic)
            pitch = float(np.sqrt(np.median(np.asarray(machine.area))))
            offsets = np.asarray(operator.wall_unit_offsets)
            row = {
                "case": case_name,
                "realised_cells": len(machine.node),
                "state_values": len(analytic),
                "characteristic_pitch_m": pitch,
                "reference_nulls": reference,
                "arms": [],
                "wall_units": [
                    [int(start), int(stop), bool(closed), kind]
                    for start, stop, closed, kind in zip(
                        offsets[:-1],
                        offsets[1:],
                        operator.wall_unit_closed,
                        operator.wall_unit_kinds,
                        strict=True,
                    )
                ],
                "analytic_saddle_cluster": _cluster(
                    operator, analytic, reference, census
                ),
                "state_file": f"{case_name}-trip-arms.npz",
            }
            receipt["cases"].append(row)
            states = {}
            for name, arm_request in (
                ("one_trip", one_request),
                ("default_policy", request),
            ):
                arm, state = _arm(
                    profile, arm_request, operator, reference, pitch, name
                )
                row["arms"].append(arm)
                states[name] = state
                _write_json(output / "terminal-state-trip-arms.json", receipt)
            row["one_trip_reproduction"] = {
                "expected_residual": expected,
                "relative_tolerance": 0.01,
                "matches": abs(row["arms"][0]["residual"] / expected - 1) <= 0.01,
            }
            default = row["arms"][1]
            if default["converged"]:
                require_converged(default)
                row["disposition"] = "default_policy_converged"
            else:
                row["disposition"] = "default_policy_nonconverged"
                row["solver_owner"] = "gs-absolute-accuracy-gates"
            np.savez(
                output / row["state_file"],
                coordinates=coordinates,
                wall=machine.wall_node,
                analytic=analytic,
                **states,
            )
            mutation, _ = _arm(
                profile,
                one_request,
                operator,
                reference,
                pitch,
                "default_policy_forced_one_trip",
            )
            try:
                require_converged(mutation)
            except ValueError as error:
                with negative_log.open("a") as stream:
                    stream.write(f"case={case_name} {error}\n")
                receipt["negative_control"].append(
                    {
                        "case": case_name,
                        "refused": True,
                        "reason": str(error),
                        "receipt": mutation,
                    }
                )
            else:
                raise AssertionError("forced-one-trip negative control did not refuse")
            _write_json(output / "terminal-state-trip-arms.json", receipt)
            assert row["one_trip_reproduction"]["matches"], (
                "one-trip residual does not reproduce"
            )
            assert row["arms"][0]["trip_count"] == 1
        render(receipt, output)
        receipt["figure_src"] = (
            "/nova/figures/plasma-cell-read-fidelity/terminal-state-trip-arms.png"
        )
        _write_json(output / "terminal-state-trip-arms.json", receipt)
        print(
            "MEASUREMENT_COMPLETE " + str(output / "terminal-state-trip-arms.json"),
            flush=True,
        )
    finally:
        set_support_clip_mode(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-control-log", type=Path)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--draw-order", choices=DRAW_ORDERS, default="reference-last")
    arguments = parser.parse_args()
    if arguments.render_only:
        receipt_path = arguments.output / "terminal-state-trip-arms.json"
        receipt = json.loads(receipt_path.read_text())
        render(receipt, arguments.output, draw_order=arguments.draw_order)
        _write_json(receipt_path, receipt)
    else:
        if arguments.negative_control_log is None:
            parser.error("--negative-control-log is required for measurement")
        measure(arguments.output, arguments.negative_control_log)


if __name__ == "__main__":
    main()
