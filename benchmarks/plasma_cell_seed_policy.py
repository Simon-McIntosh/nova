"""Compare two seed policies on the exact-clip certificate route.

The certificate production route is measured at the reduced rung for both
declared cases under two seed policies: the explicit analytic flux sample
the route already carries, and the current-centroid disc seed the last
converging receipt was launched from.  Every solve is the certificate's
own construction -- its own operator, closed-form current target and
solve request.  Per arm the receipt records the seed policy, the seed
state digest, the converged flag, the trip count, the per-trip residual
history, the terminal residual, the terminal and reference saddle with
their distance in metres, and the count of non-finite clipped-support
current moments at the terminal state.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium.forward import (
    ColdSeedConstruction,
    ForwardProfile,
)
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.equilibrium.forward_operator import (
    set_support_clip_mode,
    support_clip_mode,
)
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.sources.frame import WallUnit
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery

CASES = ("diverted-single-null", "weak-rotation-reactor-static")
REQUESTED_CELLS = -110
SEED_POLICIES = ("analytic", "current_centroid_disc")
NEGATIVE_CONTROL = (
    "run the analytic-seed arm of the diverted case with the active-set "
    "budget forced to one trip and observe the receipt read converged "
    "false with residual 0.7166 at trip one, reproducing the production "
    "arm of the exact-mode construction comparison within one percent"
)
EXPECTED_ONE_TRIP_RESIDUAL = 0.7166
ONE_TRIP_RELATIVE_TOLERANCE = 0.01
FIGURE_STEM = "seed-policy-at-main"
FIGURE_URL = "/nova/figures/plasma-cell-read-fidelity-seed"
MOMENT_FIELDS = ("cell_current", "radial_moment", "vertical_moment")


def _digest(state):
    return hashlib.sha256(np.asarray(state, dtype=np.float64).tobytes()).hexdigest()


def _point(value):
    point = np.asarray(value, dtype=np.float64)
    if np.all(np.isfinite(point)):
        return point.tolist()
    return None


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _nulls(operator, state):
    _, topology = operator.read(jnp.asarray(state, dtype=jnp.float64))
    grid = operator._fixed_design_topology.grid
    census = grid.candidate_table_status(operator.null_flux_pool(jnp.asarray(state)))
    assert int(census["retained_count"][0]) > 0, "axis control empty"
    saddles = np.asarray(census["retained_candidate"])[1]
    valid = np.asarray(census["retained_valid"])[1]
    return {
        "axis_rz_m": _point(topology.axis),
        "x_point_rz_m": _point(topology.x_point),
        "qualified_saddles_rz_m": saddles[valid, :2].tolist(),
        "retained_count": np.asarray(census["retained_count"]).tolist(),
        "axis_flux_wb": float(topology.axis_flux),
        "boundary_flux_wb": float(topology.boundary_flux),
    }


def _terminal_moment_counts(operator, state):
    """Count non-finite clipped-support current moments at a state."""
    moments = operator.cell_current_moments(jnp.asarray(state, dtype=jnp.float64))
    counts = {}
    total = 0
    for name in MOMENT_FIELDS:
        field = np.asarray(getattr(moments, name), dtype=np.float64)
        count = int(np.count_nonzero(~np.isfinite(field)))
        counts["nonfinite_" + name] = count
        total += count
    counts["nonfinite_moment_total"] = total
    counts["moment_values"] = int(np.asarray(moments.cell_current).size)
    return counts


def _analytic_seed(analytic):
    state = np.asarray(analytic, dtype=np.float64)
    receipt = {
        "seed_policy": "analytic",
        "seed_state_digest": _digest(state),
    }
    return jnp.asarray(state, dtype=jnp.float64), receipt


def _disc_seed(profile, target_current, centroid):
    """Build the current-centroid disc seed with no diverted geometry."""
    portfolio = profile.cold_seed_portfolio(
        target_current, centroid, diverted_geometry=None
    )
    branches = portfolio.branches
    wanted = int(ColdSeedConstruction.CURRENT_CENTROID_DISC)
    index = branches.construction.tolist().index(wanted)
    state = np.asarray(branches.flux[index], dtype=np.float64)
    receipt = {
        "seed_policy": "current_centroid_disc",
        "seed_state_digest": _digest(state),
        "seed_construction": "current_centroid_disc",
        "seed_branch_index": index,
        "seed_radius_m": float(np.asarray(branches.radius)[index]),
        "supported_cell_count": int(np.asarray(branches.supported_cells)[index]),
        "stored_flux_samples_used": bool(
            np.asarray(branches.stored_flux_samples_used)[index]
        ),
    }
    return jnp.asarray(state, dtype=jnp.float64), receipt


def _arm(profile, request, operator, reference, pitch, seed_receipt):
    started = perf_counter()
    receipt = profile.solve(request)
    equilibrium = receipt.equilibrium
    state = np.asarray(jax.block_until_ready(equilibrium.flux), dtype=np.float64)
    assert np.all(np.isfinite(state)), "terminal state carries non-finite flux"
    history = equilibrium.fixed_point
    trips = int(history.active_set_iterations)
    residuals = np.asarray(history.active_set_residuals)[:trips]
    assert len(residuals) == trips and np.all(np.isfinite(residuals))
    nulls = _nulls(operator, state)
    reference_x = reference["x_point_rz_m"]
    terminal_x = nulls["x_point_rz_m"]
    distance = None
    if reference_x is not None and terminal_x is not None:
        distance = float(np.linalg.norm(np.asarray(terminal_x) - reference_x))
    row = {
        "seed_policy": seed_receipt["seed_policy"],
        "seed_receipt": seed_receipt,
        "active_set_budget": request.policy.active_set_steps,
        "terminal_residual": float(history.residual),
        "converged": bool(history.converged),
        "qualified": bool(receipt.qualified),
        "trip_count": trips,
        "per_trip_residual_history": [
            {"trip": index + 1, "residual": float(value)}
            for index, value in enumerate(residuals)
        ],
        "terminal_axis_rz_m": nulls["axis_rz_m"],
        "terminal_x_point_rz_m": terminal_x,
        "reference_x_point_rz_m": reference_x,
        "x_point_distance_m": distance,
        "certificate_position_bound_m": (
            certificate.TOPOLOGY_POSITION_BOUND_PITCHES * pitch
        ),
        "certificate_residual_bound": certificate.TERMINAL_RESIDUAL_BOUND,
        "jax_platform": jax.default_backend(),
        "terminal_nulls": nulls,
        "termination_reason": int(receipt.termination_reason),
        "wall_seconds": perf_counter() - started,
        "compilation_cache_hit": bool(receipt.compilation_cache_hit),
    }
    row.update(_terminal_moment_counts(operator, state))
    print("ARM " + json.dumps(row, allow_nan=False), flush=True)
    return row, state


def require_converged(row):
    """Refuse a convergence claim the solve receipt does not support."""
    if (
        not row["converged"]
        or not row["qualified"]
        or not np.isfinite(row["terminal_residual"])
        or row["terminal_residual"] > row["certificate_residual_bound"]
    ):
        raise ValueError(
            "converged assertion refused: "
            f"{row['seed_policy']} residual={row['terminal_residual']:.12g} "
            f"converged={row['converged']} trips={row['trip_count']}"
        )
    if row["reference_x_point_rz_m"] is not None:
        distance = row["x_point_distance_m"]
        if distance is None or distance > row["certificate_position_bound_m"]:
            raise ValueError(
                "converged assertion refused: terminal saddle "
                "exceeds the certificate position bound"
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


def render(receipt, output):
    """Draw the analytic state beside both terminal states per case."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(receipt["cases"]), 3, figsize=(12, 9), constrained_layout=True
    )
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
            states = [data["analytic"], data["analytic_seed"], data["disc_seed"]]
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
            radial, height, raster = certificate._raster_field(coordinates, state, wall)
            contour = poloidal.draw_flux_contours(
                axis, radial, height, raster, levels, color="#444444"
            )
            segments = sum(len(part) > 1 for group in contour.allsegs for part in group)
            assert segments > 0, "contour positive control is empty"
            poloidal.draw_wall(axis, units=units)
            if column == 0:
                nulls = reference
            else:
                nulls = row["arms"][column - 1]["terminal_nulls"]
            reference_tally = _draw_nulls(axis, reference, units, reference_style)
            panel_tally = _draw_nulls(axis, nulls, units, panel_style)
            poloidal_axes(axis)
            if column == 0:
                caption = "Analytic input\nresidual=n/a; converged=n/a; trips=0"
            else:
                arm = row["arms"][column - 1]
                caption = (
                    f"seed={arm['seed_policy']}\n"
                    f"residual={arm['terminal_residual']:.6g}; "
                    f"converged={arm['converged']}; trips={arm['trip_count']}"
                )
            axis.set_title(caption, fontsize=10)
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
        "Exact clip: analytic input and both seed-policy terminal states",
        fontsize=15,
    )
    figure.supxlabel(
        "Blue large markers: analytic-state read; "
        "red small markers: panel-state read.\n"
        "Filled triangle: axis; filled cross: admitted saddle; "
        "hollow crosses: other qualified saddles. One shared level array.",
        fontsize=10,
    )
    figure.savefig(output / (FIGURE_STEM + ".png"), dpi=160)
    figure.savefig(output / (FIGURE_STEM + ".svg"))
    plt.close(figure)


def _construction(case_name):
    """Build the certificate route for one case at the reduced rung."""
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, REQUESTED_CELLS)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(case_name, exact, coordinates)
    empty = oracle_fixture.forward_operator(source_case, machine)
    exact_physical, exterior, _cache = oracle_fixture.cached_fixture_exterior(
        source_case, exact, machine, empty, analytic
    )
    operator = oracle_fixture.forward_operator(source_case, machine, exterior)
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        case_name, source_case, operator, exact_physical
    )
    return {
        "case_name": case_name,
        "operator": operator,
        "profile": profile,
        "machine": machine,
        "coordinates": coordinates,
        "analytic": analytic,
        "target_current": float(target_current),
        "centroid": np.asarray(centroid, dtype=np.float64),
        "current_receipt": current_receipt,
    }


def _run_case(built, receipt, output, negative_log):
    case_name = built["case_name"]
    operator = built["operator"]
    profile = built["profile"]
    machine = built["machine"]
    reference = _nulls(operator, built["analytic"])
    pitch = float(np.sqrt(np.median(np.asarray(machine.area))))
    offsets = np.asarray(operator.wall_unit_offsets)
    row = {
        "case": case_name,
        "requested_cells": REQUESTED_CELLS,
        "realised_cells": len(machine.node),
        "characteristic_pitch_m": pitch,
        "reference_nulls": reference,
        "arms": [],
        "state_file": case_name + "-seed-policy.npz",
        "wall_units": [
            [int(a), int(b), bool(c), k]
            for a, b, c, k in zip(
                offsets[:-1],
                offsets[1:],
                operator.wall_unit_closed,
                operator.wall_unit_kinds,
                strict=True,
            )
        ],
    }
    receipt["cases"].append(row)
    analytic_seed, analytic_receipt = _analytic_seed(built["analytic"])
    disc_seed, disc_receipt = _disc_seed(
        profile, built["target_current"], built["centroid"]
    )
    seeds = {
        "analytic": (analytic_seed, analytic_receipt),
        "current_centroid_disc": (disc_seed, disc_receipt),
    }
    carrier = f"solovev:{case_name}:{REQUESTED_CELLS}"
    states = {}
    for policy in SEED_POLICIES:
        seed, seed_receipt = seeds[policy]
        request = certificate._certificate_solve_request(
            profile, seed, float(built["target_current"]), carrier_identity=carrier
        )
        arm, state = _arm(profile, request, operator, reference, pitch, seed_receipt)
        row["arms"].append(arm)
        states["analytic_seed" if policy == "analytic" else "disc_seed"] = state
        _write_json(output / (FIGURE_STEM + ".json"), receipt)
    analytic_arm, disc_arm = row["arms"][0], row["arms"][1]
    if disc_arm["converged"] and not analytic_arm["converged"]:
        verdict = "disc_converges_where_analytic_does_not"
    elif analytic_arm["converged"] and not disc_arm["converged"]:
        verdict = "analytic_converges_where_disc_does_not"
    elif analytic_arm["converged"] and disc_arm["converged"]:
        verdict = "both_converge"
    else:
        verdict = "both_fail"
    row["seed_policy_verdict"] = verdict
    print("VERDICT " + case_name + " " + verdict, flush=True)
    np.savez(
        output / row["state_file"],
        coordinates=built["coordinates"],
        wall=machine.wall_node,
        analytic=built["analytic"],
        **states,
    )
    if case_name == CASES[0] and analytic_arm["seed_policy"] == "analytic":
        request = certificate._certificate_solve_request(
            profile,
            seeds["analytic"][0],
            float(built["target_current"]),
            carrier_identity=carrier,
        )
        one_request = replace(
            request, policy=replace(request.policy, active_set_steps=1)
        )
        mutation, _state = _arm(
            profile, one_request, operator, reference, pitch, seeds["analytic"][1]
        )
        _one_trip_control(mutation, row, receipt, negative_log)
    _write_json(output / (FIGURE_STEM + ".json"), receipt)


def _one_trip_control(mutation, row, receipt, negative_log):
    """The declared control: one trip must refuse the convergence claim."""
    residual = mutation["terminal_residual"]
    relative = abs(residual / EXPECTED_ONE_TRIP_RESIDUAL - 1.0)
    record = {
        "case": row["case"],
        "declared_control": NEGATIVE_CONTROL,
        "one_trip_residual": residual,
        "expected_one_trip_residual": EXPECTED_ONE_TRIP_RESIDUAL,
        "relative_tolerance": ONE_TRIP_RELATIVE_TOLERANCE,
        "reproduces_production_arm": relative <= ONE_TRIP_RELATIVE_TOLERANCE,
        "converged": mutation["converged"],
        "refused": False,
    }
    try:
        require_converged(mutation)
    except ValueError as error:
        record["refused"] = True
        record["reason"] = str(error)
        with negative_log.open("a") as stream:
            stream.write("case=" + row["case"] + " " + str(error) + "\n")
    else:
        raise AssertionError("forced-one-trip control did not refuse")
    assert record["reproduces_production_arm"], "one-trip residual differs"
    assert mutation["trip_count"] == 1
    record["receipt"] = mutation
    receipt["negative_control"].append(record)


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
        "requested_cells": REQUESTED_CELLS,
        "cases": [],
        "negative_control": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    negative_log.write_text(NEGATIVE_CONTROL + "\n")
    previous = support_clip_mode()
    set_support_clip_mode("exact")
    try:
        for case_name in CASES:
            print("BUILD case=" + case_name, flush=True)
            built = _construction(case_name)
            _run_case(built, receipt, output, negative_log)
        render(receipt, output)
        receipt["figure_src"] = FIGURE_URL + "/" + FIGURE_STEM + ".png"
        _write_json(output / (FIGURE_STEM + ".json"), receipt)
    finally:
        set_support_clip_mode(previous)
    print("MEASUREMENT_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-control-log", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.negative_control_log.parent.is_dir():
        parser.error("negative control log directory is absent")
    if arguments.render_only:
        receipt_path = arguments.output / (FIGURE_STEM + ".json")
        receipt = json.loads(receipt_path.read_text())
        render(receipt, arguments.output)
        _write_json(receipt_path, receipt)
        return
    if arguments.negative_control_log is None:
        parser.error("--negative-control-log is required for measurement")
    measure(arguments.output, arguments.negative_control_log)


if __name__ == "__main__":
    main()
