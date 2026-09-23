"""Measure certificate production solves across isolated source revisions.

The observer copies selected trip states from the native reconcile function.
It leaves every operand, branch and return value intact; support health is
recomputed after the solve from those states with the revision's own integrator.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import traceback


REVISIONS = (
    "b43714114",
    "08dd0dda1",
    "ee17f9570",
    "2e38dc877",
    "337eb81ee",
    "23a8522b6",
    "450bab78a",
    "a17431153",
    "f5af729a9",
)
CASES = ("diverted-single-null", "weak-rotation-reactor-static")
STEM = "production-route-ladder"
NEGATIVE_CONTROL = (
    "run the main rung with the production route in chord clip mode (the current "
    "production default) and observe whether it converges, so the receipt "
    "separates exact clip mode from the route itself"
)


def strict_json(value):
    """Retain missing numerical values as JSON nulls, never nonstandard tokens."""
    if isinstance(value, dict):
        return {key: strict_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [strict_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "tolist"):
        return strict_json(value.tolist())
    return value


def write_json(path, payload):
    temporary = path.with_suffix(".pending")
    temporary.write_text(
        json.dumps(strict_json(payload), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def finite_float(value):
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def nulls(operator, state):
    import jax.numpy as jnp
    import numpy as np

    state = jnp.asarray(state)
    _, topology = operator.read(state)
    if hasattr(operator, "null_flux_pool"):
        pool = operator.null_flux_pool(state)
    elif hasattr(operator, "_null_flux_pool"):
        pool = operator._null_flux_pool(state)
    else:
        pool, _ = operator.topology.split_flux_map(state)
    census = operator._fixed_design_topology.grid.candidate_table_status(pool)
    assert int(census["retained_count"][0]) > 0, "known-present axis was not read"
    candidates = np.asarray(census["retained_candidate"])[1]
    valid = np.asarray(census["retained_valid"])[1]

    def point(value):
        array = np.asarray(value)
        return array.tolist() if np.all(np.isfinite(array)) else None

    return {
        "axis_rz_m": point(topology.axis),
        "x_point_rz_m": point(topology.x_point),
        "qualified_saddles_rz_m": candidates[valid, :2].tolist(),
        "retained_count": np.asarray(census["retained_count"]).tolist(),
        "axis_flux_wb": finite_float(topology.axis_flux),
        "boundary_flux_wb": finite_float(topology.boundary_flux),
    }


def observe_trips():
    """Insert a host observation at the native selected-state boundary."""
    import numpy as np
    from nova.equilibrium import fixed_point

    states = {}

    def capture(active, index, state, residual):
        if bool(active):
            states[int(index)] = (np.asarray(state).copy(), float(residual))

    native = fixed_point._active_set_newton_krylov
    source = inspect.getsource(native)
    marker = "        if stream_active_set:\n"
    assert source.count(marker) == 1, "trip observation boundary is ambiguous"
    insertion = (
        "        jax.debug.callback(_capture_trip, trip_active, index, "
        "selected_state, selected_residual, ordered=True)\n"
    )
    namespace = dict(native.__globals__, _capture_trip=capture)
    exec(
        compile(
            source.replace(marker, insertion + marker), inspect.getfile(native), "exec"
        ),
        namespace,
    )
    fixed_point._active_set_newton_krylov = namespace[native.__name__]
    return states, {
        "method": "host callback copies selected state at native trip reconcile",
        "native_function_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "inserted_statement": insertion.strip(),
        "solver_values_modified": False,
    }


def support_reader(operator):
    """Return a compiled native-moment census with explicit overflow provenance."""
    import jax
    import jax.numpy as jnp
    from nova.equilibrium.forward_operator import flux_field_polynomial

    def count_nonfinite(values):
        return sum(jnp.sum(~jnp.isfinite(value)) for value in values)

    positive = int(count_nonfinite((jnp.array([0.0, jnp.nan, jnp.inf]),)))
    assert positive == 2, "non-finite counter missed two known-present entries"

    @jax.jit
    def read(state):
        masks, _, samples, support = operator._support_partition(state)
        if hasattr(operator, "_moment_support_masks"):
            masks = operator._moment_support_masks(masks, support)
        counts = []
        cuts = []

        def integrate(profile, centroid_flux, sample_flux, selected_support):
            values = operator.support_current_moments(
                profile, centroid_flux, sample_flux, selected_support
            )
            field = flux_field_polynomial(
                operator._support_moment_stencils, centroid_flux, sample_flux
            )
            selected = field.active & (selected_support.vertex_count >= 3)
            cuts.append(jnp.sum(selected & selected_support.boundary))
            counts.append(count_nonfinite(values))
            return values

        moments = operator.source.current_moments(
            masks, integrate, support, sample_flux=samples
        )
        assert counts, "native clipped-support moment callable was not reached"
        return jnp.stack(counts), jnp.stack(cuts), count_nonfinite(moments)

    source = inspect.getsource(type(operator).support_current_moments)
    return read, {
        "counter_positive_control": {"nonfinite_entries": positive, "expected": 2},
        "integrator": source,
        "capacity_receipt": (
            "no public overflow receipt; recomputed native selected boundary "
            "count > cut-cell bank capacity"
        ),
        "cut_cell_capacity": int(operator._cut_cell_bank_capacity),
    }


def arm(args):
    row = {
        "revision": args.revision,
        "case": args.case,
        "clip_mode": args.mode,
        "status": "not-measured",
        "exception": None,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    started = perf_counter()
    try:
        import nova
        import jax
        from nova.jax.config import configure_dtypes

        configure_dtypes()
        assert jax.config.jax_enable_x64 is True
        import jax.numpy as jnp
        import numpy as np
        from benchmarks import solovev_certificate as certificate
        from nova.equilibrium.forward import ForwardProfile
        from nova.equilibrium.forward_operator import set_support_clip_mode
        from nova.equilibrium.stencil_mesh import StencilMesh
        from scripts.analytic_oracle_fixtures import measure as fixture
        from scripts.oracle_rebaseline import measure as recovery

        row["nova_file"] = str(Path(nova.__file__).resolve())
        row["full_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        assert Path(row["nova_file"]).is_relative_to(Path.cwd())
        assert row["full_revision"].startswith(args.revision)
        assert jax.default_backend() == "gpu"
        print(f"NOVA_FILE {row['nova_file']}", flush=True)
        row["devices"] = [str(device) for device in jax.devices()]
        set_support_clip_mode(args.mode)
        carrier, source, exact = certificate._case(args.case)
        machine = certificate._case_machine(args.case, carrier, exact, -110)
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        analytic = certificate._exact_state(args.case, exact, coordinates)
        empty = fixture.forward_operator(source, machine)
        moments, exterior, fixture_cache = fixture.cached_fixture_exterior(
            source, exact, machine, empty, analytic
        )
        operator = fixture.forward_operator(source, machine, exterior)
        profile = ForwardProfile(
            operator,
            StencilMesh(machine.node, machine.stencil, machine.area),
            newton_steps=recovery.NEWTON_STEPS,
        )
        target, centroid, current_receipt = certificate._closed_form_current_target(
            args.case, source, operator, moments
        )
        seed, _, seed_receipt = certificate._production_seed(
            profile, args.case, target, centroid, current_receipt
        )
        seed_moments = operator.cell_current_moments(seed)
        seed_amplitude = float(
            operator.current_normalisation_amplitude(
                target, jnp.sum(seed_moments.cell_current)
            )
        )
        identity = f"solovev:{args.case}:-110"
        request = certificate._certificate_solve_request(
            profile, seed, float(target), carrier_identity=identity
        )
        reference = nulls(operator, analytic)
        row.update(
            {
                "realised_cells": len(machine.node),
                "carrier_identity": identity,
                "reference_nulls": reference,
                "seed_receipt": seed_receipt,
                "seed_amplitude": finite_float(seed_amplitude),
                "fixture_cache": fixture_cache,
                "current_receipt": current_receipt,
                "seed_sha256": hashlib.sha256(np.asarray(seed).tobytes()).hexdigest(),
                "active_set_budget": request.policy.active_set_steps,
            }
        )
        write_json(args.output, row)
        trip_states, row["observer"] = observe_trips()
        solve = profile.solve(request)
        terminal = np.asarray(jax.block_until_ready(solve.equilibrium.flux))
        jax.effects_barrier()
        history = solve.equilibrium.fixed_point
        trips = int(history.active_set_iterations)
        residuals = np.asarray(history.active_set_residuals)[:trips]
        assert sorted(trip_states) == list(range(trips)), (
            "trip state census is incomplete"
        )
        np.testing.assert_allclose(
            [trip_states[i][1] for i in range(trips)], residuals, rtol=1e-13, atol=0
        )
        terminal_nulls = nulls(operator, terminal)
        reference_x, terminal_x = (
            reference["x_point_rz_m"],
            terminal_nulls["x_point_rz_m"],
        )
        row.update(
            {
                "status": "measured",
                "converged": bool(history.converged),
                "qualified": bool(solve.qualified),
                "trip_count": trips,
                "terminal_residual": finite_float(history.residual),
                "per_trip_residual_history": [
                    {"trip": i + 1, "residual": finite_float(r)}
                    for i, r in enumerate(residuals)
                ],
                "terminal_nulls": terminal_nulls,
                "terminal_axis_rz_m": terminal_nulls["axis_rz_m"],
                "terminal_x_point_rz_m": terminal_x,
                "reference_axis_rz_m": reference["axis_rz_m"],
                "reference_x_point_rz_m": reference_x,
                "x_point_distance_m": float(
                    np.linalg.norm(np.asarray(terminal_x) - reference_x)
                )
                if reference_x is not None and terminal_x is not None
                else None,
                "x_point_distance_status": "measured"
                if reference_x is not None and terminal_x is not None
                else "no_admitted_reference_saddle"
                if reference_x is None
                else "terminal_saddle_absent",
                "resolved_defaults": solve.resolved_defaults.to_dict(),
                "termination_reason": int(solve.termination_reason),
                "wall_units": [
                    [int(a), int(b), bool(c), k]
                    for a, b, c, k in zip(
                        operator.wall_unit_offsets[:-1],
                        operator.wall_unit_offsets[1:],
                        operator.wall_unit_closed,
                        operator.wall_unit_kinds,
                        strict=True,
                    )
                ],
            }
        )
        state_path = args.output.with_suffix(".npz")
        np.savez(
            state_path,
            coordinates=coordinates,
            wall=machine.wall_node,
            analytic=analytic,
            terminal=terminal,
            trip_states=np.stack([trip_states[i][0] for i in range(trips)]),
        )
        row["state_file"] = state_path.name
        write_json(args.output, row)
        reader, health = support_reader(operator)
        health["per_trip"] = []
        for index in range(trips):
            counts, cuts, total = jax.device_get(
                reader(jnp.asarray(trip_states[index][0]))
            )
            health["per_trip"].append(
                {
                    "trip": index + 1,
                    "nonfinite_clipped_moment_entries": int(np.sum(counts)),
                    "nonfinite_entries_per_profile": counts.tolist(),
                    "nonfinite_combined_moment_entries": int(total),
                    "selected_cut_cells_per_profile": cuts.tolist(),
                    "cut_cell_capacity_overflow": bool(
                        np.any(cuts > health["cut_cell_capacity"])
                    ),
                }
            )
        health["max_nonfinite_clipped_moment_entries"] = max(
            x["nonfinite_clipped_moment_entries"] for x in health["per_trip"]
        )
        row["support_health"] = health
        row["support_health_status"] = "measured"
        print(
            f"RESULT converged={row['converged']} trips={trips} "
            f"residual={row['terminal_residual']} "
            f"nonfinite={health['max_nonfinite_clipped_moment_entries']}",
            flush=True,
        )
    except Exception:
        row["exception"] = traceback.format_exc()
        if row["status"] == "measured":
            row["support_health_status"] = "not-measured"
        print(row["exception"], flush=True)
    row["wall_seconds"] = perf_counter() - started
    write_json(args.output, row)
    return int(
        row["status"] != "measured" or row.get("support_health_status") != "measured"
    )


def attribution(rows):
    result = {}
    for case in CASES:
        ordered = [
            next(
                (
                    r
                    for r in rows
                    if r["case"] == case
                    and r["revision"] == rev
                    and r["clip_mode"] == "exact"
                ),
                None,
            )
            for rev in REVISIONS
        ]
        measured = [r for r in ordered if r and r["status"] == "measured"]
        transitions = [
            {"predecessor": a["revision"], "first_nonconverged": b["revision"]}
            for a, b in zip(ordered, ordered[1:])
            if a
            and b
            and a["status"] == b["status"] == "measured"
            and a["converged"]
            and not b["converged"]
        ]
        verdict = (
            "converged_to_nonconverged_transition"
            if transitions
            else "all_measured_revisions_converged"
            if measured and all(r["converged"] for r in measured)
            else "all_measured_revisions_unconverged"
            if measured and all(not r["converged"] for r in measured)
            else "no_adjacent_transition_attributed"
        )
        result[case] = {
            "verdict": verdict,
            "first_transition": transitions[0] if transitions else None,
            "attempted": sum(r is not None for r in ordered),
            "measured": len(measured),
            "unmeasured_revisions": [
                rev
                for rev, r in zip(REVISIONS, ordered, strict=True)
                if not r or r["status"] != "measured"
            ],
        }
    return result


def draw_nulls(axis, reading, units, style):
    import numpy as np
    from nova.media import poloidal

    admitted = reading["x_point_rz_m"]
    others = np.asarray(reading["qualified_saddles_rz_m"]).reshape(-1, 2)
    if admitted is not None:
        others = others[np.linalg.norm(others - admitted, axis=1) > 1e-10]
    return poloidal.draw_nulls(
        axis,
        magnetic_axis=reading["axis_rz_m"],
        x_points=admitted,
        other_x_points=others,
        contain=units,
        style=style,
    )


def render(payload, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from benchmarks import solovev_certificate as certificate
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes
    from nova.media.sources.frame import WallUnit

    rows = {
        r["revision"]: r
        for r in payload["rows"]
        if r["case"] == CASES[0] and r["clip_mode"] == "exact"
    }
    measured = [
        rows[r] for r in REVISIONS if r in rows and rows[r]["status"] == "measured"
    ]
    if not measured:
        return
    ref = measured[0]["reference_nulls"]
    with np.load(output / measured[0]["state_file"]) as data:
        levels = poloidal.contour_levels(
            data["analytic"],
            count=15,
            axis=ref["axis_flux_wb"],
            boundary=ref["boundary_flux_wb"],
        )
    payload["shared_contour_levels_wb"] = levels.tolist()
    figure = plt.figure(figsize=(15, 14), constrained_layout=True)
    grid = figure.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.65])
    reference_style = DEFAULT_INK.variant(
        axis_color="#3366cc",
        xpoint_color="#3366cc",
        axis_markersize=9,
        xpoint_markersize=11,
    )
    terminal_style = DEFAULT_INK.variant(axis_markersize=4, xpoint_markersize=5)
    for index, revision in enumerate(REVISIONS):
        axis = figure.add_subplot(grid[index // 3, index % 3])
        poloidal_axes(axis)
        row = rows.get(revision)
        if row is None or row["status"] != "measured":
            axis.text(
                0.5,
                0.5,
                "Not measured\n"
                + ("pending" if row is None else "construction exception; see receipt"),
                transform=axis.transAxes,
                ha="center",
            )
            axis.set_title(revision)
            continue
        with np.load(output / row["state_file"]) as data:
            units = tuple(
                WallUnit(data["wall"][a:b, 0], data["wall"][a:b, 1], closed=c, kind=k)
                for a, b, c, k in row["wall_units"]
            )
            segments = []
            for field, color in (("analytic", "#91ace2"), ("terminal", "#444444")):
                radial, vertical, raster = certificate._raster_field(
                    data["coordinates"], data[field], data["wall"]
                )
                contours = poloidal.draw_flux_contours(
                    axis, radial, vertical, raster, levels, color=color
                )
                count = sum(
                    len(part) > 1 for group in contours.allsegs for part in group
                )
                assert count > 0, "known-present contour field was not rendered"
                segments.append(count)
            poloidal.draw_wall(axis, units=units)
            reference_marks = draw_nulls(
                axis, row["reference_nulls"], units, reference_style
            )
            terminal_marks = draw_nulls(
                axis, row["terminal_nulls"], units, terminal_style
            )
        bad = row.get("support_health", {}).get(
            "max_nonfinite_clipped_moment_entries", "unmeasured"
        )
        caption = (
            f"{revision} | converged={row['converged']}\n"
            f"trips={row['trip_count']} | "
            f"residual={row['terminal_residual']:.6g}\n"
            f"non-finite moment entries (max/trip)={bad}"
        )
        axis.set_title(caption, fontsize=9)
        row["panel"] = {
            "caption": caption,
            "axis_off": not axis.axison,
            "contour_segments": segments,
            "reference_markers": reference_marks,
            "terminal_markers": terminal_marks,
        }
    trend = figure.add_subplot(grid[3, :])
    for case, color in zip(CASES, ("#444444", "#bb5533"), strict=True):
        values = {
            r["revision"]: r.get("terminal_residual")
            for r in payload["rows"]
            if r["case"] == case and r["clip_mode"] == "exact"
        }
        trend.plot(
            range(len(REVISIONS)),
            [values.get(r) or np.nan for r in REVISIONS],
            "o-",
            color=color,
            label=case,
        )
    trend.set_xticks(range(len(REVISIONS)), REVISIONS, rotation=25)
    trend.set_yscale("log")
    trend.set_ylabel("Terminal residual")
    trend.legend(fontsize=9)
    trend.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Certificate production route in exact clip mode — diverted case", fontsize=16
    )
    figure.supxlabel(
        "Blue contours and large blue nulls: analytic state. "
        "Gray contours and small red nulls: terminal state.\n"
        "Triangles: axes; filled crosses: admitted saddles; "
        "hollow crosses: other qualified saddles. One shared physical level array.",
        fontsize=10,
    )
    figure.savefig(output / f"{STEM}.png", dpi=160)
    figure.savefig(output / f"{STEM}.svg")
    plt.close(figure)


def measure(args):
    args.output.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{STEM}.json"
    payload = {
        "revision_ladder": list(REVISIONS),
        "execution_order": list(reversed(REVISIONS)),
        "requested_cells": 110,
        "route": "certificate production current-moment seed",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "negative_control": NEGATIVE_CONTROL,
        "rows": [],
    }
    write_json(path, payload)
    tasks = [(r, c, "exact") for r in reversed(REVISIONS) for c in CASES]
    tasks[2:2] = [(REVISIONS[-1], c, "chord") for c in CASES]
    for revision, case, mode in tasks:
        tree = args.scratch / revision
        name = f"{revision}-{case}-{mode}"
        row_path = args.output / f"{name}.json"
        log = args.logs / f"{name}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "arm",
            "--revision",
            revision,
            "--case",
            case,
            "--mode",
            mode,
            "--output",
            str(row_path),
        ]
        with log.open("w") as stream:
            if mode == "chord":
                stream.write(NEGATIVE_CONTROL + "\n")
            stream.write(
                f"revision={revision} tree={tree} command={json.dumps(command)}\n"
            )
            stream.flush()
            result = subprocess.run(
                command,
                cwd=tree,
                env=dict(os.environ, PYTHONPATH=str(tree)),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        row = (
            json.loads(row_path.read_text())
            if row_path.exists()
            else {
                "revision": revision,
                "case": case,
                "clip_mode": mode,
                "status": "not-measured",
                "exception": f"subprocess exited {result.returncode} without receipt",
            }
        )
        row.update(log=str(log), exit_status=result.returncode)
        payload["rows"].append(row)
        payload["attribution"] = attribution(payload["rows"])
        write_json(path, payload)
        render(payload, args.output)
        write_json(path, payload)
        print(
            f"ROW {revision} {case} {mode}: {row['status']} "
            f"residual={row.get('terminal_residual')}",
            flush=True,
        )
    payload["gate_passed"] = evidence_complete(payload)
    payload["status"] = "complete" if payload["gate_passed"] else "incomplete"
    write_json(path, payload)
    print(f"MEASUREMENT_COMPLETE gate_passed={payload['gate_passed']}", flush=True)
    return int(not payload["gate_passed"])


def evidence_complete(payload):
    """Refuse a complete ladder when an expected arm or health census is absent."""
    optional_build_failures = {"08dd0dda1", "ee17f9570", "2e38dc877"}
    expected = {(revision, case, "exact") for revision in REVISIONS for case in CASES}
    expected |= {(REVISIONS[-1], case, "chord") for case in CASES}
    rows = payload["rows"]
    identities = {(row["revision"], row["case"], row["clip_mode"]) for row in rows}
    if identities != expected or len(rows) != len(expected):
        return False
    return all(
        (
            row["status"] == "measured"
            and row.get("support_health_status") == "measured"
            and len(row["support_health"]["per_trip"]) == row["trip_count"]
            and row["exit_status"] == 0
        )
        or (
            row["revision"] in optional_build_failures
            and row["status"] == "not-measured"
            and bool(row.get("exception"))
        )
        for row in rows
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    single = sub.add_parser("arm")
    single.add_argument("--revision", required=True)
    single.add_argument("--case", required=True, choices=CASES)
    single.add_argument("--mode", required=True, choices=("exact", "chord"))
    single.add_argument("--output", type=Path, required=True)
    all_rows = sub.add_parser("measure")
    all_rows.add_argument("--scratch", type=Path, required=True)
    all_rows.add_argument("--output", type=Path, required=True)
    all_rows.add_argument("--logs", type=Path, required=True)
    validation = sub.add_parser("validate")
    validation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        payload = json.loads(args.output.read_text())
        passed = evidence_complete(payload)
        print(
            "COMPLETE" if passed else "REFUSED: incomplete production ladder evidence"
        )
        return int(not passed)
    return arm(args) if args.command == "arm" else measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
