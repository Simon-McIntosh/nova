"""Measure an exact-support fixed point across isolated source revisions."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
import importlib.util
import json
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
NEGATIVE_CONTROL = (
    "run the b43714114 arm with active_set_steps forced to one and observe its "
    "receipt read converged false with the one-trip residual, distinguishing "
    "the ladder measurement from a policy artefact"
)
STEM = "exact-mode-fixed-point-attribution"
COMPARISON_STEM = "exact-mode-construction-comparison"


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def load_driver(path):
    spec = importlib.util.spec_from_file_location("terminal_state_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compatible_nulls(driver, operator, state):
    """Read each operator's native census input without changing its solve."""
    if hasattr(operator, "null_flux_pool"):
        return driver._native_nulls(operator, state)
    np, jnp = driver.np, driver.jnp
    _, topology = operator.read(jnp.asarray(state, dtype=jnp.float64))
    if hasattr(operator, "_null_flux_pool"):
        pool = operator._null_flux_pool(jnp.asarray(state))
    else:
        pool, _ = operator.topology.split_flux_map(jnp.asarray(state))
    census = operator._fixed_design_topology.grid.candidate_table_status(pool)
    assert int(census["retained_count"][0]) > 0, "axis positive control is empty"
    saddles = np.asarray(census["retained_candidate"])[1]
    valid = np.asarray(census["retained_valid"])[1]
    return {
        "axis_rz_m": driver._point(topology.axis),
        "x_point_rz_m": driver._point(topology.x_point),
        "qualified_saddles_rz_m": saddles[valid, :2].tolist(),
        "retained_count": np.asarray(census["retained_count"]).tolist(),
        "axis_flux_wb": float(topology.axis_flux),
        "boundary_flux_wb": float(topology.boundary_flux),
    }, census


def moment_diagnostic(driver, operator, state):
    """Read finite moments and the capacity predicate from the terminal support."""
    from nova.equilibrium.clip_quadrature import clipped_support_current_moments
    from nova.equilibrium.forward_operator import flux_field_polynomial

    np, jnp = driver.np, driver.jnp
    masks, _, sample_flux, support = operator._support_partition(jnp.asarray(state))
    field = flux_field_polynomial(
        operator._support_moment_stencils, masks.psi_norm, sample_flux
    )
    selected = field.active & (jnp.asarray(support.vertex_count) >= 3)
    boundary = selected & jnp.asarray(support.boundary, dtype=bool)
    cut_count = int(jnp.sum(boundary))
    capacity = int(operator._cut_cell_bank_capacity)
    values = []

    def integrate(profile, *_args):
        moments = clipped_support_current_moments(
            support, selected, field, profile, cut_cell_capacity=capacity
        )
        values.append(moments)
        return moments

    direct = operator.source.current_moments(
        masks, integrate, support, sample_flux=sample_flux
    )
    actual = operator.cell_current_moments(jnp.asarray(state))
    assert values, "moment-call positive control is empty"
    return {
        "cut_cell_capacity": capacity,
        "selected_cut_cell_count": cut_count,
        "cut_cell_capacity_overflow": cut_count > capacity,
        "capacity_receipt": (
            "recomputed operator integrator predicate: cut_count > capacity"
        ),
        "clipped_support_current_moments_finite": all(
            bool(np.all(np.isfinite(np.asarray(value)))) for value in direct
        ),
        "operator_current_moments_finite": all(
            bool(np.all(np.isfinite(np.asarray(value)))) for value in actual
        ),
        "moment_calls": len(values),
    }


def arm(arguments):
    row = {
        "revision": arguments.revision,
        "case": arguments.case,
        "arm": "forced_one_trip" if arguments.one_trip else "default_policy",
        "status": "not-measured",
        "nova_file": None,
        "exception": None,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    started = perf_counter()
    try:
        import nova

        row["nova_file"] = str(Path(nova.__file__).resolve())
        row["full_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        assert Path(row["nova_file"]).is_relative_to(Path.cwd())
        print(f"NOVA_FILE {row['nova_file']}", flush=True)
        driver = load_driver(arguments.driver)
        driver.configure_dtypes()
        assert driver.jax.config.jax_enable_x64 is True
        assert driver.jax.default_backend() == "gpu"
        driver.set_support_clip_mode("exact")
        driver._native_nulls = driver._nulls
        driver._nulls = lambda operator, state: compatible_nulls(
            driver, operator, state
        )
        np, jnp, certificate = driver.np, driver.jnp, driver.certificate
        carrier, source, exact = certificate._case(arguments.case)
        machine = certificate._case_machine(arguments.case, carrier, exact, -110)
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        analytic = certificate._exact_state(arguments.case, exact, coordinates)
        empty = driver.oracle_fixture.forward_operator(source, machine)
        moments, exterior, _ = driver.oracle_fixture.cached_fixture_exterior(
            source, exact, machine, empty, analytic
        )
        operator = driver.oracle_fixture.forward_operator(source, machine, exterior)
        profile = driver.ForwardProfile(
            operator,
            driver.StencilMesh(machine.node, machine.stencil, machine.area),
            newton_steps=driver.recovery.NEWTON_STEPS,
        )
        target, _, _ = certificate._closed_form_current_target(
            arguments.case, source, operator, moments
        )
        request = certificate._certificate_solve_request(
            profile,
            jnp.asarray(analytic, dtype=jnp.float64),
            float(target),
            carrier_identity=f"analytic-hex:{arguments.case}:-110",
        )
        if arguments.one_trip:
            request = replace(
                request, policy=replace(request.policy, active_set_steps=1)
            )
        reference, _ = driver._nulls(operator, analytic)
        row.update(
            {
                "realised_cells": len(machine.node),
                "reference_nulls": reference,
                "diagnostic_census_input": "null_flux_pool"
                if hasattr(operator, "null_flux_pool")
                else "_null_flux_pool"
                if hasattr(operator, "_null_flux_pool")
                else "native_grid_flux",
            }
        )
        write_json(arguments.output, row)
        measurement, terminal = driver._arm(
            profile,
            request,
            operator,
            reference,
            float(np.sqrt(np.median(np.asarray(machine.area)))),
            row["arm"],
        )
        row.update(measurement)
        row["status"] = "measured"
        row["terminal_residual"] = row["residual"]
        row["wall_units"] = [
            [int(start), int(stop), bool(closed), kind]
            for start, stop, closed, kind in zip(
                operator.wall_unit_offsets[:-1],
                operator.wall_unit_offsets[1:],
                operator.wall_unit_closed,
                operator.wall_unit_kinds,
                strict=True,
            )
        ]
        state_file = arguments.output.with_suffix(".npz")
        np.savez(
            state_file,
            coordinates=coordinates,
            wall=machine.wall_node,
            analytic=analytic,
            terminal=terminal,
        )
        row["state_file"] = state_file.name
        write_json(arguments.output, row)
        try:
            row["moment_diagnostic"] = moment_diagnostic(driver, operator, terminal)
        except Exception:
            row["moment_diagnostic"] = {
                "status": "not-measured",
                "exception": traceback.format_exc(),
            }
        if arguments.one_trip:
            assert row["trip_count"] == 1 and not row["converged"]
            try:
                driver.require_converged(row)
            except ValueError as error:
                row["convergence_refusal"] = str(error)
                print(str(error), flush=True)
            else:
                raise AssertionError("one-trip control did not refuse convergence")
    except Exception:
        row["exception"] = traceback.format_exc()
        row["status"] = "not-measured"
        print(row["exception"], flush=True)
    row["total_wall_seconds"] = perf_counter() - started
    write_json(arguments.output, row)
    return int(row["status"] != "measured")


def attribution(rows):
    results = {}
    for case in CASES:
        selected = [
            row
            for row in rows
            if row["case"] == case and row["arm"] == "default_policy"
        ]
        transitions = [
            {"predecessor": left["revision"], "first_nonconverged": right["revision"]}
            for left, right in zip(selected, selected[1:])
            if left["status"] == right["status"] == "measured"
            and left["converged"]
            and not right["converged"]
        ]
        complete = len(selected) == len(REVISIONS) and all(
            row["status"] == "measured" for row in selected
        )
        verdict = (
            "converged_to_nonconverged_transition"
            if transitions
            else "incomplete_ladder"
            if not complete
            else "every_revision_converged"
            if all(row["converged"] for row in selected)
            else "every_revision_unconverged"
            if all(not row["converged"] for row in selected)
            else "no_converged_to_nonconverged_transition"
        )
        results[case] = {
            "verdict": verdict,
            "first_transition": transitions[0] if transitions else None,
            "complete": complete,
        }
    return results


def measure(arguments):
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "revision_ladder": list(REVISIONS),
        "requested_cells": 110,
        "support_clip_mode": "exact",
        "driver": str(arguments.driver),
        "driver_sha256": hashlib.sha256(arguments.driver.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "rows": [],
    }
    tasks = [(revision, case, False) for revision in REVISIONS for case in CASES]
    tasks.insert(1, (REVISIONS[0], CASES[0], True))
    for revision, case, one_trip in tasks:
        name = f"{revision}-{case}" + ("-one-trip" if one_trip else "")
        tree = arguments.scratch / revision
        row_path = output / f"{name}.json"
        log = arguments.logs / f"{name}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "arm",
            "--driver",
            str(arguments.driver),
            "--revision",
            revision,
            "--case",
            case,
            "--output",
            str(row_path),
        ]
        if one_trip:
            command.append("--one-trip")
        environment = dict(os.environ, PYTHONPATH=str(tree))
        with log.open("w") as stream:
            if one_trip:
                stream.write(NEGATIVE_CONTROL + "\n")
            stream.write(
                f"revision={revision} tree={tree} command={json.dumps(command)}\n"
            )
            stream.flush()
            result = subprocess.run(
                command,
                cwd=tree,
                env=environment,
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
                "arm": "forced_one_trip" if one_trip else "default_policy",
                "status": "not-measured",
                "exception": f"subprocess exited {result.returncode} without receipt",
            }
        )
        row.update({"log": str(log), "exit_status": result.returncode})
        payload["rows"].append(row)
        payload["attribution"] = attribution(payload["rows"])
        write_json(output / f"{STEM}.json", payload)
        print(
            f"ROW {revision} {case} {row['status']} residual={row.get('residual')}",
            flush=True,
        )
    render(payload, output, arguments.driver)
    write_json(output / f"{STEM}.json", payload)
    print("MEASUREMENT_COMPLETE", flush=True)
    return int(any(row["status"] != "measured" for row in payload["rows"]))


def render(payload, output, driver_path):
    """Overlay the reference and terminal fields on one physical level array."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    driver = load_driver(driver_path)
    rows = [
        row
        for row in payload["rows"]
        if row["case"] == CASES[0] and row["arm"] == "default_policy"
    ]
    figure = plt.figure(figsize=(15, 14), constrained_layout=True)
    grid = figure.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.7])
    reference_style = driver.DEFAULT_INK.variant(
        axis_color="#3366cc",
        xpoint_color="#3366cc",
        axis_markersize=8,
        xpoint_markersize=10,
    )
    terminal_style = driver.DEFAULT_INK.variant(axis_markersize=4, xpoint_markersize=5)
    levels = None
    for index, row in enumerate(rows):
        axis = figure.add_subplot(grid[index // 3, index % 3])
        driver.poloidal_axes(axis)
        if row["status"] != "measured":
            axis.text(
                0.5,
                0.5,
                "Not measured\nSee exception in receipt",
                transform=axis.transAxes,
                ha="center",
            )
            axis.set_title(row["revision"])
            continue
        with np.load(output / row["state_file"]) as data:
            units = tuple(
                driver.WallUnit(
                    data["wall"][start:stop, 0],
                    data["wall"][start:stop, 1],
                    closed=closed,
                    kind=kind,
                )
                for start, stop, closed, kind in row["wall_units"]
            )
            if levels is None:
                levels = driver.poloidal.contour_levels(
                    data["analytic"],
                    count=15,
                    axis=row["reference_nulls"]["axis_flux_wb"],
                    boundary=row["reference_nulls"]["boundary_flux_wb"],
                )
            counts = []
            for field, color in (("analytic", "#91ace2"), ("terminal", "#444444")):
                radial, height, raster = driver.certificate._raster_field(
                    data["coordinates"], data[field], data["wall"]
                )
                contours = driver.poloidal.draw_flux_contours(
                    axis, radial, height, raster, levels, color=color
                )
                count = sum(
                    len(part) > 1 for group in contours.allsegs for part in group
                )
                assert count > 0, "contour positive control is empty"
                counts.append(count)
            driver.poloidal.draw_wall(axis, units=units)
            reference_marks = driver._draw_nulls(
                axis, row["reference_nulls"], units, reference_style
            )
            terminal_marks = driver._draw_nulls(
                axis, row["nulls"], units, terminal_style
            )
            caption = (
                f"{row['revision']} | converged={row['converged']}\n"
                f"trips={row['trip_count']} | residual={row['residual']:.6g}"
            )
            axis.set_title(caption, fontsize=10)
            row["panel"] = {
                "caption": caption,
                "contour_segments": counts,
                "axis_off": not axis.axison,
                "reference_markers": reference_marks,
                "terminal_markers": terminal_marks,
            }
    trend = figure.add_subplot(grid[3, :])
    trend.plot(
        range(len(rows)),
        [row.get("residual", np.nan) for row in rows],
        "o-",
        color="#444444",
    )
    trend.set_xticks(range(len(rows)), [row["revision"] for row in rows], rotation=25)
    trend.set_ylabel("Terminal residual")
    trend.set_yscale("log")
    trend.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Diverted exact-support fixed point across source revisions", fontsize=16
    )
    figure.supxlabel(
        "Blue: analytic contours and large null markers. "
        "Gray/red: terminal contours and small null markers.\n"
        "Shared physical levels; triangles mark axes, filled crosses admitted saddles, "
        "hollow crosses other qualified saddles.",
        fontsize=10,
    )
    payload["shared_contour_levels_wb"] = None if levels is None else levels.tolist()
    figure.savefig(output / f"{STEM}.png", dpi=160)
    figure.savefig(output / f"{STEM}.svg")
    plt.close(figure)


def describe_value(value):
    """Retain scalar values and exact array identities for construction comparison."""
    import numpy as np

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(value)
        if array.ndim == 0:
            return describe_value(array.item())
        if array.dtype.hasobject:
            raise TypeError("object arrays have no portable byte identity")
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "finite": bool(np.all(np.isfinite(array))),
        }
    if isinstance(value, dict):
        return {str(key): describe_value(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [describe_value(child) for child in value]
    if is_dataclass(value):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            **{
                entry.name: describe_value(getattr(value, entry.name))
                for entry in fields(value)
            },
        }
    if isinstance(value, type | np.dtype):
        return str(value)
    if isinstance(value, Enum):
        return describe_value(value.value)
    if callable(value):
        from nova.equilibrium.forward_operator import _callable_semantic_identity

        return {"callable_identity": _callable_semantic_identity(value)}
    if hasattr(value, "__dict__"):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "attributes": describe_value(vars(value)),
        }
    raise TypeError(f"No construction serializer for {type(value)}")


def compare_fields(left, right, prefix=""):
    """Emit every differing field, including nested request and operator fields."""
    differences = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left.keys() | right.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.append(
                    {
                        "field": name,
                        "production": left.get(key),
                        "analytic_hex": right.get(key),
                    }
                )
            else:
                differences.extend(compare_fields(left[key], right[key], name))
    elif left != right:
        differences.append({"field": prefix, "production": left, "analytic_hex": right})
    return differences


def build_construction(driver, case_name, route):
    """Build each route independently with its native seed and carrier identity."""
    certificate, np, jnp = driver.certificate, driver.np, driver.jnp
    carrier, source, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier, exact, -110)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(case_name, exact, coordinates)
    empty = driver.oracle_fixture.forward_operator(source, machine)
    moments, exterior, fixture_cache = driver.oracle_fixture.cached_fixture_exterior(
        source, exact, machine, empty, analytic
    )
    operator = driver.oracle_fixture.forward_operator(source, machine, exterior)
    profile = driver.ForwardProfile(
        operator,
        driver.StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=driver.recovery.NEWTON_STEPS,
    )
    target, centroid, current_receipt = certificate._closed_form_current_target(
        case_name, source, operator, moments
    )
    if route == "production":
        seed, requested_class, seed_receipt = certificate._production_seed(
            profile, case_name, target, centroid, current_receipt
        )
        seed_moments = operator.cell_current_moments(seed)
        seed_amplitude = float(
            operator.current_normalisation_amplitude(
                target, jnp.sum(seed_moments.cell_current)
            )
        )
        identity = f"solovev:{case_name}:-110"
    else:
        seed = jnp.asarray(analytic, dtype=jnp.float64)
        requested_class, seed_moments, seed_amplitude = None, None, None
        seed_receipt = {"factory": "certificate._exact_state"}
        identity = f"analytic-hex:{case_name}:-110"
    request = certificate._certificate_solve_request(
        profile, seed, float(target), carrier_identity=identity
    )
    snapshot = {
        "request": describe_value(request),
        "operator": {
            entry.name: describe_value(getattr(operator, entry.name))
            for entry in fields(operator)
        },
        "profile_newton_steps": profile.newton_steps,
        "policy": describe_value(request.policy),
        "active_set_budget": request.policy.active_set_steps,
        "newton_steps": request.policy.newton_steps,
        "exterior_source": (
            "cached_fixture_exterior: analytic total minus exact-density plasma image"
        ),
        "exterior": describe_value(exterior),
        "moments": describe_value(moments),
        "target_current": float(target),
        "clip_mode": driver.support_clip_mode(),
        "cell_polygons": describe_value(machine.cell_polygons),
        "seed_state": describe_value(seed),
        "carrier_identity": identity,
    }
    snapshot["cell_polygons_digest"] = hashlib.sha256(
        json.dumps(snapshot["cell_polygons"], sort_keys=True).encode()
    ).hexdigest()
    provenance = {
        "seed_receipt": describe_value(seed_receipt),
        "requested_class_from_seed": requested_class,
        "seed_moments": describe_value(seed_moments),
        "seed_amplitude": seed_amplitude,
        "fixture_cache": describe_value(fixture_cache),
        "current_receipt": describe_value(current_receipt),
    }
    return {
        "profile": profile,
        "operator": operator,
        "request": request,
        "machine": machine,
        "coordinates": coordinates,
        "analytic": analytic,
        "snapshot": snapshot,
        "provenance": provenance,
    }


def compare_constructions(arguments):
    """Measure both independently built constructions in this one GPU process."""
    import nova

    driver = load_driver(arguments.driver)
    driver.configure_dtypes()
    assert driver.jax.config.jax_enable_x64 is True
    assert driver.jax.default_backend() == "gpu"
    driver.set_support_clip_mode("exact")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert revision.startswith("f5af729a9"), revision
    assert Path(nova.__file__).resolve().is_relative_to(Path.cwd())
    np = driver.np
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{COMPARISON_STEM}.json"
    payload = {
        "revision": revision,
        "nova_file": str(Path(nova.__file__).resolve()),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(arguments.driver.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "process_id": os.getpid(),
        "jax_platform": driver.jax.default_backend(),
        "devices": [str(device) for device in driver.jax.devices()],
        "x64": True,
        "requested_cells": 110,
        "clip_mode": "exact",
        "cases": [],
    }
    write_json(path, payload)
    for case_name in CASES:
        row = {"case": case_name, "arms": [], "differences": None}
        payload["cases"].append(row)
        constructions = {}
        for route in ("production", "analytic_hex"):
            print(f"BUILD {case_name} {route}", flush=True)
            arm_row = {"route": route, "status": "not-measured"}
            row["arms"].append(arm_row)
            try:
                built = build_construction(driver, case_name, route)
                constructions[route] = built
                arm_row.update(
                    {
                        "construction": built["snapshot"],
                        "provenance": built["provenance"],
                    }
                )
                write_json(path, payload)
            except Exception:
                arm_row["exception"] = traceback.format_exc()
                print(arm_row["exception"], flush=True)
                write_json(path, payload)
        if len(constructions) == 2:
            row["differences"] = compare_fields(
                constructions["production"]["snapshot"],
                constructions["analytic_hex"]["snapshot"],
            )
            row["differing_fields"] = [entry["field"] for entry in row["differences"]]
            print(
                f"DIFFERENCES {case_name} {json.dumps(row['differing_fields'])}",
                flush=True,
            )
            write_json(path, payload)
        for arm_row in row["arms"]:
            route = arm_row["route"]
            if route not in constructions:
                continue
            built = constructions[route]
            try:
                reference, _ = driver._nulls(built["operator"], built["analytic"])
                measurement, terminal = driver._arm(
                    built["profile"],
                    built["request"],
                    built["operator"],
                    reference,
                    float(np.sqrt(np.median(np.asarray(built["machine"].area)))),
                    route,
                )
                arm_row.update(measurement)
                arm_row.update(
                    {
                        "status": "measured",
                        "reference_nulls": reference,
                        "realised_cells": len(built["machine"].node),
                        "carrier_identity": built["request"].carrier_identity,
                    }
                )
                operator = built["operator"]
                arm_row["wall_units"] = [
                    [int(start), int(stop), bool(closed), kind]
                    for start, stop, closed, kind in zip(
                        operator.wall_unit_offsets[:-1],
                        operator.wall_unit_offsets[1:],
                        operator.wall_unit_closed,
                        operator.wall_unit_kinds,
                        strict=True,
                    )
                ]
                state_file = f"{case_name}-{route}-construction.npz"
                np.savez(
                    output / state_file,
                    coordinates=built["coordinates"],
                    analytic=built["analytic"],
                    terminal=terminal,
                    seed=np.asarray(built["request"].seed_policy.state),
                    wall=built["machine"].wall_node,
                )
                arm_row["state_file"] = state_file
            except Exception:
                arm_row["exception"] = traceback.format_exc()
                print(arm_row["exception"], flush=True)
            write_json(path, payload)
        measured = [arm for arm in row["arms"] if arm["status"] == "measured"]
        if len(measured) != 2:
            row["conclusion"] = "comparison_not_measured"
        elif all(not arm["converged"] for arm in measured):
            row["conclusion"] = "both_constructions_fail_to_converge"
        elif measured[0]["converged"] and not measured[1]["converged"]:
            row["conclusion"] = "production_converges_analytic_hex_does_not"
            row["candidate_cause_fields"] = row["differing_fields"]
        elif all(arm["converged"] for arm in measured):
            row["conclusion"] = "both_constructions_converge"
        else:
            row["conclusion"] = "analytic_hex_converges_production_does_not"
        write_json(path, payload)
        print(f"CASE_COMPLETE {case_name} {row['conclusion']}", flush=True)
    render_constructions(driver, payload, output)
    payload["status"] = "complete"
    write_json(path, payload)
    print("CONSTRUCTION_COMPARISON_COMPLETE", flush=True)
    return int(
        any(row["conclusion"] == "comparison_not_measured" for row in payload["cases"])
    )


def render_constructions(driver, payload, output):
    """Draw production and analytic-seed terminals with shared physical contours."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(11, 11), constrained_layout=True)
    reference_style = driver.DEFAULT_INK.variant(
        axis_color="#3366cc",
        xpoint_color="#3366cc",
        axis_markersize=8,
        xpoint_markersize=10,
    )
    terminal_style = driver.DEFAULT_INK.variant(axis_markersize=4, xpoint_markersize=5)
    for index, row in enumerate(payload["cases"]):
        levels = None
        for column, arm_row in enumerate(row["arms"]):
            axis = axes[index, column]
            driver.poloidal_axes(axis)
            if arm_row["status"] != "measured":
                axis.set_title(f"{row['case']} / {arm_row['route']}\nNot measured")
                continue
            with np.load(output / arm_row["state_file"]) as data:
                reference = arm_row["reference_nulls"]
                if levels is None:
                    levels = driver.poloidal.contour_levels(
                        data["analytic"],
                        count=15,
                        axis=reference["axis_flux_wb"],
                        boundary=reference["boundary_flux_wb"],
                    )
                units = tuple(
                    driver.WallUnit(
                        data["wall"][start:stop, 0],
                        data["wall"][start:stop, 1],
                        closed=closed,
                        kind=kind,
                    )
                    for start, stop, closed, kind in arm_row["wall_units"]
                )
                counts = []
                for name, color in (("analytic", "#91ace2"), ("terminal", "#444444")):
                    radial, height, raster = driver.certificate._raster_field(
                        data["coordinates"], data[name], data["wall"]
                    )
                    contours = driver.poloidal.draw_flux_contours(
                        axis, radial, height, raster, levels, color=color
                    )
                    count = sum(
                        len(part) > 1 for group in contours.allsegs for part in group
                    )
                    assert count > 0, "contour positive control is empty"
                    counts.append(count)
                driver.poloidal.draw_wall(axis, units=units)
                reference_markers = driver._draw_nulls(
                    axis, reference, units, reference_style
                )
                terminal_markers = driver._draw_nulls(
                    axis, arm_row["nulls"], units, terminal_style
                )
                caption = (
                    f"{row['case']} / {arm_row['route']}\n"
                    f"converged={arm_row['converged']}; trips={arm_row['trip_count']}\n"
                    f"residual={arm_row['residual']:.6g}"
                )
                axis.set_title(caption, fontsize=10)
                arm_row["panel"] = {
                    "caption": caption,
                    "axis_off": not axis.axison,
                    "contour_segments": counts,
                    "reference_markers": reference_markers,
                    "terminal_markers": terminal_markers,
                }
        row["shared_contour_levels_wb"] = None if levels is None else levels.tolist()
    figure.suptitle(
        "Exact clip mode: certificate production and analytic-hex construction",
        fontsize=14,
    )
    figure.supxlabel(
        "Blue contours and large markers: analytic-state read. "
        "Gray contours and small red markers: returned state.\n"
        "Triangles mark axes; filled crosses admitted saddles; "
        "hollow crosses other qualified saddles.",
        fontsize=10,
    )
    figure.savefig(output / f"{COMPARISON_STEM}.png", dpi=160)
    figure.savefig(output / f"{COMPARISON_STEM}.svg")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("measure", "arm", "compare"))
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--one-trip", action="store_true")
    arguments = parser.parse_args()
    if arguments.mode == "compare":
        return compare_constructions(arguments)
    return arm(arguments) if arguments.mode == "arm" else measure(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
