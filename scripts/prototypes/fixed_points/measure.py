"""Measure moving stationary points, bounded-row storage and split membership."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import hashlib
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, topology
from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures.measure import analytic_profile
from scripts.prototypes.fixed_points.solve import (
    CachedField,
    KernelField,
    build_rows,
    select_cache_nodes,
    solve,
    solve_adaptive,
)
from scripts.prototypes.read_route.bell import build_mesh
from scripts.prototypes.read_route.check_element import stack_jet
from tests.equilibrium import test_topology_read as fixture

assert jax.config.jax_enable_x64 is True
ROOT = Path(__file__).resolve().parents[3]


def persist(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")
    temporary.replace(path)


def compile_solver(field, seeds, pitch, adaptive=False):
    start = perf_counter()
    executable = (
        jax.jit(solve_adaptive if adaptive else solve)
        .lower(field, seeds, pitch)
        .compile()
    )
    cold = perf_counter() - start
    try:
        size = len(executable.runtime_executable().serialize())
    except AttributeError, RuntimeError:
        size = None
    jax.block_until_ready(executable(field, seeds, pitch))
    return executable, cold, size


def run_sweep(
    executable, fields, initial, pitch, truth, reference=None, position_gate=True
):
    seed = initial
    rows = []
    for index, field in enumerate(fields):
        result = executable(field, seed, pitch)
        jax.block_until_ready(result)
        position = np.asarray(result.position)
        target = (
            truth[index]
            if reference is None
            else np.asarray(reference[index]["positions"])
        )
        errors = np.linalg.norm(position - target, axis=-1) / float(pitch)
        determinants = np.linalg.det(np.asarray(result.jet.hessian))
        axis_is_maximum = np.all(
            np.linalg.eigvalsh(np.asarray(result.jet.hessian)[0]) < 0
        )
        classification = axis_is_maximum and (len(position) == 1 or determinants[1] < 0)
        failures = []
        if not np.all(np.asarray(result.valid)):
            failures.append("no-convergence")
        if not np.isfinite(position).all():
            failures.append("point-lost-or-cache-miss")
        if not classification:
            failures.append("wrong-stationary-type")
        if position_gate and np.max(errors) > 1:
            failures.append("wrong-saddle-or-position")
        rows.append(
            dict(
                state=index,
                warm_seed=np.asarray(seed).tolist(),
                positions=position.tolist(),
                error_pitches=errors.tolist(),
                valid=np.asarray(result.valid).tolist(),
                determinants=determinants.tolist(),
                iterations=np.asarray(getattr(result, "iterations", 6)).tolist(),
                jet_evaluations_per_point=int(getattr(result, "evaluations", 7)),
                failures=failures,
            )
        )
        # Preserve a failed result in the record, but do not poison later states.
        accepted = result.valid & jnp.all(jnp.isfinite(result.position), axis=-1)
        seed = jnp.where(accepted[:, None], result.position, seed)
    return rows


def latency(executable, fields, sweep, pitch, calls):
    # Device operands and prior-state seeds are prepared before starting any clock.
    seeds = [jnp.asarray(row["warm_seed"]) for row in sweep]
    operands = [(fields[i], seeds[i], pitch) for i in range(len(fields))]
    jax.block_until_ready(operands)
    jax.block_until_ready(executable(*operands[0]))
    times = []
    for index in range(calls):
        args = operands[index % len(operands)]
        start = perf_counter()
        jax.block_until_ready(executable(*args))
        times.append(perf_counter() - start)
    return dict(
        calls=calls,
        median_seconds=float(np.median(times)),
        p99_seconds=float(np.percentile(times, 99)),
        seconds=times,
    )


def membership(values, indices, geometry, fixed, sigma, radius, diverted):
    sampled = values[indices]
    coefficient = jnp.einsum("nij,nj->ni", geometry.fit_inverse, sampled)
    form = topology.saddle_normal_form(
        fixed.position[-1], sigma * fixed.jet.hessian[-1], jnp.zeros((2, 2, 2)), 1e-12
    )
    level = fixed.jet.value[-1] if diverted else jnp.asarray(0.0)
    policy = TopologyPolicy(normal_form_radius=radius)
    support = topology._support_at_level(
        geometry, coefficient, sigma, level, fixed.position[0], form, diverted, policy
    )
    area, _, _, _, selected, _, valid, _ = support
    fraction = jnp.sum(jnp.where(selected, area, 0.0), axis=1) / geometry.full_area
    return fraction, valid


def membership_measure(
    args, geometry, analytic, oracle, reference, moments, nodes, indices, fixed
):
    from scripts.prototypes.read_route.jets import _source_jets

    start = perf_counter()
    evaluate = jax.jit(
        jax.vmap(
            lambda point, coupling: _source_jets(point, coupling)[..., 0],
            in_axes=(0, None),
        )
    )
    matrix = np.empty((len(nodes), len(geometry.centre), 3))
    batch = 16
    for first in range(0, len(nodes), batch):
        points = nodes[first : first + batch]
        padded = np.concatenate(
            (points, np.repeat(points[-1:], batch - len(points), axis=0))
        )
        matrix[first : first + len(points)] = np.asarray(
            evaluate(jnp.asarray(padded), reference.coupling)
        )[: len(points)]
    matrix = jnp.asarray(matrix)
    exact = jnp.stack(tuple(reference.moments), -1)
    exterior = jax.jit(jax.vmap(analytic.value))(jnp.asarray(nodes)) - jnp.einsum(
        "ncm,cm->n", matrix, exact
    )
    jax.block_until_ready(exterior)
    build_wall = perf_counter() - start
    sigma = topology.TopologyConvention.from_cocos(17, 1.0).sigma
    pitch = float(np.median(np.asarray(geometry.pitch)))
    radius = 1.5 * pitch

    def step(matrix, moments, exterior, geometry, fixed):
        values = exterior + jnp.einsum("ncm,cm->n", matrix, moments)
        return membership(
            values,
            jnp.asarray(indices),
            geometry,
            fixed,
            sigma,
            radius,
            args.kind == "diverted",
        )

    executable = (
        jax.jit(step).lower(matrix, moments, exterior, geometry, fixed).compile()
    )
    operands = matrix, moments, exterior, geometry, fixed
    jax.block_until_ready(executable(*operands))
    walls = []
    for _ in range(10):
        start = perf_counter()
        result = executable(*operands)
        jax.block_until_ready(result)
        walls.append(perf_counter() - start)
    truth, uncertainty = fixture._reference_cell_fractions(args.kind, oracle, geometry)
    area = np.asarray(geometry.full_area)
    error = np.abs(np.asarray(result[0]) - truth) * area / np.median(area)
    near = (
        np.linalg.norm(
            np.asarray(geometry.centre) - np.asarray(fixed.position[-1]), axis=1
        )
        <= radius
        if args.kind == "diverted"
        else np.zeros(len(area), bool)
    )
    return dict(
        valid=bool(result[1]),
        median_seconds=float(np.median(walls)),
        value_coupling_bytes=matrix.nbytes,
        build_seconds=build_wall,
        smooth_error=float(np.max(error[~near])),
        saddle_error=float(np.max(error[near])) if near.any() else None,
        reference_uncertainty=float(np.max(uncertainty)),
        normal_form_radius_m=radius,
    )


def measure(args):
    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs(args.kind)
    print("BUILD_GEOMETRY", args.kind, args.cells, flush=True)
    geometry = fixture._realised_hex_geometry(wall, args.cells)
    pitch = jnp.median(geometry.pitch)
    positions = np.asarray([axis] if saddle is None else [axis, saddle])
    shifts = np.linspace(-2, 2, args.states) * float(pitch)
    truth = positions[None] + np.stack((np.zeros(args.states), shifts), -1)[:, None]
    fields = [replace(analytic, shift=jnp.asarray((0.0, shift))) for shift in shifts]
    initial = jnp.asarray(truth[0] + (0.07 * float(pitch)))
    closed, cold, size = compile_solver(fields[0], initial, pitch)
    closed_rows = run_sweep(closed, fields, initial, pitch, truth)
    positive = max(max(row["error_pitches"]) for row in closed_rows)
    print("CLOSED_FORM_CONTROL max_error_pitches=", positive, flush=True)
    assert positive < 1e-9, closed_rows
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    indices = inverse.reshape(samples.shape[:-1])
    mesh = build_mesh(nodes, indices, geometry)
    selected = select_cache_nodes(mesh, truth.reshape(-1, 2), float(pitch))
    lookup = np.full(len(nodes), -1, dtype=np.int32)
    lookup[selected] = np.arange(len(selected))
    print("BUILD_COUPLING cache_nodes=", len(selected), flush=True)
    reference = fixture._kernel_backed_field(args.kind, oracle, analytic, geometry)
    start = perf_counter()
    rows = build_rows(nodes[selected], reference.coupling)
    jax.block_until_ready(rows)
    build_wall = perf_counter() - start
    print("ROWS_READY bytes=", rows.nbytes, "seconds=", build_wall, flush=True)
    exact = jnp.stack(tuple(reference.moments), -1)
    active_cache_nodes = len(selected)
    padding = 512 - len(selected)
    rows = jnp.pad(rows, ((0, padding), (0, 0), (0, 0), (0, 0)))
    reference_jets = jnp.einsum("ncmq,cm->nq", rows, exact)
    # State construction is setup, outside the timed solve.
    nodal_jets = jax.jit(
        lambda field, points: stack_jet(jax.vmap(field.evaluate)(points))
    )
    nodal = [nodal_jets(field, jnp.asarray(nodes[selected])) for field in fields]
    exterior = [jnp.pad(jet, ((0, padding), (0, 0))) - reference_jets for jet in nodal]
    receipt = dict(
        kind=args.kind,
        cells=args.cells,
        pitch_m=float(pitch),
        states=args.states,
        imposed_travel_pitches=4.0,
        closed_form_max_error_pitches=positive,
        closed_form_cold_seconds=cold,
        closed_form_executable_bytes=size,
        cache_nodes=active_cache_nodes,
        cache_cap_nodes=512,
        cache_bytes=rows.nbytes,
        cache_build_seconds=build_wall,
        rows=[],
        complete=False,
    )
    out = args.out / f"{args.kind}-{args.cells}.json"
    persist(out, receipt)
    print("BOOKING_STATES", flush=True)
    _, source, _ = certificate._case(
        "weak-rotation-reactor-static" if saddle is None else "diverted-single-null",
        clip_mode="exact",
    )
    profile = analytic_profile(source)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996 if saddle is not None else 0.0
    )
    reader = jax.jit(topology.read)
    book = jax.jit(
        lambda psi, reading, geometry, quadrature: current.book(
            psi, reading, geometry, profile, quadrature
        )
    )
    # Hold the input state fixed and vary the prescribed exterior. Each output
    # is one Picard image of this input under its own exterior supply.
    reading = reader(analytic, geometry, convention, policy)
    psi = jax.jit(lambda field, points: jax.vmap(jax.vmap(field.value))(points))(
        analytic, geometry.sample_points
    )
    booked = book(
        psi,
        reading,
        geometry,
        current.Quadrature(sigma=convention.sigma, segments=64),
    )
    jax.block_until_ready(booked)
    bookings = [jnp.stack(tuple(booked[:3]), -1)] * args.states
    if args.shifted_bookings:
        bookings = []
        for field in fields:
            reading = reader(field, geometry, convention, policy)
            psi = jax.jit(
                lambda field, points: jax.vmap(jax.vmap(field.value))(points)
            )(field, geometry.sample_points)
            shifted = book(
                psi,
                reading,
                geometry,
                current.Quadrature(sigma=convention.sigma, segments=64),
            )
            jax.block_until_ready(shifted)
            bookings.append(jnp.stack(tuple(shifted[:3]), -1))
    receipt["booking_valid"] = bool(booked.valid)
    receipt["picard_input"] = (
        "independently shifted analytic inputs"
        if args.shifted_bookings
        else "unshifted analytic state; varying exterior supply"
    )
    receipt["booked_moment_relative_difference"] = float(
        jnp.max(jnp.abs(bookings[0] - exact)) / jnp.max(jnp.abs(exact))
    )
    persist(out, receipt)
    print(
        "BOOKED",
        bool(booked.valid),
        receipt["booked_moment_relative_difference"],
        flush=True,
    )
    programs = {}
    for state_kind, all_moments in [
        ("analytic", [exact] * args.states),
        ("picard", bookings),
    ]:
        cached_fields = [
            CachedField(mesh, jnp.asarray(lookup), rows, moments, ext, jnp.asarray(1.0))
            for moments, ext in zip(all_moments, exterior)
        ]
        kernel_fields = [
            KernelField(
                field, reference.coupling, ClippedCurrentMoments(*(moments - exact).T)
            )
            for field, moments in zip(fields, all_moments)
        ]
        reference_rows = None
        for name, arm_fields in [
            ("kernel", kernel_fields),
            ("cached-jet", cached_fields),
            ("cached-adaptive", cached_fields),
        ]:
            print("COMPILE_SOLVE", state_kind, name, flush=True)
            if name not in programs:
                programs[name] = compile_solver(
                    arm_fields[0], initial, pitch, adaptive=name == "cached-adaptive"
                )
            executable, cold, size = programs[name]
            sweep = run_sweep(
                executable,
                arm_fields,
                initial,
                pitch,
                truth,
                reference_rows,
                position_gate=not (state_kind == "picard" and name == "kernel"),
            )
            if name == "kernel":
                reference_rows = sweep if state_kind == "picard" else None
            timed = (
                latency(executable, arm_fields, sweep, pitch, args.calls)
                if name != "kernel"
                else dict(calls=0, median_seconds=None, p99_seconds=None, seconds=[])
            )
            if name == "cached-adaptive" and state_kind == "analytic":
                profile_path = args.out / f"{args.kind}-{args.cells}-hlo.txt"
                profile_path.write_text(executable.as_text())
                receipt["profile"] = dict(
                    cost=executable.cost_analysis(), hlo_path=str(profile_path)
                )
            travel = float(
                np.linalg.norm(
                    np.asarray(sweep[-1]["positions"])[-1]
                    - np.asarray(sweep[0]["positions"])[-1]
                )
                / float(pitch)
            )
            row = dict(
                arm=name,
                measured_travel_pitches=travel,
                state_kind=state_kind,
                position_reference=(
                    "closed-form displacement"
                    if state_kind == "picard" and name == "kernel"
                    else "kernel"
                    if state_kind == "picard"
                    else "closed form"
                ),
                cold_compile_seconds=cold,
                reused_executable=state_kind == "picard",
                executable_bytes=size,
                iterations_per_call=[row["iterations"] for row in sweep],
                sweep=sweep,
                **timed,
            )
            if args.shifted_bookings and name == "cached-adaptive":
                from scripts.prototypes.fixed_points.cache import solve_with_refresh

                repaired = []
                seed = initial
                for index, field in enumerate(arm_fields):
                    result, _, events = solve_with_refresh(
                        executable,
                        field,
                        seed,
                        pitch,
                        nodes,
                        reference.coupling,
                        exact,
                        fields[index],
                    )
                    target = (
                        truth[index]
                        if state_kind == "analytic"
                        else np.asarray(reference_rows[index]["positions"])
                    )
                    repaired.append(
                        dict(
                            state=index,
                            valid=np.asarray(result.valid).tolist(),
                            error_pitches=(
                                np.linalg.norm(
                                    np.asarray(result.position) - target, axis=-1
                                )
                                / float(pitch)
                            ).tolist(),
                            refresh_events=events,
                        )
                    )
                    seed = jnp.where(result.valid[:, None], result.position, seed)
                row["recovered_sweep"] = repaired
                if state_kind == "picard":
                    from functools import partial

                    extended = (
                        jax.jit(partial(solve_adaptive, iterations=24))
                        .lower(arm_fields[0], initial, pitch)
                        .compile()
                    )
                    extended_rows = []
                    seed = initial
                    for index, field in enumerate(arm_fields):
                        result, _, events = solve_with_refresh(
                            extended,
                            field,
                            seed,
                            pitch,
                            nodes,
                            reference.coupling,
                            exact,
                            fields[index],
                        )
                        target = np.asarray(reference_rows[index]["positions"])
                        extended_rows.append(
                            dict(
                                state=index,
                                valid=np.asarray(result.valid).tolist(),
                                iterations=np.asarray(result.iterations).tolist(),
                                hessian_eigenvalues=np.linalg.eigvalsh(
                                    np.asarray(result.jet.hessian)
                                ).tolist(),
                                error_pitches=(
                                    np.linalg.norm(
                                        np.asarray(result.position) - target, axis=-1
                                    )
                                    / float(pitch)
                                ).tolist(),
                                refresh_events=events,
                            )
                        )
                        seed = jnp.where(result.valid[:, None], result.position, seed)
                    row["extended_recovered_sweep"] = extended_rows
            receipt["rows"].append(row)
            persist(out, receipt)
            print(
                "SOLVE_ROW",
                json.dumps(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("sweep", "seconds")
                    }
                ),
                flush=True,
            )
            if name == "cached-jet" and state_kind == "analytic" and saddle is not None:
                mutated = [
                    replace(field, hessian_factor=jnp.asarray(0.0))
                    for field in arm_fields
                ]
                negative = run_sweep(executable, mutated, initial, pitch, truth)
                receipt["zero_hessian_control"] = negative
                persist(out, receipt)
    with jax.profiler.trace(str(args.out / f"{args.kind}-{args.cells}-profile")):
        jax.block_until_ready(
            programs["cached-adaptive"][0](cached_fields[0], initial, pitch)
        )
    if args.coverage and saddle is not None:
        receipt["coverage"] = coverage_measure(
            args,
            cached_fields[0],
            nodes,
            reference,
            exact,
            analytic,
            positions,
            pitch,
            programs["cached-adaptive"][0],
            programs["kernel"][0],
            bookings[0],
        )
        persist(out, receipt)
        print("COVERAGE_ROW", json.dumps(receipt["coverage"]), flush=True)
    if args.membership and args.cells in (2000, 5000):
        unshifted = closed(analytic, jnp.asarray(positions), pitch)
        receipt["membership"] = membership_measure(
            args,
            geometry,
            analytic,
            oracle,
            reference,
            exact,
            nodes,
            indices,
            unshifted,
        )
        persist(out, receipt)
        print("MEMBERSHIP_ROW", json.dumps(receipt["membership"]), flush=True)
    receipt["complete"] = True
    persist(out, receipt)
    print("ROW_COMPLETE", str(out), flush=True)


def coverage_measure(
    args,
    field,
    nodes,
    reference,
    exact,
    analytic,
    positions,
    pitch,
    executable,
    kernel,
    moments,
):
    from scripts.prototypes.fixed_points.cache import solve_with_refresh, _nodal_jets

    seed = jnp.asarray(positions + np.asarray((0.0, -2 * float(pitch))))
    kernel_seed = seed
    records = []
    for index, shift in enumerate(np.linspace(-2, 6, args.states) * float(pitch)):
        analytic_state = replace(analytic, shift=jnp.asarray((0.0, shift)))
        lookup = np.asarray(field.lookup)
        selected = np.flatnonzero(lookup >= 0)
        ext = (
            jnp.zeros_like(field.exterior)
            .at[lookup[selected]]
            .set(
                _nodal_jets(analytic_state, jnp.asarray(nodes[selected]))
                - jnp.einsum("ncmq,cm->nq", field.rows[lookup[selected]], exact)
            )
        )
        field = replace(field, moments=moments, exterior=ext)
        direct = KernelField(
            analytic_state,
            reference.coupling,
            ClippedCurrentMoments(*(moments - exact).T),
        )
        truth = kernel(direct, kernel_seed, pitch)
        jax.block_until_ready(truth)
        start = perf_counter()
        result, field, events = solve_with_refresh(
            executable,
            field,
            seed,
            pitch,
            nodes,
            reference.coupling,
            exact,
            analytic_state,
        )
        wall = perf_counter() - start
        records.append(
            dict(
                state=index,
                imposed_shift_pitches=shift / float(pitch),
                positions=np.asarray(result.position).tolist(),
                valid=np.asarray(result.valid).tolist(),
                kernel_valid=np.asarray(truth.valid).tolist(),
                error_pitches=(
                    np.linalg.norm(
                        np.asarray(result.position - truth.position), axis=-1
                    )
                    / float(pitch)
                ).tolist(),
                refresh_events=events,
                total_seconds=wall,
            )
        )
        seed = jnp.where(result.valid[:, None], result.position, seed)
        kernel_seed = jnp.where(truth.valid[:, None], truth.position, kernel_seed)
    return dict(
        imposed_travel_pitches=8,
        states=records,
        refresh_count=sum(len(row["refresh_events"]) for row in records),
        semantics=(
            "host window refresh retries whole compiled solves; "
            "miss latency is separate"
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--membership", action="store_true")
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--shifted-bookings", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"REVISION={revision} TREE={ROOT} COMMAND={sys.executable} "
        "-m scripts.prototypes.fixed_points.measure " + " ".join(sys.argv[1:]),
        flush=True,
    )
    print(
        f"NOVA_MODULE={nova.__file__} CWD={Path.cwd().resolve()} "
        f"DEVICES={jax.devices()}",
        flush=True,
    )
    print(
        "SOURCE_DIGESTS",
        json.dumps(
            {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in ("measure.py", "solve.py", "cache.py")
            }
        ),
        flush=True,
    )
    assert jax.default_backend() == "gpu"
    measure(args)


if __name__ == "__main__":
    main()
