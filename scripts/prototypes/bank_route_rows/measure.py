"""Measure the fast route and the present route on Solovev certificate rows.

The fast route is one row: the bank-backed Newton step iterated from the
certificate seed (the analytic read, booked) until its fresh-read residual is
at most 1e-10.  The present route is measured in the same process through the
certificate benchmark, so both routes see the same carrier geometry and the
same analytic oracle.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import current, fixed_points, support_read, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.bank_newton_converge.fresh_solve import analytic_oracle
from scripts.prototypes.bank_newton_step.measure import machine as seeded_machine

CASES = {
    "diverted": "diverted-single-null",
    "limited": "weak-rotation-reactor-static",
}
# The present route publishes only these certificate sizes; each fast-route
# cell target maps to the nearest published row.
PRESENT_CELLS = {550: -500, 1000: -1000, 2000: -2500, 2500: -2500}
ROOT = Path(__file__).resolve().parents[3]


def emit(name, **values):
    print(name + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def _case_and_profile(kind):
    from benchmarks import solovev_certificate as certificate
    from scripts.analytic_oracle_fixtures.measure import analytic_profile

    _, source, _ = certificate._case(CASES[kind], clip_mode="exact")
    return certificate, analytic_profile(source)


def build(kind, cells):
    """Build the certificate machine at ``cells`` carrier cells."""
    from scripts.prototypes.read_route.jets import build_jet_matrix
    import tests.equilibrium.test_topology_read as fixture

    certificate, profile = _case_and_profile(kind)
    if kind == "diverted" and cells == 550:
        (
            operands,
            cache,
            roots,
            convention,
            policy,
            profile,
            quadrature,
            moments,
            analytic,
        ) = seeded_machine("diverted")
        return dict(
            operands=operands,
            cache=cache,
            roots=roots,
            convention=convention,
            policy=policy,
            profile=profile,
            quadrature=quadrature,
            moments=moments,
            analytic=analytic,
            bank_seconds=0.0,
        )
    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs(kind)
    geometry = fixture._realised_hex_geometry(wall, cells)
    if jax.default_backend() != "gpu":
        raise RuntimeError("analytic operator construction needs a GPU")
    reference = fixture._kernel_backed_field(kind, oracle, analytic, geometry)
    matrix, exterior, nodes, sample_index, seconds = build_jet_matrix(
        reference.coupling, geometry, analytic, reference.moments, persist=False
    )
    operands = support_read.SupportOperands.from_rows(
        geometry, matrix, exterior, nodes, sample_index, bucket=len(nodes)
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = (
        TopologyPolicy(normal_form_radius=0.054806712567833996)
        if kind == "diverted"
        else TopologyPolicy()
    )
    quadrature = current.Quadrature(sigma=convention.sigma, segments=64)
    roots = jnp.asarray([axis] if saddle is None else [axis, saddle])
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    analytic_read = jax.jit(topology.read)(analytic, geometry, convention, policy)
    booked = current.book(psi, analytic_read, geometry, profile, quadrature)
    moments = booked[:3]
    cache = fixed_points.RowCache(
        operands.mesh,
        operands.nodes,
        jnp.arange(len(operands.nodes)),
        operands.rows,
        operands.exterior,
    )
    emit("BANK_BUILD", kind=kind, cells=cells, seconds=seconds, nodes=len(nodes))
    return dict(
        operands=operands,
        cache=cache,
        roots=roots,
        convention=convention,
        policy=policy,
        profile=profile,
        quadrature=quadrature,
        moments=moments,
        analytic=analytic,
        bank_seconds=seconds,
    )


def fast_route(built, max_steps=12):
    operands = built["operands"]
    cache = built["cache"]
    roots = built["roots"]
    convention = built["convention"]
    policy = built["policy"]
    profile = built["profile"]
    quadrature = built["quadrature"]
    moments = built["moments"]
    seeds = roots + 0.03 * jnp.median(operands.geometry.pitch)

    @jax.jit
    def read(bank, window, state):
        return support_read.read(bank, window, state, seeds, roots, convention, policy)

    @jax.jit
    def residual(bank, base, candidate, support):
        return support_read.step_residual(
            bank, base, candidate, support, profile, quadrature
        )

    @jax.jit
    def step(bank, state, support):
        return support_read.newton_step(
            bank,
            state,
            support,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
            fresh_geometry=True,
        )

    def magnitude(result):
        return float(jnp.max(jnp.abs(jnp.concatenate(result.residual))))

    warm_walls = []
    trace = []
    index = 0
    support = read(operands, cache, moments)
    for index in range(max_steps):
        support = read(operands, cache, moments)
        current_result = residual(operands, moments, moments, support)
        before = magnitude(current_result)
        emit("STEP", index=index, fresh_residual=before, valid=bool(support.valid))
        trace.append(before)
        if before <= 1e-10 and bool(support.valid):
            break
        started = perf_counter()
        correction = step(operands, moments, support)
        jax.block_until_ready(correction)
        warm_walls.append(1e3 * (perf_counter() - started))
        accepted = False
        for half in range(13):
            length = 2.0**-half
            candidate = tuple(
                value + length * change
                for value, change in zip(moments, correction.update, strict=True)
            )
            next_support = read(operands, cache, candidate)
            refreshed = residual(operands, candidate, candidate, next_support)
            fresh_norm = magnitude(refreshed)
            if (
                bool(next_support.valid)
                and np.isfinite(fresh_norm)
                and fresh_norm < before
            ):
                emit("ACCEPT", index=index, length=length, fresh_residual=fresh_norm)
                moments = candidate
                accepted = True
                break
        if not accepted:
            emit("NO_DESCENT", index=index, residual=before)
            break

    support = read(operands, cache, moments)
    result = residual(operands, moments, moments, support)
    terminal = magnitude(result)
    converged = bool(support.valid) and terminal <= 1e-10
    reading = support.reading
    active = int(operands.active_nodes)
    nodes = np.asarray(operands.nodes[:active])
    solved_flux = np.asarray(support_read.nodal_jets(operands, moments)[:active, 0])
    emit("FINAL", steps=index, fresh_residual=terminal, converged=converged)
    return dict(
        steps=int(index),
        converged=bool(converged),
        fresh_residual=terminal,
        trace=trace,
        warm_step_ms=float(np.median(warm_walls)) if warm_walls else None,
        support=support,
        moments=moments,
        nodes=nodes,
        solved_flux=solved_flux,
        reading=reading,
        operands=operands,
        analytic=built["analytic"],
        roots=built["roots"],
    )


def present_route(kind, cells, requested_cells=None, root=None):
    """Run the certificate row through the default present route in-process."""
    import benchmarks.solovev_certificate as certificate

    root = root or Path("present")
    certificate.PART_ROOT = root / "parts"
    certificate.FIGURE_ROOT = root / "figures"
    certificate.DIAGNOSTIC_ROOT = root / "diagnostics"
    for directory in (
        certificate.PART_ROOT,
        certificate.FIGURE_ROOT,
        certificate.DIAGNOSTIC_ROOT,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    saved = sys.argv
    sys.argv = [
        "benchmarks/solovev_certificate.py",
        "--case",
        CASES[kind],
        "--requested-cells",
        str(requested_cells or PRESENT_CELLS[cells]),
    ]
    started = perf_counter()
    try:
        certificate.main()
    finally:
        sys.argv = saved
    part = json.loads(
        certificate._part_path(
            CASES[kind], requested_cells or PRESENT_CELLS[cells]
        ).read_text()
    )
    solver = part["solver"]
    render = part.get("render_data") or {}
    emit(
        "PRESENT",
        kind=kind,
        cells=cells,
        realised=part["realised_cells"],
        converged=solver["converged"],
        terminal=solver["terminal_fixed_point_residual"],
        wall=perf_counter() - started,
    )
    return dict(
        converged=bool(solver["converged"]),
        terminal=float(solver["terminal_fixed_point_residual"]),
        newton_steps=solver.get("newton_steps"),
        solve_wall=solver.get("solve_wall_seconds"),
        row_wall=perf_counter() - started,
        realised_cells=part["realised_cells"],
        topology=render.get("terminal_topology"),
        coordinates=np.asarray(render["coordinates_rz_m"]) if render else None,
        flux=np.asarray(render["terminal_flux_wb"]) if render else None,
        wall_units=render.get("wall_units_rz_m"),
        part_path=str(
            certificate._part_path(CASES[kind], requested_cells or PRESENT_CELLS[cells])
        ),
    )


def row_oracle(kind, analytic, roots, reading, wall):
    """Judge one read against stationary points found on analytic flux alone.

    A limited row has no X-point, so its boundary is the limiter contact: the
    wall node carrying the largest analytic flux, which is where a closed flux
    surface first touches the wall.
    """
    from scripts.prototypes.tracked_read.measure import newton_reference

    if kind == "diverted":
        oracle, analytic_roots = analytic_oracle(analytic, roots, reading, wall)
        return oracle, analytic_roots
    axis, axis_flux = newton_reference(analytic, roots[0])
    gradient = np.asarray(analytic.evaluate(jnp.asarray(axis)).gradient)
    assert np.linalg.norm(gradient) < 1e-10
    wall_points = np.asarray(wall)
    wall_flux = np.asarray(jax.vmap(analytic.value)(jnp.asarray(wall_points)))
    contact = wall_points[int(np.argmax(wall_flux))]
    boundary_flux = float(analytic.value(jnp.asarray(contact)))
    wall = wall_points
    radius = 0.5 * float(np.ptp(wall[:, 0]))
    span = abs(boundary_flux - axis_flux)
    assert radius > 0 and span > 0
    axis_error = float(np.linalg.norm(np.asarray(reading.axis) - axis))
    boundary_error = abs(float(reading.boundary_flux) - boundary_flux)
    return (
        {
            "axis_error": axis_error,
            "minor_radius": radius,
            "axis_ratio": axis_error / radius,
            "saddle_error": None,
            "saddle_ratio": None,
            "boundary_error": boundary_error,
            "flux_span": span,
            "boundary_ratio": boundary_error / span,
            "boundary_point": contact.tolist(),
            "x_point_note": "limited row: no magnetic island boundary",
        },
        np.stack((axis, contact)),
    )


def asymptote_angles(analytic, saddle):
    """Four separatrix asymptote bearings from the Hessian eigenvalue ratio.

    In the Hessian's own eigenbasis the level set of a saddle is
    ``l1 u**2 + l2 v**2 = const``, so its asymptotes are the directions
    ``(1, +-sqrt(l1 / -l2))`` when ``l1 > 0 > l2``.  The eigenvectors
    themselves lie 45 degrees off those asymptotes whenever the two
    eigenvalues have equal magnitude, which is why the ratio is used.
    """

    def gradient(point):
        return analytic.evaluate(jnp.asarray(point)).gradient

    hessian = np.asarray(jax.jacobian(gradient)(jnp.asarray(saddle, dtype=jnp.float64)))
    values, vectors = np.linalg.eigh(hessian)
    order = np.argsort(values)
    positive = vectors[:, order[-1]]
    values = values[order]
    if not (values[-1] > 0 > values[0]):
        return {"hessian_eigenvalues": values.tolist(), "class": "not_a_saddle"}
    cross = np.sqrt(values[-1] / -values[0])
    bearings = sorted(
        float(
            np.degrees(
                np.arctan2(*(positive + sign * cross * vectors[:, order[0]])[::-1])
            )
            % 180.0
        )
        for sign in (1.0, -1.0)
    )
    return {
        "hessian_eigenvalues": values.tolist(),
        "eigenvalue_ratio": float(cross),
        "asymptote_bearings_deg": sorted(
            bearings + [bearing + 180.0 for bearing in bearings]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=tuple(CASES), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--present", action="store_true")
    parser.add_argument("--present-only", action="store_true")
    parser.add_argument("--present-cells", type=int)
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()
    import subprocess

    emit(
        "HEADER",
        revision=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        nova=nova.__file__,
        tree=str(ROOT),
        kind=args.kind,
        cells=args.cells,
    )
    if args.present_only:
        present = present_route(
            args.kind,
            args.cells,
            args.present_cells or PRESENT_CELLS[args.cells],
            args.output.parent / "present",
        )
        topology = present.get("topology") or {}
        row = {
            "kind": args.kind,
            "case": CASES[args.kind],
            "cells": args.cells,
            "present": {
                key: value
                for key, value in present.items()
                if key not in ("coordinates", "flux")
            },
            "present_x_point_rz_m": topology.get("x_point_rz_m"),
            "present_class": topology.get("class"),
            "present_read_status": topology.get("read_status"),
            "present_axis_rz_m": topology.get("axis_rz_m"),
            "present_axis_flux_wb": topology.get("axis_flux_wb"),
            "present_boundary_flux_wb": topology.get("boundary_flux_wb"),
        }
        args.output.write_text(json.dumps(row, indent=2, default=float) + "\n")
        emit(
            "PRESENT_ONLY_ROW",
            kind=args.kind,
            cells=args.cells,
            converged=present["converged"],
        )
        return
    built = build(args.kind, args.cells)
    fast = fast_route(built, args.max_steps)
    from tests.equilibrium import test_topology_read as fixture

    _, _, wall, _, _ = fixture._analytic_inputs(args.kind)
    oracle, analytic_roots = row_oracle(
        args.kind, fast["analytic"], built["roots"], fast["reading"], wall
    )
    emit("ORACLE", **oracle)
    metric_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        metric_path,
        nodes=fast["nodes"],
        solved_flux=fast["solved_flux"],
        analytic_roots=analytic_roots,
        solved_roots=np.asarray(
            jnp.stack((fast["reading"].axis, fast["reading"].boundary))
        ),
        wall=np.asarray(wall),
        converged=np.asarray(fast["converged"]),
        residual_sup=np.asarray(fast["fresh_residual"]),
    )
    row = {
        "kind": args.kind,
        "case": CASES[args.kind],
        "cells": args.cells,
        "realised_cells": int(len(np.asarray(built["operands"].geometry.vertices))),
        "bank_seconds": built["bank_seconds"],
        "fast": {
            "converged": fast["converged"],
            "updates": fast["steps"],
            "fresh_residual": fast["fresh_residual"],
            "residual_trace": fast["trace"],
            "warm_step_ms": fast["warm_step_ms"],
        },
        "oracle": oracle,
        "budget": {
            "axis": oracle["axis_ratio"] <= 5e-4,
            "xpoint": (
                None
                if oracle.get("saddle_ratio") is None
                else oracle["saddle_ratio"] <= 5e-4
            ),
            "boundary": oracle["boundary_ratio"] <= 1e-4,
            "residual": fast["fresh_residual"] <= 1e-10,
        },
        "metric_path": str(metric_path),
    }
    if args.present:
        present = present_route(
            args.kind,
            args.cells,
            args.present_cells or PRESENT_CELLS[args.cells],
            args.output.parent / "present",
        )
        row["present"] = {
            k: v for k, v in present.items() if k not in ("coordinates", "flux")
        }
        topology_res = present.get("topology") or {}
        x_point = topology_res.get("x_point_rz_m")
        if x_point is not None:
            saddle = np.asarray(fast["reading"].boundary, dtype=float)
            gap_mm = 1e3 * float(np.linalg.norm(saddle - np.asarray(x_point)))
            row["xpoint_gap_lattice_mm"] = gap_mm
        else:
            row["xpoint_gap_lattice_mm"] = None
            row["xpoint_gap_reason"] = (
                f"present route admits no X-point: class={topology_res.get('class')} "
                f"read_status={topology_res.get('read_status')} "
                f"x_candidate_count={topology_res.get('x_candidate_count')}"
            )
        if present.get("coordinates") is not None:
            from scipy.spatial import cKDTree

            tree = cKDTree(present["coordinates"])
            distance, index = tree.query(fast["nodes"], k=1)
            pitch = float(np.median(built["operands"].geometry.pitch))
            matched = distance < 1e-3 * pitch
            if matched.any():
                difference = np.abs(
                    present["flux"][index[matched]] - fast["solved_flux"][matched]
                )
                row["flux_gap"] = {
                    "matched_nodes": int(matched.sum()),
                    "largest_abs_wb": float(difference.max()),
                    "largest_ratio_of_span": float(
                        difference.max() / oracle["flux_span"]
                    ),
                    "coordinate_tolerance_m": 1e-3 * pitch,
                }
            else:
                row["flux_gap"] = {"matched_nodes": 0}
    if args.kind == "diverted" and fast["reading"].boundary is not None:
        row["fast_asymptotes"] = asymptote_angles(
            fast["analytic"], fast["reading"].boundary
        )
    elif args.kind != "diverted":
        row["fast_asymptotes"] = {"class": "not_a_saddle"}
    args.output.write_text(json.dumps(row, indent=2, default=float) + "\n")
    emit("ROW", **{k: row[k] for k in ("kind", "cells", "realised_cells")})
    emit("EXIT_ROW", ok=True)


if __name__ == "__main__":
    main()
