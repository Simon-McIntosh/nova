"""Measure a complete fixed-shape moment correction on a certificate carrier."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, fixed_points, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fast_read.program import map_moments
from scripts.prototypes.lean_map.jets import build_jets
from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def save(path, result):
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("CHECKPOINT " + json.dumps(result, default=str), flush=True)


def wall(program, arguments, repetitions=5):
    jax.block_until_ready(program(*arguments))
    elapsed = []
    for _ in range(repetitions):
        started = perf_counter()
        jax.block_until_ready(program(*arguments))
        elapsed.append(1000 * (perf_counter() - started))
    return float(np.median(elapsed))


def generated_bytes(program):
    executable = program.runtime_executable()
    method = getattr(executable, "size_of_generated_code_in_bytes", None)
    return method() if callable(method) else None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CachedExterior:
    """Return stored exterior jets at the same nodes retained by row maintenance."""

    nodes: jax.Array
    jets: jax.Array

    def evaluate(self, point):
        distance = jnp.sum((self.nodes - point) ** 2, axis=1)
        row = self.jets[jnp.argmin(distance)]
        return topology.FieldJet(
            row[0],
            row[1:3],
            jnp.stack((jnp.stack((row[3], row[4])), jnp.stack((row[4], row[5])))),
        )


def run(args):
    kind, cache, output = args.kind, args.cache, args.output
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    machine, key, builder = _machine(cache, carrier, 250)
    result = {"kind": kind, "cells": len(machine.node), "builder": builder}
    save(output, result)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(analytic_field, kind)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    reading = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    assert bool(reading.valid) and bool(reading.qualified)
    samples = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda psi: current.book(
            psi,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
    )(samples)
    assert bool(booked.valid)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = triangle_rule(geometry)
    fit, design = quadratic_fit(rule, geometry)
    values, values_receipt = build_values(machine, rule, field, reference)
    result["values"] = values_receipt
    save(output, result)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, jets_receipt = build_jets(coupling, geometry, field, reference)
    result["jets"] = jets_receipt
    save(output, result)
    roots = np.vstack(
        (
            np.asarray(reading.axis)[None],
            np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        )
    )
    cached_rows = fixed_points.RowCache(
        jets.mesh,
        jets.nodes,
        jnp.arange(len(jets.nodes), dtype=jnp.int32),
        jets.rows,
        jets.exterior,
    )
    generators = np.asarray(jets.mesh.generators)
    margin = float(
        np.max(
            np.min(np.linalg.norm(generators[:, None] - roots[None], axis=-1), axis=1)
        )
        + 2 * np.max(geometry.pitch)
    )
    maintained, maintenance = fixed_points.maintain_rows(
        geometry,
        coupling,
        CachedExterior(jets.nodes, jets.exterior),
        jnp.asarray(roots),
        margin,
        cache=cached_rows,
        capacity=len(jets.nodes),
    )
    window = cached_rows
    result["window"] = {
        "rows": int(jnp.sum(window.lookup >= 0)),
        "capacity": len(window.rows),
        "owner_rows": int(jnp.sum(maintained.lookup >= 0)),
        "maintenance": str(maintenance),
    }
    save(output, result)
    axis_sign = float(jnp.sign(field.evaluate(jnp.asarray(roots[0])).hessian[0, 0]))
    seeds = jnp.asarray(roots)
    operands = (
        values,
        jets,
        window,
        rule,
        fit,
        design,
        geometry,
        seeds,
        convention,
        booked.raw_current,
    )

    def mapped(moments, *args):
        values, jets, window, rule, fit, design, geometry, seeds, convention, target = (
            args
        )
        return map_moments(
            moments,
            values,
            jets,
            window,
            rule,
            fit,
            design,
            geometry,
            profile,
            seeds,
            convention,
            policy,
            target,
            axis_sign,
        )

    pitch = float(np.median(np.asarray(geometry.pitch)))
    scale = jnp.asarray(
        (float(booked.raw_current) / len(reference)) * np.asarray((1.0, pitch, pitch))
    )

    def equation(scaled, args):
        state = mapped(scaled.reshape(reference.shape) * scale, *args)
        return (state.moment_residual / scale).reshape(-1)

    map_program = jax.jit(mapped)
    centre = map_program(reference, *operands)
    jax.block_until_ready(centre)
    assert bool(centre.booking_valid)
    result["centre_residual"] = float(jnp.max(jnp.abs(centre.moment_residual / scale)))
    result["centre_class"] = int(centre.boundary_class)
    result["centre_support"] = int(centre.graph_cells)
    save(output, result)

    if args.taylor:
        direction = jnp.sin(jnp.arange(reference.size).reshape(reference.shape) + 1)
        direction = direction / jnp.sqrt(reference.size)
        base = reference / scale
        value, tangent = jax.jvp(lambda x: equation(x, operands), (base,), (direction,))
        jax.block_until_ready(tangent)
        rows = []
        for h in np.logspace(-2, -8, 7):
            upper = equation(base + h * direction, operands)
            lower = equation(base - h * direction, operands)
            symmetric = (upper + lower) / 2 - value
            derivative_error = (upper - lower) / (2 * h) - tangent
            jax.block_until_ready((symmetric, derivative_error))
            block = np.asarray(symmetric).reshape(reference.shape)
            error = np.asarray(derivative_error).reshape(reference.shape)
            rows.append(
                {
                    "step": float(h),
                    "symmetric_remainder": [
                        float(np.linalg.norm(block[:, k])) for k in range(3)
                    ],
                    "derivative_error": [
                        float(np.linalg.norm(error[:, k])) for k in range(3)
                    ],
                }
            )
        result["taylor"] = rows
        save(output, result)
        return

    def krylov_step(factor, rhs):
        length = jnp.linalg.norm(rhs)
        basis = (
            jnp.zeros((5, rhs.size), dtype=rhs.dtype)
            .at[0]
            .set(rhs / jnp.where(length > 0, length, 1))
        )
        hessenberg = jnp.zeros((5, 4), dtype=rhs.dtype)
        for column in range(4):
            vector = factor(basis[column])
            projections = jnp.einsum("ij,j->i", basis, vector)
            occupied = jnp.arange(5) <= column
            projection = jnp.where(occupied, projections, 0)
            vector = vector - jnp.einsum("i,ij->j", projection, basis)
            next_length = jnp.linalg.norm(vector)
            basis = basis.at[column + 1].set(
                vector / jnp.where(next_length > 0, next_length, 1)
            )
            hessenberg = hessenberg.at[:, column].set(
                projection.at[column + 1].set(next_length)
            )
        target = jnp.zeros((5,), dtype=rhs.dtype).at[0].set(length)
        coefficients = jnp.linalg.lstsq(hessenberg, target, rcond=None)[0]
        return jnp.einsum("i,ij->j", coefficients, basis[:4]), 4

    from scipy.linalg import lu_factor, lu_solve
    from scipy.sparse.linalg import LinearOperator, gmres

    def linear_step(point, measure=True):
        started = perf_counter()
        residual, factor = jax.linearize(
            lambda x: equation(x, operands), point.reshape(-1)
        )
        jax.block_until_ready(residual)
        linearize_ms = 1000 * (perf_counter() - started)
        rhs = -residual
        build_start = perf_counter()
        if args.method == "krylov":
            step, iterations = krylov_step(factor, rhs)
            jax.block_until_ready(step)
            build_ms = 0.0
        elif args.method == "gmres":
            product = jax.jit(factor)
            counter = [0]

            def multiply(x):
                return np.asarray(product(jnp.asarray(x)))

            def count(_):
                counter[0] += 1

            operator = LinearOperator((rhs.size, rhs.size), matvec=multiply)
            step, info = gmres(
                operator,
                np.asarray(rhs),
                rtol=1e-10,
                atol=0.0,
                restart=50,
                maxiter=20,
                callback=count,
                callback_type="pr_norm",
            )
            iterations = counter[0]
            result["gmres_info"] = int(info)
            build_ms = 0.0
        else:
            columns = jnp.eye(rhs.size, dtype=rhs.dtype)
            products = jax.jit(jax.vmap(factor))
            blocks = []
            for start in range(0, rhs.size, args.block):
                stop = min(start + args.block, rhs.size)
                blocks.append(np.asarray(products(columns[start:stop])))
            matrix = np.concatenate(blocks, axis=0).T
            decomposition = lu_factor(matrix)
            build_ms = 1000 * (perf_counter() - build_start)
            apply_start = perf_counter()
            step = lu_solve(decomposition, np.asarray(rhs))
            apply_ms = 1000 * (perf_counter() - apply_start)
            iterations = 1
        if args.method != "dense":
            apply_ms = 1000 * (perf_counter() - build_start)
        candidate = point.reshape(-1) + jnp.asarray(step)
        original = float(jnp.max(jnp.abs(residual)))
        accepted = False
        factor_scale = 1.0
        new_residual = original
        for _ in range(12 if args.line_search else 1):
            trial = point.reshape(-1) + factor_scale * jnp.asarray(step)
            trial_residual = float(jnp.max(jnp.abs(equation(trial, operands))))
            if np.isfinite(trial_residual) and trial_residual < original:
                candidate = trial
                new_residual = trial_residual
                accepted = True
                break
            factor_scale *= 0.5
        row = {
            "residual": new_residual,
            "original": original,
            "accepted": accepted,
            "factor_scale": factor_scale if accepted else 0.0,
            "linearize_ms": linearize_ms,
            "build_ms": build_ms,
            "apply_ms": apply_ms,
            "iterations": iterations,
            "linear_residual": float(
                jnp.linalg.norm(factor(jnp.asarray(step)) - rhs)
                / jnp.maximum(jnp.linalg.norm(rhs), 1e-30)
            ),
        }
        return candidate.reshape(reference.shape), row

    point = reference / scale
    history = []
    for index in range(args.max_steps):
        point, row = linear_step(point)
        row["correction"] = index + 1
        history.append(row)
        result["history"] = history
        save(output, result)
        if row["residual"] <= 1e-12 or not row["accepted"]:
            break
    result["converged"] = history[-1]["residual"] <= 1e-12
    result["terminal_state"] = {
        "moments": np.asarray(point * scale).tolist(),
        "roots": np.asarray(mapped(point * scale, *operands).roots).tolist(),
    }
    save(output, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--method", choices=("krylov", "gmres", "dense"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--line-search", action="store_true")
    parser.add_argument("--taylor", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
