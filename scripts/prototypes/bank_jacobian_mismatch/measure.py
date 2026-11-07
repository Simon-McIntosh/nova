"""Compare the assembled step Jacobian with central differences of the fresh read.

The Newton matrix the fast step solves is ``I - d(booked)/du``. The residual the
solve evaluates comes from reading the state afresh and booking at that read, so
that is the map the matrix must differentiate. This script central differences
the fresh-read residual at the state where the 1e-2 drive stall begins and at
the converged certificate state, tables the relative mismatch per direction
class and step size, and splits the mismatch between the analytic Jacobian terms
and the read refresh inside the stencil.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_solve_profile import fixture as profile_fixture
from scripts.prototypes.bank_warm_start import streams

TOLERANCE = 1e-10
STEPS = (1e-5, 1e-6, 1e-7)
SEED_ITERATIONS = 12
CLASSES = ("cut", "normal", "whole", "axis", "saddle")


def _default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return [_default(item) for item in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    raise TypeError(type(value))


def emit(label, **values):
    print(label + " " + json.dumps(values, sort_keys=True, default=_default), flush=True)


def residual_vector(operands, base, state, support, profile, quadrature):
    """The residual at ``state`` under ``support`` as one vector; invalid cells zeroed."""
    result = support_read.step_residual(
        operands, base, state, support, profile, quadrature
    )
    vector = jnp.concatenate(tuple(jnp.ravel(value) for value in result.residual))
    return jnp.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def read_at(operands, cache, state, seeds, previous, convention, policy, axis_sign):
    return support_read.read(
        operands,
        cache,
        state,
        seeds,
        previous,
        convention,
        policy,
        axis_sign,
        iterations=SEED_ITERATIONS,
    )


def decision_signature(reading):
    membership = np.asarray(reading.membership)
    return {
        "cut": (membership > 0) & (membership < 1 - 1e-14),
        "normal": np.asarray(reading.normal_form_cells),
        "fragments": np.asarray(reading.fragment_selected),
        "ownership": np.rint(membership),
        "saddle": np.asarray(reading.saddle_form.position),
    }


def signature_deltas(before, after):
    return {
        "cut": int(np.sum(before["cut"] != after["cut"])),
        "normal": int(np.sum(before["normal"] != after["normal"])),
        "fragments": int(np.sum(before["fragments"] != after["fragments"])),
        "ownership": int(np.sum(before["ownership"] != after["ownership"])),
        "saddle_move": float(np.max(np.abs(after["saddle"] - before["saddle"]))),
    }


def relative(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else float("inf")


def mismatch(vector, reference):
    error = float(jnp.linalg.norm(vector - reference))
    return {
        "abs": error,
        "rel": relative(error, float(jnp.linalg.norm(reference))),
        "rel_j": relative(error, float(jnp.linalg.norm(vector))),
    }


def class_cells(reading, centre):
    membership = np.asarray(reading.membership)

    def nearest(point):
        return int(np.argmin(np.sum((centre - np.asarray(point)) ** 2, axis=-1)))

    return {
        "cut": np.flatnonzero((membership > 0) & (membership < 1 - 1e-14)),
        "normal": np.flatnonzero(np.asarray(reading.normal_form_cells)),
        "whole": np.flatnonzero(membership >= 1 - 1e-14),
        "axis": np.asarray([nearest(reading.axis)]),
        "saddle": np.asarray([nearest(reading.saddle_form.position)]),
    }


def direction_list(cells_of_class, count, cells, rng):
    built = []
    for name in CLASSES:
        pool = cells_of_class[name]
        if pool.size == 0:
            continue
        for index in range(count):
            cell = int(pool[index % pool.size])
            built.append((f"{name}-c{cell}-b{index % 3}", name, cell, index % 3))
    for index in range(8):
        cell = int(rng.integers(0, cells))
        built.append((f"random-c{cell}-b{index % 3}", "random", cell, index % 3))
    return built


def unit_direction(cell, block, cells):
    blocks = [jnp.zeros(cells) for _ in range(3)]
    blocks[block] = blocks[block].at[cell].set(1.0)
    return tuple(blocks)


def perturb(state, direction, scale, eps, sign):
    return tuple(
        value + sign * eps * scale_b * step
        for value, scale_b, step in zip(state, scale, direction, strict=True)
    )


def measure_state(label, state, support, operands, cache, seeds, previous, convention,
                  policy, profile, quadrature, gradient, axis_sign, count, rng,
                  control_gradient=None, control_count=0):
    cells = int(state[0].size)
    scale = tuple(jnp.maximum(jnp.max(jnp.abs(value), axis=-1), 1e-30) for value in state)

    @jax.jit
    def assemble_fresh(moments_state, moments_support):
        return support_read.step_jacobian(
            operands, moments_state, moments_support, profile, gradient, quadrature,
            fresh_geometry=True,
        )

    @jax.jit
    def assemble_static(moments_state, moments_support):
        return support_read.step_jacobian(
            operands, moments_state, moments_support, profile, gradient, quadrature,
            fresh_geometry=False,
        )

    @jax.jit
    def read_one(one):
        return read_at(
            operands, cache, one, seeds, previous, convention, policy, axis_sign
        )

    @jax.jit
    def residual_one(one, one_support):
        return residual_vector(operands, one, one, one_support, profile, quadrature)

    @jax.jit
    def residual_frozen(one, one_support):
        return residual_vector(operands, state, one, one_support, profile, quadrature)

    matrix_full = assemble_fresh(state, support)
    matrix_static = assemble_static(state, support)
    matrix_control = None
    if control_gradient is not None:
        @jax.jit
        def assemble_control(moments_state, moments_support):
            return support_read.step_jacobian(
                operands, moments_state, moments_support, profile, control_gradient,
                quadrature, fresh_geometry=True,
            )

        matrix_control = assemble_control(state, support)
    classify = class_cells(support.reading, np.asarray(operands.geometry.centre))
    picks = direction_list(classify, count, cells, rng)
    base_residual = residual_vector(operands, state, state, support, profile, quadrature)
    rows, decisions = [], []
    for name, klass, cell, block in picks:
        direction = unit_direction(cell, block, cells)
        flat = jnp.concatenate(direction)
        analytic_full = matrix_full @ flat
        analytic_static = matrix_static @ flat
        analytic_control = None if matrix_control is None else matrix_control @ flat
        row = {"class": klass, "cell": cell, "block": block, "direction": name}
        for eps in STEPS:
            plus = perturb(state, direction, scale, eps, 1.0)
            minus = perturb(state, direction, scale, eps, -1.0)
            plus_support = read_one(plus)
            minus_support = read_one(minus)
            fresh_cd = -(
                residual_one(plus, plus_support) - residual_one(minus, minus_support)
            ) / (2.0 * eps)
            frozen_cd = -(
                residual_frozen(plus, support) - residual_frozen(minus, support)
            ) / (2.0 * eps)
            refresh = fresh_cd - frozen_cd
            hybrids = {
                "analytic-full": analytic_full,
                "analytic-static": analytic_static,
                "booking-to-cd": frozen_cd + (analytic_full - analytic_static),
                "geometry-to-cd": analytic_static + refresh,
            }
            tag = format(eps, "g")
            for option, vector in hybrids.items():
                for key, value in mismatch(vector, fresh_cd).items():
                    row[f"{option}:h{tag}:{key}"] = value
            row[f"refresh:h{tag}:rel"] = relative(
                float(jnp.linalg.norm(refresh)), float(jnp.linalg.norm(fresh_cd))
            )
            row[f"static-vs-frozen:h{tag}:rel"] = relative(
                float(jnp.linalg.norm(analytic_static - frozen_cd)),
                float(jnp.linalg.norm(frozen_cd)),
            )
            if analytic_control is not None and len(rows) < control_count:
                row[f"booking-to-cd-control:h{tag}:rel"] = mismatch(
                    frozen_cd + (analytic_control - analytic_static), fresh_cd
                )["rel"]
                row[f"matrix-control-change:h{tag}"] = relative(
                    float(jnp.linalg.norm(analytic_control - analytic_full)),
                    float(jnp.linalg.norm(analytic_full)),
                )
            decisions.append(
                {
                    "class": klass,
                    "direction": name,
                    "step": eps,
                    "plus": signature_deltas(
                        decision_signature(support.reading),
                        decision_signature(plus_support.reading),
                    ),
                    "minus": signature_deltas(
                        decision_signature(support.reading),
                        decision_signature(minus_support.reading),
                    ),
                    "plus_valid": bool(np.all(np.asarray(plus_support.valid))),
                    "minus_valid": bool(np.all(np.asarray(minus_support.valid))),
                }
            )
        rows.append(row)
    return {
        "label": label,
        "cells": cells,
        "base_residual_max": float(jnp.max(jnp.abs(base_residual))),
        "target_current": None
        if quadrature.target_current is None
        else float(quadrature.target_current),
        "class_counts": {name: int(len(classify[name])) for name in CLASSES},
        "directions": rows,
        "decisions": decisions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", type=int, default=5)
    args = parser.parse_args()
    output = Path(args.output)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = "unversioned-archive"
    machine = streams.build()
    emit(
        "HEADER",
        revision=revision,
        tree=os.getcwd(),
        cwd=os.getcwd(),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        devices=[str(device) for device in jax.devices()],
        x64=bool(jax.config.jax_enable_x64),
        job=os.environ.get("SLURM_JOB_ID"),
    )
    assert jax.config.jax_enable_x64
    operands, cache = machine.operands, machine.cache
    convention, policy = machine.convention, machine.policy
    quadrature = machine.quadrature
    gradient = profile_fixture.gradient
    axis_sign = -1.0
    lane = 0
    seed_state = tuple(jnp.asarray(value[lane]) for value in machine.cold_states)
    seeds = jnp.asarray(machine.cold_seeds[lane])
    previous = seeds

    def solve(amplitude, max_steps):
        profile = streams.scaled_profile(machine.profile, amplitude)

        @jax.jit
        def run(one):
            return support_read.solve_batch(
                operands,
                cache,
                tuple(value[None] for value in one),
                seeds[None],
                previous[None],
                convention,
                policy,
                profile,
                gradient,
                quadrature,
                axis_sign,
                max_steps=max_steps,
                reuse_factor=False,
                contraction_limit=0.5,
                tolerance=TOLERANCE,
            )

        return run(seed_state)

    certificate = solve(1.0, 60)
    certificate_state = tuple(jnp.asarray(value[0]) for value in certificate.moments)
    path = np.asarray(certificate.residual_path)[0]
    emit(
        "CERTIFICATE",
        residual=float(np.asarray(certificate.residual)[0]),
        steps=int(np.asarray(certificate.steps)[0]),
        converged=bool(np.asarray(certificate.converged)[0]),
        reads=int(np.asarray(certificate.reads)[0]),
        path=path[np.isfinite(path)].tolist(),
    )
    probe = solve(1.01, 1)
    probe_path = np.asarray(probe.residual_path)[0]
    emit(
        "STALL-PROBE",
        residual_before=float(probe_path[0]),
        residual_after_first_search=float(probe_path[1]),
        reads=int(np.asarray(probe.reads)[0]),
    )
    # The stalled state is the seed itself: the first Jacobian the solve assembles
    # is built there, and every later trial is a damped retreat from it.
    stalled_state = seed_state

    states = (
        ("stalled-1e-2", stalled_state, streams.scaled_profile(machine.profile, 1.01)),
        ("certificate", certificate_state, machine.profile),
    )
    def profile_probe(one_profile, amplitude, points, flux):
        """The flux derivative of the booked source at the probe points."""
        step = 1e-6
        high = np.asarray(one_profile.p_prime(flux + step))
        low = np.asarray(one_profile.p_prime(flux - step))
        field_high = np.asarray(one_profile.ff_prime(flux + step))
        field_low = np.asarray(one_profile.ff_prime(flux - step))
        provided = np.asarray(gradient(points, flux))
        return {
            "amplitude": float(amplitude),
            "p_prime_value": float(np.max(np.abs(np.asarray(one_profile.p_prime(flux))))),
            "p_prime_flux_derivative": float(np.max(np.abs((high - low) / (2 * step)))),
            "ff_prime_flux_derivative": float(
                np.max(np.abs((field_high - field_low) / (2 * step)))
            ),
            "supplied_gradient_max": float(np.max(np.abs(provided))),
            "supplied_gradient_shape": list(provided.shape),
        }

    stalled_profile = streams.scaled_profile(machine.profile, 1.01)
    probe_flux = jnp.linspace(0.0, 1.0, 9)
    probe_points = jnp.full_like(probe_flux, 1.0)
    gradient_check = {
        "stalled": profile_probe(stalled_profile, 1.01, probe_points, probe_flux),
        "certificate": profile_probe(machine.profile, 1.0, probe_points, probe_flux),
    }
    emit("GRADIENT", **gradient_check["stalled"])
    receipt_gradient = gradient_check
    def control(radius, flux):
        return jnp.full_like(radius, 1e-3)

    rng = np.random.default_rng(20261107)
    receipt = {
        "revision": revision,
        "module": support_read.__file__,
        "steps": list(STEPS),
        "gradient_check": receipt_gradient,
        "states": [],
    }
    for label, state, profile in states:
        support = read_at(operands, cache, state, seeds, previous, convention, policy, axis_sign)
        result = measure_state(
            label,
            state,
            support,
            operands,
            cache,
            seeds,
            previous,
            convention,
            policy,
            profile,
            quadrature,
            gradient,
            axis_sign,
            args.classes,
            rng,
            control_gradient=control,
            control_count=6,
        )
        receipt["states"].append(result)
        output.write_text(json.dumps(receipt, indent=2))
        emit("STATE", state=label, base_residual_max=result["base_residual_max"],
             **{f"count_{name}": count for name, count in result["class_counts"].items()})
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()