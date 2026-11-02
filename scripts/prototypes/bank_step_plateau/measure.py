"""Compare Newton's assembled direction with fresh and held-read differences."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import subprocess
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine


def emit(label, **values):
    print(label, json.dumps(values, sort_keys=True, default=float), flush=True)


def parts(flat):
    return tuple(jnp.split(flat, 3))


def flat(values):
    return jnp.concatenate(tuple(jnp.ravel(value) for value in values))


def norm(value):
    return float(np.linalg.norm(np.asarray(value)))


def error(predicted, observed):
    return norm(predicted - observed) / max(norm(observed), 1e-30)


def setup():
    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine(
        "diverted"
    )
    seed = roots + 0.03 * jnp.median(bank.geometry.pitch)
    quadrature = replace(quadrature, target_current=jnp.sum(moments[0]))

    @jax.jit
    def read(state):
        return support_read.read(
            bank, cache, parts(state), seed, roots, convention, policy
        )

    @jax.jit
    def residual(state, support):
        return flat(
            support_read.step_residual(
                bank, parts(state), parts(state), support, profile, quadrature
            ).residual
        )

    @jax.jit
    def fresh(state):
        support = read(state)
        return residual(state, support), support.valid

    @jax.jit
    def held(state, anchor, support):
        return flat(
            support_read.step_residual(
                bank, parts(anchor), parts(state), support, profile, quadrature
            ).residual
        )

    @partial(jax.jit, static_argnames=("moving",))
    def matrix(state, support, moving):
        return support_read.step_jacobian(
            bank,
            parts(state),
            support,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
            fresh_geometry=moving,
        )

    @jax.jit
    def step(state, support):
        return support_read.newton_step(
            bank,
            parts(state),
            support,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
            fresh_geometry=True,
            condition_diagnostics=False,
        )

    @jax.jit
    def flux(state):
        return support_read.nodal_jets(bank, parts(state))[bank.sample_index, 0]

    return bank, flat(moments), read, fresh, held, matrix, step, flux


def record(path):
    _, state, read, fresh, _, _, step, _ = setup()
    states = []
    directions = []
    accepted = []
    for index in range(30):
        before, valid = fresh(state)
        magnitude = float(jnp.max(jnp.abs(before)))
        emit("ITERATE", index=index, residual=magnitude, valid=bool(valid))
        if magnitude <= 1e-10 and bool(valid):
            break
        support = read(state)
        correction = step(state, support)
        direction = flat(correction.update)
        states.append(np.asarray(state))
        directions.append(np.asarray(direction))
        for half in range(13):
            length = 2.0**-half
            candidate = state + length * direction
            value, admissible = fresh(candidate)
            after = float(jnp.max(jnp.abs(value)))
            if bool(admissible) and np.isfinite(after) and after < magnitude:
                accepted.append(length)
                emit(
                    "ACCEPT",
                    index=index,
                    length=length,
                    residual=after,
                    linear_residual=float(correction.linear_residual),
                )
                state = candidate
                break
        else:
            emit("NO_DESCENT", index=index, residual=magnitude)
            break
    else:
        raise AssertionError("fresh solve exceeded 30 steps")
    terminal, valid = fresh(state)
    terminal_residual = float(jnp.max(jnp.abs(terminal)))
    np.savez(
        path,
        states=np.stack(states),
        directions=np.stack(directions),
        accepted=np.asarray(accepted),
        terminal=np.asarray(state),
    )
    emit(
        "FINAL",
        steps=len(accepted),
        residual=terminal_residual,
        valid=bool(valid),
        trajectory=str(path),
    )
    assert bool(valid) and terminal_residual <= 1e-10


def compare(path, output, mutation):
    bank, _, read, fresh, held, matrix, _, flux = setup()
    trajectory = np.load(path)
    rows = []
    ladder = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
    for index, (x, d) in enumerate(
        zip(trajectory["states"], trajectory["directions"], strict=True)
    ):
        state = jnp.asarray(x)
        direction = jnp.asarray(d)
        support = read(state)
        fresh_matrix = matrix(state, support, True)
        held_matrix = matrix(state, support, False)
        if mutation and index == 2:
            column = int(jnp.argmax(jnp.abs(direction)))
            fresh_matrix = fresh_matrix.at[0, column].add(1e-3)
            held_matrix = held_matrix.at[0, column].add(1e-3)
        full_action = fresh_matrix @ direction
        held_action = held_matrix @ direction
        geometry_action = full_action - held_action
        flux_action = support_read.nodal_jets(bank, parts(direction))[
            bank.sample_index, 0
        ]
        base, valid = fresh(state)
        emit(
            "BASE",
            index=index,
            residual=float(jnp.max(jnp.abs(base))),
            valid=bool(valid),
            direction_norm=norm(direction),
        )
        for relative_size in ladder:
            scale = relative_size * norm(state) / norm(direction)
            plus = state + scale * direction
            minus = state - scale * direction
            high, high_valid = fresh(plus)
            low, low_valid = fresh(minus)
            full_difference = (high - low) / (2 * scale)
            held_difference = (
                held(plus, state, support) - held(minus, state, support)
            ) / (2 * scale)
            geometry_difference = full_difference - held_difference
            flux_difference = (flux(plus) - flux(minus)) / (2 * scale)
            row = {
                "iterate": index,
                "h_over_x": relative_size,
                "h": scale,
                "fresh_residual": float(jnp.max(jnp.abs(base))),
                "plus_valid": bool(high_valid),
                "minus_valid": bool(low_valid),
                "full": error(full_action, full_difference),
                "booking": error(held_action, held_difference),
                "geometry": error(geometry_action, geometry_difference),
                "flux": error(flux_action, flux_difference),
                "full_abs": norm(full_action - full_difference),
                "booking_abs": norm(held_action - held_difference),
                "geometry_abs": norm(geometry_action - geometry_difference),
                "flux_abs": norm(flux_action - flux_difference),
                "geometry_reference_norm": norm(geometry_difference),
            }
            rows.append(row)
            emit("DIFFERENCE", **row)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    if mutation:
        target = next(
            row for row in rows if row["iterate"] == 2 and row["h_over_x"] == 1e-5
        )
        assert target["booking"] > 1e-6, target
        emit("NEGATIVE_CONTROL", booking_error=target["booking"], flagged=True)
    emit(
        "SUMMARY",
        rows=len(rows),
        iterates=len(trajectory["states"]),
        output=str(output),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("record", "compare"))
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mutation", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    emit(
        "HEADER",
        revision=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        tree=str(root),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        command=" ".join(__import__("sys").argv),
    )
    assert jax.config.jax_enable_x64
    if args.mode == "record":
        record(args.trajectory)
    else:
        if args.output is None:
            parser.error("compare requires --output")
        compare(args.trajectory, args.output, args.mutation)
