"""Measure the drive barrier's soft mode and test a centroid constraint."""

# ruff: noqa: E402
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import drive_continuation, support_read
from nova.equilibrium.constraint import VerticalPositionPin, uniform_radial_field_jet
from scripts.prototypes.bank_drive_continuation import measure as predecessor
from scripts.prototypes.bank_solve_profile import fixture as profile_fixture
from scripts.prototypes.bank_warm_start import streams


def emit(label, **values):
    print(label + " " + json.dumps(values, sort_keys=True), flush=True)


def centroid(state):
    return jnp.sum(state[2], axis=-1) / jnp.sum(state[0], axis=-1)


def shifted(machine, field):
    operands = replace(
        machine.operands,
        exterior=machine.operands.exterior
        + field * uniform_radial_field_jet(machine.operands.nodes),
    )
    cache = replace(
        machine.cache,
        exterior=machine.cache.exterior
        + field * uniform_radial_field_jet(machine.cache.nodes),
    )
    return operands, cache


def run_unpinned(machine, moments, seeds):
    predecessor._MACHINE = machine
    accepted = []

    def solve(
        _op,
        _cache,
        state,
        seed,
        warm,
        _convention,
        _policy,
        drive,
        _gradient,
        _quadrature,
        _axis_sign=-1.0,
        *,
        max_steps=60,
        tolerance=1e-10,
        **_options,
    ):
        attempt = predecessor.solve(
            drive, state, seed, warm, max_steps=max_steps, tolerance=tolerance
        )
        if (
            bool(jnp.all(attempt.converged))
            and float(jnp.max(attempt.residual)) <= tolerance
        ):
            accepted.append((float(drive), attempt.moments, centroid(attempt.moments)))
        return attempt

    result = drive_continuation.advance_drive(
        None,
        None,
        moments,
        seeds,
        seeds,
        None,
        None,
        lambda drive: drive,
        profile_fixture.gradient,
        None,
        start_drive=1.0,
        target_drive=1.01,
        solve=solve,
    )
    return result, accepted


def inspect_null(machine, state, seeds, drive):
    one = tuple(part[0] for part in state)
    op = machine.operands
    support = support_read.read(
        op,
        machine.cache,
        one,
        seeds[0],
        seeds[0],
        machine.convention,
        machine.policy,
    )
    profile = streams.scaled_profile(machine.profile, jnp.asarray(drive))
    residual = support_read.step_residual(
        op, one, one, support, profile, machine.quadrature
    )
    matrix = support_read.step_jacobian(
        op,
        one,
        support,
        profile,
        profile_fixture.gradient,
        machine.quadrature,
        booked=residual.booked,
        fresh_geometry=True,
    )
    balanced, _, column = support_read._equilibrate(matrix, one, residual.booked)
    _, singular, vh = jnp.linalg.svd(balanced, full_matrices=False)
    null_state = column * vh[-1]
    cells = one[0].size
    parts = tuple(jnp.split(null_state, (cells, 2 * cells)))
    null_jets = support_read.nodal_jets(op, parts) - op.exterior
    plasma_jets = support_read.nodal_jets(op, one) - op.exterior
    valid = np.arange(op.nodes.shape[0]) < int(op.active_nodes)
    r = np.asarray(op.nodes)[valid, 0]
    z = np.asarray(op.nodes)[valid, 1]
    psi = np.asarray(null_jets)[valid, 0]
    vertical = -np.asarray(plasma_jets)[valid, 2]
    radial = -np.asarray(plasma_jets)[valid, 1]

    def cosine(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    payload = dict(
        drive=drive,
        singular=float(singular[-1]),
        cosine_vertical=cosine(psi, vertical),
        cosine_radial=cosine(psi, radial),
        centroid=float(centroid(one)),
    )
    emit("NULL", **payload)
    return dict(
        payload,
        r=r.tolist(),
        z=z.tolist(),
        flux=psi.tolist(),
        wall=np.asarray(op.geometry.wall.coordinate).tolist(),
        wall_offsets=np.asarray(op.geometry.wall_unit_offsets).tolist(),
        wall_closed=np.asarray(op.geometry.wall_unit_closed).tolist(),
    )


def pinned_step(machine, state, seed, drive, field, target):
    op, cache = shifted(machine, field)
    profile = streams.scaled_profile(machine.profile, drive)
    support = support_read.read(
        op, cache, state, seed, seed, machine.convention, machine.policy
    )
    residual = support_read.step_residual(
        op, state, state, support, profile, machine.quadrature
    )
    matrix = support_read.step_jacobian(
        op,
        state,
        support,
        profile,
        profile_fixture.gradient,
        machine.quadrature,
        booked=residual.booked,
        fresh_geometry=True,
    )
    field_delta = 1e-6
    op_plus, cache_plus = shifted(machine, field + field_delta)
    support_plus = support_read.read(
        op_plus, cache_plus, state, seed, seed, machine.convention, machine.policy
    )
    residual_plus = support_read.step_residual(
        op_plus, state, state, support_plus, profile, machine.quadrature
    )

    def flat(value):
        return jnp.concatenate(value)

    rhs = -flat(residual.residual)
    field_column = (flat(residual_plus.residual) + rhs) / field_delta
    balanced, row, column = support_read._equilibrate(matrix, state, residual.booked)
    both = jnp.linalg.solve(
        balanced, jnp.stack((rhs, field_column), axis=-1) / row[:, None]
    )
    u, v = column * both[:, 0], column * both[:, 1]
    total = jnp.sum(state[0])
    zc = jnp.sum(state[2]) / total
    n = state[0].size
    g = jnp.concatenate(
        (jnp.full(n, -zc / total), jnp.zeros(n), jnp.full(n, 1 / total))
    )
    field_update = (-jnp.dot(g, u) - (zc - target)) / -jnp.dot(g, v)
    updates = u - v * field_update
    pieces = jnp.split(updates, (n, 2 * n), axis=-1)
    return tuple(pieces), field_update, support.valid


def pinned_measure(machine, initial, seeds, targets):
    pin = VerticalPositionPin(
        targets, source_scale=1e-3, row_scale=1e-3, tolerance=1e-6, active=True
    )
    assert pin.active
    print("SCRATCH_PIN_ACTIVE=true", flush=True)
    emit(
        "PIN_CONFIG",
        value="true",
        row_scale=float(pin.row_scale),
        source_scale=float(pin.source_scale),
        target=np.asarray(targets).tolist(),
    )

    @jax.jit
    def step(state, field, drive):
        return jax.vmap(
            lambda one, seed, strength, target: pinned_step(
                machine, one, seed, drive, strength, target
            )
        )(state, seeds, field, targets)

    @jax.jit
    def evaluate(state, field, drive):
        profile = streams.scaled_profile(machine.profile, drive)

        def one(one_state, seed, strength, target):
            op, cache = shifted(machine, strength)
            support = support_read.read(
                op, cache, one_state, seed, seed, machine.convention, machine.policy
            )
            residual = support_read.step_residual(
                op, one_state, one_state, support, profile, machine.quadrature
            )
            norm = jnp.max(jnp.abs(jnp.concatenate(residual.residual)))
            return norm, centroid(one_state) - target, support.valid

        return jax.vmap(one)(state, seeds, field, targets)

    state = initial
    field = jnp.zeros(len(targets), dtype=initial[0].dtype)
    drive = 1.0
    sub_step = 1e-3
    rows = []
    reads = updates = 0
    for _ in range(256):
        if drive >= 1.01:
            break
        change = min(sub_step, 1.01 - drive)
        requested = drive + change
        tolerance = 1e-10 if requested >= 1.01 else 1e-6
        trial_state, trial_field = state, field
        for iteration in range(6):
            norms, heights, valid = evaluate(trial_state, trial_field, requested)
            reads += 1
            norms, heights, valid = map(np.asarray, (norms, heights, valid))
            if np.all(valid & (norms <= tolerance) & (np.abs(heights) <= 1e-6)):
                break
            direction, field_direction, step_valid = step(
                trial_state, trial_field, requested
            )
            updates += 1
            best_score = np.maximum(norms / tolerance, np.abs(heights) / 1e-6)
            best_state, best_field = trial_state, trial_field
            for divisor in (1, 2, 4, 8, 16, 32, 64, 128):
                candidate_state = tuple(
                    a + b / divisor for a, b in zip(trial_state, direction)
                )
                candidate_field = trial_field + field_direction / divisor
                candidate_norms, candidate_heights, candidate_valid = evaluate(
                    candidate_state, candidate_field, requested
                )
                reads += 1
                score = np.maximum(
                    np.asarray(candidate_norms) / tolerance,
                    np.abs(np.asarray(candidate_heights)) / 1e-6,
                )
                improve = np.asarray(candidate_valid) & (score < best_score)
                best_score = np.where(improve, score, best_score)
                best_state = tuple(
                    jnp.where(improve[:, None], a, b)
                    for a, b in zip(candidate_state, best_state)
                )
                best_field = jnp.where(improve, candidate_field, best_field)
            trial_state, trial_field = best_state, best_field
        accepted = bool(
            np.all(valid & (norms <= tolerance) & (np.abs(heights) <= 1e-6))
        )
        rows.append(
            dict(
                drive=requested,
                accepted=accepted,
                residual_max=float(np.max(norms)),
                height_error_max=float(np.max(np.abs(heights))),
                field_max=float(np.max(np.abs(np.asarray(trial_field)))),
                iteration=iteration + 1,
            )
        )
        emit("PINNED_STEP", **rows[-1])
        if accepted:
            state, field, drive = trial_state, trial_field, requested
            sub_step = min(sub_step * 2, 1.0)
        else:
            sub_step *= 0.5
            if sub_step < 1e-9:
                break
    outcome = dict(
        drive=drive,
        certified=drive >= 1.01,
        reads=reads,
        updates=updates,
        steps=rows,
        final_field=np.asarray(field).tolist(),
    )
    emit(
        "PINNED",
        drive=drive,
        certified=outcome["certified"],
        reads=reads,
        updates=updates,
        substeps=len(rows),
    )
    return outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=16)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    root = Path(nova.__file__).resolve().parent.parent
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    emit(
        "HEADER",
        revision=revision,
        cwd=str(Path.cwd().resolve()),
        module=support_read.__file__,
        prototype=__file__,
        backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64),
        lanes=args.lanes,
    )
    print("PIN_ACTIVE=absent", flush=True)
    emit("PIN_CONFIG", value="absent", forward_default="false", bank_row_count=0)
    machine = streams.build()
    with np.load(args.streams) as archive:
        moments = tuple(
            jnp.asarray(archive[name][: args.lanes]) for name in ("m0", "m1", "m2")
        )
        seeds = jnp.asarray(archive["stationary"][: args.lanes])
    target = centroid(moments)
    unpinned, accepted = run_unpinned(machine, moments, seeds)
    emit(
        "UNPINNED",
        drive=unpinned.drive,
        certified=unpinned.certified,
        reads=unpinned.reads,
        updates=unpinned.updates,
        substeps=len(unpinned.substeps),
        centroid_drift=(np.asarray(accepted[-1][2] - target)).tolist(),
    )
    null = inspect_null(machine, unpinned.result.moments, seeds, unpinned.drive)
    receipt = dict(
        null=null,
        unpinned=dict(
            drive=unpinned.drive,
            certified=unpinned.certified,
            reads=unpinned.reads,
            updates=unpinned.updates,
            centroid_start=np.asarray(target).tolist(),
            centroid_accepted=[
                dict(drive=d, height=np.asarray(h).tolist()) for d, _, h in accepted
            ],
        ),
    )
    args.output.write_text(json.dumps(receipt, sort_keys=True))
    pinned = pinned_measure(machine, moments, seeds, target)
    receipt["pinned"] = pinned
    args.output.write_text(json.dumps(receipt, sort_keys=True))
    emit("DONE", output=str(args.output))


if __name__ == "__main__":
    main()
