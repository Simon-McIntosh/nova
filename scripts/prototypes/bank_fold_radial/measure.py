"""Compare drive continuation with and without a current-centroid radius pin."""

# ruff: noqa: E402
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import drive_continuation, support_read
from scripts.prototypes.bank_fold_vertical import measure as shared
from scripts.prototypes.bank_solve_profile import fixture as profile_fixture
from scripts.prototypes.bank_warm_start import streams


emit = shared.emit
accepted_read = shared.accepted_read
bank_pin_status = shared.bank_pin_status


def centroid(state, centres):
    """Major-radius centroid from cell current and cell-relative first moments."""
    return jnp.sum(state[0] * centres + state[1], axis=-1) / jnp.sum(state[0], axis=-1)


def centroid_row(state, centres):
    """Differentiate physical radius with respect to the three moment blocks."""
    total = jnp.sum(state[0])
    radius = centroid(state, centres)
    count = state[0].size
    return jnp.concatenate(
        ((centres - radius) / total, jnp.full(count, 1 / total), jnp.zeros(count))
    )


ROW_SCALE = 1e-3
SOURCE_SCALE = 1e-3


@dataclass(frozen=True)
class RadialPositionPin:
    """Seed current-centroid radius held by a uniform vertical-field source."""

    target_radius: object
    source_scale: float = SOURCE_SCALE
    row_scale: float = ROW_SCALE
    tolerance: float = 1e-6
    active: bool = True


def uniform_vertical_field_jet(points):
    """Flux, gradient and Hessian from one tesla of vertical field."""
    radius = jnp.asarray(points)[..., 0]
    zeros = jnp.zeros_like(radius)
    return jnp.stack(
        (
            jnp.pi * radius**2,
            2 * jnp.pi * radius,
            zeros,
            2 * jnp.pi * jnp.ones_like(radius),
            zeros,
            zeros,
        ),
        axis=-1,
    )


def shifted(machine, field):
    operands = replace(
        machine.operands,
        exterior=machine.operands.exterior
        + field * uniform_vertical_field_jet(machine.operands.nodes),
    )
    cache = replace(
        machine.cache,
        exterior=machine.cache.exterior
        + field * uniform_vertical_field_jet(machine.cache.nodes),
    )
    return operands, cache


def run_unpinned(machine, moments, seeds):
    """Reuse the bank replay, adapting only its recorded centroid coordinate."""
    result, states = shared.run_unpinned(machine, moments, seeds)
    centres = machine.operands.geometry.centre[:, 0]
    return result, [
        (drive, state, centroid(state, centres)) for drive, state, _ in states
    ]


def linearized_system(machine, state, seed, drive, field):
    """Assemble the fresh bank Jacobian and physical vertical-field column."""
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
    return matrix, residual.booked, rhs, field_column, support.valid, support_plus.valid


def pinned_step(machine, state, seed, drive, field, target, payload):
    matrix, booked, rhs, field_column, valid, _ = linearized_system(
        machine, state, seed, drive, field
    )
    field_column = field_column * jnp.asarray(payload) * SOURCE_SCALE
    balanced, row, column = support_read._equilibrate(matrix, state, booked)
    both = jnp.linalg.solve(
        balanced, jnp.stack((rhs, field_column), axis=-1) / row[:, None]
    )
    u, v = column * both[:, 0], column * both[:, 1]
    centres = machine.operands.geometry.centre[:, 0]
    radius = centroid(state, centres)
    n = state[0].size
    g = centroid_row(state, centres) * jnp.asarray(payload) * ROW_SCALE
    field_update = (-jnp.dot(g, u) - ROW_SCALE * (radius - target)) / -jnp.dot(g, v)
    updates = u - v * field_update
    pieces = jnp.split(updates, (n, 2 * n), axis=-1)
    row_present = jnp.any(g != 0) & jnp.any(field_column != 0)
    return tuple(pieces), field_update * SOURCE_SCALE, valid, row_present


def correct_pinned(
    step,
    evaluate,
    trial_state,
    trial_field,
    requested,
    tolerance,
    max_steps,
    report_pin,
):
    """Correct one bordered sub-step, including the read after its final update."""
    reads = updates = 0
    for iteration in range(max_steps):
        norms, heights, valid = evaluate(trial_state, trial_field, requested)
        reads += 1
        norms, heights, valid = map(np.asarray, (norms, heights, valid))
        if accepted_read(valid, norms, tolerance, height=heights):
            break
        direction, field_direction, step_valid, row_present = step(
            trial_state, trial_field, requested
        )
        report_pin(row_present)
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
    else:
        # The last update must receive the same fresh read as earlier ones.
        norms, heights, valid = evaluate(trial_state, trial_field, requested)
        reads += 1
        norms, heights, valid = map(np.asarray, (norms, heights, valid))
    accepted = accepted_read(valid, norms, tolerance, height=heights)
    return SimpleNamespace(
        state=trial_state,
        field=trial_field,
        residual=norms,
        position_error=heights,
        valid=valid,
        accepted=accepted,
        steps=updates,
        reads=reads,
        iteration=iteration + 1,
    )


def advance_pinned(initial, field, seeds, solve, target_drive, terminal_tolerance):
    """Delegate adaptive drive scheduling to the bank continuation owner."""
    return drive_continuation.advance_drive(
        None,
        None,
        (initial, field),
        seeds,
        seeds,
        None,
        None,
        lambda drive: drive,
        profile_fixture.gradient,
        None,
        start_drive=1.0,
        target_drive=target_drive,
        certification_tolerance=terminal_tolerance,
        solve=solve,
    )


def pinned_measure(
    machine,
    initial,
    seeds,
    targets,
    *,
    target_drive=1.01,
    terminal_tolerance=1e-10,
    return_state=False,
):
    pin = RadialPositionPin(
        targets, source_scale=1e-3, row_scale=1e-3, tolerance=1e-6, active=True
    )
    emit(
        "PIN_CONFIG",
        row_scale=float(pin.row_scale),
        source_scale=float(pin.source_scale),
        target=np.asarray(targets).tolist(),
    )

    @jax.jit
    def step(state, field, drive):
        return jax.vmap(
            lambda one, seed, strength, target: pinned_step(
                machine, one, seed, drive, strength, target, pin.active
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
            return (
                norm,
                centroid(one_state, machine.operands.geometry.centre[:, 0]) - target,
                support.valid,
            )

        return jax.vmap(one)(state, seeds, field, targets)

    rows = []
    accepted_states = []
    pin_reported = False

    def report_pin(row_present):
        nonlocal pin_reported
        if pin_reported:
            return
        active = bool(np.all(np.asarray(row_present)))
        print(f"PIN_ACTIVE={str(active).lower()}", flush=True)
        emit("PIN_CONFIG", value=str(active).lower(), row_count=int(active))
        if not active:
            raise AssertionError("centroid-radius row or vertical-field column absent")
        pin_reported = True

    def solve(
        _op,
        _cache,
        current,
        _seed,
        _warm,
        _convention,
        _policy,
        requested,
        _gradient,
        _quadrature,
        _axis_sign=-1.0,
        *,
        max_steps,
        tolerance,
    ):
        state, field = current
        attempt = correct_pinned(
            step, evaluate, state, field, requested, tolerance, max_steps, report_pin
        )
        rows.append(
            dict(
                drive=requested,
                accepted=attempt.accepted,
                residual_max=float(np.max(attempt.residual)),
                radius_error_max=float(np.max(np.abs(attempt.position_error))),
                field_max=float(np.max(np.abs(np.asarray(attempt.field)))),
                iteration=attempt.iteration,
            )
        )
        emit("PINNED_STEP", **rows[-1])
        if attempt.accepted and return_state:
            accepted_states.append((requested, attempt.state, attempt.field))
        return SimpleNamespace(
            moments=(attempt.state, attempt.field),
            converged=attempt.accepted,
            residual=attempt.residual,
            steps=attempt.steps,
            reads=attempt.reads,
            support=SimpleNamespace(stationary=SimpleNamespace(position=seeds)),
        )

    field = jnp.zeros(len(targets), dtype=initial[0].dtype)
    result = advance_pinned(
        initial, field, seeds, solve, target_drive, terminal_tolerance
    )
    state, field = result.result.moments if result.result else (initial, field)
    drive, reads, updates = result.drive, result.reads, result.updates
    outcome = dict(
        drive=drive,
        certified=drive >= target_drive,
        reads=reads,
        updates=updates,
        substeps=len(rows),
        residual_max=rows[-1]["residual_max"],
        accepted_residual_max=next(
            (r["residual_max"] for r in reversed(rows) if r["accepted"]), None
        ),
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
    if return_state:
        return outcome, state, field, accepted_states
    return outcome


def condition_rows(machine, seeds, states, arm):
    """Measure both balanced matrices at retained states of one branch."""
    from scripts.prototypes.bank_fold_radial.condition import condition

    measure = jax.jit(
        lambda one, field, drive: condition(machine, one, seeds[0], field, drive)
    )
    rows = []
    for drive, batch, fields in states:
        one = tuple(part[0] for part in batch)
        bordered, original, valid = measure(one, fields[0], jnp.asarray(drive))
        row = dict(
            drive=float(drive),
            smallest=float(bordered if arm == "pinned" else original),
            original=float(original),
            bordered=float(bordered),
            valid=bool(valid),
        )
        if not row["valid"] or not np.isfinite(row["smallest"]):
            raise ValueError(f"invalid {arm} conditioning at drive {drive}")
        rows.append(row)
        emit("SINGULAR", arm=arm, **row)
    return rows


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
        row_scale=ROW_SCALE,
        source_scale=SOURCE_SCALE,
        revision=revision,
        cwd=str(Path.cwd().resolve()),
        module=support_read.__file__,
        prototype=__file__,
        backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64),
        lanes=args.lanes,
    )
    machine = streams.build()
    with np.load(args.streams) as archive:
        moments = tuple(
            jnp.asarray(archive[name][: args.lanes]) for name in ("m0", "m1", "m2")
        )
        seeds = jnp.asarray(archive["stationary"][: args.lanes])
    target = centroid(moments, machine.operands.geometry.centre[:, 0])
    bank_pin_status(machine, moments, seeds)
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
    receipt = dict(
        unpinned=dict(
            drive=unpinned.drive,
            certified=unpinned.certified,
            reads=unpinned.reads,
            updates=unpinned.updates,
            substeps=len(unpinned.substeps),
            residual_max=float(np.max(np.asarray(unpinned.result.residual))),
            centroid_start=np.asarray(target).tolist(),
            centroid_accepted=[
                dict(drive=d, radius=np.asarray(h).tolist()) for d, _, h in accepted
            ],
        ),
    )
    args.output.write_text(json.dumps(receipt, sort_keys=True))
    unpinned_states = [
        (drive, state, jnp.zeros(args.lanes, dtype=moments[0].dtype))
        for drive, state, _ in [accepted[0], *accepted[-3:]]
    ]
    receipt["condition"] = dict(
        unpinned=condition_rows(machine, seeds, unpinned_states, "unpinned")
    )
    args.output.write_text(json.dumps(receipt, sort_keys=True))
    pinned, _, _, pinned_states = pinned_measure(
        machine, moments, seeds, target, return_state=True
    )
    receipt["pinned"] = pinned
    args.output.write_text(json.dumps(receipt, sort_keys=True))
    selected = [
        (1.0, moments, jnp.zeros(args.lanes, dtype=moments[0].dtype)),
        *pinned_states[:: max(1, len(pinned_states) // 3)],
    ]
    if pinned_states and selected[-1][0] != pinned_states[-1][0]:
        selected.append(pinned_states[-1])
    receipt["condition"]["pinned"] = condition_rows(machine, seeds, selected, "pinned")
    args.output.write_text(json.dumps(receipt, sort_keys=True))
    emit("DONE", output=str(args.output))


if __name__ == "__main__":
    main()
