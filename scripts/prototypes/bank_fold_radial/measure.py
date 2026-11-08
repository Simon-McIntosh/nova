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

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import drive_continuation, support_read
from scripts.prototypes.bank_drive_continuation import measure as predecessor
from scripts.prototypes.bank_solve_profile import fixture as profile_fixture
from scripts.prototypes.bank_warm_start import streams


def emit(label, **values):
    print(label + " " + json.dumps(values, sort_keys=True), flush=True)


def centroid(state, centres):
    """Major-radius centroid from cell current and cell-relative first moments."""
    return jnp.sum(state[0] * centres + state[1], axis=-1) / jnp.sum(state[0], axis=-1)


def accepted_read(valid, residual, tolerance, *, height=None):
    """Apply the same fresh-read acceptance predicate to either branch."""
    accepted = np.asarray(valid, dtype=bool) & (np.asarray(residual) <= tolerance)
    if height is not None:
        accepted &= np.abs(np.asarray(height)) <= 1e-6
    return bool(np.all(accepted))


def bank_pin_status(machine, moments, seeds):
    """Count the rows assembled by the bank route at the supplied state."""
    one = tuple(part[0] for part in moments)
    op = machine.operands
    support = support_read.read(
        op, machine.cache, one, seeds[0], seeds[0], machine.convention, machine.policy
    )
    profile = streams.scaled_profile(machine.profile, jnp.asarray(1.0))
    residual = support_read.step_residual(
        op, one, one, support, profile, machine.quadrature
    )
    state_count = sum(part.size for part in one)
    row_count = sum(part.size for part in residual.residual)
    extra_rows = row_count - state_count
    if extra_rows not in (0, 1):
        raise ValueError(f"unexpected bank constraint row count: {extra_rows}")
    status = "true" if extra_rows else "absent"
    print(f"BANK_PIN_ACTIVE={status}", flush=True)
    emit("PIN_CONFIG", value=status, bank_row_count=extra_rows)


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
        if accepted_read(attempt.converged, attempt.residual, tolerance):
            accepted.append(
                (
                    float(drive),
                    attempt.moments,
                    centroid(attempt.moments, machine.operands.geometry.centre[:, 0]),
                )
            )
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


def pinned_step(machine, state, seed, drive, field, target, payload):
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
    field_column = field_column * jnp.asarray(payload) * SOURCE_SCALE
    balanced, row, column = support_read._equilibrate(matrix, state, residual.booked)
    both = jnp.linalg.solve(
        balanced, jnp.stack((rhs, field_column), axis=-1) / row[:, None]
    )
    u, v = column * both[:, 0], column * both[:, 1]
    total = jnp.sum(state[0])
    centres = machine.operands.geometry.centre[:, 0]
    zc = centroid(state, centres)
    n = state[0].size
    g = (
        jnp.concatenate(((centres - zc) / total, jnp.full(n, 1 / total), jnp.zeros(n)))
        * jnp.asarray(payload)
        * ROW_SCALE
    )
    field_update = (-jnp.dot(g, u) - ROW_SCALE * (zc - target)) / -jnp.dot(g, v)
    updates = u - v * field_update
    pieces = jnp.split(updates, (n, 2 * n), axis=-1)
    row_present = jnp.any(g != 0) & jnp.any(field_column != 0)
    return tuple(pieces), field_update * SOURCE_SCALE, support.valid, row_present


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

    state = initial
    field = jnp.zeros(len(targets), dtype=initial[0].dtype)
    drive = 1.0
    sub_step = 1e-3
    rows = []
    accepted_states = []
    reads = updates = 0
    pin_reported = False
    for _ in range(256):
        if drive >= target_drive:
            break
        change = min(sub_step, target_drive - drive)
        requested = drive + change
        tolerance = terminal_tolerance if requested >= target_drive else 1e-6
        trial_state, trial_field = state, field
        for iteration in range(6):
            norms, heights, valid = evaluate(trial_state, trial_field, requested)
            reads += 1
            norms, heights, valid = map(np.asarray, (norms, heights, valid))
            if accepted_read(valid, norms, tolerance, height=heights):
                break
            direction, field_direction, step_valid, row_present = step(
                trial_state, trial_field, requested
            )
            if not pin_reported:
                active = bool(np.all(np.asarray(row_present)))
                print(f"PIN_ACTIVE={str(active).lower()}", flush=True)
                emit(
                    "PIN_CONFIG",
                    value=str(active).lower(),
                    row_count=1 if active else 0,
                )
                if not active:
                    raise AssertionError(
                        "centroid-radius row or vertical-field column absent"
                    )
                pin_reported = True
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
        rows.append(
            dict(
                drive=requested,
                accepted=accepted,
                residual_max=float(np.max(norms)),
                radius_error_max=float(np.max(np.abs(heights))),
                field_max=float(np.max(np.abs(np.asarray(trial_field)))),
                iteration=iteration + 1,
            )
        )
        emit("PINNED_STEP", **rows[-1])
        if accepted:
            state, field, drive = trial_state, trial_field, requested
            if return_state:
                accepted_states.append((drive, state, field))
            sub_step = min(sub_step * 2, 1.0)
        else:
            sub_step *= 0.5
            if sub_step < 1e-9:
                break
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
