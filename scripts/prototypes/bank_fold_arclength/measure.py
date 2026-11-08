"""Follow the bank equilibrium branch with a secant arclength corrector."""

# ruff: noqa: E402
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from nova.jax.config import configure_dtypes

configure_dtypes()
import argparse
import json
import subprocess
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from nova.equilibrium import support_read
from scripts.prototypes.bank_fold_vertical.measure import run_unpinned
from scripts.prototypes.bank_solve_profile import fixture
from scripts.prototypes.bank_warm_start import streams


def emit(label, **values):
    print(label + " " + json.dumps(values, sort_keys=True), flush=True)


def accepted(norm, valid, arc):
    return bool(
        np.all(np.asarray(valid))
        and np.max(np.asarray(norm)) <= 1e-10
        and np.max(np.abs(np.asarray(arc))) <= 1e-12
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    root = Path(__file__).resolve().parents[3]
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
        devices=[str(d) for d in jax.devices()],
        x64=True,
    )
    assert jax.default_backend() == "gpu"
    machine = streams.build()
    with np.load(args.streams) as source:
        initial = tuple(jnp.asarray(source[name][:16]) for name in ("m0", "m1", "m2"))
        seeds = jnp.asarray(source["stationary"][:16])
    replay, _ = run_unpinned(machine, initial, seeds)
    emit(
        "REPLAY",
        drive=replay.drive,
        certified=bool(replay.certified),
        reads=replay.reads,
        updates=replay.updates,
    )
    assert not replay.certified and abs(replay.drive - 1.0089341297149657) < 1e-9
    receipt = dict(revision=revision, replay_drive=replay.drive, rows=[], attempts=[])
    args.output.write_text(json.dumps(receipt))
    n = initial[0].shape[-1]
    # The fixed metric gives every moment block its seed maximum and RMS weight.
    scale = jnp.concatenate(
        [jnp.full(n, jnp.max(jnp.abs(a[0])) * jnp.sqrt(n)) for a in initial]
    )
    scale = jnp.maximum(scale, 1e-12)
    emit(
        "METRIC",
        block_scales=np.asarray(scale[::n]).tolist(),
        drive_scale=1.0,
        lanes=16,
    )

    def unpack(y):
        return tuple(jnp.split(y[:-1] * scale, (n, 2 * n)))

    def read_one(y, seed):
        state = unpack(y)
        support = support_read.read(
            machine.operands,
            machine.cache,
            state,
            seed,
            seed,
            machine.convention,
            machine.policy,
        )
        profile = streams.scaled_profile(machine.profile, y[-1])
        residual = support_read.step_residual(
            machine.operands, state, state, support, profile, machine.quadrature
        )
        return state, support, residual

    def residual_one(y, seed):
        _, support, result = read_one(y, seed)
        return jnp.concatenate(result.residual), support.valid & result.valid

    def system_one(y, seed):
        state, support, result = read_one(y, seed)
        profile = streams.scaled_profile(machine.profile, y[-1])
        matrix = support_read.step_jacobian(
            machine.operands,
            state,
            support,
            profile,
            fixture.gradient,
            machine.quadrature,
            booked=result.booked,
            fresh_geometry=True,
        )
        balanced, row, column = support_read._equilibrate(matrix, state, result.booked)

        def at_drive(drive):
            return jnp.concatenate(
                support_read.step_residual(
                    machine.operands,
                    state,
                    state,
                    support,
                    streams.scaled_profile(machine.profile, drive),
                    machine.quadrature,
                ).residual
            )

        drive_column = jax.jvp(at_drive, (y[-1],), (jnp.ones_like(y[-1]),))[1]
        return balanced, row, column, drive_column, jnp.concatenate(result.residual)

    @jax.jit
    def evaluate(y, prediction, tangent):
        residual, valid = jax.vmap(residual_one)(y, seeds)
        arc = jnp.sum((y - prediction) * tangent, axis=-1)
        return jnp.max(jnp.abs(residual), axis=-1), valid, arc

    @jax.jit
    def update(y, prediction, tangent):
        def one(value, seed, pred, tan):
            balanced, row, column, drive_column, residual = system_one(value, seed)
            border = jnp.block(
                [
                    [balanced, (drive_column / row)[:, None]],
                    [(tan[:-1] * column / scale)[None, :], tan[-1:].reshape(1, 1)],
                ]
            )
            rhs = jnp.concatenate((-residual / row, -jnp.dot(value - pred, tan)[None]))
            delta = jnp.linalg.solve(border, rhs)
            return jnp.concatenate((delta[:-1] * column / scale, delta[-1:]))

        return jax.vmap(one)(y, seeds, prediction, tangent)

    @jax.jit
    def spectrum(y, tangent):
        balanced, row, column, drive_column, _ = system_one(y, seeds[0])
        border = jnp.block(
            [
                [balanced, (drive_column / row)[:, None]],
                [(tangent[:-1] * column / scale)[None, :], tangent[-1:].reshape(1, 1)],
            ]
        )
        return jnp.linalg.svd(balanced, compute_uv=False)[-1], jnp.linalg.svd(
            border, compute_uv=False
        )[-1]

    def correct(prediction, tangent, limit=16):
        y = prediction
        for iteration in range(limit + 1):
            norms, valid, arc = evaluate(y, prediction, tangent)
            if accepted(norms, valid, arc):
                return y, iteration, float(jnp.max(norms)), True
            if iteration == limit:
                return y, iteration, float(jnp.max(norms)), False
            delta = update(y, prediction, tangent)
            old_score = np.maximum(
                np.asarray(norms) / 1e-10, np.abs(np.asarray(arc)) / 1e-12
            )
            best, best_score = y, old_score
            for divisor in (1, 2, 4, 8, 16, 32, 64, 128):
                candidate = y + delta / divisor
                cn, cv, ca = evaluate(candidate, prediction, tangent)
                score = np.maximum(
                    np.asarray(cn) / 1e-10, np.abs(np.asarray(ca)) / 1e-12
                )
                improve = np.asarray(cv) & np.isfinite(score) & (score < best_score)
                best = jnp.where(improve[:, None], candidate, best)
                best_score = np.where(improve, score, best_score)
                if np.all(improve) and divisor == 1:
                    break
            y = best
        raise AssertionError("unreachable corrector exit")

    flat = jnp.concatenate(initial, axis=-1) / scale
    y = jnp.concatenate((flat, jnp.ones((16, 1))), axis=-1)
    vertical = jnp.zeros_like(y).at[:, -1].set(1)
    y, iterations, norm, ok = correct(y, vertical)
    assert ok, f"seed refinement failed: {norm}"
    previous = y
    bootstrap = y.at[:, -1].set(1.0002)
    y, iterations, norm, ok = correct(bootstrap, vertical)
    assert ok, f"bootstrap refinement failed: {norm}"
    tangent = (y - previous) / jnp.linalg.norm(y - previous, axis=-1, keepdims=True)
    step = float(jnp.min(jnp.linalg.norm(y - previous, axis=-1)))
    length = step
    states = []

    def record(value, tan, residual, iterations):
        small, augmented = spectrum(value[0], tan[0])
        state = unpack(value[0])
        centres = machine.operands.geometry.centre
        total = jnp.sum(state[0])
        radius = jnp.sum(centres[:, 0] * state[0] + state[1]) / total
        height = jnp.sum(centres[:, 1] * state[0] + state[2]) / total
        row = dict(
            arclength=length,
            drive=float(value[0, -1]),
            tangent_drive=float(tan[0, -1]),
            singular=float(small),
            augmented=float(augmented),
            radius=float(radius),
            height=float(height),
            residual=residual,
            updates=iterations,
            drive_spread=float(jnp.ptp(value[:, -1])),
        )
        assert np.isfinite(row["singular"]) and row["augmented"] > 0
        receipt["rows"].append(row)
        states.append(np.asarray(value))
        args.output.write_text(json.dumps(receipt))
        np.savez(
            args.output.with_suffix(".npz"),
            states=np.asarray(states),
            scale=np.asarray(scale),
            seeds=np.asarray(seeds),
        )
        emit("POINT", **row)

    record(y, tangent, norm, iterations)
    negative_points = 0
    for attempt in range(300):
        prediction = y + step * tangent
        trial, iterations, norm, ok = correct(prediction, tangent)
        receipt["attempts"].append(
            dict(step=step, accepted=ok, residual=norm, updates=iterations)
        )
        emit("CORRECTOR", attempt=attempt, **receipt["attempts"][-1])
        if not ok:
            step *= 0.5
            args.output.write_text(json.dumps(receipt))
            if step < 1e-8:
                break
            continue
        secant = trial - y
        new_tangent = secant / jnp.linalg.norm(secant, axis=-1, keepdims=True)
        if bool(jnp.any(jnp.sum(new_tangent * tangent, axis=-1) <= 0)):
            raise AssertionError("branch orientation reversed")
        length += float(jnp.linalg.norm(secant[0]))
        previous, y, tangent = y, trial, new_tangent
        record(y, tangent, norm, iterations)
        if float(y[0, -1]) >= 1.01 and float(tangent[0, -1]) > 0:
            receipt["verdict"] = "passes 1.01"
            break
        if float(tangent[0, -1]) < 0:
            negative_points += 1
        if negative_points >= 6:
            rows = receipt["rows"]
            peak = max(rows, key=lambda r: r["drive"])
            receipt["verdict"] = f"turning-point at {peak['drive']:.16g}"
            receipt["turning_bracket"] = [
                rows[max(0, rows.index(peak) - 1)]["drive"],
                rows[min(len(rows) - 1, rows.index(peak) + 1)]["drive"],
            ]
            break
        step = min(step * (1.3 if iterations <= 5 else 0.8), 0.01)
    args.output.write_text(json.dumps(receipt))
    if "verdict" not in receipt:
        raise AssertionError("continuation exhausted without a classification")
    print("FOLD_VERDICT " + receipt["verdict"], flush=True)


if __name__ == "__main__":
    main()
