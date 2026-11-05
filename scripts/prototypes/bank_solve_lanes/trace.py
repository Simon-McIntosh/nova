"""Trace every lane of the batched bank solve iterate by iterate.

The trace is a host-side transcription of ``support_read.solve_batch``'s
``advance`` body: it calls the same ``read_batch``, ``newton_step`` and
``step_residual`` the driver calls, and it carries the same frozen-decision
state machine. Every quantity the driver keeps only as a per-lane terminal
counter is recorded per iterate here. The transcription is only admissible if
it reproduces the driver, so the run asserts that each lane's traced step count
equals ``solve_batch``'s own and that its traced terminal fresh residual equals
the driver's within a relative bound.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_solve_read import measure


ROOT = Path(__file__).resolve().parents[3]


def emit(label, **values):
    print(
        label + " " + json.dumps(values, sort_keys=True, default=_default), flush=True
    )


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, jax.Array):
        return np.asarray(value).tolist()
    return float(value)


def magnitude(operands, state, reading, profile, quadrature):
    """The driver's own fresh-residual magnitude, one lane at a time."""

    def one(single, single_read):
        result = support_read.step_residual(
            operands, single, single, single_read, profile, quadrature
        )
        return jnp.max(jnp.abs(jnp.concatenate(result.residual, axis=-1)), axis=-1)

    return jax.vmap(one)(state, reading)


def selection_changed(reading, cut_set, normal_set, fragments, lanes):
    """The driver's discrete-decision comparison, verbatim."""
    cut = (reading.membership > 0) & (reading.membership < 1 - 1e-14)
    return (
        jnp.any(cut != cut_set, axis=-1)
        | jnp.any(reading.normal_form_cells != normal_set, axis=-1)
        | jnp.any((reading.fragment_selected != fragments).reshape(lanes, -1), axis=-1)
    )


def lane_slice(tree, lane):
    return jax.tree.map(lambda value: value[lane], tree)


def trace_lanes(
    operands,
    cache,
    moments,
    seeds,
    previous,
    convention,
    policy,
    profile,
    gradient,
    quadrature,
    *,
    max_steps=60,
    tolerance=1e-10,
    freeze_threshold=1e-6,
    freeze_active_set=True,
    iterations=12,
):
    """Record every iterate the driver's own state machine takes."""
    lanes, _cells = moments[0].shape
    support = support_read.read_batch(
        operands,
        cache,
        moments,
        seeds,
        previous,
        convention,
        policy,
        -1.0,
        iterations=iterations,
    )
    norm = magnitude(operands, moments, support, profile, quadrature)
    accepted = support.valid & jnp.isfinite(norm) & (norm <= tolerance)
    cut_set = jnp.zeros_like(support.reading.normal_form_cells)
    normal_set = jnp.zeros_like(support.reading.normal_form_cells)
    locked_fragments = support.reading.fragment_selected
    cut_now = (support.reading.membership > 0) & (
        support.reading.membership < 1 - 1e-14
    )
    normal_now = support.reading.normal_form_cells
    done = accepted
    frozen = jnp.zeros(lanes, bool)
    steps = jnp.zeros(lanes, jnp.int32)
    refactors = jnp.zeros(lanes, jnp.int32)
    freezes = jnp.zeros(lanes, jnp.int32)
    unfreezes = jnp.zeros(lanes, jnp.int32)
    history = [
        {
            "iterate": 0,
            "residual": [float(value) for value in norm],
            "update": [None] * lanes,
            "frozen": [False] * lanes,
            "start_freeze": [False] * lanes,
            "drift": [False] * lanes,
            "cut_changes": [0] * lanes,
            "normal_changes": [0] * lanes,
            "fragment_changes": [0] * lanes,
            "condition": [None] * lanes,
            "done": [bool(value) for value in done],
        }
    ]

    def one_fresh(items):
        one, one_read = items
        step = support_read.newton_step(
            operands,
            one,
            one_read,
            profile,
            gradient,
            quadrature,
            fresh_geometry=True,
            condition_diagnostics=False,
        )
        diagonal = jnp.abs(jnp.diagonal(step.factor.lu))
        condition = jnp.max(diagonal) / jnp.maximum(jnp.min(diagonal), 1e-300)
        update = jnp.max(jnp.abs(jnp.concatenate(step.update, axis=-1)))
        return step, condition, update

    for iterate in range(max_steps):
        if not bool(jnp.any(~done)):
            break
        active = ~done
        changed = frozen & selection_changed(
            support.reading, cut_set, normal_set, locked_fragments, lanes
        )
        start_freeze = (
            active & (~frozen) & (~changed) & support.valid & (norm < freeze_threshold)
        )
        if not freeze_active_set:
            start_freeze = jnp.zeros_like(start_freeze)
        lock = (frozen & ~changed) | start_freeze
        cut_set = jnp.where(start_freeze[:, None], cut_now, cut_set)
        normal_set = jnp.where(start_freeze[:, None], normal_now, normal_set)
        locked_fragments = jnp.where(
            start_freeze.reshape((lanes,) + (1,) * (locked_fragments.ndim - 1)),
            support.reading.fragment_selected,
            locked_fragments,
        )
        step_reading = support.reading._replace(
            normal_form_cells=jnp.where(lock[:, None], normal_set, normal_now),
            fragment_selected=jnp.where(
                lock.reshape(
                    (lanes,) + (1,) * (support.reading.fragment_selected.ndim - 1)
                ),
                locked_fragments,
                support.reading.fragment_selected,
            ),
        )
        step_support = support._replace(reading=step_reading)
        stepped, condition, update_norm = jax.lax.map(
            one_fresh, (moments, step_support)
        )
        candidate = tuple(
            jnp.where(active[:, None], value + change, value)
            for value, change in zip(moments, stepped.update, strict=True)
        )
        next_read = support_read.read_batch(
            operands,
            cache,
            candidate,
            seeds,
            previous,
            convention,
            policy,
            -1.0,
            iterations=iterations,
        )
        next_norm = magnitude(operands, candidate, next_read, profile, quadrature)
        drift = lock & selection_changed(
            next_read.reading, cut_set, normal_set, locked_fragments, lanes
        )
        next_done = done | (
            next_read.valid
            & jnp.isfinite(next_norm)
            & (next_norm <= tolerance)
            & ~drift
        )
        cut_after = (next_read.reading.membership > 0) & (
            next_read.reading.membership < 1 - 1e-14
        )
        history.append(
            {
                "iterate": iterate + 1,
                "residual": [float(value) for value in next_norm],
                "update": [
                    float(value) if bool(active[lane]) else None
                    for lane, value in enumerate(update_norm)
                ],
                "frozen": [bool(value) for value in lock],
                "start_freeze": [bool(value) for value in start_freeze],
                "drift": [bool(value) for value in drift],
                "cut_changes": [
                    int(value) for value in jnp.sum(cut_after != cut_set, axis=-1)
                ],
                "normal_changes": [
                    int(value)
                    for value in jnp.sum(
                        next_read.reading.normal_form_cells != normal_set, axis=-1
                    )
                ],
                "fragment_changes": [
                    int(value)
                    for value in jnp.sum(
                        (
                            next_read.reading.fragment_selected != locked_fragments
                        ).reshape(lanes, -1),
                        axis=-1,
                    )
                ],
                "condition": [
                    float(value) if bool(active[lane]) else None
                    for lane, value in enumerate(condition)
                ],
                "done": [bool(value) for value in next_done],
            }
        )
        cut_now = cut_after
        normal_now = next_read.reading.normal_form_cells
        moments = candidate
        support = next_read
        norm = next_norm
        done = next_done
        frozen = lock & ~drift
        steps = steps + active.astype(jnp.int32)
        refactors = refactors + active.astype(jnp.int32)
        freezes = freezes + start_freeze.astype(jnp.int32)
        unfreezes = unfreezes + (changed | drift).astype(jnp.int32)
    records = {
        lane: [
            {
                "iterate": row["iterate"],
                "residual": row["residual"][lane],
                "update": row["update"][lane],
                "frozen": row["frozen"][lane],
                "start_freeze": row["start_freeze"][lane],
                "drift": row["drift"][lane],
                "cut_changes": row["cut_changes"][lane],
                "normal_changes": row["normal_changes"][lane],
                "fragment_changes": row["fragment_changes"][lane],
                "condition": row["condition"][lane],
                "done": row["done"][lane],
            }
            for row in history
        ]
        for lane in range(lanes)
    }
    return {
        "steps": np.asarray(steps),
        "residual": np.asarray(norm),
        "converged": np.asarray(done),
        "refactors": np.asarray(refactors),
        "freezes": np.asarray(freezes),
        "unfreezes": np.asarray(unfreezes),
        "records": records,
        "moments": moments,
        "support": support,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--host-trace", action="store_true")
    args = parser.parse_args()
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        revision = "unversioned-archive"
    emit(
        "HEADER",
        arm=args.arm,
        revision=revision,
        tree=str(ROOT),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64),
    )
    assert jax.config.jax_enable_x64
    bank, cache, roots, convention, policy, profile, quadrature, moments, analytic = (
        measure.machine("diverted")
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    lanes = 16
    scales = jnp.linspace(2e-5, 6e-5, lanes)
    states = tuple(value[None] * (1 + scales[:, None]) for value in moments)
    seeds_batch = jnp.broadcast_to(seeds, (lanes, *seeds.shape))
    roots_batch = jnp.broadcast_to(roots, (lanes, *roots.shape))

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    @jax.jit
    def batch(state):
        return support_read.solve_batch(
            bank,
            cache,
            state,
            seeds_batch,
            roots_batch,
            convention,
            policy,
            profile,
            gradient,
            quadrature,
            max_steps=60,
            reuse_factor=False,
            contraction_limit=0.5,
        )

    batch(states)
    jax.block_until_ready(batch(states))
    first = batch(states)
    jax.block_until_ready(first)
    second = batch(states)
    jax.block_until_ready(second)
    first_steps = np.asarray(first.steps)
    second_steps = np.asarray(second.steps)
    first_residual = np.asarray(first.residual)
    second_residual = np.asarray(second.residual)
    emit(
        "REPEAT",
        steps_first=first_steps.tolist(),
        steps_second=second_steps.tolist(),
        repeatable=bool(np.array_equal(first_steps, second_steps)),
        residual_first=first_residual.tolist(),
        residual_second=second_residual.tolist(),
        residual_identical=bool(first_residual.tobytes() == second_residual.tobytes()),
        max_relative_gap=float(
            np.max(np.abs(first_residual - second_residual) / first_residual)
        ),
    )
    emitted = {
        "steps": first_steps.tolist(),
        "residual": first_residual.tolist(),
        "converged": np.asarray(first.converged).tolist(),
        "refactors": np.asarray(first.refactors).tolist(),
        "freezes": np.asarray(first.freezes).tolist(),
        "unfreezes": np.asarray(first.unfreezes).tolist(),
    }
    emit("SOLVE_BATCH", **emitted)
    residual_path = np.asarray(first.residual_path)
    update_path = np.asarray(first.update_path)
    emit(
        "PATHS",
        residual_path=residual_path.tolist(),
        update_path=update_path.tolist(),
        steps=first_steps.tolist(),
    )
    payload = {
        "arm": args.arm,
        "revision": revision,
        "lanes": lanes,
        "perturbation": [float(value) for value in scales],
        "solve_batch": emitted,
        "repeat": {
            "steps_first": first_steps.tolist(),
            "steps_second": second_steps.tolist(),
            "repeatable": bool(np.array_equal(first_steps, second_steps)),
            "residual_identical": bool(
                first_residual.tobytes() == second_residual.tobytes()
            ),
            "residual_max_relative_gap": float(
                np.max(np.abs(first_residual - second_residual) / first_residual)
            ),
        },
        "paths": {
            "residual": residual_path.tolist(),
            "update": update_path.tolist(),
        },
    }

    if not args.host_trace:
        Path(args.out).write_text(json.dumps(payload, indent=1))
        emit("OK", out=args.out)
        return
    trace = trace_lanes(
        bank,
        cache,
        states,
        seeds_batch,
        roots_batch,
        convention,
        policy,
        profile,
        gradient,
        quadrature,
    )
    steps = trace["steps"]
    residual = trace["residual"]
    emit(
        "TRACE_SUMMARY",
        steps=steps.tolist(),
        residual=residual.tolist(),
        converged=trace["converged"].tolist(),
        freezes=trace["freezes"].tolist(),
        unfreezes=trace["unfreezes"].tolist(),
        refactors=trace["refactors"].tolist(),
    )
    gaps = np.abs(residual - first_residual) / np.maximum(first_residual, 1e-300)
    emit(
        "AGREEMENT",
        steps_equal=bool(np.array_equal(steps, first_steps)),
        steps_offset=bool(np.array_equal(steps, first_steps + 1)),
        residual_max_relative_gap=float(np.max(gaps)),
        residual_equal=bool(residual.tobytes() == first_residual.tobytes()),
    )
    payload["trace"] = {
        "steps": steps.tolist(),
        "residual": residual.tolist(),
        "converged": trace["converged"].tolist(),
        "freezes": trace["freezes"].tolist(),
        "unfreezes": trace["unfreezes"].tolist(),
        "refactors": trace["refactors"].tolist(),
        "records": trace["records"],
    }
    payload["agreement"] = {
        "steps_equal": bool(np.array_equal(steps, first_steps)),
        "steps_offset": bool(np.array_equal(steps, first_steps + 1)),
        "residual_max_relative_gap": float(np.max(gaps)),
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    assert np.array_equal(steps, first_steps), (steps.tolist(), first_steps.tolist())
    assert np.max(gaps) <= 1e-9, float(np.max(gaps))
    emit("OK", out=args.out)


if __name__ == "__main__":
    main()
