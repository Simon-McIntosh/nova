"""Observe production search decisions and compare with the unobserved solve."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from pytest import MonkeyPatch

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_solve_cycle.observe import observe_updates
from scripts.prototypes.bank_solve_profile import fixture


def masks(reading):
    return (
        (reading.membership > 0) & (reading.membership < 1 - 1e-14),
        reading.normal_form_cells,
        reading.fragment_selected,
    )


def signatures(values):
    return [hashlib.sha256(np.packbits(lane).tobytes()).hexdigest() for lane in values]


def install(patch, events):
    original_loop = jax.lax.while_loop
    original_search = support_read._damped_scale

    def record(kind, names, *values):
        events.append(
            dict(
                kind=kind,
                **{
                    key: np.asarray(value).tolist()
                    for key, value in zip(names, values, strict=True)
                },
            )
        )

    def decisions(kind, *values):
        row = {"kind": kind}
        for key, value in zip(("cut", "normal", "fragments"), values, strict=True):
            array = np.asarray(value).reshape(16, -1)
            row[key] = signatures(array)
            row[key + "_count"] = np.count_nonzero(array, axis=1).tolist()
            altered = array.copy()
            altered[:, 0] = ~altered[:, 0]
            assert all(a != b for a, b in zip(signatures(array), signatures(altered)))
        events.append(row)

    def callback(kind, names, *values):
        jax.debug.callback(lambda *v: record(kind, names, *v), *values, ordered=True)

    def loop(condition, body, initial):
        name = body.__qualname__
        if name.endswith(
            "observe_updates.<locals>.observed_loop.<locals>.observed_advance"
        ):

            def advance(carried):
                callback(
                    "begin",
                    ("steps", "residual", "done", "frozen"),
                    carried[11],
                    carried[2],
                    carried[3],
                    carried[4],
                )
                jax.debug.callback(
                    lambda *v: decisions("before", *v),
                    *masks(carried[1].reading),
                    ordered=True,
                )
                following = body(carried)
                callback(
                    "end",
                    ("steps", "residual", "done", "refactors", "reads", "releases"),
                    following[11],
                    following[2],
                    following[3],
                    following[12] - carried[12],
                    following[13],
                    following[15] - carried[15],
                )
                jax.debug.callback(
                    lambda *v: decisions("after", *v),
                    *masks(following[1].reading),
                    ordered=True,
                )
                return following

            return original_loop(condition, advance, initial)
        if name == "_damped_scale.<locals>.body":

            def trial(carried):
                following = body(carried)
                callback(
                    "trial_rule",
                    ("scale", "pending", "pending_after", "best_scale", "best_norm"),
                    carried[4],
                    carried[6],
                    following[6],
                    following[3],
                    following[2],
                )
                return following

            return original_loop(condition, trial, initial)
        return original_loop(condition, body, initial)

    def search(evaluate, state, step, current, auxiliary, **kwargs):
        def evaluated(trial):
            norm, valid, read = evaluate(trial)
            callback("trial_norm", ("norm", "valid", "incumbent"), norm, valid, current)
            jax.debug.callback(
                lambda *v: decisions("trial_decisions", *v),
                *masks(read.reading),
                ordered=True,
            )
            return norm, valid, read

        result = original_search(evaluated, state, step, current, auxiliary, **kwargs)
        callback(
            "search",
            ("scale", "residual", "evaluations"),
            result.scale,
            result.residual,
            result.evaluations,
        )
        return result

    patch.setattr(jax.lax, "while_loop", loop)
    patch.setattr(support_read, "_damped_scale", search)
    observe_updates(patch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(
        "HEADER",
        json.dumps(
            dict(
                revision=args.revision,
                nova=nova.__file__,
                module=support_read.__file__,
                cwd=str(Path.cwd().resolve()),
                device=str(jax.devices()[0]),
                x64=jax.config.jax_enable_x64,
            )
        ),
        flush=True,
    )
    assert jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
    batch = fixture.build(16)
    inputs = (
        batch.operands,
        batch.cache,
        batch.states,
        batch.seeds_batch,
        batch.previous_batch,
    )

    def solve(bank, cache, state, seeds, previous):
        return support_read.solve_batch(
            bank,
            cache,
            state,
            seeds,
            previous,
            batch.convention,
            batch.policy,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            max_steps=60,
            reuse_factor=False,
            contraction_limit=0.5,
        )

    program = jax.jit(solve).lower(*inputs).compile()
    jax.block_until_ready(program(*inputs))
    walls = []
    for _ in range(3):
        started = perf_counter()
        plain = jax.block_until_ready(program(*inputs))
        walls.append(1000 * (perf_counter() - started))
    fields = ("steps", "reads", "refactors", "converged", "residual", "residual_path")
    receipt = dict(
        revision=args.revision,
        walls_ms=walls,
        initial_hash=hashlib.sha256(
            b"".join(np.asarray(x).tobytes() for x in batch.states)
        ).hexdigest(),
        plain={name: np.asarray(getattr(plain, name)).tolist() for name in fields},
    )
    args.output.write_text(json.dumps(receipt))
    events = []
    with MonkeyPatch.context() as patch:
        install(patch, events)

        def observed(bank, cache, state, seeds, previous):
            return solve(bank, cache, state, seeds, previous)

        traced = jax.jit(observed).lower(*inputs).compile()
        result = jax.block_until_ready(traced(*inputs))
        jax.effects_barrier()
    receipt.update(
        events=events,
        observed={name: np.asarray(getattr(result, name)).tolist() for name in fields},
        same_reads=bool(np.array_equal(result.reads, plain.reads)),
        same_path=bool(
            np.array_equal(result.residual_path, plain.residual_path, equal_nan=True)
        ),
        largest_moment_gap=max(
            float(np.max(np.abs(a - b))) for a, b in zip(result.moments, plain.moments)
        ),
    )
    args.output.write_text(json.dumps(receipt))
    print(
        "SUMMARY",
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "revision",
                    "walls_ms",
                    "same_reads",
                    "same_path",
                    "largest_moment_gap",
                )
            }
        ),
        "READS",
        np.asarray(plain.reads),
        "OBSERVED_READS",
        np.asarray(result.reads),
        flush=True,
    )
    assert np.all(result.converged) and np.all(plain.converged)
    assert sum(event["kind"] == "begin" for event in events) == max(result.steps)
    assert any(event["kind"] == "trial_norm" for event in events)


if __name__ == "__main__":
    main()
