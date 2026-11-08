"""Measure the bordered centroid-row spectrum along one pinned lane."""

# ruff: noqa: E402
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import support_read
from scripts.prototypes.bank_fold_vertical.measure import (
    centroid,
    pinned_measure,
    shifted,
)
from scripts.prototypes.bank_solve_profile import fixture as profile_fixture
from scripts.prototypes.bank_warm_start import streams


def condition(machine, state, seed, field, drive):
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
    balanced, row, column = support_read._equilibrate(matrix, state, residual.booked)
    op_plus, cache_plus = shifted(machine, field + 1e-6)
    support_plus = support_read.read(
        op_plus, cache_plus, state, seed, seed, machine.convention, machine.policy
    )
    residual_plus = support_read.step_residual(
        op_plus, state, state, support_plus, profile, machine.quadrature
    )
    def flat(value):
        return jnp.concatenate(value)
    field_column = (flat(residual_plus.residual) - flat(residual.residual)) / 1e-6
    n = state[0].size
    total = jnp.sum(state[0])
    height = centroid(state)
    height_row = jnp.concatenate(
        (jnp.full(n, -height / total), jnp.zeros(n), jnp.full(n, 1 / total))
    )
    bordered = jnp.block(
        [
            [balanced, (field_column * 1e-3 / row)[:, None]],
            [(height_row * column * 1e-3)[None, :], jnp.zeros((1, 1))],
        ]
    )
    original = jnp.linalg.svd(balanced, compute_uv=False)
    pinned = jnp.linalg.svd(bordered, compute_uv=False)
    return pinned[-1], original[-1], support.valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--barrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    machine = streams.build()
    with np.load(args.streams) as archive:
        state = tuple(jnp.asarray(archive[name][:1]) for name in ("m0", "m1", "m2"))
        seeds = jnp.asarray(archive["stationary"][:1])
    target = centroid(state)
    outcome, _, _, accepted = pinned_measure(
        machine,
        state,
        seeds,
        target,
        target_drive=1.00155,
        terminal_tolerance=1e-6,
        return_state=True,
    )
    chosen = [(1.0, state, jnp.zeros(1)), *accepted[:: max(1, len(accepted) // 3)]]
    if accepted and chosen[-1][0] != accepted[-1][0]:
        chosen.append(accepted[-1])
    measure = jax.jit(
        lambda one, seed, field, drive: condition(machine, one, seed, field, drive)
    )
    rows = []
    for drive, batch, fields in chosen:
        one = tuple(part[0] for part in batch)
        smallest, original, valid = measure(
            one, seeds[0], fields[0], jnp.asarray(drive)
        )
        row = dict(
            drive=float(drive),
            smallest=float(smallest),
            original=float(original),
            valid=bool(valid),
        )
        rows.append(row)
        print("SINGULAR " + json.dumps(row, sort_keys=True), flush=True)
    unpinned = []
    for line in args.barrier.read_text().splitlines():
        if line.startswith("CONDITION "):
            item = json.loads(line.removeprefix("CONDITION "))
            unpinned.append(
                dict(drive=item["drive"], smallest=item["balanced_smallest_singular"])
            )
    receipt = dict(pinned=rows, unpinned=unpinned, outcome=outcome)
    args.output.write_text(json.dumps(receipt, sort_keys=True))
    print(
        "SINGULAR_DONE "
        + json.dumps(
            dict(rows=len(rows), pinned_reached=outcome["drive"]), sort_keys=True
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
