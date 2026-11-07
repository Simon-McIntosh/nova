"""Ask whether equilibration explains the product path's Krylov stall.

The assembled path balances its matrix (moment scales, then three rounds of row
and column norms) before block GMRES; the product path carries the moment scales
alone, and it stalls at the 400-iteration allowance. This script separates the
two questions the stall leaves open. It takes the row and column scales and the
3x3 cell-block inverses from an assembled matrix and applies them around
``step_jacobian_vector``, running the same GMRES over the same product. If the
product then converges in about forty iterations, the equilibration is what was
missing; if it still stalls, the defect is in the preconditioner or the product.

Two arms, because a solve cannot assemble the matrix it wants to avoid:

``same``   scales and blocks taken from the assembled matrix at the state being
           solved, which is the best any equilibration could do
``stale``  scales and blocks taken from an assembled matrix at another state,
           which is what a solve could actually carry forward

Each case prints one ``RESULT`` JSON line. This script needs the revision that
carries ``support_read._block_gmres_operator``; against a tree without it the
product arm cannot run and the script says so.

usage: diagnose.py <case>   with case in {trial, converged, previous}
"""

import argparse
import json
import os

import jax
import jax.numpy as jnp

from nova.jax.config import configure_dtypes

configure_dtypes()

from nova.equilibrium import support_read  # noqa: E402
from scripts.prototypes.bank_solve_profile import fixture  # noqa: E402


def _header():
    import nova

    print(f"# nova={nova.__file__}", flush=True)
    print(f"# platforms={[str(device) for device in jax.devices()]}", flush=True)
    print(f"# revision={_revision()}", flush=True)


def _revision():
    root = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".git")
    try:
        with open(os.path.join(root, "HEAD"), encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return "unknown"


def _states():
    """The certificate pair the equality test uses: the saved seed and converged."""
    batch = fixture.build(1)

    @jax.jit
    def converge(bank, cache, states, seeds, previous):
        return support_read.solve_batch(
            bank,
            cache,
            states,
            seeds,
            previous,
            batch.convention,
            batch.policy,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
        )

    result = jax.block_until_ready(
        converge(
            batch.operands,
            batch.cache,
            batch.states,
            batch.seeds_batch,
            batch.previous_batch,
        )
    )
    trial = tuple(value[0] for value in batch.states)
    converged = tuple(value[0] for value in result.moments)
    return batch, {"trial": trial, "converged": converged}


def _read(batch, moments):
    return jax.jit(
        lambda bank, cache, values: support_read.read(
            bank,
            cache,
            values,
            batch.seeds,
            batch.roots,
            batch.convention,
            batch.policy,
        )
    )(batch.operands, batch.cache, moments)


def _system(batch, moments):
    """The support, the residual and the assembled matrix at one state."""
    support = _read(batch, moments)
    residual = support_read.step_residual(
        batch.operands,
        moments,
        moments,
        support,
        batch.profile,
        batch.quadrature,
    )
    matrix = support_read.step_jacobian(
        batch.operands,
        moments,
        support,
        batch.profile,
        fixture.gradient,
        batch.quadrature,
        residual.booked,
        fresh_geometry=True,
    )
    return support, residual, matrix


def _scales(moments, booked, matrix):
    balanced, row, column = support_read._equilibrate(matrix, moments, booked)
    inverse, indices = support_read._block_preconditioner(balanced)
    return row, column, inverse, indices


def _split(update, cells):
    return tuple(jnp.split(update, (cells, 2 * cells)))


def _dense_update(batch, moments, support, residual, row, column, inverse, indices):
    cells = moments[0].size
    right = -jnp.concatenate(tuple(jnp.ravel(value) for value in residual.residual))
    balanced = (
        support_read.step_jacobian(
            batch.operands,
            moments,
            support,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            residual.booked,
            fresh_geometry=True,
        )
        * column[None, :]
        / row[:, None]
    )
    solution, _, _, iterations = support_read._block_gmres(
        balanced, right / row, inverse, indices
    )
    return _split(column * solution, cells), int(iterations)


def _product_update(batch, moments, support, residual, row, column, inverse, indices):
    cells = moments[0].size
    right = -jnp.concatenate(tuple(jnp.ravel(value) for value in residual.residual))

    def operator(value):
        direction = tuple(
            (column * value)[block * cells : (block + 1) * cells] for block in range(3)
        )
        product = support_read.step_jacobian_vector(
            batch.operands,
            moments,
            direction,
            support,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            residual.booked,
            fresh_geometry=True,
        )
        return jnp.concatenate(tuple(product)) / row

    solution, condition, krylov_residual, iterations = (
        support_read._block_gmres_operator(operator, right / row, inverse, indices)
    )
    return (
        _split(column * solution, cells),
        int(iterations),
        float(condition),
        float(krylov_residual),
    )


def _error(update, reference):
    change = jnp.concatenate(tuple(update))
    target = jnp.concatenate(tuple(reference))
    return float(jnp.linalg.norm(change - target) / jnp.linalg.norm(target))


def run(case):
    if not hasattr(support_read, "_block_gmres_operator"):
        raise SystemExit(
            "this tree has no callable-operator GMRES; run the diagnostic against "
            "the revision that carries the product path"
        )
    batch, pair = _states()
    if case == "previous":
        # A genuine previous update: the seed stepped once is the state whose
        # matrix a solve at the new state could carry forward.
        support, residual, matrix = _system(batch, pair["trial"])
        row, column, inverse, indices = _scales(pair["trial"], residual.booked, matrix)
        first, _ = _dense_update(
            batch, pair["trial"], support, residual, row, column, inverse, indices
        )
        moments = tuple(value + change for value, change in zip(pair["trial"], first))
        previous = pair["trial"]
        source = "the seed state, one Newton update behind"
    else:
        moments = pair[case]
        other = "converged" if case == "trial" else "trial"
        previous, source = pair[other], f"the {other} certificate state"

    support, residual, matrix = _system(batch, moments)
    row, column, inverse, indices = _scales(moments, residual.booked, matrix)
    dense, dense_iterations = _dense_update(
        batch, moments, support, residual, row, column, inverse, indices
    )
    same_update, same_iterations, same_condition, same_residual = _product_update(
        batch, moments, support, residual, row, column, inverse, indices
    )
    stale_support, stale_residual, stale_matrix = _system(batch, previous)
    stale_row, stale_column, stale_inverse, stale_indices = _scales(
        previous, stale_residual.booked, stale_matrix
    )
    stale_update, stale_iterations, stale_condition, stale_residual_norm = (
        _product_update(
            batch,
            moments,
            support,
            residual,
            stale_row,
            stale_column,
            stale_inverse,
            stale_indices,
        )
    )
    return {
        "case": case,
        "cells": int(moments[0].size),
        "stale_source": source,
        "dense_iterations": dense_iterations,
        "same_iterations": same_iterations,
        "same_error": _error(same_update, dense),
        "same_condition": same_condition,
        "same_residual": same_residual,
        "stale_iterations": stale_iterations,
        "stale_error": _error(stale_update, dense),
        "stale_condition": stale_condition,
        "stale_residual": stale_residual_norm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["trial", "converged", "previous"])
    arguments = parser.parse_args()
    _header()
    print("RESULT " + json.dumps(run(arguments.case)), flush=True)


if __name__ == "__main__":
    main()
