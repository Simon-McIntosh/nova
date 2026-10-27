"""The solve drivers refuse a non-finite cell moment by name at their host loops.

Every production solve route is traced, so a raise written inside the mapped
step never runs.  The traced step returns the per-cell mask of non-finite
moments as an output instead, and the host loop that drives the step refuses,
naming the cells.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import scripts.prototypes.fast_map_production.solver as certificate_solver
from nova.equilibrium.reduced_newton import (
    NonFiniteCellMomentError,
    _drive_trips,
    non_finite_cell_mask,
    refuse_non_finite_cell_moments,
)

CELLS = 256
PLANTED = 103


def planted_mask_program():
    """Compile a program flagging one cell of a per-cell moment array."""

    def moments(value):
        planted = jnp.nan * jnp.sum(jnp.asarray(value))
        return (
            jnp.zeros((CELLS, 2), dtype=jnp.asarray(value).dtype)
            .at[PLANTED, 0]
            .set(planted)
        )

    return jax.jit(lambda value: non_finite_cell_mask(moments(value)))


def test_non_finite_cell_mask_names_only_the_planted_cell():
    mask = planted_mask_program()(jnp.asarray([1.0, 2.0]))
    assert np.flatnonzero(np.asarray(mask)).tolist() == [PLANTED]


def test_non_finite_cell_mask_reads_every_moment_leaf():
    current = jnp.ones((CELLS,))
    moment = jnp.zeros((CELLS, 3)).at[7, 2].set(jnp.inf)
    mask = jax.jit(non_finite_cell_mask)((current, moment))
    assert np.flatnonzero(np.asarray(mask)).tolist() == [7]


def test_certificate_solver_host_loop_refuses_the_planted_cell():
    def step(point):
        return point, {"residual": 1.0, "accepted": True}

    with pytest.raises(NonFiniteCellMomentError, match=str(PLANTED)):
        certificate_solver.drive_linear_steps(
            step,
            jnp.asarray([1.0, 2.0]),
            4,
            cell_moments=planted_mask_program(),
        )


def test_reduced_newton_host_loop_refuses_the_planted_cell():
    mask_program = planted_mask_program()

    def moment_finiteness(reduced, shadow, base_state):
        return mask_program(reduced)

    with pytest.raises(NonFiniteCellMomentError, match=str(PLANTED)):
        _drive_trips(
            {"moment_finiteness": moment_finiteness},
            jnp.zeros((4,)),
            None,
            jnp.zeros((4,), dtype=bool),
            tolerance=1e-12,
            newton_steps=1,
            active_set_steps=1,
            fused=False,
            scoring="first",
            regather=lambda state: state,
            dispatched_boundary=None,
            stream=False,
        )


def test_the_refusal_names_every_flagged_cell():
    mask = jnp.zeros((CELLS,), dtype=bool).at[jnp.asarray([2, 9])].set(True)
    with pytest.raises(NonFiniteCellMomentError) as raised:
        refuse_non_finite_cell_moments(mask)
    assert raised.value.cells == (2, 9)
    assert "2, 9" in str(raised.value)
