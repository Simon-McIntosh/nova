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
from nova.equilibrium.forward_operator import CellCurrentMoments
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


def test_certificate_host_loop_refuses_the_state_the_final_step_returns():
    """Only the step that terminates the loop produces the non-finite state
    the run would record as its terminal state."""

    program = jax.jit(non_finite_cell_mask)
    steps = {"count": 0}

    def step(point):
        steps["count"] += 1
        return point.at[PLANTED].set(jnp.nan), {
            "residual": 1e-13,
            "accepted": True,
        }

    with pytest.raises(NonFiniteCellMomentError, match=str(PLANTED)):
        certificate_solver.drive_linear_steps(
            step,
            jnp.zeros(CELLS),
            8,
            cell_moments=program,
        )
    assert steps["count"] == 1


def test_certificate_cell_moments_flags_a_planted_moment():
    reference = np.zeros((CELLS, 3))
    scale = np.ones(3)
    cell_moments = certificate_solver.certificate_cell_moments(reference, scale)
    state = jnp.zeros(CELLS * 3).at[PLANTED * 3 + 1].set(jnp.nan)
    assert np.flatnonzero(np.asarray(cell_moments(state))).tolist() == [PLANTED]


def test_certificate_host_loop_refuses_through_the_production_moments():
    reference = np.zeros((CELLS, 3))
    scale = jnp.asarray(np.ones(3))
    cell_moments = certificate_solver.certificate_cell_moments(reference, scale)

    def step(point):
        return point.at[PLANTED * 3 + 2].set(jnp.nan), {
            "residual": 1.0,
            "accepted": True,
        }

    with pytest.raises(NonFiniteCellMomentError, match=str(PLANTED)):
        certificate_solver.drive_linear_steps(
            step,
            jnp.zeros(CELLS * 3),
            4,
            cell_moments=cell_moments,
        )


def test_reduced_newton_host_loop_refuses_a_moment_the_production_mask_flags():
    program = jax.jit(non_finite_cell_mask)

    def moment_finiteness(reduced, shadow, base_state):
        current = jnp.zeros(CELLS).at[PLANTED].set(jnp.nan)
        return program(CellCurrentMoments(current, jnp.zeros(CELLS), jnp.zeros(CELLS)))

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
