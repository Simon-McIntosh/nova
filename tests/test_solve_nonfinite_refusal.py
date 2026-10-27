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
from nova.equilibrium import reduced_newton
from nova.equilibrium.reduced_newton import (
    NonFiniteCellMomentError,
    _drive_trips,
    non_finite_cell_mask,
    refuse_non_finite_cell_moments,
)
from tests.test_reduced_newton import machine  # noqa: F401

CELLS = 256
PLANTED = 103
PLANTED_NODE = 40


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


def _production_kernels(profile, seed):
    """Build the reduced kernels and the readings one trip opens with."""
    operator = profile.operator
    state = jnp.asarray(seed)
    coordinates = reduced_newton.reduced_coordinates(operator, state)
    kernels = reduced_newton._reduced_kernels(
        operator, coordinates, operator.external(), None, None
    )
    shadow = jnp.ravel(
        jnp.asarray(operator.residual_shadow_mask(state, None), dtype=bool)
    )

    def regather(flux):
        moments = reduced_newton._current_moments(operator, flux, None, None)
        return reduced_newton._gather(coordinates, moments)

    return kernels, state, shadow, regather


def test_the_production_moment_closure_flags_a_planted_moment(machine):  # noqa: F811
    """The closure the solve builds finds a non-finite moment in its own input.

    The NaN is planted in the flux state the closure reconstructs from and
    reads through ``_current_moments``, so the derivation of
    ``cell_current_moments`` is what has to notice it.  The same call on the
    unplanted state returns no flag, which is the control that the mask is
    reading this input rather than reporting a constant.
    """
    profile, seed = machine
    kernels, state, shadow, regather = _production_kernels(profile, seed)
    amplitudes = regather(state)

    clean = np.asarray(kernels["moment_finiteness"](amplitudes, shadow, state))
    assert not clean.any()

    planted = state.at[PLANTED_NODE].set(jnp.nan)
    flagged = np.asarray(
        kernels["moment_finiteness"](regather(planted), shadow, planted)
    )
    assert flagged.any()
    with pytest.raises(NonFiniteCellMomentError):
        refuse_non_finite_cell_moments(jnp.asarray(flagged))


def test_reduced_newton_refuses_a_moment_only_the_final_trip_produces(
    machine,  # noqa: F811
):
    """The driver reads the mask on the state its last trip returns.

    The trip opens on a finite state, so the head-of-trip check passes; the
    boundary step then returns a state carrying a non-finite moment, which no
    later trip would ever inspect.  The closure that finds it is the
    production one, over the real operator.
    """
    profile, seed = machine
    kernels, state, shadow, regather = _production_kernels(profile, seed)
    planted = state.at[PLANTED_NODE].set(jnp.nan)

    def boundary(reduced, trip_shadow, base_state):
        return planted, trip_shadow, 0, 0.5, 0.0

    with pytest.raises(NonFiniteCellMomentError):
        _drive_trips(
            kernels,
            state,
            None,
            shadow,
            tolerance=1e-12,
            newton_steps=0,
            active_set_steps=1,
            fused=False,
            scoring=reduced_newton.LADDER_SCORING,
            regather=regather,
            dispatched_boundary=boundary,
            stream=False,
        )


def test_the_refusal_names_every_flagged_cell():
    mask = jnp.zeros((CELLS,), dtype=bool).at[jnp.asarray([2, 9])].set(True)
    with pytest.raises(NonFiniteCellMomentError) as raised:
        refuse_non_finite_cell_moments(mask)
    assert raised.value.cells == (2, 9)
    assert "2, 9" in str(raised.value)
