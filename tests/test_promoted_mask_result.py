"""The reduced solve's terminal promoted mask on its result and its receipt.

A promoted node is one the closing trip of a reduced solve kept at the trial
state instead of the map's image, so the flux there is a piece of the iterate
that no field of the cell currents reproduces.  That set is computed inside
the trip and was reported nowhere, which left a producer holding a terminal
state with no record of which nodes carry that piece.  These tests pin it on
the result of both solve routes and on the profile receipt, and pin the
all-false mask a solve that promotes nothing returns.

The machine is the bootstrapped Solov'ev free-boundary problem
``tests/test_reduced_newton.py`` builds, so the masks are read off a real
free-boundary map rather than an algebraic identity.
"""

from __future__ import annotations

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.equilibrium import reduced_newton
    from tests.test_reduced_newton import (  # noqa: F401
        REDUCED_NEWTON_STEPS,
        SOLVE_TOLERANCE,
        machine,
    )


#: Trip budget the mask-equality solves run under.  A single trip fixes the
#: previous shadow the closing read sees at the seed's own mask, which is a
#: value the test computes for itself, so the expected set is reconstructed
#: from the terminal state alone rather than read back off the result.
SINGLE_TRIP = 1
#: Trip budget the full solves run under, matching the file the machine
#: fixture lives in.
ACTIVE_SET_STEPS = 12


def _seed_shadow(operator, seed):
    """Return the residual shadow the solve reads at its own seed."""
    return jnp.ravel(jnp.asarray(operator.residual_shadow_mask(seed), dtype=bool))


def _promoted_set(operator, state, previous_shadow):
    """Return the promoted set a closing trip reads at one terminal state."""
    return jnp.ravel(
        jnp.asarray(
            operator.residual_shadow_mask(state, previous_shadow=previous_shadow),
            dtype=bool,
        )
    )


def _solve_arguments(**overrides):
    """Return the solve policy both routes are driven under."""
    arguments = {
        "tolerance": SOLVE_TOLERANCE,
        "newton_steps": REDUCED_NEWTON_STEPS,
    }
    arguments.update(overrides)
    return arguments


def test_host_route_reports_the_promoted_set_its_closing_trip_read(machine):
    """The terminal mask is the set the dispatched closing read produced.

    The result field is compared against the mask the operator's own read
    returned on the solve's final trip, recorded as the solve runs, so a
    result that carried the seed's shadow, a stale trip's shadow or an
    all-false array fails here.
    """
    profile, seed = machine
    operator = profile.operator
    reads = []
    original = operator.residual_shadow_mask

    def recording(psi, requested_class=None, previous_shadow=None):
        mask = original(psi, requested_class, previous_shadow)
        reads.append(np.asarray(mask, dtype=bool))
        return mask

    operator.residual_shadow_mask = recording
    try:
        result = reduced_newton.solve_reduced_newton(
            operator,
            seed,
            active_set_steps=ACTIVE_SET_STEPS,
            trip_boundary=reduced_newton.DISPATCHED_TRIP_BOUNDARY,
            **_solve_arguments(),
        )
    finally:
        del operator.residual_shadow_mask

    assert result.promoted_mask is not None
    returned = np.asarray(result.promoted_mask)
    assert returned.dtype == bool
    assert returned.shape == np.shape(result.state)
    assert len(reads) >= 2, reads
    assert np.array_equal(returned, reads[-1])
    assert returned.sum() > 0


def test_compiled_route_reports_the_promoted_set_of_its_terminal_state(machine):
    """One compiled trip returns exactly the set its own closing read forms."""
    profile, seed = machine
    operator = profile.operator
    previous_shadow = _seed_shadow(operator, seed)
    result = reduced_newton.solve_reduced_newton_compiled(
        operator,
        seed,
        active_set_steps=SINGLE_TRIP,
        **_solve_arguments(),
    )
    expected = _promoted_set(operator, result.state, previous_shadow)
    returned = np.asarray(result.promoted_mask)
    assert returned.dtype == bool
    assert np.array_equal(returned, np.asarray(expected, dtype=bool))
    assert returned.sum() > 0


def test_both_routes_report_the_same_promoted_set_on_one_trip(machine):
    """The host and compiled trips compute one set for one fixture."""
    profile, seed = machine
    operator = profile.operator
    host = reduced_newton.solve_reduced_newton(
        operator,
        seed,
        active_set_steps=SINGLE_TRIP,
        trip_boundary=reduced_newton.DISPATCHED_TRIP_BOUNDARY,
        **_solve_arguments(),
    )
    compiled = reduced_newton.solve_reduced_newton_compiled(
        operator, seed, active_set_steps=SINGLE_TRIP, **_solve_arguments()
    )
    assert np.array_equal(
        np.asarray(host.promoted_mask), np.asarray(compiled.promoted_mask)
    )


def test_a_solve_that_promotes_nothing_reports_an_all_false_mask(machine):
    """A map that promotes no node returns an all-false mask, not a full one."""
    profile, seed = machine
    operator = profile.operator
    shape = np.shape(np.ravel(np.asarray(operator.residual_shadow_mask(seed))))

    operator.residual_shadow_mask = lambda *arguments, **keywords: jnp.zeros(
        shape, dtype=bool
    )
    try:
        host = reduced_newton.solve_reduced_newton(
            operator,
            seed,
            active_set_steps=SINGLE_TRIP,
            trip_boundary=reduced_newton.DISPATCHED_TRIP_BOUNDARY,
            **_solve_arguments(),
        )
        compiled = reduced_newton.solve_reduced_newton_compiled(
            operator, seed, active_set_steps=SINGLE_TRIP, **_solve_arguments()
        )
    finally:
        del operator.residual_shadow_mask

    for result in (host, compiled):
        returned = np.asarray(result.promoted_mask)
        assert returned.shape == shape
        assert not returned.any()


def test_the_profile_solve_carries_the_promoted_mask(machine):
    """The public reduced route puts the mask on the forward equilibrium."""
    profile, seed = machine
    equilibrium = profile.solve(
        seed,
        route="reduced_newton",
        active_set_steps=SINGLE_TRIP,
        **_solve_arguments(),
    )
    returned = np.asarray(equilibrium.promoted_mask)
    expected = _promoted_set(
        profile.operator, equilibrium.flux, _seed_shadow(profile.operator, seed)
    )
    assert np.array_equal(returned, np.asarray(expected, dtype=bool))
    assert returned.sum() > 0
