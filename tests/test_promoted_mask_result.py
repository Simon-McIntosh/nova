"""The reduced solve's terminal promoted mask on its result and its receipt.

A promoted node is one the closing trip of a reduced solve kept at the trial
state instead of the map's image, so the flux there is a piece of the iterate
that no field of the cell currents reproduces.  That set is computed inside
the trip and was reported nowhere, which left a producer holding a terminal
state with no record of which nodes carry that piece.  These tests pin it on
the result of both solve routes and on the profile receipt, and pin the
all-false mask a solve that promotes nothing returns.

The fast lane reads the bootstrapped Solov'ev free-boundary problem
``tests/test_reduced_newton.py`` builds, which carries no private-flux region
and so promotes no node.  Those tests therefore pin what that fixture can show
— the returned mask's dtype, shape and equality with the solve's own terminal
promoted set, which is all-false there — and the requirement that the mask name
at least one promoted node lives in the slow test.  The promoting case is the
warm MAST coil-edit machine the coil-edit sweep prepares: its converged edit
state has a qualified saddle and a private-flux region, so its closing trip
promotes nodes.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from benchmarks import coil_edit_latency, mast_response_carrier_warm
    from nova.equilibrium import reduced_newton
    from nova.equilibrium.topology import TopologyClass
    from nova.jax.config import configure_dtypes
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
#: One-based coil edit whose converged state the non-convergence diagnosis
#: replayed; the sweep's fraction array is zero-based, so edit 12 is its
#: eleventh entry.
DIAGNOSED_EDIT = 12


def _seed_shadow(operator, seed, requested_class=None):
    """Return the residual shadow the solve reads at its own seed."""
    return jnp.ravel(
        jnp.asarray(operator.residual_shadow_mask(seed, requested_class), dtype=bool)
    )


def _promoted_set(operator, state, previous_shadow, requested_class=None):
    """Return the promoted set a closing trip reads at one terminal state."""
    return jnp.ravel(
        jnp.asarray(
            operator.residual_shadow_mask(
                state, requested_class, previous_shadow=previous_shadow
            ),
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
    result that carried the seed's shadow or an earlier trip's shadow fails
    here.  This fixture promotes no node, so the compared set is all-false and
    a mask that named nodes would fail too.
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
    assert returned.shape == np.shape(reads[-1])
    assert np.array_equal(returned, reads[-1])


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
    assert returned.shape == np.shape(result.state)
    assert np.array_equal(returned, np.asarray(expected, dtype=bool))


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


def test_a_solve_given_no_trip_reports_an_all_false_mask(machine):
    """With no trip to close, the result carries no set at all.

    The operator's read is stubbed to name one node, so the seed's shadow the
    solve is handed is not empty and an all-false result cannot be the seed's
    mask passed through: the two differ, and the zero-trip path is what makes
    the difference.  One trip on the same stub then returns that named node,
    which shows the field is not a constant.
    """
    profile, seed = machine
    operator = profile.operator
    shape = np.shape(np.ravel(np.asarray(operator.residual_shadow_mask(seed))))
    named = jnp.zeros(shape, dtype=bool).at[0].set(True)

    operator.residual_shadow_mask = lambda *arguments, **keywords: named
    try:
        untripped = reduced_newton.solve_reduced_newton(
            operator,
            seed,
            active_set_steps=0,
            trip_boundary=reduced_newton.DISPATCHED_TRIP_BOUNDARY,
            **_solve_arguments(),
        )
        tripped = reduced_newton.solve_reduced_newton(
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

    returned = np.asarray(untripped.promoted_mask)
    assert returned.shape == shape
    assert not returned.any()
    assert np.asarray(tripped.promoted_mask).sum() == 1
    assert np.array_equal(np.asarray(compiled.promoted_mask), np.asarray(named))


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
    assert returned.dtype == bool
    assert returned.shape == np.shape(expected)
    assert np.array_equal(returned, np.asarray(expected, dtype=bool))


@pytest.mark.slow
def test_a_warm_coil_edit_state_carries_its_promoted_nodes():
    """The converged warm coil-edit state promotes nodes and reports them.

    The machine, its seed and the edited current vector come from the
    coil-edit sweep's own preparation, so the state this asserts on is the one
    that diagnosed the non-convergence: a diverted MAST frame whose closing
    trip passes a set of nodes through instead of mapping them.  The solve is
    warm, seeded from the prepared frame and chained through the earlier edits
    of the sweep, which is the sequence that made that frame converge.  The
    returned mask is compared against the mask recorded at the solve's own
    closing read, so a result carrying the seed's shadow, an earlier edit's
    set or an all-false array fails here.  The same warm edit is then replayed
    through the compiled route, where the returned mask is compared against
    that route's own terminal set, so a compiled carry that returned the seed
    mask or an all-false array fails here too.
    """
    configure_dtypes()
    profile, prepared, _carrier = coil_edit_latency._prepare_case(
        mast_response_carrier_warm.DEFAULT_CARRIER,
        impose_vertical_centroid=False,
    )
    operator = profile.operator
    requested_class = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    base_current = np.asarray(prepared["prescribed_current"], dtype=np.float64)
    circuit_index = int(prepared["circuit_index"])
    target_current = float(prepared["target_current"])
    reads: list[np.ndarray] = []
    original = operator.residual_shadow_mask

    def recording(psi, requested_class=None, previous_shadow=None):
        mask = original(psi, requested_class, previous_shadow)
        reads.append(np.asarray(mask, dtype=bool))
        return mask

    state = prepared["initial"]
    warm_seed = state
    latest_current = None
    operator.residual_shadow_mask = recording
    try:
        for fraction in coil_edit_latency.EDIT_FRACTIONS[:DIAGNOSED_EDIT]:
            current = base_current.copy()
            current[circuit_index] *= 1.0 + fraction
            warm_seed = state
            latest_current = jnp.asarray(current)
            reads.clear()
            result = reduced_newton.solve_reduced_newton(
                operator,
                state,
                requested_class=requested_class,
                target_current=target_current,
                prescribed_current=latest_current,
                tolerance=coil_edit_latency.COMPILED_SLICE_TOLERANCE,
                newton_steps=coil_edit_latency.COMPILED_SLICE_NEWTON_STEPS,
                active_set_steps=coil_edit_latency.COMPILED_SLICE_ACTIVE_SET_STEPS,
                trip_boundary=reduced_newton.DISPATCHED_TRIP_BOUNDARY,
            )
            state = result.state
    finally:
        del operator.residual_shadow_mask

    returned = np.asarray(result.promoted_mask)
    assert returned.dtype == bool
    assert bool(result.converged), float(result.terminal_residual)
    assert returned.sum() > 0
    assert np.array_equal(returned, reads[-1])

    # The same warm edit through the compiled route: its own terminal set is
    # what a single compiled trip forms at the state it returns, read against
    # the shadow its seed carries.
    compiled = reduced_newton.solve_reduced_newton_compiled(
        operator,
        warm_seed,
        requested_class=requested_class,
        target_current=target_current,
        prescribed_current=latest_current,
        tolerance=coil_edit_latency.COMPILED_SLICE_TOLERANCE,
        newton_steps=coil_edit_latency.COMPILED_SLICE_NEWTON_STEPS,
        active_set_steps=SINGLE_TRIP,
    )
    compiled_returned = np.asarray(compiled.promoted_mask)
    compiled_expected = _promoted_set(
        operator,
        compiled.state,
        _seed_shadow(operator, warm_seed),
        requested_class,
    )
    assert compiled_returned.dtype == bool
    assert compiled_returned.shape == np.shape(compiled_expected)
    assert compiled_returned.sum() > 0
    assert np.array_equal(compiled_returned, np.asarray(compiled_expected, dtype=bool))
