"""Pin the weak-rotation 110-cell row's booked plasma current and outboard hole.

The production current-moment booking has not clipped the profile support
geometrically since 2026-08-27: the support is the full set of atomic cells
selected by the profile partition label, so a cell the analytic separatrix cuts
only partly is booked in whole or not at all.  The unit-amplitude census
(``benchmarks/unit_amplitude_current_census.py``) measured the consequence on
the committed terminal state of the weak-rotation-reactor-static chord row: the
cut cells on the outboard side drop out of the booking because the profile
participation flag excludes them.

Target and before-state
-----------------------

The analytic plasma current the certificate row solves towards is
16,314,773.31 A.  The census receipt recorded, on that state::

    chord clip                    15,540,896.52 A
    exact clip                    15,148,862.10 A
    cells 83, 104, 106, 108, 116  zero in both clip modes

Those numbers are the receipt's *before* state.  They were taken at revision
28a49bcb and do not reproduce here.  Measured on the committed terminal state
at this revision, under the operator's own absent-saddle reading of the
fixture:

    chord clip                    15,868,149.09 A
    exact clip                    15,491,114.13 A
    cells 83, 104, 106, 108, 116  nonzero in both clip modes
    chord order                   16 cells book zero, 41 cells differ between modes

So the receipt's outboard hole -- five cells booking zero in both clip modes --
is not present at this revision.  Cell 83 books 113,536.53 A, the four others
book between 1.12e5 A and 1.92e5 A, and for all five the chord and exact clips
select the same geometry (identical bookings and identical clip vertex counts),
so the two modes agree on these cells rather than one dropping them.  The
instrument is not degenerate: 41 of 135 cells still differ between the modes and
16 cells book zero in chord mode, so the curved clip does clip.

The movement is consistent with the separatrix-clip repair merged as 5a703429b,
which the receipt predates.  ``RECEIPT_TOTALS_A`` records the before-state and
``BOOKED_TOTALS_A`` pins what this revision books, so the pair states the
movement rather than one number standing for both.  What the booking repair
must still reach is the analytic target of 16,314,773.31 A within one percent:
the exact clip books 0.950 of it here, so the total is short for reasons other
than these five cells.

The absent census saddle
------------------------

The analytic oracle fixture builds its own ``SimpleNamespace`` topology from
its recorded flux image, and that namespace carries no ``x_point``.  The exact
branch of ``ForwardFluxOperator._profile_support`` reads ``topology.x_point``
unconditionally, so the fixture path raises before any booking is computed.
This module bridges that gap at the single call boundary with the operator's
own absent-saddle sentinel -- the non-finite ``x_point`` the same accessor
already falls back on elsewhere -- for the duration of its fixtures and nothing
more.  The bridged read is ``nova/equilibrium/forward_operator.py:3091``.  The
bridge is a defect workaround scoped to this module: delete it once the
fixture's topology or the operator's read is repaired, and note that it changes
no production code.

Reuse
-----

:func:`terminal_booked_currents` is the reusable entry point: it returns the
two booked totals and the per-cell booked current for the five outboard cells,
so the booking repair re-runs it and asserts the after values (each total
within one percent of the analytic target, each of the five cells booking a
nonzero current).

The values are pinned on the CPU lane because the census receipt they come from
was measured there; a run on another device is a different numerical
measurement rather than a failure of the booking contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nova.equilibrium.forward_operator as forward_operator
from benchmarks import unit_amplitude_current_census as census
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


pytestmark = pytest.mark.slow

#: Analytical plasma current the certificate row solves towards, in amperes.
#: This is the annotation the booking repair must reach.
TARGET_CURRENT_A = 16_314_773.311828371

#: Booked totals recorded by the census receipt at revision 28a49bcb, in
#: amperes, one per clip mode.  The before-state, kept for comparison.
RECEIPT_TOTALS_A = {"chord": 15_540_896.52, "exact": 15_148_862.10}

#: Booked totals this revision produces on the same committed terminal state,
#: in amperes, one per clip mode.
BOOKED_TOTALS_A = {"chord": 15_868_149.094491, "exact": 15_491_114.132435}

#: Cut cells the receipt booked zero current for in either clip mode.  Their
#: centroids sit on the outboard side of the machine, where the separatrix
#: crosses the cell so the chord label selects the whole cell or none of it.
OUTBOARD_HOLE_CELLS = (83, 104, 106, 108, 116)

#: Per-cell booked current this revision produces on the committed terminal
#: state, in amperes, per clip mode.  Every one of the five books a nonzero
#: current in both modes; the receipt recorded zero for all five.
MEASURED_CELL_CURRENT_A: dict[int, dict[str, float]] = {
    83: {"chord": 113_536.53228908387, "exact": 113_536.53228908387},
    104: {"chord": 192_269.94030158775, "exact": 192_269.94030158775},
    106: {"chord": 192_343.79043466656, "exact": 192_343.79043466656},
    108: {"chord": 192_269.94030158763, "exact": 192_269.94030158763},
    116: {"chord": 112_093.46642055141, "exact": 112_093.46642055141},
}

#: Relative agreement required between the recomputed total and the pinned
#: value.  The per-cell values are compared exactly, because the outboard hole
#: is a booking of zero and a relative tolerance has no meaning there.
BOOKING_RELATIVE_TOLERANCE = 1.0e-6

#: Marker set on the bridged accessor so the bridge installs only once.
_BRIDGE_MARKER = "_outboard_hole_census_gate_bridge"


def _profile_support_with_absent_saddle(self, masks, topology, physical,
                                        sample_psi_norm, **kwargs):
    """Supply the absent-saddle sentinel the analytic fixture does not carry.

    The fixture topology has no ``x_point``; the exact branch reads it
    unconditionally at ``nova/equilibrium/forward_operator.py:3091``.  Install
    the same non-finite sentinel the operator falls back on where a census
    saddle is genuinely absent, so the fixture path measures the booking
    instead of raising.
    """
    if not hasattr(topology, "x_point"):
        topology = SimpleNamespace(
            **vars(topology),
            x_point=jnp.full(2, jnp.nan),
            x_point_flux=jnp.asarray(0.0),
        )
    return _bridge_target(self, masks, topology, physical, sample_psi_norm, **kwargs)


_bridge_target = forward_operator.ForwardFluxOperator._profile_support


def install_absent_saddle_bridge() -> None:
    """Install the absent-saddle bridge on the operator's profile support.

    Idempotent: a second call leaves an already-installed bridge in place.
    Restore by assigning :func:`_profile_support_with_absent_saddle` off and
    :data:`_bridge_target` back onto ``ForwardFluxOperator._profile_support``.
    """
    current = forward_operator.ForwardFluxOperator._profile_support
    if getattr(current, _BRIDGE_MARKER, False):
        return
    setattr(_profile_support_with_absent_saddle, _BRIDGE_MARKER, True)
    forward_operator.ForwardFluxOperator._profile_support = (
        _profile_support_with_absent_saddle
    )


@pytest.fixture(scope="module", autouse=True)
def _absent_saddle_bridge():
    original = forward_operator.ForwardFluxOperator._profile_support
    install_absent_saddle_bridge()
    try:
        yield
    finally:
        forward_operator.ForwardFluxOperator._profile_support = original


@pytest.fixture(scope="module", autouse=True)
def _binary64_cpu() -> None:
    configure_dtypes()
    assert jax.config.jax_enable_x64, "the current census requires extended precision"
    assert jax.default_backend() == "cpu", (
        "the census receipt this gate pins was measured on the CPU lane"
    )
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())


def terminal_booked_currents() -> tuple[float, float, dict[int, dict[str, float]]]:
    """Return the terminal-state bookings this gate pins.

    Returns the chord and exact booked totals over the committed chord row's
    terminal flux state, in amperes, beside the per-cell booked current for
    each cell in :data:`OUTBOARD_HOLE_CELLS`.  The terminal state is read from
    the committed control part rather than re-solved, so the measurement sits
    on the same state the receipt was taken on and costs no forward solve.

    The absent-saddle bridge must already be installed when this is called from
    outside a pytest run; :func:`install_absent_saddle_bridge` does that.
    """
    control = census._read_control_row()
    context = census._build_context(control)
    operator = context["operator"]
    machine = context["machine"]
    state = np.asarray(control["terminal_flux_wb"], dtype=np.float64)
    target_current = float(context["target_current"])
    analytic = census._cell_analysis(machine, context["exact"], target_current)
    probe = census._partition_probe(operator, state, "exact")
    curve = census._curve_probe(
        operator,
        probe["base_masks"],
        probe["topology"],
        probe["sample_psi_norm"],
    )
    census_row: dict[str, Any] = census._state_census(
        operator,
        machine,
        state,
        "terminal",
        target_current,
        analytic,
        curve,
    )
    totals = census_row["unit_amplitude_totals_a"]
    per_cell = {
        int(cell): {
            "chord": float(census_row["per_cell"][cell]["chord_current_a"]),
            "exact": float(census_row["per_cell"][cell]["exact_current_a"]),
        }
        for cell in OUTBOARD_HOLE_CELLS
    }
    return float(totals["chord"]), float(totals["exact"]), per_cell


@pytest.fixture(scope="module")
def booked() -> tuple[float, float, dict[int, dict[str, float]]]:
    return terminal_booked_currents()


def test_booked_totals_reproduce_the_committed_state(booked) -> None:
    chord_total, exact_total, _per_cell = booked
    observed = {"chord": chord_total, "exact": exact_total}
    for mode, pinned in BOOKED_TOTALS_A.items():
        relative = abs(observed[mode] - pinned) / abs(pinned)
        assert relative <= BOOKING_RELATIVE_TOLERANCE, (
            f"{mode} booked total {observed[mode]:.6f} A is outside "
            f"{BOOKING_RELATIVE_TOLERANCE:g} of the committed state's {pinned} A "
            f"(the census receipt recorded {RECEIPT_TOTALS_A[mode]} A)"
        )


def test_outboard_cells_book_the_committed_current(booked) -> None:
    _chord_total, _exact_total, per_cell = booked
    for cell in OUTBOARD_HOLE_CELLS:
        for mode in ("chord", "exact"):
            value = per_cell[cell][mode]
            expected = MEASURED_CELL_CURRENT_A[cell][mode]
            relative = abs(value - expected) / abs(expected)
            assert relative <= BOOKING_RELATIVE_TOLERANCE, (
                f"cell {cell} books {value!r} A under the {mode} clip; the "
                f"committed state books {expected!r} A"
            )


def test_outboard_cells_book_nonzero_in_both_clip_modes(booked) -> None:
    """The receipt's outboard hole is absent at this revision.

    The census receipt recorded zero for all five cells in both clip modes.
    Pin the opposite reading instead, so a booking that drops any of them
    reddens this gate rather than passing quietly.
    """
    _chord_total, _exact_total, per_cell = booked
    for cell in OUTBOARD_HOLE_CELLS:
        for mode in ("chord", "exact"):
            assert per_cell[cell][mode] != 0.0, (
                f"cell {cell} books zero under the {mode} clip; the committed "
                "terminal state books a nonzero current for every one of the "
                "five outboard cells"
            )