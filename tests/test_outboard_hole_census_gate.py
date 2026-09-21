"""Pin the weak-rotation 110-cell row's booked plasma current and outboard hole.

The production current-moment booking has not clipped the profile support
geometrically since 2026-08-27: the support is the full set of atomic cells
selected by the profile partition label, so a cell the analytic separatrix cuts
only partly is booked in whole or not at all.  The unit-amplitude census
(``benchmarks/unit_amplitude_current_census.py``) measured the consequence on
the committed terminal state of the weak-rotation-reactor-static chord row:
against an analytic target of 16,314,773.31 A the chord clip books
15,540,896.52 A and the curved (exact) clip books 15,148,862.10 A, and both
clip modes book exactly zero current for five cut cells on the outboard side
because the profile participation flag excludes them.

This module recomputes those bookings through the census helpers and pins the
values the census receipt records, so the booking defect and its repair are
both measurable on a committed state.  :func:`terminal_booked_currents` is the
reusable entry point: it returns the two booked totals and the per-cell booked
current for the five outboard cells, so the booking repair re-runs it and
asserts the after values (each total within one percent of the analytic
target, each of the five cells booking a nonzero current).

The values are pinned on the CPU lane because the census receipt they come from
was measured there; a run on another device is a different numerical
measurement rather than a failure of the booking contract.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from benchmarks import unit_amplitude_current_census as census
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


pytestmark = pytest.mark.slow

#: Analytical plasma current the certificate row solves towards, in amperes.
TARGET_CURRENT_A = 16_314_773.311828371

#: Booked totals recorded by the census receipt for the committed terminal
#: state, in amperes, one per clip mode.
BOOKED_TOTALS_A = {"chord": 15_540_896.52, "exact": 15_148_862.10}

#: Cut cells the census books no current for in either clip mode.  Their
#: centroids sit on the outboard side of the machine, where the separatrix
#: crosses the cell so the chord label selects the whole cell or none of it.
OUTBOARD_HOLE_CELLS = (83, 104, 106, 108, 116)

#: Relative agreement required between the recomputed booking and the receipt.
BOOKING_RELATIVE_TOLERANCE = 1.0e-6


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


def test_booked_totals_reproduce_the_census_receipt(booked) -> None:
    chord_total, exact_total, _per_cell = booked
    observed = {"chord": chord_total, "exact": exact_total}
    for mode, recorded in BOOKED_TOTALS_A.items():
        relative = abs(observed[mode] - recorded) / abs(recorded)
        assert relative <= BOOKING_RELATIVE_TOLERANCE, (
            f"{mode} booked total {observed[mode]:.6f} A is outside "
            f"{BOOKING_RELATIVE_TOLERANCE:g} of the census receipt's {recorded} A"
        )


def test_outboard_hole_cells_book_zero_in_both_clip_modes(booked) -> None:
    _chord_total, _exact_total, per_cell = booked
    for cell in OUTBOARD_HOLE_CELLS:
        for mode in ("chord", "exact"):
            value = per_cell[cell][mode]
            assert value == 0.0, (
                f"cell {cell} books {value:.6f} A under the {mode} clip; the "
                "census receipt records zero in both modes"
            )
