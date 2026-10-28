"""Guards on the diverted 300-cell certificate row.

Two things must hold before the row's terminal residual means anything.

The banked part receipt must belong to the carrier the current tree builds.
``solovev_certificate._measure`` returns a persisted part without entering the
solver, so a receipt whose declared target current was computed from another
carrier reports a residual that no run of this tree produced.

And the row itself must converge inside the certificate's qualification bound.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import solovev_certificate as certificate
from nova.jax.config import configure_dtypes

CASE = "diverted-single-null"
REQUESTED_CELLS = -300
PART = (
    Path(__file__).resolve().parents[1]
    / "docs/figures/gs-absolute-accuracy/solovev/production-route-parts"
    / f"{CASE}-production-route-cells-300.json"
)


def _current_carrier_target_current() -> float:
    """Return the target current the certificate builder declares right now."""

    configure_dtypes()
    carrier_case, source_case, exact = certificate._case(CASE)
    machine = certificate._case_machine(CASE, carrier_case, exact, REQUESTED_CELLS)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(CASE, exact, coordinates)
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture

    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical, _, _ = oracle_fixture.cached_fixture_exterior(
        source_case, exact, machine, empty_operator, oracle_state
    )
    operator = oracle_fixture.forward_operator(source_case, machine)
    target_current, _, _ = certificate._closed_form_current_target(
        CASE, source_case, operator, exact_physical
    )
    return float(target_current)


def _measure_row_local() -> float:
    """Return the row's terminal residual from a fresh production solve."""

    configure_dtypes()
    carrier_case, source_case, exact = certificate._case(CASE)
    machine = certificate._case_machine(CASE, carrier_case, exact, REQUESTED_CELLS)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(CASE, exact, coordinates)
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture
    from nova.equilibrium.stencil_mesh import StencilMesh
    from scripts.oracle_rebaseline import measure as recovery

    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical, fixture_exterior, _ = oracle_fixture.cached_fixture_exterior(
        source_case, exact, machine, empty_operator, oracle_state
    )
    operator = oracle_fixture.forward_operator(source_case, machine, fixture_exterior)
    mesh = StencilMesh(machine.node, machine.stencil, machine.area)
    profile = certificate.ForwardProfile(
        operator, mesh, newton_steps=recovery.NEWTON_STEPS
    )
    target_current, centroid, receipt = certificate._closed_form_current_target(
        CASE, source_case, operator, exact_physical
    )
    seed, _, _ = certificate._production_seed(
        profile, CASE, target_current, centroid, receipt
    )
    request = certificate._certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"solovev:{CASE}:{REQUESTED_CELLS}",
    )
    solve = profile.solve(request)
    import jax

    jax.block_until_ready(solve.equilibrium.flux)
    return float(solve.equilibrium.fixed_point.residual)


def test_banked_part_receipt_matches_the_current_carrier() -> None:
    """The receipt the driver would return describes the carrier we build."""

    persisted = json.loads(PART.read_text(encoding="utf-8"))
    recorded = float(persisted["solver"]["target_current_a"])
    current = _current_carrier_target_current()
    assert recorded == pytest.approx(current, rel=1.0e-9), (
        "the banked part receipt declares a target current computed from another "
        f"carrier: recorded {recorded!r} A against {current!r} A on this tree"
    )


@pytest.mark.slow
def test_diverted_300_row_converges() -> None:
    """The diverted 300-cell row reaches the certificate's residual bound."""

    residual = _measure_row_local()
    assert residual <= certificate.TERMINAL_RESIDUAL_BOUND, (
        f"terminal residual {residual!r} above {certificate.TERMINAL_RESIDUAL_BOUND!r}"
    )
