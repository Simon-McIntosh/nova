"""The closed-form row budget is one tenth of the row's other error term.

The gate each row is scored against is a tenth of the smallest error term the
row carries, and writing the term where the tenth belongs is a defect that has
recurred, so the relation is pinned here rather than left to the constant's
docstring. The terms themselves are the studies' measurements: the second-order
coupling's frozen-image error of the row's own component
(``route_metrics.order_two.all.rms_over_span``) on the four rows the coupling
study froze, and the fan's own refinement floor, per moment, where it did not.
"""

from __future__ import annotations

import importlib

import pytest

MODULE = importlib.import_module("benchmarks.exact_clip_closed_form_floor")

COUPLING_ROW_OTHER_ERROR_TERM = {
    ("weak-rotation-reactor-static", -110): 1.147e-4,
    ("weak-rotation-reactor-static", -300): 3.244e-5,
    ("moderate-rotation-conventional-static", -110): 1.040e-4,
    ("moderate-rotation-conventional-static", -300): 3.348e-5,
}
FAN_REFINEMENT_FLOOR = {
    "current": 2.924e-16,
    "radial": 5.109e-15,
    "vertical": 5.064e-15,
}
TENTH_RELATIVE_TOLERANCE = 1e-12


def test_coupling_budget_is_one_tenth_of_the_row_terms():
    assert set(MODULE.COUPLING_BUDGET) == set(COUPLING_ROW_OTHER_ERROR_TERM)
    for row, term in sorted(COUPLING_ROW_OTHER_ERROR_TERM.items()):
        budget = MODULE.COUPLING_BUDGET[row]
        assert budget == pytest.approx(term / 10.0, rel=TENTH_RELATIVE_TOLERANCE)
        assert budget != term, "the row's own error term is carried as its budget"


def test_fallback_budget_is_one_tenth_of_the_fan_floors():
    assert set(MODULE.FALLBACK_BUDGET) == set(FAN_REFINEMENT_FLOOR)
    for moment, floor in sorted(FAN_REFINEMENT_FLOOR.items()):
        budget = MODULE.FALLBACK_BUDGET[moment]
        assert budget == pytest.approx(floor / 10.0, rel=TENTH_RELATIVE_TOLERANCE)
        assert budget != floor, "the fan floor itself is carried as the budget"


def test_row_budget_gives_every_coupling_row_the_tenth_and_the_rest_the_triple():
    for row, budget in sorted(MODULE.COUPLING_BUDGET.items()):
        expected = dict.fromkeys(MODULE.MOMENT_NAMES, budget)
        assert MODULE._row_budget(*row) == expected
    for case in MODULE.SWEEP_CASES:
        for cells in MODULE.CELL_REQUESTS:
            if (case, cells) in MODULE.COUPLING_BUDGET:
                continue
            assert MODULE._row_budget(case, cells) == MODULE.FALLBACK_BUDGET


def test_production_boundary_reduction_has_no_gauss_evaluations():
    """The endpoint route samples its density fit once and has no edge nodes."""
    from nova.equilibrium.clip_quadrature import cut_cell_moment_evaluation_bound

    assert cut_cell_moment_evaluation_bound() == 25


def test_production_boundary_lowering_uses_endpoint_recurrence():
    """Trace a live density with every monomial to expose the integration body."""
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.clip_quadrature import _sampled_arc_polynomial_moments
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64
    vertices = jnp.asarray([[[0.0, 0.0], [0.8, -0.2], [0.1, 0.7]]])
    lowered = jax.jit(_sampled_arc_polynomial_moments).lower(
        vertices,
        jnp.asarray([3]),
        jnp.zeros((1, 2)),
        jnp.ones((1, 2)),
        jnp.arange(1.0, 16.0)[None],
        jnp.zeros((1, 2)),
    )
    text = lowered.as_text(debug_info=True)
    assert "_sampled_arc_polynomial_moments" in text
    assert "_straight_edge_monomial_moments" in text
    assert "_monomial_antiderivative" not in text
