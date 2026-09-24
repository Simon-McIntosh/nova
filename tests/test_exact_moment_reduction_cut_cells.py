"""Regression coverage for sampled-arc cut-cell moment reduction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from nova.jax.config import configure_dtypes


configure_dtypes()

import jax  # noqa: E402

from benchmarks import exact_cut_cell_moment_stages as stages  # noqa: E402
from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.equilibrium.forward_operator import set_support_clip_mode  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402


assert jax.config.jax_enable_x64 is True

ROOT = Path(__file__).resolve().parents[1]
RECORDED_CUT_CELLS = (
    ("weak-rotation-reactor-static", 110, 108),
    ("weak-rotation-reactor-static", 300, 187),
    ("diverted-single-null", 110, 110),
)


@lru_cache(maxsize=None)
def _recorded_row(case_name: str, requested_cells: int):
    carrier, source, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier, exact, -requested_cells)
    operator = fixture.forward_operator(source, machine)
    label = f"{case_name}-cells-{requested_cells}"
    with np.load(
        ROOT
        / "docs/figures/plasma-cell-read-fidelity/map-fidelity"
        / f"{label}-exact.npz"
    ) as bank:
        state = bank["analytic"]
    set_support_clip_mode("exact")
    jax.clear_caches()
    return stages._evaluate_path(operator, state)


@pytest.mark.parametrize(("case_name", "requested_cells", "cell"), RECORDED_CUT_CELLS)
def test_sampled_arc_moments_match_order_doubled_reference(
    case_name: str,
    requested_cells: int,
    cell: int,
):
    _incoming, effective, production, field, _topology, _selected, profile = (
        _recorded_row(case_name, requested_cells)
    )
    vertices = np.asarray(effective.support_vertices)[cell]
    count = int(np.asarray(effective.vertex_count)[cell])
    centre = np.asarray(effective.centroids)[cell]
    assert count > 24

    rings = [[vertices[:count]]]
    cells = np.asarray([cell], dtype=np.int32)
    centres = centre[None]
    reference_order_8 = stages._profile_reference(
        field,
        profile.confined,
        rings,
        centres,
        cells,
        8,
        positive_orientation=True,
    )[0]
    reference_order_16 = stages._profile_reference(
        field,
        profile.confined,
        rings,
        centres,
        cells,
        16,
        positive_orientation=True,
    )[0]
    np.testing.assert_allclose(
        reference_order_8,
        reference_order_16,
        rtol=2e-11,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        np.asarray(production)[cell],
        reference_order_16,
        rtol=2e-6,
        atol=2e-8,
    )
    assert abs(np.asarray(production)[cell, 0] / reference_order_16[0] - 1.0) < 1e-8


def test_recorded_cells_are_distinct_rows():
    assert len(set(RECORDED_CUT_CELLS)) == len(RECORDED_CUT_CELLS)
    assert {case for case, _cells, _cell in RECORDED_CUT_CELLS} == {
        "weak-rotation-reactor-static",
        "diverted-single-null",
    }
    assert {cells for _case, cells, _cell in RECORDED_CUT_CELLS} == {110, 300}
