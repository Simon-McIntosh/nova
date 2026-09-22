"""Production saddle admission on cell carriers near a separatrix edge."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import dual_stencil_census as census_benchmark
from benchmarks import solovev_certificate as certificate
from nova.jax.config import configure_dtypes


@pytest.mark.parametrize("requested,realised", [(110, 132), (300, 340), (500, 550)])
def test_production_read_admits_analytic_saddle(requested, realised):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    machine, operator, state, _exact = census_benchmark._machine_and_field(
        certificate.DIVERTED_CASE_NAME, requested
    )
    assert len(machine.node) == realised
    pitch = np.sqrt(np.median(np.asarray(machine.area, dtype=np.float64)))
    reference = np.asarray(certificate.DIVERTED_REFERENCE.x_point)
    assert np.ptp(state[:realised]) > 1.0e-10
    masks, topology = operator.read(jnp.asarray(state, dtype=jnp.float64))
    error = float(np.linalg.norm(np.asarray(topology.x_point) - reference) / pitch)
    grid = operator._fixed_design_topology.grid
    pool = operator.null_flux_pool(jnp.asarray(state, dtype=jnp.float64))
    table = grid.candidate_table_status(pool)
    crossing = np.asarray(table["ring_crossing_count"])
    print(
        f"saddle_admission realised={realised} error_pitch={error:.12g} "
        f"retained={np.asarray(table['retained_count']).tolist()} "
        f"crossing_count_shape={crossing.shape} labelled_cells={masks.label.size}",
        flush=True,
    )
    assert crossing.size > 0
    assert np.any(crossing == 2)
    assert np.isfinite(error) and error <= 0.10


def _hex_locator(*, wall_half_width=3.0):
    from nova.biot.null import Null2D
    from nova.equilibrium.forward_operator import _FixedDesignNull2D

    configure_dtypes()
    angle = np.arange(6) * np.pi / 3.0
    vertices = np.column_stack((np.cos(angle), np.sin(angle)))
    coordinate = np.vstack((np.zeros(2), vertices)).astype(np.float64)
    wall = wall_half_width * np.asarray([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    locator = Null2D.from_coordinates(coordinate, np.arange(7)[None, :])
    fixed = _FixedDesignNull2D.from_locator(
        locator,
        cell_polygons=(vertices,) * len(coordinate),
        cell_area=np.ones(len(coordinate)),
        wall_coordinate=wall,
    )
    return fixed, coordinate


@pytest.mark.parametrize("centre", [(0.2, 0.1), (1.15, 0.0)])
def test_source_cell_admits_interior_and_quarter_pitch_near_saddles(centre):
    fixed, coordinate = _hex_locator()
    x, z = coordinate.T
    flux = (x - centre[0]) ** 2 - (z - centre[1]) ** 2
    table = fixed.candidate_table_status(jnp.asarray(flux))
    assert int(table["retained_count"][1]) == 1
    np.testing.assert_allclose(
        table["retained_candidate"][1, 0, :2], centre, atol=1e-12
    )


@pytest.mark.parametrize(
    "kind", ["outside_source", "outside_wall", "singular", "definite", "nonfinite"]
)
def test_saddle_prefilter_refuses_ineligible_quadratic_roots(kind):
    fixed, coordinate = _hex_locator(
        wall_half_width=0.1 if kind == "outside_wall" else 3.0
    )
    x, z = coordinate.T
    centre = (1.3, 0.0) if kind == "outside_source" else (0.2, 0.1)
    flux = (x - centre[0]) ** 2 - (z - centre[1]) ** 2
    if kind == "singular":
        flux = x + z
    elif kind == "definite":
        flux = (x - centre[0]) ** 2 + (z - centre[1]) ** 2
    elif kind == "nonfinite":
        flux[0] = np.nan
    table = fixed.candidate_table_status(jnp.asarray(flux))
    print(f"saddle_prefilter case={kind} retained={int(table['retained_count'][1])}")
    assert int(table["retained_count"][1]) == 0
    assert not np.any(np.asarray(table["quadratic_admitted_mask"])[1])


def test_raster_label_parity_over_both_analytic_fields():
    """Measure every label independently of boundary-coordinate roundoff."""
    from tests.test_plasma_cell_topology_read import (
        _limited_flux,
        _raster_label_oracle,
        _raster_operator,
        _single_null_flux,
    )

    configure_dtypes()
    compared = 0
    differing = 0
    for flux in (_limited_flux, _single_null_flux):
        operator = _raster_operator((17, 19))
        points = np.vstack((operator.grid.coordinate, operator.wall.coordinate))
        state = jnp.asarray(flux(points[:, 0], points[:, 1]), dtype=jnp.float64)
        actual, _topology, _connected, admitted = operator._fixed_design_read(state)
        expected, _stationary = _raster_label_oracle(operator, state)
        assert bool(admitted)
        compared += actual.label.size
        differing += np.count_nonzero(np.asarray(actual.label) != np.asarray(expected))
    print(f"raster_oracle_compared={compared} differing={differing}")
    assert compared == 646
    assert differing == 0


def test_compatibility_census_reports_prepolish_work_slot_exhaustion():
    """Three disjoint saddle roots exceed a two-slot polish work table."""
    from nova.biot.null import Null2D
    from nova.equilibrium.forward_operator import _FixedDesignNull2D

    configure_dtypes()
    angles = np.arange(6) * np.pi / 3.0
    patch = np.vstack((np.zeros(2), np.column_stack((np.cos(angles), np.sin(angles)))))
    centres = np.asarray(((0.0, 0.0), (4.0, 0.2), (8.0, -0.1)))
    coordinates = (patch[None, :, :] + centres[:, None, :]).reshape((-1, 2))
    stencil = np.arange(len(coordinates)).reshape((-1, 7))
    values = jnp.asarray(np.tile(patch[:, 0] ** 2 - patch[:, 1] ** 2, len(centres)))
    locator = Null2D.from_coordinates(coordinates, stencil, maxsize=1)
    fixed = _FixedDesignNull2D.from_locator(locator)
    assert not fixed.structured
    table = fixed.candidate_table_status(values)
    control = _FixedDesignNull2D.from_locator(
        locator.with_capacity(3)
    ).candidate_table_status(values)
    print(
        f"compatibility_work_slots roots={int(table['typed_count'][1])} "
        f"exhausted={bool(table['census_slots_exhausted'])} "
        f"control_exhausted={bool(control['census_slots_exhausted'])}",
        flush=True,
    )
    assert int(table["typed_count"][1]) == 3
    assert int(table["candidate_count"][1]) == 3
    assert int(table["retained_count"][1]) == 1
    assert bool(table["overflow"][1])
    assert bool(table["census_slots_exhausted"])
    assert not bool(control["census_slots_exhausted"])


def test_banked_consumer_uses_public_complete_flux_pool(monkeypatch):
    from benchmarks.diiid_forward_gs_match import candidate_flux_margins

    configure_dtypes()
    _machine, operator, state, _exact = census_benchmark._machine_and_field(
        certificate.DIVERTED_CASE_NAME, 110
    )
    state = jnp.asarray(state, dtype=jnp.float64)
    calls = []
    builder = operator.null_flux_pool

    def record_pool(argument):
        calls.append(np.asarray(argument))
        return builder(argument)

    monkeypatch.setattr(operator, "null_flux_pool", record_pool)
    expected = operator._fixed_design_topology.grid.candidate_table_status(
        builder(state)
    )
    receipt = candidate_flux_margins(operator, state, polarity=operator.polarity)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], state)
    assert receipt["o_candidate_count"] == int(expected["candidate_count"][0])
    assert receipt["x_candidate_count"] == int(expected["candidate_count"][1])
    assert receipt["x_candidate_count"] > 0
    print(f"banked_consumer complete_pool=True counts={receipt}", flush=True)
    with pytest.raises(ValueError, match="direct sampling flux values"):
        candidate_flux_margins(
            operator, state[: operator.physical_node_number], polarity=operator.polarity
        )
