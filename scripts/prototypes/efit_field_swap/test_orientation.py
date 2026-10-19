"""Pin the reference-sign guard: refuse a grid that contradicts its declaration.

The guard reads the flux difference between the stored axis and the stored
boundary, a quantity an additive gauge constant cannot change, against the
extremum type the declared convention fixes. These tests pin the refusal on a
synthetic field whose grid is the negation of its scalars, the pass on the
agreeing field, and the invariances the criterion rests on.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.prototypes.efit_field_swap import orientation

GRID = np.linspace(-1.0, 1.0, 41)
RR, ZZ = np.meshgrid(GRID, GRID, indexing="ij")
AXIS = (0.0, 0.0)
CORNER = (1.0, 1.0)


def _scalar_field():
    """A flux with a maximum at the axis, as a well-formed reference stores it."""
    return 1.0 - (RR**2 + ZZ**2)


def _wall_ring():
    ring = np.asarray(
        (
            (-0.8, -0.8),
            (0.8, -0.8),
            (0.8, 0.8),
            (-0.8, 0.8),
            (-0.8, -0.8),
        )
    )
    wall_r = np.column_stack((ring[:-1, 0], ring[1:, 0]))
    wall_z = np.column_stack((ring[:-1, 1], ring[1:, 1]))
    return wall_r, wall_z


def _record(polarity, psi):
    wall_r, wall_z = _wall_ring()
    return SimpleNamespace(
        slice=7,
        polarity=polarity,
        axis_r=AXIS[0],
        axis_z=AXIS[1],
        axis_psi=1.0,
        saddle_refined=-1.0,
        rgrid=GRID,
        zgrid=GRID,
        psi=psi,
        wall_r=wall_r,
        wall_z=wall_z,
    )


def test_cocos_and_current_fix_the_same_axis_type():
    """DDv3 COCOS 11 with negative Ip and DDv4 COCOS 17 with positive Ip agree."""
    cocos11 = orientation.read_convention(dd_version="3.41.0", plasma_current=-1.0e5)
    cocos17 = orientation.read_convention(dd_version="4.1.0", plasma_current=1.0e5)
    assert float(cocos11.sigma) == pytest.approx(1.0)
    assert float(cocos17.sigma) == pytest.approx(1.0)
    assert float(cocos17.sigma) == pytest.approx(float(cocos11.sigma))


def test_grid_value_uses_the_read_spline():
    r = np.linspace(0.0, 2.0, 5)
    z = np.linspace(0.0, 2.0, 5)
    psi = 2.0 * r[:, None] + z[None, :]
    assert orientation.grid_value(r, z, psi, (0.0, 0.0)) == pytest.approx(0.0)
    assert orientation.grid_value(r, z, psi, (1.0, 1.0)) == pytest.approx(3.0)
    assert orientation.grid_value(r, z, psi, (2.0, 2.0)) == pytest.approx(6.0)


def test_agreeing_grid_is_read():
    convention = orientation.read_convention(polarity=1)
    signs = orientation.check_axis_boundary_sign(
        convention, GRID, GRID, _scalar_field(), AXIS, (CORNER,)
    )
    assert signs["grid"] == 1


def test_negated_grid_against_its_scalars_is_refused():
    """The stored grid is the negation of its scalars: the axis is a minimum."""
    convention = orientation.read_convention(dd_version="3.41.0", plasma_current=-1.0e5)
    negated = -_scalar_field() + 0.7
    with pytest.raises(orientation.ReferenceSignError) as refused:
        orientation.check_axis_boundary_sign(
            convention,
            GRID,
            GRID,
            negated,
            AXIS,
            (CORNER,),
            scalar_axis_psi=1.0,
            scalar_boundary_psi=-1.0,
            slice_index=7,
            declared="COCOS 11, Ip -100000 A",
        )
    message = str(refused.value)
    assert "slice 7" in message
    assert "COCOS 11, Ip -100000 A" in message
    assert "grid psi axis - boundary" in message and "sign -1" in message
    assert "scalar psi axis - boundary" in message and "sign +1" in message


def test_additive_constant_leaves_the_verdict_unchanged():
    convention = orientation.read_convention(polarity=1)
    negated = -_scalar_field()
    for offset in (0.0, 5.0, -123.0, 1.0e10):
        with pytest.raises(orientation.ReferenceSignError):
            orientation.check_axis_boundary_sign(
                convention, GRID, GRID, negated + offset, AXIS, (CORNER,)
            )
    orientation.check_axis_boundary_sign(
        convention, GRID, GRID, _scalar_field() + 7.5, AXIS, (CORNER,)
    )


def test_scalars_contradicting_the_declaration_are_refused():
    """A grid that agrees with the declaration cannot hide scalars that do not."""
    convention = orientation.read_convention(polarity=1)
    with pytest.raises(orientation.ReferenceSignError) as refused:
        orientation.check_axis_boundary_sign(
            convention,
            GRID,
            GRID,
            _scalar_field(),
            AXIS,
            (CORNER,),
            scalar_axis_psi=-1.0,
            scalar_boundary_psi=1.0,
            slice_index=3,
            declared="polarity +1",
        )
    assert "scalar psi axis - boundary" in str(refused.value)


def test_trace_record_with_contradicting_polarity_is_refused():
    record = _record(polarity=1, psi=-_scalar_field() + 0.5)
    with pytest.raises(orientation.ReferenceSignError) as refused:
        orientation.check_trace_record(record)
    assert "polarity +1" in str(refused.value)


def test_trace_record_with_agreeing_polarity_passes():
    record = _record(polarity=1, psi=_scalar_field() + 0.5)
    signs = orientation.check_trace_record(record)
    assert signs["grid"] == 1


def test_crosscheck_refuses_a_bad_trace_before_building_the_field(monkeypatch):
    from scripts.prototypes.efit_field_swap import crosscheck

    class UnreachableField:
        @classmethod
        def from_record(cls, record):
            pytest.fail("comparison built a field before checking the trace sign")

    monkeypatch.setattr(crosscheck, "TraceSplineField", UnreachableField)
    record = _record(polarity=1, psi=-_scalar_field() + 0.5)
    with pytest.raises(orientation.ReferenceSignError) as refused:
        crosscheck._decision("contradictory", "synthetic", record, (16,))
    assert "slice 7" in str(refused.value)
    assert "polarity +1" in str(refused.value)


def test_probe_refuses_a_bad_trace_before_building_the_field(monkeypatch):
    from scripts.prototypes.efit_field_swap import probe_read

    class UnreachableField:
        @classmethod
        def from_record(cls, record):
            pytest.fail("probe built a field before checking the trace sign")

    monkeypatch.setattr(probe_read, "TraceSplineField", UnreachableField)
    record = _record(polarity=1, psi=-_scalar_field() + 0.5)
    with pytest.raises(orientation.ReferenceSignError) as refused:
        probe_read.probe_record("contradictory", record)
    assert "slice 7" in str(refused.value)
    assert "polarity +1" in str(refused.value)


def test_trace_record_without_an_in_grid_wall_uses_axis_curvature():
    record = _record(polarity=1, psi=-_scalar_field())
    record.wall_r = np.asarray([[9.0, 9.5]])
    record.wall_z = np.asarray([[9.0, 9.5]])
    with pytest.raises(orientation.ReferenceSignError) as refused:
        orientation.check_trace_record(record)
    assert "grid curvature" in str(refused.value)

    record.psi = _scalar_field()
    assert orientation.check_trace_record(record)["grid curvature"] == 1


def test_undecidable_grid_span_is_refused_not_read_as_agreement():
    """A grid span below the sign tolerance fixes no extremum and is refused."""
    convention = orientation.read_convention(polarity=1)
    flat = 2.0 + 1e-15 * _scalar_field()
    with pytest.raises(orientation.UndecidableSignError) as refused:
        orientation.check_axis_boundary_sign(
            convention, GRID, GRID, flat, AXIS, (CORNER,)
        )
    message = str(refused.value)
    assert "slice None" in message
    assert "grid span" in message
    assert "sign tolerance" in message
    assert "undecidable" in message
    assert f"{orientation.sign_tolerance(2.0):.3e}" in message


def test_undecidable_grid_span_is_refused_with_slice_and_declaration():
    convention = orientation.read_convention(dd_version="3.41.0", plasma_current=-1.0e5)
    flat = 1.0 + 1e-15 * _scalar_field()
    with pytest.raises(orientation.UndecidableSignError) as refused:
        orientation.check_axis_boundary_sign(
            convention,
            GRID,
            GRID,
            flat,
            AXIS,
            (CORNER,),
            scalar_axis_psi=1.0,
            scalar_boundary_psi=-1.0,
            slice_index=7,
            declared="COCOS 11, Ip -100000 A",
        )
    message = str(refused.value)
    assert "slice 7" in message and "COCOS 11, Ip -100000 A" in message
    assert "sign tolerance" in message and "undecidable" in message


def test_zero_scalar_span_is_undecidable_even_when_the_grid_agrees():
    """A zero scalar span fixes no extremum, so a grid that agrees cannot hide it."""
    convention = orientation.read_convention(polarity=1)
    with pytest.raises(orientation.UndecidableSignError) as refused:
        orientation.check_axis_boundary_sign(
            convention,
            GRID,
            GRID,
            _scalar_field(),
            AXIS,
            (CORNER,),
            scalar_axis_psi=0.5,
            scalar_boundary_psi=0.5,
            slice_index=9,
            declared="polarity +1",
        )
    assert "scalar span" in str(refused.value)


def test_undecidable_trace_record_is_refused():
    record = _record(polarity=1, psi=2.0 + 1e-15 * _scalar_field())
    with pytest.raises(orientation.UndecidableSignError) as refused:
        orientation.check_trace_record(record)
    assert "polarity +1" in str(refused.value)
    assert "sign tolerance" in str(refused.value)


def test_crosscheck_refuses_an_undecidable_trace_before_building_the_field(monkeypatch):
    from scripts.prototypes.efit_field_swap import crosscheck

    class UnreachableField:
        @classmethod
        def from_record(cls, record):
            pytest.fail("comparison built a field before checking the trace sign")

    monkeypatch.setattr(crosscheck, "TraceSplineField", UnreachableField)
    record = _record(polarity=1, psi=2.0 + 1e-15 * _scalar_field())
    with pytest.raises(orientation.UndecidableSignError):
        crosscheck._decision("undecidable", "synthetic", record, (16,))


def test_probe_refuses_an_undecidable_trace_before_building_the_field(monkeypatch):
    from scripts.prototypes.efit_field_swap import probe_read

    class UnreachableField:
        @classmethod
        def from_record(cls, record):
            pytest.fail("probe built a field before checking the trace sign")

    monkeypatch.setattr(probe_read, "TraceSplineField", UnreachableField)
    record = _record(polarity=1, psi=2.0 + 1e-15 * _scalar_field())
    with pytest.raises(orientation.UndecidableSignError):
        probe_read.probe_record("undecidable", record)


def test_missing_ids_boundary_is_refused_with_slice_and_declaration():
    convention = orientation.read_convention(dd_version="3.41.0", plasma_current=-1.0e5)
    with pytest.raises(orientation.ReferenceSignError) as refused:
        orientation.check_axis_boundary_sign(
            convention,
            GRID,
            GRID,
            _scalar_field(),
            AXIS,
            None,
            scalar_axis_psi=1.0,
            scalar_boundary_psi=-1.0,
            slice_index=7,
            declared="COCOS 11, Ip -100000 A",
        )
    assert "slice 7" in str(refused.value)
    assert "COCOS 11, Ip -100000 A" in str(refused.value)
    assert "no boundary" in str(refused.value)
