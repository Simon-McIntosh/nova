"""Round-trip nova's certified carrier exports through the version-2 codec.

Each exported record is written, read back through the codec, and re-read by
nova's own topology read. The axis, the declared polarity and the boundary
level must reproduce within the tolerances below, and the boundary must sit
within the analytic certificate's bound.

The negated grid is the negative-control state: a record whose flux sign
disagrees with the polarity it declares must be refused as an unqualified axis,
and declaring the opposite polarity must restore the read, so the guard is the
declared COCOS-17 orientation rather than a broken record. A gate run drives
the same state through :data:`export.MUTATION_ENV`, which corrupts the record
on its way to disk so the round-trip test below is the thing that refuses it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from nova.equilibrium.topology import TopologyReason
from scripts.prototypes.certified_carrier_export import export_records as export
from scripts.prototypes.efit_field_swap.trace_record import read_trace_record
from tests.equilibrium.test_topology_read import _analytic_inputs

# Measured agreement, and the margin allowed over it. The axis reproduces the
# analytic certificate axis to 1.6e-7 m (limited) and 1.1e-7 m (diverted), so
# the tolerance is a ~60x margin on the larger of the two. The boundary sits
# 2.2e-14 Wb inside the certificate bound (limited) and 1.4e-9 Wb (diverted),
# so the tolerance is a ~70x margin on the larger.
AXIS_TOLERANCE = 1.0e-5
BOUNDARY_TOLERANCE = 1.0e-7
X_POINT_TOLERANCE = 1.0e-3
# The axis the record itself stored is read back unchanged, so it reproduces to
# the codec's own write precision rather than to a physics tolerance.
STORED_TOLERANCE = 1.0e-12
GRID = export.GRID
POLARITY = export.POLARITY


def _certificate_geometry(kind: str) -> tuple[np.ndarray, np.ndarray | None]:
    _oracle, _field, _wall, axis, saddle = _analytic_inputs(kind)
    return np.asarray(axis, dtype=np.float64), (
        None if saddle is None else np.asarray(saddle, dtype=np.float64)
    )


@pytest.mark.parametrize("kind", ["limited", "diverted"])
def test_round_trip(kind, tmp_path):
    """The written record re-reads to the axis, polarity and boundary level."""
    built = export.build_export(kind)
    path = export.write_export(built, tmp_path)
    record = read_trace_record(path)

    assert record.polarity == POLARITY
    assert record.rgrid.size == GRID and record.zgrid.size == GRID
    assert record.psi.shape == (GRID, GRID)
    assert len(record.wall_rings) == 1

    reading = export.read_record(record)
    assert bool(reading.valid), int(reading.reason)
    assert bool(reading.qualified)
    assert int(np.asarray(reading.boundary_class)) == built.boundary_class

    axis = np.asarray(reading.axis, dtype=np.float64)
    np.testing.assert_allclose(
        axis, (record.axis_r, record.axis_z), atol=STORED_TOLERANCE
    )
    boundary = float(np.asarray(reading.boundary_flux))
    assert abs(boundary - record.fed_back_psi) <= STORED_TOLERANCE

    analytic_axis, analytic_saddle = _certificate_geometry(kind)
    if kind == "limited":
        np.testing.assert_array_equal(analytic_axis, [6.2, 0.0])
    np.testing.assert_allclose(axis, analytic_axis, atol=AXIS_TOLERANCE)
    assert abs(boundary - 0.0) <= BOUNDARY_TOLERANCE
    if kind == "diverted":
        assert analytic_saddle is not None
        np.testing.assert_allclose(
            np.asarray(reading.boundary), analytic_saddle, atol=X_POINT_TOLERANCE
        )
        valid_points = np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)]
        assert len(valid_points) == 1
        np.testing.assert_allclose(
            valid_points[0], analytic_saddle, atol=X_POINT_TOLERANCE
        )
    else:
        assert analytic_saddle is None
        assert not np.any(np.asarray(reading.x_point_valid))


@pytest.mark.parametrize("kind", ["limited", "diverted"])
def test_negated_flux_is_refused_on_declared_polarity(kind):
    """A negated grid is refused on the polarity the record declares.

    The refusal is checked by its type, not by a disjunction: the read returns
    ``NO_QUALIFIED_AXIS`` and an unqualified axis. Declaring the opposite
    polarity restores the read with the same boundary class, which is what
    makes the refusal specific to the declared COCOS-17 orientation: it is the
    disagreement between the grid sign and the declaration that is refused, so
    a grid of the wrong sign cannot reproduce the decision its own record
    carries.
    """
    built = export.build_export(kind)
    negated = replace(built.record, psi=-built.record.psi)

    reading = export.read_record(negated)
    assert not bool(reading.valid)
    assert int(np.asarray(reading.reason)) == int(TopologyReason.NO_QUALIFIED_AXIS)
    assert not bool(reading.qualified)

    agreeing = export.read_record(replace(negated, polarity=-built.record.polarity))
    assert bool(agreeing.valid), int(agreeing.reason)
    assert int(np.asarray(agreeing.boundary_class)) == built.boundary_class
    agreeing_flux = float(np.asarray(agreeing.boundary_flux))
    assert abs(agreeing_flux + built.boundary_flux) <= BOUNDARY_TOLERANCE
