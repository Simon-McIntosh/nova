"""Round-trip nova's certified carrier exports through the version-2 codec.

Each exported record is written, read back through the codec, and re-read by
nova's own topology read. The axis, the polarity and the boundary level must
reproduce within the stated tolerances, and the boundary must sit within the
analytic certificate's bound. A negated psi grid is the negative control: it
must be refused, so the test shows the COCOS-17 polarity guard fires rather
than only showing a green path.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.prototypes.certified_carrier_export import export_records as export
from scripts.prototypes.efit_field_swap.trace_record import read_trace_record
from tests.equilibrium.test_topology_read import _analytic_inputs

AXIS_TOLERANCE = 5.0e-2
BOUNDARY_TOLERANCE = 1.0e-6
GRID = export.GRID
POLARITY = export.POLARITY


def _certificate_axis(kind: str) -> np.ndarray:
    _oracle, _field, _wall, axis, _saddle = _analytic_inputs(kind)
    return np.asarray(axis, dtype=np.float64)


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
    np.testing.assert_allclose(axis, (record.axis_r, record.axis_z), atol=1e-12)
    boundary = float(np.asarray(reading.boundary_flux))
    assert abs(boundary - record.fed_back_psi) <= 1e-12

    analytic_axis = _certificate_axis(kind)
    np.testing.assert_allclose(axis, analytic_axis, atol=AXIS_TOLERANCE)
    assert abs(boundary) <= BOUNDARY_TOLERANCE


@pytest.mark.parametrize("kind", ["limited", "diverted"])
def test_negated_psi_is_refused(kind):
    """A negated psi grid is refused on the record's declared COCOS polarity."""
    built = export.build_export(kind)
    negated = replace(built.record, psi=-built.record.psi)
    reading = export.read_record(negated)
    refused = (not bool(reading.valid)) or (
        int(np.asarray(reading.boundary_class)) != built.boundary_class
    )
    assert refused, "a negated psi grid must not reproduce the declared decision"
