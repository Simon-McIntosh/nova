"""Guards for the probe-row convention and the wall ring this study writes.

The probe rows are the measurement: a writer that projects along
``Br cos(theta) + Bz sin(theta)`` writes rows imas-efit reads as their
reflection, and every fit then lands a metre from the machine's own axis.  The
tests below pin the row a probe of a known orientation must carry, and pin the
wall against the reader that owns the limiter-unit walk, so a sign flip or a
second wall reader would redden rather than quietly move the reconstruction.
"""

from __future__ import annotations
import numpy as np
import pytest

import study
from nova.equilibrium.measurement import Magnetics

# A field of 1 T radially and 0.5 T vertically, at a 30 deg orientation.
BR = 1.0
BZ = 0.5
ANGLE = 30.0


def project_row(angle_deg: float) -> float:
    """Return the row nova's sensor reports for a probe at ``angle_deg``."""

    sensors = Magnetics(
        r=np.array([2.0]),
        z=np.array([0.0]),
        angle=np.array([angle_deg], dtype=float),
        flux_loop=np.array([False]),
    )
    return float(
        sensors.project(
            np.zeros(1), np.array([BR], dtype=float), np.array([BZ], dtype=float)
        )[0]
    )


@pytest.mark.parametrize("angle", [0.0, 30.0, 75.0, -40.0])
def test_probe_row_carries_the_imas_axis(angle: float) -> None:
    """The written row is Br cos(theta) - Bz sin(theta) at the dictionary angle."""

    written = project_row(float(study.probe_row_angle(np.array([angle]))[0]))
    theta = np.deg2rad(angle)
    assert written == pytest.approx(BR * np.cos(theta) - BZ * np.sin(theta), abs=1e-12)


def test_prior_convention_is_the_reflection() -> None:
    """Writing the dictionary angle unchanged reflects the vertical field."""

    theta = np.deg2rad(ANGLE)
    corrected = project_row(float(study.probe_row_angle(np.array([ANGLE]))[0]))
    prior = project_row(ANGLE)
    assert prior == pytest.approx(BR * np.cos(theta) + BZ * np.sin(theta))
    assert prior - corrected == pytest.approx(2.0 * BZ * np.sin(theta))


def test_wall_ring_is_the_machines_own_limiter_unit(tmp_path) -> None:
    """The written wall is the first limiter contour the shared reader returns."""

    import imas

    from nova.equilibrium.wall_mask import wall_units_from_ids

    entry = imas.DBEntry(study.MACHINE_URI, "r", dd_version=study.DD_VERSION)
    try:
        units = wall_units_from_ids(entry.get("wall"))
    finally:
        entry.close()
    expected = next(unit.vertices for unit in units if unit.vertices.shape[0] >= 3)
    path = study.wall_npz(tmp_path / "wall.npz")
    with np.load(path) as stored:
        wall = np.asarray(stored["wall"], dtype=float)
    assert wall.shape[0] >= 3
    assert np.array_equal(wall, expected)
