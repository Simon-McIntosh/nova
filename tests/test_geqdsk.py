"""Round-trip the G-EQDSK interchange reader and writer.

The tolerance here is a property of the format, not a choice. G-EQDSK is
fixed-width text: the four header lines are written with nine decimal places
(`{:16.9f}`), which quantises them to an absolute 5e-10 in whatever SI unit the
field carries, and every array is written with nine digits after the point in
exponential form (`{:16.9e}`), which quantises them to a relative 5e-10. A
comparison at rtol=atol=1e-9 therefore sits just outside the format's own
resolution and would tighten only if the field widths changed.

Two things deliberately do not survive a round trip: `name` (the writer stamps
the current date onto it) and `header` (the verbatim first line). A header
scalar smaller than about 1e-9 also does not survive, because the `{:16.9f}`
fields carry absolute rather than relative precision.
"""

from pathlib import Path

import numpy as np
import pytest

from nova.io import geqdsk

FIXTURE = Path(__file__).parent / "data" / "geqdsk" / "analytic.geqdsk"

#: Format-set round-trip tolerance -- see the module docstring.
TOLERANCE = {"rtol": 1e-9, "atol": 1e-9}

#: Reconstructed by read() from the grid geometry rather than stored verbatim.
DERIVED = ("x", "z", "pnorm")

#: Overwritten by the writer (name) or the verbatim input line (header).
NOT_ROUND_TRIPPED = ("name", "header")


def analytic_equilibrium():
    """Return a small equilibrium with no symmetry a transpose could hide.

    The grid is deliberately non-square and psi is deliberately not symmetric
    under exchanging its axes, so a flattening in the wrong order fails on both
    shape and value.
    """
    nx, nz = 9, 11
    xgrid1, xdim = 3.5, 2.0
    zmid, zdim = 0.25, 3.0
    x = xgrid1 + xdim * np.arange(nx) / (nx - 1)
    z = (zmid - 0.5 * zdim) + zdim * np.arange(nz) / (nz - 1)
    xmagx, zmagx = 4.4, 0.3
    # an elongated, radially sheared flux function: distinct coefficients on the
    # two coordinates plus a cross term
    psi = (
        0.7 * (x[:, None] - xmagx) ** 2
        + 0.3 * (z[None, :] - zmagx) ** 2
        + 0.11 * (x[:, None] - xmagx) * (z[None, :] - zmagx)
    )
    pnorm = np.linspace(0, 1, nx)
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    return {
        "name": "analytic",
        "nx": nx,
        "nz": nz,
        "x": x,
        "z": z,
        "xdim": xdim,
        "zdim": zdim,
        "xcentr": 4.5,
        "bcentr": -5.3,
        "xgrid1": xgrid1,
        "zmid": zmid,
        "xmagx": xmagx,
        "zmagx": zmagx,
        "simagx": -1.75,
        "sibdry": 0.625,
        "Ip": 1.234567e6,
        "psi": psi,
        "fpol": -5.3 * 4.5 * (1 + 0.05 * pnorm),
        "pressure": 1.2e5 * (1 - pnorm) ** 2,
        "ffprim": -3.1 * (1 - 0.4 * pnorm),
        "pprime": -8.7e4 * (1 - 0.6 * pnorm),
        "qpsi": 0.95 + 3.4 * pnorm**2,
        "pnorm": pnorm,
        "nbdry": len(theta),
        "xbdry": xmagx + 0.9 * np.cos(theta),
        "zbdry": zmagx + 1.4 * np.sin(theta),
        "nlim": 5,
        "xlim": np.array([3.6, 5.3, 5.3, 3.6, 3.6]),
        "zlim": np.array([-1.2, -1.2, 1.7, 1.7, -1.2]),
        "ncoil": 3,
        "xc": np.array([3.2, 5.8, 5.8]),
        "zc": np.array([0.0, 1.9, -1.9]),
        "dxc": np.array([0.4, 0.3, 0.3]),
        "dzc": np.array([0.5, 0.35, 0.35]),
        "It": np.array([-1.1e6, 4.4e5, 4.4e5]),
    }


def assert_equilibria_agree(expected, actual):
    """Assert every round-trippable field agrees to the format's resolution."""
    compared = [key for key in expected if key not in NOT_ROUND_TRIPPED]
    assert set(compared) <= set(actual)
    for key in compared:
        assert np.allclose(
            np.asarray(actual[key], dtype=float),
            np.asarray(expected[key], dtype=float),
            **TOLERANCE,
        ), key


def test_write_then_read_returns_the_equilibrium(tmp_path):
    """Every numeric field survives a write and a read."""
    equilibrium = analytic_equilibrium()
    path = tmp_path / "written.geqdsk"
    geqdsk.write(str(path), equilibrium)
    assert_equilibria_agree(equilibrium, geqdsk.read(str(path)))


def test_read_write_read_is_idempotent(tmp_path):
    """A second pass through the format changes nothing further."""
    equilibrium = analytic_equilibrium()
    first = tmp_path / "first.geqdsk"
    geqdsk.write(str(first), equilibrium)
    once = geqdsk.read(str(first))
    second = tmp_path / "second.geqdsk"
    geqdsk.write(str(second), once)
    assert_equilibria_agree(once, geqdsk.read(str(second)))


def test_psi_keeps_its_radial_and_vertical_axes(tmp_path):
    """psi comes back as psi[x, z], not transposed and not reshaped."""
    equilibrium = analytic_equilibrium()
    path = tmp_path / "written.geqdsk"
    geqdsk.write(str(path), equilibrium)
    psi = geqdsk.read(str(path))["psi"]
    assert psi.shape == (equilibrium["nx"], equilibrium["nz"])
    assert np.allclose(psi, equilibrium["psi"], **TOLERANCE)


def test_committed_fixture_reads_as_the_analytic_equilibrium():
    """The stored file pins the reader independently of the writer."""
    assert FIXTURE.is_file(), f"missing round-trip fixture {FIXTURE}"
    assert_equilibria_agree(analytic_equilibrium(), geqdsk.read(str(FIXTURE)))


def test_committed_fixture_survives_a_further_round_trip(tmp_path):
    """Reading the stored file and writing it again preserves it."""
    once = geqdsk.read(str(FIXTURE))
    path = tmp_path / "rewritten.geqdsk"
    geqdsk.write(str(path), once)
    assert_equilibria_agree(once, geqdsk.read(str(path)))


@pytest.mark.parametrize("key", DERIVED)
def test_derived_grid_is_reconstructed(key, tmp_path):
    """The grid coordinates and normalised flux are rebuilt on read."""
    equilibrium = analytic_equilibrium()
    path = tmp_path / "written.geqdsk"
    geqdsk.write(str(path), equilibrium)
    assert np.allclose(geqdsk.read(str(path))[key], equilibrium[key], **TOLERANCE)
