"""Invariance property tests.

Fits and proxies must respect the physical symmetries of the problem: the
Fourier gap proxy is blind to a global gap offset and rotates its phase (not
its amplitude) under a cyclic coil relabelling; the cylindrical transform
preserves radius under a global clocking rotation. These hold independently of
any recorded fixture, so they run in the fast lane.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_fourier_proxy_invariant_under_global_offset():
    """Adding a constant to every coil's gap leaves the spectrum unchanged.

    Mean removal in ``ModelData.fft`` kills the offset, so the real, imaginary
    and amplitude coefficients are identical. The DC-mode *phase* is excluded:
    it is ``np.angle`` of a ~1e-16 rounding residual and flips between 0 and pi,
    a meaningless quantity that no golden should ever gate on.
    """
    from nova.assembly.gap import GapData

    meaningful = ["real", "imag", "amplitude"]
    base = GapData(["s"], gap=np.random.default_rng(3).normal(size=(1, 18)))
    gap = base.data.gap.values
    got = GapData(["s"], gap=gap + 4.2).data.fft.sel(coefficient=meaningful).values
    ref = base.data.fft.sel(coefficient=meaningful).values
    assert np.allclose(ref, got, atol=1e-9)


def test_fourier_amplitude_invariant_under_cyclic_relabelling():
    """A cyclic roll of the coils preserves the amplitude spectrum."""
    from nova.assembly.gap import GapData

    rng = np.random.default_rng(5)
    gap = rng.normal(size=(1, 18))
    rolled = np.roll(gap, 3, axis=1)

    amp = lambda g: (  # noqa: E731
        GapData(["s"], gap=g)
        .data.fft.sel(coefficient="amplitude")
        .isel(simulation=0, signal=0)
        .values
    )
    assert np.allclose(amp(gap), amp(rolled), atol=1e-9)


def test_cylindrical_radius_invariant_under_clocking():
    """Clocking a point set about z preserves each point's cylindrical radius."""
    xr = pytest.importorskip("xarray")
    from nova.assembly.transform import Rotate

    rng = np.random.default_rng(17)
    points = xr.DataArray(
        rng.random((10, 3)),
        dims=("point", "cartesian"),
        coords=dict(cartesian=list("xyz")),
    )
    rotate = Rotate(ncoil=18)

    radius = rotate.to_cylindrical(points).isel(cylindrical=0).values
    clocked = points.copy()
    clocked.values = rotate.clock(points.values)
    radius_clocked = rotate.to_cylindrical(clocked).isel(cylindrical=0).values
    assert np.allclose(radius, radius_clocked, atol=1e-9)


def test_gp_prediction_shifts_with_constant_signal_offset():
    """A constant offset in the data shifts the GP mean by the same constant."""
    from nova.assembly.gaussianprocessregressor import GaussianProcessRegressor

    x = np.linspace(0.0, 1.0, 18, endpoint=False)
    y = 0.4 * np.sin(2 * np.pi * x)
    grid = np.linspace(0.0, 1.0, 40)

    base = GaussianProcessRegressor(x, variance=1e-4)
    base.fit(y)
    ref = base.predict(grid)

    offset = GaussianProcessRegressor(x, variance=1e-4)
    offset.fit(y + 3.0)
    got = offset.predict(grid)

    assert np.max(np.abs((got - 3.0) - ref)) < 1e-3
