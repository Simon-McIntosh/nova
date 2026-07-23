"""Kernel unit tests -- synthetic, millisecond, no recorded data files.

These pin the mathematical primitives the fitting cluster is built on:
the periodic Gaussian-process regressor, the clocking / cylindrical
transforms, and the Fourier gap-proxy coefficient identity.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_gp_regression_recovers_periodic_signal():
    """A near-interpolating periodic GP recovers a known cyclic waveform."""
    from nova.assembly.gaussianprocessregressor import GaussianProcessRegressor

    x = np.linspace(0.0, 1.0, 24, endpoint=False)
    y = 2.0 + 0.5 * np.sin(2 * np.pi * x)
    gpr = GaussianProcessRegressor(x, variance=1e-6)
    prediction = gpr.evaluate(x, y)
    assert prediction.shape == x.shape
    # With negligible assumed noise the periodic kernel tracks the signal.
    assert np.max(np.abs(prediction - y)) < 5e-2


def test_gp_regression_is_deterministic():
    """Two identical fits produce byte-identical predictions (random_state pin)."""
    from nova.assembly.gaussianprocessregressor import GaussianProcessRegressor

    x = np.linspace(0.0, 1.0, 16, endpoint=False)
    y = np.cos(2 * np.pi * x)
    first = GaussianProcessRegressor(x, variance=1e-4).evaluate(x, y)
    second = GaussianProcessRegressor(x, variance=1e-4).evaluate(x, y)
    assert np.array_equal(first, second)


def test_gp_prediction_order_invariant():
    """Permuting the fiducial input order leaves the prediction unchanged."""
    from nova.assembly.gaussianprocessregressor import GaussianProcessRegressor

    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 20, endpoint=False)
    y = 1.0 + 0.3 * np.sin(2 * np.pi * x) - 0.1 * np.cos(4 * np.pi * x)
    grid = np.linspace(0.0, 1.0, 50)

    base = GaussianProcessRegressor(x, variance=1e-4)
    base.fit(y)
    ref = base.predict(grid)

    order = rng.permutation(len(x))
    shuffled = GaussianProcessRegressor(x[order], variance=1e-4)
    shuffled.fit(y[order])
    got = shuffled.predict(grid)

    assert np.max(np.abs(got - ref)) < 1e-6


def test_clock_anticlock_round_trip():
    """Clocking then anticlocking is the identity rotation."""
    from nova.assembly.transform import Rotate

    vector = [1.0, 2.0, 3.0]
    assert np.allclose(Rotate().anticlock(Rotate().clock(vector)), vector)
    assert np.allclose(Rotate().clock(Rotate().anticlock(vector)), vector)


def test_clock_composition_across_coil_counts():
    """Two 18-coil clocks equal one 9-coil clock (half-angle addition)."""
    from nova.assembly.transform import Rotate

    rotate_9 = Rotate(ncoil=9)
    rotate_18 = Rotate(ncoil=18)
    vector = [1.0, 2.0, 3.0]
    assert np.allclose(
        rotate_9.anticlock(rotate_18.clock(rotate_18.clock(vector))), vector
    )


def test_cartesian_cylindrical_round_trip():
    """Cartesian -> cylindrical -> cartesian recovers the input points."""
    xr = pytest.importorskip("xarray")
    from nova.assembly.transform import Rotate

    rng = np.random.default_rng(2025)
    points = xr.DataArray(
        rng.random((12, 3)),
        dims=("point", "cartesian"),
        coords=dict(cartesian=list("xyz")),
    )
    rotate = Rotate(ncoil=18)
    cylindrical = rotate.to_cylindrical(points)
    recovered = rotate.to_cartesian(cylindrical)
    assert np.allclose(recovered.values, points.values)
    assert not np.allclose(cylindrical.values, recovered.values)


def test_fiducial_transform_error_vector_recompute():
    """FiducialTransform.error_vector agrees with an independent rms recompute."""
    from nova.assembly.fiducialtransform import FiducialTransform

    delta = np.random.default_rng(13).normal(size=(2, 6, 3))
    got = FiducialTransform.error_vector(delta, "rms")
    expected = np.array(
        [
            np.mean(delta[:, [5, 3, 4], 0] ** 2),
            np.mean(delta[..., 1] ** 2),
            np.mean(delta[:, [2, 1, -1, -2], 2] ** 2),
        ]
    )
    assert got.shape == (3,)
    assert np.all(got >= 0.0)
    assert np.allclose(got, expected, atol=1e-12)


def test_rotate_to_angle_z_rotation():
    """rotate_to_angle applies a right-handed rotation about z."""
    from nova.assembly.fiducialpit import rotate_to_angle

    x_axis = np.array([1.0, 0.0, 0.0])
    quarter = rotate_to_angle(x_axis, np.pi / 2)
    assert np.allclose(quarter, [0.0, 1.0, 0.0], atol=1e-12)

    # Rotating back recovers the original vector.
    there_and_back = rotate_to_angle(quarter, -np.pi / 2)
    assert np.allclose(there_and_back, x_axis, atol=1e-12)

    # z is unchanged by a rotation about z.
    z_axis = np.array([0.0, 0.0, 1.0])
    assert np.allclose(rotate_to_angle(z_axis, 0.7), z_axis, atol=1e-12)


def test_fourier_gap_proxy_coefficient_identity():
    """The stored rfft real/imag coefficients invert back to the mean-removed signal."""
    from nova.assembly.gap import GapData

    ncoil = 18
    rng = np.random.default_rng(7)
    gap = rng.normal(size=(1, ncoil))
    data = GapData(["s0"], gap=gap).data

    real = data.fft.sel(coefficient="real").isel(simulation=0, signal=0).values
    imag = data.fft.sel(coefficient="imag").isel(simulation=0, signal=0).values
    reconstructed = np.fft.irfft(real + 1j * imag, n=ncoil)

    mean_removed = gap[0] - gap[0].mean()
    assert np.allclose(reconstructed, mean_removed, atol=1e-9)


def test_fiducial_fit_configuration_constants():
    """Pin the fit-configuration constants that shape every coil fit."""
    from nova.assembly.fiducialfit import FiducialFit

    assert FiducialFit.radial_offset == pytest.approx((33.04 - 36) / (2 * np.pi))
    assert FiducialFit.weights == [1, 1, 0.25]
    assert FiducialFit.fiducial_index == {
        "radial": [5, 3, 4],
        "toroidal": [0, 5, 3, 4],
        "vertical": [2, 1, -1, -2],
    }


def _reference_error_vector(delta, method):
    """Independent re-implementation of the error vector for cross-checking."""
    from nova.assembly.fiducialfit import FiducialFit

    idx = FiducialFit.fiducial_index
    picks = [
        (delta[..., idx["radial"], 0]),
        (delta[..., idx["toroidal"], 1]),
        (delta[..., idx["vertical"], 2]),
    ]
    if method == "rms":
        return np.array([np.mean(p**2) for p in picks])
    if method == "max":
        return np.array([np.max(np.abs(p)) for p in picks])
    raise ValueError(method)


def test_error_vector_matches_independent_recompute():
    """error_vector agrees with an independent recompute for rms and max."""
    from nova.assembly.fiducialfit import FiducialFit

    delta = np.random.default_rng(7).normal(size=(2, 8, 3))
    for method in ("rms", "max"):
        got = FiducialFit.error_vector(delta, method)
        assert got.shape == (3,)
        assert np.all(got >= 0.0)
        assert np.allclose(got, _reference_error_vector(delta, method), atol=1e-12)


def test_error_vector_rejects_unknown_method():
    from nova.assembly.fiducialfit import FiducialFit

    with pytest.raises(NotImplementedError):
        FiducialFit.error_vector(np.zeros((8, 3)), "quadratic")


def test_fourier_amplitude_phase_consistent_with_real_imag():
    """Amplitude/phase coefficients agree with the real/imag pair."""
    from nova.assembly.gap import GapData

    rng = np.random.default_rng(11)
    gap = rng.normal(size=(1, 18))
    data = GapData(["s0"], gap=gap).data
    fft = data.fft.isel(simulation=0, signal=0)

    real = fft.sel(coefficient="real").values
    imag = fft.sel(coefficient="imag").values
    phase = fft.sel(coefficient="phase").values
    assert np.allclose(np.angle(real + 1j * imag), phase, atol=1e-9)
