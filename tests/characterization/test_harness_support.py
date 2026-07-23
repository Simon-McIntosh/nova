"""Self-tests for the characterization support layer.

These exercise the canonicaliser, tolerance classes and manifest round-trip
without importing the assembly code, so they stay in the fast lane and give an
immediate signal that the harness machinery itself is sound.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import _canonical, _environment, _tolerance


def test_canonicalize_flattens_mapping_sorted():
    result = {"b": np.array([1.0, 2.0]), "a": np.array([[3.0]])}
    canon = _canonical.canonicalize(result)
    assert list(canon) == ["a", "b"]
    assert canon["a"].dtype == np.float64


def test_canonicalize_xarray_dataset():
    xr = pytest.importorskip("xarray")
    ds = xr.Dataset({"gap": ("index", np.arange(3.0))}, coords={"index": [1, 2, 3]})
    canon = _canonical.canonicalize(ds)
    assert "gap" in canon
    assert np.allclose(canon["gap"], [0.0, 1.0, 2.0])


def test_canonicalize_drops_object_arrays():
    result = {"poly": np.array([object(), object()]), "x": np.array([1.0])}
    canon = _canonical.canonicalize(result)
    assert "poly" not in canon
    assert "x" in canon


def test_npz_round_trip_is_byte_stable():
    arrays = {"x": np.linspace(0, 1, 5), "y": np.array([[1.0, 2.0], [3.0, 4.0]])}
    payload_a = _canonical.to_npz_bytes(arrays)
    payload_b = _canonical.to_npz_bytes(dict(reversed(list(arrays.items()))))
    # Sorted-key serialization makes ordering irrelevant to the bytes.
    assert payload_a == payload_b
    loaded = _canonical.load_npz(payload_a)
    assert np.allclose(loaded["x"], arrays["x"])
    assert loaded["y"].shape == (2, 2)


def test_tolerance_length_mm_passes_at_micron():
    golden = np.array([1.0, 2.0, 3.0])
    candidate = golden + 5e-4  # half a micron on a millimetre
    result = _tolerance.compare(candidate, golden, "length_mm")
    assert result.passed
    assert result.max_abs_dev < 1e-3


def test_tolerance_length_mm_fails_above_micron():
    golden = np.array([1.0, 2.0, 3.0])
    candidate = golden + 2e-3  # two microns -- over the gate
    result = _tolerance.compare(candidate, golden, "length_mm")
    assert not result.passed
    assert "exceeds" in result.detail


def test_tolerance_shape_mismatch_fails():
    result = _tolerance.compare(np.zeros(3), np.zeros(4), "default")
    assert not result.passed
    assert "shape mismatch" in result.detail


def test_tolerance_nan_pattern_must_match():
    golden = np.array([1.0, np.nan, 3.0])
    good = np.array([1.0, np.nan, 3.0])
    bad = np.array([1.0, 2.0, 3.0])
    assert _tolerance.compare(good, golden, "length_mm").passed
    assert not _tolerance.compare(bad, golden, "length_mm").passed


def test_env_lock_is_stable_within_a_run():
    assert _environment.env_lock() == _environment.env_lock()


def test_package_versions_include_numpy():
    versions = _environment.package_versions()
    assert "numpy" in versions
