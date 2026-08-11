"""Precision and pytree contracts for traced null locators."""

import subprocess
import sys

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.biot.null import Null2D
    from nova.jax.config import Precision, configure_dtypes


def _geometry(offset=(6.2, -3.7), spacing=(0.012, 0.009)):
    """Return one centre-first physical stencil with ITER-scale offsets."""
    local = np.array(
        [
            (0, 0),
            (-1, 0),
            (0, -1),
            (1, -1),
            (1, 0),
            (0, 1),
            (-1, 1),
        ],
        dtype=np.float64,
    )
    coordinate = np.asarray(offset, dtype=np.float64) + local * np.asarray(
        spacing, dtype=np.float64
    )
    return coordinate, np.arange(7, dtype=np.int32)[None]


def test_normalized_single_precision_reconstructs_iter_scale_null():
    """The local fp32 fit retains a precise physical-coordinate reconstruction."""
    configure_dtypes()
    coordinate, stencil = _geometry()
    locator = Null2D.from_coordinates(
        coordinate, stencil, maxsize=1, precision=Precision.SINGLE
    )
    reference = Null2D.from_coordinates(
        coordinate, stencil, maxsize=1, precision=Precision.DOUBLE
    )
    truth = np.array([6.2031, -3.6982])
    psi = (coordinate[:, 0] - truth[0]) ** 2 - 1.4 * (coordinate[:, 1] - truth[1]) ** 2
    local_cluster = locator.local_coordinate_stencil[0]
    cluster = jnp.concatenate(
        (local_cluster, jnp.asarray(psi, dtype=locator.fit_dtype)[:, None]), axis=1
    )[None]

    result = np.asarray(
        locator.interpolate(
            jnp.asarray(1),
            cluster,
            locator.physical_origin[:1],
            locator.physical_scale[:1],
        )
    )
    reference_cluster = jnp.concatenate(
        (
            reference.local_coordinate_stencil[0],
            jnp.asarray(psi, dtype=reference.fit_dtype)[:, None],
        ),
        axis=1,
    )[None]
    reference_result = np.asarray(
        reference.interpolate(
            jnp.asarray(1),
            reference_cluster,
            reference.physical_origin[:1],
            reference.physical_scale[:1],
        )
    )

    spacing = np.asarray([0.012, 0.009])
    assert locator.precision is Precision.SINGLE
    assert reference.precision is Precision.DOUBLE
    assert locator.local_coordinate_stencil.dtype == jnp.float32
    assert reference.local_coordinate_stencil.dtype == jnp.float64
    assert locator.physical_origin.dtype == jnp.float64
    assert result.shape == (1, 4)
    assert np.max(np.abs((result[0, :2] - truth) / spacing)) < 1e-3
    assert np.max(np.abs((reference_result[0, :2] - truth) / spacing)) < 1e-8
    assert np.max(np.abs((result[0, :2] - reference_result[0, :2]) / spacing)) < 1e-3
    assert result[0, 3] == 0


def test_an_unqualified_locator_fits_on_the_double_ladder():
    """The fit dtype a locator defaults to is the ladder its flux reads land on.

    Everything read through the locator — the null flux, and whatever is
    normalised against it — is quantised at one step of this dtype, so the
    unqualified fit is the one that keeps that step at the arithmetic floor.
    """
    configure_dtypes()
    coordinate, stencil = _geometry()
    locator = Null2D.from_coordinates(coordinate, stencil, maxsize=1)

    assert locator.precision is Precision.DOUBLE
    assert locator.fit_dtype == jnp.float64
    assert locator.local_coordinate_stencil.dtype == jnp.float64


def test_absolute_fp32_geometry_is_rejected_before_normalization():
    """Centering cannot recover coordinate information already lost to fp32."""
    configure_dtypes()
    coordinate, stencil = _geometry()

    with pytest.raises(TypeError, match="host float64"):
        Null2D.from_coordinates(coordinate.astype(np.float32), stencil)


def test_tree_roundtrip_preserves_normalized_geometry_and_precision():
    """Pytree reconstruction retains fit data, metadata, and static capacity."""
    configure_dtypes()
    coordinate, stencil = _geometry()
    # a locator built off the default: a roundtrip that reconstructed the
    # default instead of carrying the selection would look identical otherwise
    locator = Null2D.from_coordinates(
        coordinate, stencil, maxsize=3, precision=Precision.SINGLE
    )

    leaves, structure = jax.tree_util.tree_flatten(locator)
    restored = jax.tree_util.tree_unflatten(structure, leaves)

    assert restored.maxsize == 3
    assert restored.precision is Precision.SINGLE
    assert restored.node_number == locator.node_number
    np.testing.assert_array_equal(restored.coordinate, locator.coordinate)
    np.testing.assert_array_equal(restored.stencil, locator.stencil)
    np.testing.assert_array_equal(
        restored.local_coordinate_stencil, locator.local_coordinate_stencil
    )
    np.testing.assert_array_equal(restored.physical_origin, locator.physical_origin)
    np.testing.assert_array_equal(restored.physical_scale, locator.physical_scale)


def test_runtime_setup_supports_explicit_fp32_and_fp64_without_toggling():
    """One capability setup supports dtype-keyed fp32 and fp64 variants."""
    script = """
import jax
import jax.numpy as jnp

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.x64_enabled
single = jnp.asarray([1.25], dtype=jnp.float32)
double = jnp.asarray([1.25], dtype=jnp.float64)
assert single.dtype == jnp.float32
assert double.dtype == jnp.float64

@jax.jit
def polynomial(value):
    return (value + value.dtype.type(0.5)) ** 2

assert polynomial(single).dtype == jnp.float32
assert polynomial(double).dtype == jnp.float64
assert float(polynomial(single)[0]) == 3.0625
assert float(polynomial(double)[0]) == 3.0625
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    pytest.main([__file__])
