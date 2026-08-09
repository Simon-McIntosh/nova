"""Contract tests for traced null caller adaptation."""

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.jax.null import Null2D


def test_interpolate_adapts_stacked_cluster_to_three_arrays():
    """The locator passes each stacked column to the canonical subnull call."""
    x, z = np.meshgrid(np.arange(1, 4), np.arange(1, 4), indexing="ij")
    x, z = x.reshape(-1), z.reshape(-1)
    psi = (x - 2) ** 2 - (z - 2) ** 2
    cluster = jnp.asarray(np.c_[x, z, psi])[None, ...]
    locator = Null2D(
        coordinate=jnp.zeros((1, 2)),
        stencil=jnp.zeros((1, 1), dtype=int),
        coordinate_stencil=jnp.zeros((1, 1, 2)),
        maxsize=1,
    )

    result = np.asarray(locator.interpolate(jnp.asarray(1), cluster))

    assert result.shape == (1, 4)
    assert np.allclose(result[0, :3], [2.0, 2.0, 0.0], atol=1e-5)
    assert result[0, 3] == 0


if __name__ == "__main__":
    pytest.main([__file__])
