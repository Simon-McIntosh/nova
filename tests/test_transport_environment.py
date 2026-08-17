"""Shared-process compatibility for the embedded transport environment."""

from importlib.metadata import version
import os

from packaging.version import Version


def test_nova_transport_and_torax_share_the_expected_jax_runtime():
    """Both transport packages retain the required dtype and compiler flags."""
    import jax
    import nova.transport as nova_transport
    import torax

    assert nova_transport.CurrentDiffusion is not None
    assert Version(torax.__version__) >= Version("1.4.3")
    assert Version(version("imas-python")) >= Version("2.2")
    assert Version(version("h5py")) >= Version("3.15")
    assert jax.config.x64_enabled
    assert (
        "--xla_cpu_opt_preset=FAST_COMPILE" in os.environ.get("XLA_FLAGS", "").split()
    )
