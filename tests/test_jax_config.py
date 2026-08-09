"""Process dtype policy and per-solver precision selection."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.jax.config import Precision, configure_dtypes, resolve_precision


def test_single_and_double_coexist_with_x64_capability_enabled():
    """Process capability does not prevent an operator selecting native fp32."""
    configure_dtypes()

    assert jax.config.x64_enabled
    assert jnp.asarray(1.0, dtype=jnp.float32).dtype == jnp.float32
    assert jnp.asarray(1.0, dtype=jnp.float64).dtype == jnp.float64


def test_runtime_precision_resolution_is_explicit_and_validated():
    """Automatic policy is owned by each solver, not by device import order."""
    assert resolve_precision(Precision.AUTOMATIC, Precision.DOUBLE) is Precision.DOUBLE
    assert resolve_precision("float32", Precision.DOUBLE) is Precision.SINGLE
    with pytest.raises(ValueError):
        resolve_precision("half", Precision.DOUBLE)


def test_jit_caches_one_executable_per_selected_input_dtype():
    """Single and double precision coexist as ordinary dtype-specialised traces."""
    configure_dtypes()
    kernel = jax.jit(lambda value: value * value + value)

    single = kernel(jnp.asarray([1.0], dtype=jnp.float32))
    double = kernel(jnp.asarray([1.0], dtype=jnp.float64))

    assert single.dtype == jnp.float32
    assert double.dtype == jnp.float64
    assert kernel._cache_size() == 2


@pytest.mark.parametrize(
    "modules",
    [
        ("nova.equilibrium.connectivity_boundary", "nova.equilibrium.profile"),
        ("nova.equilibrium.profile", "nova.equilibrium.connectivity_boundary"),
        ("nova.circuit.propagate", "nova.biot.tiledassembly"),
        ("nova.biot.tiledassembly", "nova.circuit.propagate"),
    ],
)
def test_import_order_does_not_change_process_dtype_policy(modules):
    """Importing dtype-driven kernels cannot change process capability."""
    configure_dtypes()
    before = jax.config.x64_enabled
    for name in modules:
        importlib.import_module(name)
    after = jax.config.x64_enabled

    assert after is before is True
    assert np.dtype(jnp.asarray(1.0, dtype=jnp.float32).dtype) == np.dtype(np.float32)


def test_removed_global_toggle_is_not_public():
    """There is no API that changes process precision around a solve."""
    config = importlib.import_module("nova.jax.config")
    assert not hasattr(config, "enable_x64")
