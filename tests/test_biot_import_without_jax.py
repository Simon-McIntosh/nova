"""Import guarantees for the optional traced Biot operator."""

import subprocess
import sys


def test_grid_import_falls_back_to_host_without_jax():
    """The biot grid remains importable when JAX is not installed."""
    script = """
import sys

sys.modules['jax'] = None

import nova.biot.grid
from nova.biot.operate import HostOperator, Operator

assert Operator is HostOperator
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "JAX is not installed; using the host Biot operator" in result.stderr


def test_grid_import_surfaces_a_broken_jax_installation():
    """An installed JAX that fails to import must not trigger fallback."""
    script = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == 'jax':
        raise ImportError('simulated broken JAX installation')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import

import nova.biot.grid
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "simulated broken JAX installation" in result.stderr
