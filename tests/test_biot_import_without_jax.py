"""Import guarantees for the optional traced Biot operator."""

import importlib.util
import subprocess
import sys

from nova.geometry import select


def test_grid_import_falls_back_to_host_without_jax():
    """The biot grid remains importable when JAX is not installed."""
    script = """
import sys

sys.modules['jax'] = None

import nova.biot.grid
from nova.biot.operate import HostOperator, Operator
from nova.frame.coilset import CoilSet
from nova.jax.config import Precision

assert Operator is HostOperator
assert CoilSet().point.precision is Precision.DOUBLE
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


def test_select_import_without_jax_exposes_only_host_contract():
    """The colocated module retains its explicit JAX-free eager exports."""
    script = """
import sys

sys.modules['jax'] = None
from nova.geometry import select

assert 'host_subnull' in select.__all__
assert 'traced_subnull' not in select.__all__
assert not hasattr(select, 'traced_subnull')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_select_import_surfaces_missing_installed_jax_dependency():
    """Only absence of JAX itself activates the optional host-only boundary."""
    script = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == 'jax':
        raise ModuleNotFoundError('simulated missing jaxlib', name='jaxlib')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import nova.geometry.select
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "simulated missing jaxlib" in result.stderr


def test_select_exports_are_explicit_and_retired_module_is_absent():
    """Distinct implementations have named exports and no compatibility shim."""
    expected = {
        "bisect",
        "bisect_right",
        "bisect_2d",
        "length_2d",
        "wall_length",
        "wall_coordinate",
        "host_quadratic_wall",
        "traced_quadratic_wall",
        "host_wall_index",
        "traced_wall_index",
        "host_wall_flux",
        "traced_wall_flux",
        "host_quadratic_surface",
        "traced_quadratic_surface",
        "null_type",
        "null_coordinate",
        "null_flux",
        "host_subnull",
        "traced_subnull",
    }
    retired = {
        "quadratic_wall",
        "wall_index",
        "wall_flux",
        "quadratic_surface",
        "subnull",
        "null",
    }

    assert set(select.__all__) == expected
    assert not any(hasattr(select, name) for name in retired)
    assert importlib.util.find_spec("nova.jax.select") is None
