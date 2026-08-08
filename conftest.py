"""Session-wide pytest configuration."""

import os

import pytest

# Force a non-interactive matplotlib backend before any test module imports
# pyplot, so plotting tests never open a window or block on a GUI event loop
# in the fast lane or in headless CI. setdefault leaves an explicit override
# (e.g. MPLBACKEND set by a developer) untouched.
os.environ.setdefault("MPLBACKEND", "Agg")


def _imas_data_available() -> bool:
    """Return whether the canonical IMAS pulse the source doctests read resolves.

    The nova.imas source modules carry class doctests that open a live
    equilibrium pulse. When no pulse store is reachable (offline fast lane,
    CI without a data mount) those reads fail; this probe lets collection skip
    them while leaving them active whenever a store is present. It reuses the
    same validity gate the decorated database tests rely on.
    """
    try:
        from nova.imas.test_utilities import ids_attrs, load_ids
    except Exception:
        return False
    probe = load_ids(**ids_attrs["equilibrium"])
    return bool(getattr(probe, "is_valid", False))


# Test modules whose whole cost is a heavy numerical/integration workload: biot
# field validation sweeps, jax jit/vmap/grad kernels, signal-store round-trips,
# structural solves, and pulse-scale integration. The default lane excludes
# them; `pytest -m "slow or not slow"` runs everything.
_SLOW_FILES = frozenset(
    {
        "test_biotsavart.py",
        "test_biotsavart_3d.py",
        "test_biotpolygon.py",
        "test_decompose.py",
        "test_connectivity_boundary.py",
        "test_equilibrium_moment_boundary.py",
        "test_topology_boundary.py",
        "test_flux_surface_connectivity.py",
        "test_fieldnull.py",
        "test_fixed_point.py",
        "test_jax_topology.py",
        "test_jax_stencil_nulls.py",
        "test_jax_operate.py",
        "test_jax_basis.py",
        "test_io_roundtrip.py",
        "test_io_standardname.py",
        "test_ingest_import.py",
        "test_finiteframe.py",
        "test_finitebeam.py",
        "test_plasma.py",
        "test_profile_accelerated.py",
        "test_plasmapoints.py",
        "test_plasmaprofile.py",
        "test_pulsedesign.py",
        "test_extrapolate.py",
        "test_wallflux.py",
    }
)

# Individually heavy tests inside otherwise-fast modules.
_SLOW_NODEIDS = frozenset(
    {
        "tests/test_polyline.py::test_single_arc_hd",
    }
)


def pytest_collection_modifyitems(config, items):
    """Skip live-data doctests when no pulse resolves and mark the slow set.

    Everything under the ``nova/imas`` source modules listed in ``testpaths`` is
    a class doctest that needs a reachable pulse; the ordinary decorated tests
    already gate on the same validity check, so those doctests are skipped when
    no store is present (keeps the offline default lane green without weakening
    them when data is present). Separately, the curated heavy modules and tests
    are marked ``slow`` so the default ``-m 'not slow'`` invocation stays within
    the fast-lane budget while the full lane still runs them.
    """
    imas_ok = _imas_data_available()
    skip_imas = pytest.mark.skip(reason="IMAS pulse data unavailable (offline lane)")
    slow = pytest.mark.slow
    for item in items:
        if not imas_ok and item.nodeid.startswith("nova/imas/"):
            item.add_marker(skip_imas)
        filename = os.path.basename(str(getattr(item, "path", "")) or item.nodeid)
        if filename in _SLOW_FILES or item.nodeid in _SLOW_NODEIDS:
            item.add_marker(slow)
