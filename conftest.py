"""Session-wide pytest configuration."""

import os

import pytest

from nova.jax.config import bound_compilation_retention

# Force a non-interactive matplotlib backend before any test module imports
# pyplot, so plotting tests never open a window or block on a GUI event loop
# in the fast lane or in headless CI. setdefault leaves an explicit override
# (e.g. MPLBACKEND set by a developer) untouched.
os.environ.setdefault("MPLBACKEND", "Agg")

_RETENTION_CHECK_INTERVAL = 100
_LIVE_EXECUTABLE_CEILING = 1024
_DEFAULT_FAST_MARKEXPR = "not slow"
_completed_nodeids: set[str] = set()


def _command_line_sets_markexpr(config) -> bool:
    """Return whether this invocation supplied its own marker expression."""

    arguments = config.invocation_params.args
    return any(
        argument == "-m" or argument.startswith(("-m=", "--markexpr="))
        for argument in arguments
    )


def _explicit_test_paths(config) -> bool:
    """Return whether collection targets came from command-line arguments."""

    return config.args_source is config.ArgsSource.ARGS


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Let explicit collection paths override only the default fast-lane filter."""

    if (
        _explicit_test_paths(config)
        and not _command_line_sets_markexpr(config)
        and config.getoption("markexpr") == _DEFAULT_FAST_MARKEXPR
    ):
        config.option.markexpr = ""


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Make every otherwise-successful empty collection fail loudly."""

    if session.testscollected == 0 and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.NO_TESTS_COLLECTED


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    """Bound JAX compilation retention at fixed completed-test intervals."""
    terminal = report.when == "call" or report.skipped
    if not terminal or report.nodeid in _completed_nodeids:
        return

    _completed_nodeids.add(report.nodeid)
    if len(_completed_nodeids) % _RETENTION_CHECK_INTERVAL == 0:
        bound_compilation_retention(_LIVE_EXECUTABLE_CEILING)


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
# structural solves, equilibrium reproductions that solve a published reference
# machine, and pulse-scale integration. The default lane excludes them;
# `pytest -m "slow or not slow"` runs everything.
_SLOW_FILES = frozenset(
    {
        "test_biotsavart.py",
        "test_biotsavart_3d.py",
        "test_biotpolygon.py",
        "test_decompose.py",
        "test_connectivity_boundary.py",
        "test_equilibrium_forward_solve.py",
        "test_equilibrium_forward_reference.py",
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

# Individually heavy tests inside otherwise-fast modules. A parametrised test
# is named without its argument list; the collected nodeid is stripped back to
# the same form before matching.
_SLOW_NODEIDS = frozenset(
    {
        "tests/test_polyline.py::test_single_arc_hd",
        # the rotating free-boundary ladder: four bootstrapped machines, their
        # solves, and a gradient through one of them. The source-level rotation
        # contract in the same module stays in the fast lane.
        "tests/test_equilibrium_rotation.py::test_the_rotating_solve_reaches_its_fixed_point",
        "tests/test_equilibrium_rotation.py::test_the_rotating_solve_publishes_its_closure_and_conventions",
        "tests/test_equilibrium_rotation.py::test_the_rotating_solve_meets_its_conservation_tolerances",
        "tests/test_equilibrium_rotation.py::test_the_solved_axis_is_the_analytic_one_at_every_mach_number",
        "tests/test_equilibrium_rotation.py::test_rotation_pulls_the_boundary_in_by_the_analytic_shift",
        "tests/test_equilibrium_rotation.py::test_the_rotating_solve_is_differentiable_in_the_conductor_current",
        "tests/test_equilibrium_rotation.py::test_the_host_and_traced_routes_agree_on_the_rotating_map",
        # the continued free-boundary ladder: one bootstrapped machine with a
        # circular material boundary, and the solves that measure what a
        # declared scrape-off continuation does to its equilibrium. The
        # declaration-level continuation contract in the same module — anchors,
        # continuity, support bounds and the private-flux policy read on a
        # labelled diverted map — stays in the fast lane.
        "tests/test_equilibrium_sol.py::test_the_continued_solve_reaches_its_fixed_point",
        "tests/test_equilibrium_sol.py::test_the_scrape_off_current_is_published_in_its_own_ledger_row",
        "tests/test_equilibrium_sol.py::test_the_continuation_moves_the_solution_by_a_resolvable_bounded_amount",
        "tests/test_equilibrium_sol.py::test_the_continued_equilibrium_meets_its_conservation_tolerances",
        "tests/test_equilibrium_sol.py::test_no_current_appears_beyond_the_support_in_the_solved_map",
        "tests/test_equilibrium_sol.py::test_declaring_a_continuation_on_an_empty_domain_changes_nothing",
        "tests/test_equilibrium_sol.py::test_the_static_path_is_unchanged_when_no_continuation_is_declared",
        "tests/test_equilibrium_sol.py::test_the_source_evaluation_is_the_declared_gradients_unscaled",
        "tests/test_equilibrium_sol.py::test_the_host_and_traced_routes_agree_on_the_continued_map",
        "tests/test_equilibrium_sol.py::test_the_batched_continued_solve_matches_the_single_slice",
        "tests/test_equilibrium_sol.py::test_the_exponential_family_publishes_the_amplitude_it_truncates",
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
        if filename in _SLOW_FILES or item.nodeid.split("[")[0] in _SLOW_NODEIDS:
            item.add_marker(slow)
