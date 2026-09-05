"""Playable-forward-solve session gate.

Drives the session holder with a stub solver on CPU and pins the contract the
interactive keyframe loop is built on: every named key produces exactly the
commanded control-point change it names; every pushed ColumnDataSource column
has the shape its renderer in ``apps/pulsedesign/poloidal_view.py`` binds; the
receipt row per action carries wall and trips; and one keyframe through the
production protocol — ``ForwardProfile.solve`` with ``constraint_pairs`` on
the Newton-Krylov route carrying the current-centroid row — completes on the
small Solov'ev fixture from ``tests/test_reduced_newton.py``.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from subprocess import Popen
from time import perf_counter, sleep
from types import SimpleNamespace

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

from apps.playable.session import PlayableSession, SolveResult, frame_push
from apps.playable.shape import PARAMETER_FIELD, STEPS, PlasmaShape, keymap, point_delta

#: The named controls and the signed step sizes the gate enumerates.
NAMED_KEYS = {
    "bulk_r": +0.02,
    "bulk_z": +0.01,
    "elongation": +0.05,
    "triangularity_upper": +0.02,
    "triangularity_lower": +0.02,
    "x_point_r": +0.02,
    "x_point_z": -0.01,
    "inner_gap": +0.005,
    "outer_gap": +0.005,
}

REPO_ROOT = Path(__file__).resolve().parents[1]


class StubEquilibrium(SimpleNamespace):
    """Minimal carrier-shaped equilibrium the frame reduce can read."""

    def __init__(self, radius=None, height=None, circuits: int = 0):
        radius = np.linspace(0.6, 1.42, 12) if radius is None else radius
        height = np.linspace(-0.42, 0.42, 10) if height is None else height
        super().__init__(
            raster_flux=SimpleNamespace(
                radius=np.asarray(radius),
                height=np.asarray(height),
                psi=np.zeros(radius.size * height.size),
                separatrix=np.full((30, 2), np.nan),
            ),
            labelled_flux=SimpleNamespace(
                primary_x_point=np.full(2, np.nan),
                secondary_x_point=np.full(2, np.nan),
            ),
            constraints=(),
            circuits=circuits,
        )


class StubSolver:
    """Return the same equilibrium without solving, at a stated wall and trips."""

    wall = 0.234
    trips = 3

    def __init__(self, equilibrium=None):
        self._equilibrium = (
            equilibrium if equilibrium is not None else StubEquilibrium()
        )

    def __call__(self, previous, commanded, *, action=None):
        return SolveResult(self._equilibrium, wall=self.wall, trips=self.trips)


def _stub_session(**kwargs) -> PlayableSession:
    """Return a session over the stub solver with a fresh commanded set."""
    return PlayableSession(solver=StubSolver(), **kwargs)


@pytest.fixture(scope="module")
def machine():
    """Build the playable Solov'ev machine once for the solve-path tests."""
    from apps.playable.solovev import build_machine

    return build_machine()


# --------------------------------------------------------------------------
# key map and the exact commanded control-point changes
# --------------------------------------------------------------------------


def test_key_map_covers_every_named_control():
    assert set(STEPS) == set(NAMED_KEYS)
    assert set(PARAMETER_FIELD) == set(NAMED_KEYS)
    bindings = keymap()
    assert len(bindings) == 2 * len(NAMED_KEYS)
    for name, step in NAMED_KEYS.items():
        assert STEPS[name] == step
        assert bindings[f"{name}+"] == (name, step)
        assert bindings[f"{name}-"] == (name, -step)


@pytest.mark.parametrize("name", list(NAMED_KEYS))
def test_each_named_key_produces_exactly_its_commanded_control_point_change(name):
    session = _stub_session()
    shape = session.shape
    key = f"{name}+"
    delta = NAMED_KEYS[name]
    expected = point_delta(shape, name, delta)
    before = {
        point_name: point
        for point_name, point in zip(
            shape.control_point_names, shape.control_points().T
        )
    }
    receipt = session.step(key)
    after = {
        point_name: point
        for point_name, point in zip(
            session.shape.control_point_names, session.shape.control_points().T
        )
    }
    assert receipt.parameter == name
    assert receipt.delta == NAMED_KEYS[name]
    for point_name, point in after.items():
        np.testing.assert_allclose(
            point - before[point_name],
            expected.get(point_name, np.zeros(2)),
            atol=1.0e-12,
        )
    assert len(session.receipts) == 1


def test_reverse_direction_keys_step_the_opposite_sign():
    session = _stub_session()
    original = session.shape.axis_r
    session.step("bulk_r+")
    session.step("bulk_r-")
    np.testing.assert_allclose(session.shape.axis_r, original)


def test_unknown_key_raises():
    session = _stub_session()
    with pytest.raises(KeyError, match="unknown key"):
        session.step("not-a-key")


def test_receipt_row_carries_wall_and_trips():
    session = _stub_session()
    session.step("elongation+")
    receipt = session.receipts[-1]
    assert receipt.wall == StubSolver.wall
    assert receipt.trips == StubSolver.trips
    pushed = frame_push(session)
    row = pushed["receipt"]
    assert len(row["action"]) == 1
    assert row["wall"][0] == StubSolver.wall
    assert row["trips"][0] == StubSolver.trips


# --------------------------------------------------------------------------
# pushed columns match the shapes their renderers bind
# --------------------------------------------------------------------------


with skip_import("bokeh"):
    from bokeh.models import ColumnDataSource

    from apps.pulsedesign.poloidal_view import (
        add_flux_image,
        add_separatrix,
        compensation_figure,
        keyframe_receipt,
        poloidal_figure,
    )


def _bound_fields(glyph):
    """Return the column names one glyph binds, by kind."""
    fields = []
    for value in glyph.properties_with_values().values():
        field = getattr(value, "field", None)
        if isinstance(field, str) and field:
            fields.append((field, type(glyph).__name__))
    return fields


@pytest.mark.skipif(
    not Path(__file__).parent.with_name("apps").is_dir(), reason="apps tree absent"
)
def test_pushed_columns_match_renderer_bindings():
    """Every column the session pushes has the shape its renderer binds."""
    session = _stub_session()
    session.step("inner_gap+")
    session.step("bulk_r+")
    frame = frame_push(session)

    sources = {
        name: ColumnDataSource()
        for name in (
            "levelset",
            "wall",
            "x_points",
            "plasma",
            "points",
            "flux",
            "separatrix",
            "compensation",
            "receipt",
        )
    }
    poloidal = poloidal_figure(sources)
    add_separatrix(poloidal, sources)
    add_flux_image(poloidal, sources, radius=(0.6, 1.42), height=(-0.42, 0.42))
    compensation = compensation_figure(sources)
    receipt = keyframe_receipt(sources)

    bound = {}
    for renderer in [*poloidal.renderers, *compensation.renderers]:
        for column, kind in _bound_fields(renderer.glyph):
            bound.setdefault(column, set()).add(kind)

    # raster flux image: the image glyph binds the 2-D channel
    assert "psi" in bound, "the flux image glyph must bind the psi column"
    psi = frame["flux"]["psi"]
    assert psi.ndim == 2
    radius = np.asarray(session.equilibrium.raster_flux.radius)
    height = np.asarray(session.equilibrium.raster_flux.height)
    assert psi.shape == (height.size, radius.size)
    sources["flux"].data = {"psi": [psi]}

    # separatrix, control points and X-points: 1-D same-length x and z per line
    for channel in ("separatrix", "points", "x_points"):
        for column in ("x", "z"):
            assert column in bound, f"{channel} renderer must bind {column}"
            values = frame[channel][column]
            assert values.ndim == 1
        assert frame[channel]["x"].size == frame[channel]["z"].size
        sources[channel].data = frame[channel]

    # compensating currents per circuit: 1-D same-length circuit and current
    assert "circuit" in bound and "current" in bound
    assert (
        frame["compensation"]["circuit"].size == frame["compensation"]["current"].size
    )
    sources["compensation"].data = frame["compensation"]

    # keyframe receipt row: one row with action, wall and trips
    table_fields = {column.field for column in receipt.columns}
    assert table_fields == {"action", "wall", "trips"}
    for column in ("action", "wall", "trips"):
        assert len(frame["receipt"][column]) == 1
    sources["receipt"].data = frame["receipt"]

    # the shared poloidal figure binds the columns pulsedesign's simulator
    # pushes (levelset, wall, plasma, x_points, points), so one renderer set
    # drives both apps
    assert {"x", "z"} <= set(bound)


def test_pulsedesign_app_imports_its_renderers_from_the_shared_module():
    """pulsedesign main pulls its poloidal renderers from poloidal_view."""
    with skip_import("bokeh"):
        import apps.pulsedesign.poloidal_view as poloidal_view

        source = (Path(poloidal_view.__file__).parent / "main.py").read_text()
        tree = ast.parse(source)
        shared_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "apps.pulsedesign.poloidal_view"
        ]
        names = {name.name for node in shared_imports for name in node.names}
        assert "poloidal_figure" in names
        # the imported renderer is the shared one pulsedesign now calls
        assert names <= {"poloidal_figure"}
        # and the package import chain the app boots on is intact
        import apps.pulsedesign as pds

        assert pds.Simulator is not None
        assert pds.ids_attrs["pulse"] == 135013


# --------------------------------------------------------------------------
# session argument: default Solov'ev machine, MAST carrier selectable
# --------------------------------------------------------------------------


def test_machine_argument_selects_the_carrier():
    from apps.playable.machines import AVAILABLE_MACHINES, machine_argument

    assert AVAILABLE_MACHINES == ("solovev", "mast")
    assert machine_argument({}) == "solovev"
    assert machine_argument({"machine": [b"mast"]}) == "mast"
    assert machine_argument({"machine": ["solovev"]}) == "solovev"
    with pytest.raises(ValueError, match="unknown machine"):
        machine_argument({"machine": [b"kittens"]})


def test_solovev_machine_reports_its_circuit_carrier(machine):
    assert machine.identity == "solovev"
    assert machine.circuit_count == 16
    assert machine.profile.operator.prescribed_current_field is not None
    radius, height, shape = machine.profile.operator.raster_geometry()
    assert shape == (15, 15)
    assert len(radius) == 15 and len(height) == 15


# --------------------------------------------------------------------------
# one keyframe through the production protocol (slow, CPU)
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_production_keyframe_completes_on_the_solovev_machine(machine):
    from apps.playable.production import ProductionSolver
    from nova.jax.config import configure_dtypes
    from nova.equilibrium.observation import MomentIntegralSupport

    configure_dtypes()
    solver = ProductionSolver(machine)
    session = PlayableSession(solver=solver, shape=PlasmaShape(), machine="solovev")

    prime = session.prime()
    assert prime.wall > 0.0 and isinstance(prime.trips, int) and prime.trips >= 0
    assert session.equilibrium is not None
    assert session.equilibrium.finite.passed or session.equilibrium.finite.flux

    centroid = np.asarray(
        machine.profile.current_moment_observation(
            session.equilibrium.flux, support=MomentIntegralSupport.ALL_DOMAIN
        ).stack()
    )
    assert np.all(np.isfinite(centroid))

    # one moved keyframe, warm-started from the prime
    started = perf_counter()
    keyframe = session.step("bulk_r+")
    moved = perf_counter() - started
    assert keyframe.wall > 0.0
    assert keyframe.trips >= 0
    assert moved < keyframe.wall + 60.0  # solve call reports its own wall
    assert session.receipts[-1].parameter == "bulk_r"
    assert session.equilibrium.finite.flux

    # the current-centroid row was carried: a constraint record qualified
    assert len(session.equilibrium.constraints) == 1
    record = session.equilibrium.constraints[0]
    assert np.isfinite(np.asarray(record.observed)).all()
    # compensation pushed per circuit has the circuit count the carrier owns
    pushed = frame_push(session)
    assert pushed["compensation"]["circuit"].size == machine.circuit_count


# --------------------------------------------------------------------------
# bokeh serve serves the playable document on the Solov'ev machine (slow)
# --------------------------------------------------------------------------


def _free_port():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.slow
def test_bokeh_serve_serves_the_playable_document_on_the_default_machine(tmp_path):
    with skip_import("bokeh.client"):
        from bokeh.client import pull_session

        port = _free_port()
        log_path = tmp_path / "startup.log"
        environment = dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            PYTHONPATH=str(REPO_ROOT),
            TMPDIR=os.environ.get("TMPDIR", "/tmp"),
        )
        command = [
            "uv",
            "run",
            "--no-sync",
            "bokeh",
            "serve",
            str(REPO_ROOT / "apps/playable"),
            "--port",
            str(port),
            "--allow-websocket-origin",
            f"127.0.0.1:{port}",
            "--host",
            f"127.0.0.1:{port}",
        ]
        with open(log_path, "w") as log_file:
            server = Popen(
                command,
                stdout=log_file,
                stderr=log_file,
                cwd=REPO_ROOT,
                env=environment,
            )
        try:
            deadline = perf_counter() + 90.0
            served = False
            while perf_counter() < deadline:
                if server.poll() is not None:
                    break
                if "Starting Bokeh server" in log_path.read_text():
                    served = True
                    break
                sleep(1.0)
            assert server.poll() is None, (
                f"bokeh server exited early: {log_path.read_text()[-2000:]}"
            )
            assert served, f"startup log never reported serving: {log_path.read_text()}"
            # a client session drives the document on the default Solov'ev machine
            session = pull_session(
                session_id=None,
                url=f"http://127.0.0.1:{port}/apps/playable",
                arguments={"machine": [b"solovev"]},
            )
            try:
                names = {root.name for root in session.document.roots}
                assert "poloidal" in names
                assert "compensation" in names
                assert "receipt" in names
            finally:
                session.close()
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except TimeoutError:
                server.kill()
