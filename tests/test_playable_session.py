"""Playable-forward-solve session gate.

Drives the session holder with a stub solver on CPU and pins the contract the
interactive keyframe loop is built on: every named key produces exactly the
commanded control-point change it names; every pushed ColumnDataSource column
has the shape its renderer in ``apps/pulsedesign/poloidal_view.py`` binds; the
same assembled SteeringFrame reaches rendering, decoding and recording; the
recorded frames round-trip through the session store; the receipt row per
action carries wall and trips; and one keyframe through the production
inverse-forward protocol completes on the small Solov'ev fixture.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from subprocess import Popen
from time import perf_counter, sleep
from types import SimpleNamespace

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import
from nova.equilibrium.steering_frames import frames_from_session, read_session

from apps.playable.session import (
    PlayableSession,
    SolveResult,
    equilibrium_frame_receipt,
    frame_push,
)
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
        radial_count, vertical_count = radius.size, height.size
        count = radial_count * vertical_count
        separatrix = np.full((30, 2), np.nan)
        separatrix[:5] = np.array(
            [
                [0.8, -0.2],
                [1.2, -0.2],
                [1.2, 0.2],
                [0.8, 0.2],
                [0.8, -0.2],
            ]
        )
        lcfs = np.full((12, 2), np.nan)
        lcfs[:5] = separatrix[:5]
        history = SimpleNamespace(
            residual=np.asarray(0.0),
            trace=np.asarray([0.0]),
            converged=np.asarray(True),
            termination_reason=np.asarray(0, dtype=np.int32),
            active_set_iterations=np.asarray(StubSolver.trips, dtype=np.int32),
            active_set_residuals=np.asarray([0.0]),
            active_set_mask_differences=np.asarray([0], dtype=np.int32),
            shadow_mask_changes=np.asarray([0], dtype=np.int32),
            inner_iteration_decisions=np.asarray([], dtype=np.int32),
            inner_iteration_applied_factors=np.asarray([], dtype=float),
        )
        super().__init__(
            raster_flux=SimpleNamespace(
                radius=np.asarray(radius),
                height=np.asarray(height),
                shape=np.asarray([radial_count, vertical_count], dtype=np.int32),
                psi=np.arange(count, dtype=float),
                psi_norm=np.linspace(0.0, 1.0, count),
                domain_label=np.zeros(count, dtype=np.int8),
                separatrix=separatrix,
                separatrix_vertex_count=np.int32(5),
            ),
            labelled_flux=SimpleNamespace(
                o_point=np.asarray([1.0, 0.0]),
                primary_x_point=np.asarray([1.0, -0.2]),
                secondary_x_point=np.full(2, np.nan),
                strike_points=np.full((2, 2), np.nan),
                lcfs=lcfs,
                lcfs_vertex_count=np.int32(5),
            ),
            constraints=(),
            circuits=circuits,
            coil_current=np.zeros(circuits, dtype=float),
            cell_current=np.ones(count, dtype=float),
            fixed_point=history,
            finite=SimpleNamespace(passed=True, flux=True),
            normalisation=SimpleNamespace(amplitude=np.asarray(1.0)),
            topology=SimpleNamespace(
                axis=np.asarray([1.0, 0.0]),
                axis_flux=np.asarray(0.0),
                boundary_flux=np.asarray(1.0),
                flux_span=np.asarray(1.0),
                diverted=False,
            ),
            flux=np.zeros(count, dtype=float),
        )


class StubSolver:
    """Return the same equilibrium without solving, at a stated wall and trips."""

    wall = 0.234
    trips = 3

    def __init__(self, equilibrium=None):
        self._equilibrium = (
            equilibrium if equilibrium is not None else StubEquilibrium()
        )

    def __call__(self, previous, commanded, *, action=None, program=None):
        del previous, action, program
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
    assert session.recorded_frames == []
    receipt = session.receipts[-1]
    assert receipt.wall == StubSolver.wall
    assert receipt.trips == StubSolver.trips
    pushed = frame_push(session)
    row = pushed["receipt"]
    assert len(row["action"]) == 1
    assert row["wall"][0] == StubSolver.wall
    assert row["trips"][0] == StubSolver.trips


def _assert_push_equal(expected, actual):
    """Require every renderer column to survive a frame-store round trip."""
    assert actual.keys() == expected.keys()
    for channel, columns in expected.items():
        assert actual[channel].keys() == columns.keys()
        for name, values in columns.items():
            left = np.asarray(values)
            right = np.asarray(actual[channel][name])
            assert right.shape == left.shape, f"{channel}.{name} shape changed"
            if left.dtype.kind in "fc":
                assert np.array_equal(left, right, equal_nan=True), (
                    f"{channel}.{name} changed"
                )
            else:
                assert np.array_equal(left, right), f"{channel}.{name} changed"


def test_ten_recorded_keyframes_round_trip_the_pushed_channels(tmp_path):
    """Ten keyframes record the exact frame reduced into renderer channels."""
    session = _stub_session(recording=True)
    driven_keys = list(session.keys)[:10]
    expected_pushes = []
    for key in driven_keys:
        receipt = session.step(key)
        assert receipt.frame_assembly_wall >= 0.0
        expected_pushes.append(frame_push(session))

    assert len(session.recorded_frames) == 10
    assert session.frame_assembly_routes == ["frame-builder"] * 10
    assert len(session.frame_assembly_walls) == 10
    print(
        "frame_assembly_wall_seconds_cpu="
        + ",".join(f"{wall:.9f}" for wall in session.frame_assembly_walls)
    )

    session.write_recording(filename="playable", dirname=str(tmp_path))
    dataset = read_session(filename="playable", dirname=str(tmp_path))
    restored_frames = frames_from_session(dataset)
    assert len(restored_frames) == len(expected_pushes)
    for expected, restored in zip(expected_pushes, restored_frames, strict=True):
        _assert_push_equal(expected, frame_push(session, frame=restored))


def test_typed_solve_receipt_uses_the_direct_frame_branch():
    """A solver returning a typed receipt bypasses the equilibrium adapter."""

    class ReceiptSolver(StubSolver):
        def __call__(self, previous, commanded, *, action=None, program=None):
            result = super().__call__(
                previous, commanded, action=action, program=program
            )
            receipt = equilibrium_frame_receipt(
                result.equilibrium, wall_seconds=result.wall
            )
            return SolveResult(receipt, result.wall, result.trips)

    def refused_builder(equilibrium, *, wall_seconds):
        del equilibrium, wall_seconds
        raise AssertionError("the direct receipt path must not call the builder")

    session = PlayableSession(
        solver=ReceiptSolver(),
        frame_builder=refused_builder,
    )
    session.step("bulk_z+")
    assert session.frame_assembly_routes == ["receipt"]
    assert session.current_frame() is session.frame


def test_decoder_receives_the_same_frame_the_session_pushes():
    """Rendering and decoding share one assembled SteeringFrame object."""
    from apps.playable.camera import DecodedFrame

    class RecordingDecoder:
        decoder_identity = "test:recording-decoder"
        received = None

        def decode(self, frame):
            self.received = frame
            return DecodedFrame(
                image=np.zeros((2, 2, 3), dtype=np.uint8),
                decode_wall=0.001,
                decoder_identity=self.decoder_identity,
            )

    decoder = RecordingDecoder()
    session = _stub_session(decoder=decoder)
    session.step("outer_gap+")
    session.decode_frame()
    assert decoder.received is session.current_frame()


def test_document_pushes_ten_assembled_frames_to_bound_sources():
    """The playable callback reduces each assembled frame into live sources."""
    with skip_import("bokeh.document"):
        from bokeh.document import Document

        from apps.playable.main import build_document

        session = _stub_session(recording=True)
        handle = build_document(Document(), session=session)
        driven_keys = list(session.keys)[:10]
        for key in driven_keys:
            handle["on_key"](key)

        assert session.frame_index == 10
        assert len(session.recorded_frames) == 10
        expected = frame_push(session)
        assert expected.keys() <= handle["sources"].keys()
        for channel, columns in expected.items():
            actual = handle["sources"][channel].data
            assert actual.keys() == columns.keys()
            for name, values in columns.items():
                left = np.asarray(values)
                right = np.asarray(actual[name])
                assert right.shape == left.shape, f"{channel}.{name} shape changed"
        assert len(handle["sources"]["camera"].data["image"]) == 1


# --------------------------------------------------------------------------
# pushed columns match the shapes their renderers bind
# --------------------------------------------------------------------------


with skip_import("bokeh"):
    from bokeh.models import ColumnDataSource

    from apps.pulsedesign.poloidal_view import (
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
            "coil",
            "x_points",
            "x_points_secondary",
            "o_points",
            "plasma",
            "points",
            "separatrix",
            "compensation",
            "receipt",
        )
    }
    poloidal = poloidal_figure(sources)
    add_separatrix(poloidal, sources)
    compensation = compensation_figure(sources)
    receipt = keyframe_receipt(sources)

    bound = {}
    for renderer in [*poloidal.renderers, *compensation.renderers]:
        for column, kind in _bound_fields(renderer.glyph):
            bound.setdefault(column, set()).add(kind)

    # separatrix, control points and topology markers: 1-D paired coordinates
    for channel in (
        "separatrix",
        "points",
        "o_points",
        "x_points",
        "x_points_secondary",
    ):
        for column in ("x", "z"):
            assert column in bound, f"{channel} renderer must bind {column}"
            values = frame[channel][column]
            assert values.ndim == 1
        assert frame[channel]["x"].size == frame[channel]["z"].size
        sources[channel].data = frame[channel]

    # nested surfaces are fixed-shape frame fields reduced to paired lines.
    assert len(frame["levelset"]["x"]) == len(frame["levelset"]["z"])
    for radial, vertical in zip(
        frame["levelset"]["x"], frame["levelset"]["z"], strict=True
    ):
        assert len(radial) == len(vertical)
    sources["levelset"].data = frame["levelset"]

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
    assert solver.route == "reduced_newton"
    session = PlayableSession(solver=solver, shape=PlasmaShape(), machine="solovev")

    # The prime converges on the Solov'ev machine from the seed.
    prime = session.prime()
    assert prime.wall > 0.0 and isinstance(prime.trips, int) and prime.trips >= 0
    assert prime.reused is False
    assert session.program is not None
    assert session.equilibrium is not None
    assert bool(session.equilibrium.fixed_point.converged)
    assert session.equilibrium.finite.passed or session.equilibrium.finite.flux

    centroid = np.asarray(
        machine.profile.current_moment_observation(
            session.equilibrium.flux, support=MomentIntegralSupport.ALL_DOMAIN
        ).stack()
    )
    assert np.all(np.isfinite(centroid))

    # One moved keyframe solves all circuit currents, then runs the reduced
    # forward route without shape constraint pairs. A changed prescribed
    # current builds its own reduced program until that input is traced.
    keyframe = session.step("bulk_r+")
    assert keyframe.wall > 0.0
    assert keyframe.wall < 60.0
    assert keyframe.trips >= 0
    assert keyframe.reused is False
    assert session.program is not None
    assert session.receipts[-1].parameter == "bulk_r"
    assert session.equilibrium.finite.flux
    assert 1 <= len(solver.last_rounds) <= 2
    assert len(session.equilibrium.constraints) == 0

    # The reverse key executes the same bounded inverse-forward path.
    settled = session.step("bulk_r-")
    assert settled.reused is False
    assert settled.wall < 120.0
    assert session.equilibrium.finite.flux
    assert len(session.equilibrium.constraints) == 0

    # No constraint-row compensation is published on the control path.
    pushed = frame_push(session)
    assert pushed["compensation"]["circuit"].size == 0


@pytest.mark.slow
def test_newton_krylov_route_stays_reachable_as_the_reference(machine):
    from apps.playable.production import ProductionSolver
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    reference = ProductionSolver(machine, route="newton_krylov")
    assert reference.route == "newton_krylov"
    np.testing.assert_allclose(
        reference.prescribed_current,
        machine.profile.operator.prescribed_current_field.current,
    )


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
            sys.executable,
            "-m",
            "bokeh",
            "serve",
            str(REPO_ROOT / "apps/playable"),
            "--port",
            str(port),
            "--allow-websocket-origin",
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
                session_id=None, url=f"http://127.0.0.1:{port}/playable"
            )
            try:
                names = {root.name for root in session.document.roots}
                assert "panels" in names
                panels = next(
                    root for root in session.document.roots if root.name == "panels"
                )
                panel_names = {
                    child.name for child in panels.children if child.name is not None
                }
                assert {"poloidal", "camera_panel"} <= panel_names
                # the compensation chart and the keyframe receipt table live
                # inside the named receipts column, bound to the same sources
                nested = set()
                for root in session.document.roots:
                    if root.name != "receipts":
                        continue
                    for child in root.children:
                        if getattr(child, "name", None):
                            nested.add(child.name)
                        for sub in getattr(child, "children", ()) or ():
                            if getattr(sub, "name", None):
                                nested.add(sub.name)
                assert "compensation" in nested, f"receipts nested: {nested}"
                assert "receipt" in nested
            finally:
                session.close()
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except TimeoutError:
                server.kill()
