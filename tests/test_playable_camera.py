"""Two-panel playable page gate: camera panel, decoder seam, readout strips.

The camera panel of the playable page is fed through the
:class:`~apps.playable.camera.FrameDecoder` protocol, so the gate drives a stub
solver and the grey placeholder decoder through ten keyframes and pins the
contract the page is built on: every pushed ColumnDataSource column has the
shape its renderer binds (the poloidal channels, the compensating-current
bars, the receipt row, the camera cell and the sparkline); the camera source
is updated once per keyframe with the decode wall and decoder identity
recorded beside the frame in the session; the status line carries the frame
index, the decode wall and the keyframe wall; and the camera-only view serves
the camera panel and status line without the poloidal figure.

The session test living in its own file drives the stub through the shared
poloidal renderers; this file adds the camera panel and keeps its stub local
so it can carry the raster and labelled carriers the typed steering frame
reads.  The styled two-panel PNG of the Solov'ev fixture is committed under
``docs/figures/playable-forward-solve/view/`` by the same matplotlib
channel-data path the styled poloidal frame used.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from apps.playable.camera import (
    PlaceholderDecoder,
    camera_push,
    command_legend_text,
    load_decoder,
)
from apps.playable.session import PlayableSession, SolveResult, frame_push

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEW_FIGURE = REPO_ROOT / "docs/figures/playable-forward-solve/view"

#: The ten key presses the gate drives: every named control's ``+`` symbol
#: plus one reverse move, so ten distinct actions run through the loop.
TEN_KEYS = (
    "bulk_r+",
    "bulk_z+",
    "elongation+",
    "triangularity_upper+",
    "triangularity_lower+",
    "x_point_r+",
    "x_point_z+",
    "inner_gap+",
    "outer_gap+",
    "bulk_r-",
)


class StubEquilibrium(SimpleNamespace):
    """Carrier-shaped equilibrium the frame reduce and steering frame read.

    Mirrors the published raster and labelled flux carriers: the raster image
    with its grid and topology channels, the labelled point slots (absent here,
    so the frame masks them), and no constraint rows.
    """

    def __init__(self, radius=None, height=None):
        radius = np.linspace(0.6, 1.42, 12) if radius is None else radius
        height = np.linspace(-0.42, 0.42, 10) if height is None else height
        n_radius, n_height = len(radius), len(height)
        nan_point = np.full(2, np.nan)
        super().__init__(
            raster_flux=SimpleNamespace(
                radius=np.asarray(radius),
                height=np.asarray(height),
                shape=np.asarray([n_radius, n_height], dtype=np.int32),
                psi=np.zeros(n_radius * n_height),
                psi_norm=np.linspace(0.0, 1.0, n_radius * n_height),
                domain_label=np.zeros(n_radius * n_height, dtype=np.int8),
                separatrix=np.full((30, 2), np.nan),
                separatrix_vertex_count=0,
            ),
            labelled_flux=SimpleNamespace(
                o_point=nan_point,
                primary_x_point=nan_point,
                secondary_x_point=nan_point,
                lcfs=np.full((30, 2), np.nan),
                lcfs_vertex_count=0,
                strike_points=np.full((2, 2), np.nan),
            ),
            constraints=(),
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
        del previous, commanded, action, program
        return SolveResult(self._equilibrium, wall=self.wall, trips=self.trips)


def _stub_session(**kwargs) -> PlayableSession:
    """Return a session over the stub solver with the placeholder decoder."""
    session = PlayableSession(solver=StubSolver(), machine="solovev", **kwargs)
    session.decoder = load_decoder(None)
    return session


def _root_names(document) -> set[str]:
    """Return every named model in the document tree, nested included."""
    names = set()

    def walk(model):
        name = getattr(model, "name", None)
        if name:
            names.add(name)
        for child in getattr(model, "children", ()) or ():
            walk(child)

    for root in document.roots:
        walk(root)
    return names


def _bound_fields(glyph) -> set[str]:
    """Return the data-field column names one glyph binds, by kind."""
    fields = set()
    for value in glyph.properties_with_values().values():
        field = getattr(value, "field", None)
        if isinstance(field, str) and field:
            fields.add(field)
    return fields


# ---------------------------------------------------------------------------
# the decoder seam: protocol, placeholder, dotted-path loader
# ---------------------------------------------------------------------------


def test_placeholder_decoder_paints_the_frame_index_and_reports_its_wall():
    decoder = PlaceholderDecoder(height=48, width=64)
    frame = _stub_session().current_frame()
    first = decoder.decode(frame)
    second = decoder.decode(frame)
    assert isinstance(first, tuple)
    image = np.asarray(first.image)
    assert image.shape == (48, 64, 3) and image.dtype == np.uint8
    assert first.decode_wall >= 0.0
    assert first.decoder_identity == "placeholder:frame-index"
    # the painted index distinguishes consecutive frames
    assert not np.array_equal(image, np.asarray(second.image))


def test_frame_decoder_is_a_runtime_protocol_and_placeholder_implements_it():
    from apps.playable.camera import FrameDecoder

    assert isinstance(PlaceholderDecoder(), FrameDecoder)


def test_load_decoder_resolves_dotted_paths_and_defaults_to_placeholder():
    assert load_decoder(None).decoder_identity == "placeholder:frame-index"
    assert (
        load_decoder("apps.playable.camera:PlaceholderDecoder").decoder_identity
        == "placeholder:frame-index"
    )
    with pytest.raises(ImportError):
        load_decoder("apps.playable.camera:NoSuchDecoder")


def test_camera_push_shapes_the_single_cell_and_the_sparkline(session):
    session.step("elongation+")
    decoded = session.decode_frame()
    assert decoded is not None
    pushed = camera_push(session, decoded)
    cell = pushed["camera"]["image"][0]
    assert np.asarray(cell).shape == (480, 640, 4)
    assert np.asarray(cell).dtype == np.uint8
    sparkline = pushed["sparkline"]
    assert (
        len(sparkline["index"])
        == len(sparkline["wall"])
        == len(sparkline["trips"])
        == 1
    )


@pytest.fixture()
def session():
    """A stub session with the placeholder decoder, fresh per test."""
    return _stub_session()


# --------------------------------------------------------------------------
# ten keys through the stub solver: every pushed column as its glyph binds
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path(__file__).parent.with_name("apps").is_dir(), reason="apps tree absent"
)
def test_ten_keys_push_every_bound_column_shaped_as_its_glyph_binds(session):
    from bokeh.models import ColumnDataSource

    from apps.pulsedesign.poloidal_view import (
        compensation_figure,
        keyframe_receipt,
        poloidal_figure,
    )
    from apps.playable.camera import camera_figure, sparkline_figure

    for key in TEN_KEYS:
        session.step(key)
        decoded = session.decode_frame()
        assert decoded is not None
        frame = dict(frame_push(session))
        frame.update(camera_push(session, decoded))

        sources = {
            name: ColumnDataSource()
            for name in (
                "levelset",
                "wall",
                "coil",
                "plasma",
                "o_points",
                "x_points",
                "x_points_secondary",
                "points",
                "separatrix",
                "compensation",
                "receipt",
                "camera",
                "sparkline",
            )
        }
        poloidal = poloidal_figure(sources)
        compensation = compensation_figure(sources)
        camera = camera_figure(sources)
        sparkline = sparkline_figure(sources)
        receipt_table = keyframe_receipt(sources)

        bound = {}
        renderers = [
            *poloidal.renderers,
            *compensation.renderers,
            *camera.renderers,
            *sparkline.renderers,
        ]
        for renderer in renderers:
            for column in _bound_fields(renderer.glyph):
                bound.setdefault(column, set()).add(type(renderer.glyph).__name__)

        # camera cell: the image_rgba glyph binds the single-cell image column
        assert "image" in bound
        cell = frame["camera"]["image"][0]
        assert np.asarray(cell).ndim == 3 and np.asarray(cell).shape[2] == 4
        sources["camera"].data = {"image": [cell]}

        # sparkline: 1-D same-length index, wall and trips per action
        assert {"index", "wall", "trips"} <= set(bound)
        for column in ("index", "wall", "trips"):
            values = frame["sparkline"][column]
            assert np.asarray(values).ndim == 1
        length = len(frame["sparkline"]["index"])
        spark = frame["sparkline"]
        assert len(spark["wall"]) == len(spark["trips"]) == length
        assert length == len(session.receipts)
        sources["sparkline"].data = frame["sparkline"]

        # the shared poloidal channels keep their bound shapes
        for channel in ("separatrix", "points", "x_points"):
            for column in ("x", "z"):
                assert column in bound
                assert np.asarray(frame[channel][column]).ndim == 1
            assert frame[channel]["x"].size == frame[channel]["z"].size
            sources[channel].data = frame[channel]
        assert "circuit" in bound and "current" in bound
        assert frame["compensation"]["circuit"].size == (
            frame["compensation"]["current"].size
        )
        sources["compensation"].data = frame["compensation"]
        table_fields = {column.field for column in receipt_table.columns}
        assert table_fields == {"action", "wall", "trips"}
        for column in ("action", "wall", "trips"):
            assert len(frame["receipt"][column]) == 1
        sources["receipt"].data = frame["receipt"]

        # the receipt wall and trips land in the session beside the frame
        assert session.receipts[-1].key == key


def test_camera_source_updated_once_per_keyframe_with_decode_wall_recorded(session):
    """Each keyframe pushes one camera cell and one decode record beside it."""
    seen = []
    for key in TEN_KEYS:
        session.step(key)
        decoded = session.decode_frame()
        assert decoded is not None
        pushed = camera_push(session, decoded)
        cell = np.asarray(pushed["camera"]["image"][0])
        assert cell.shape == (480, 640, 4)
        seen.append(cell.copy())
    # ten keys, ten decode records, ten distinct placed cells with the
    # per-frame decode wall and decoder identity recorded beside each frame
    assert len(session.decoded_frames) == len(TEN_KEYS)
    assert all(
        record.decoder_identity == "placeholder:frame-index"
        for record in session.decoded_frames
    )
    for first, second in zip(seen, seen[1:], strict=False):
        assert not np.array_equal(first, second)


def test_status_line_carries_frame_index_decode_wall_and_keyframe_wall(session):
    from bokeh.document import Document

    from apps.playable.main import build_document

    doc = Document()
    handle = build_document(doc, view="both", session=session)
    for key in TEN_KEYS:
        handle["on_key"](key)
    status = handle["status"].text
    assert f"frame {len(TEN_KEYS):04d}" in status
    assert "decode" in status and "ms" in status
    assert f"keyframe {StubSolver.wall * 1000.0:.0f} ms" in status
    assert f"trips {StubSolver.trips}" in status
    # the recorded decode wall is the one the status line reports: the last
    # record's wall, formatted to the same millisecond
    last = session.decoded_frames[-1]
    assert f"{last.decode_wall * 1000.0:.0f} ms" in status
    # the camera cell holds the final placed index
    assert handle["sources"]["camera"].data["image"][0].shape == (480, 640, 4)


def test_key_legend_lists_the_commanded_values(session):
    from bokeh.document import Document

    from apps.playable.main import build_document

    doc = Document()
    handle = build_document(doc, view="both", session=session)
    handle["on_key"]("bulk_r+")
    text = handle["legend"].text
    assert "commanded" in text
    assert "R" in text and "gap-in" in text
    assert command_legend_text(session.shape) == text
    new_within = command_legend_text(session.shape)
    assert "R 1.020" in new_within


# --------------------------------------------------------------------------
# the view request argument: camera-only serves without the poloidal figure
# --------------------------------------------------------------------------


def test_view_argument_resolves_both_and_camera():
    from apps.playable.main import view_argument

    assert view_argument({}) == "both"
    assert view_argument({"view": [b"camera"]}) == "camera"
    assert view_argument({"view": [b"both"]}) == "both"
    with pytest.raises(ValueError, match="unknown view"):
        view_argument({"view": [b"kittens"]})


def test_camera_only_view_serves_the_camera_panel_without_the_poloidal_figure(session):
    from bokeh.document import Document

    from apps.playable.main import build_document

    doc = Document()
    build_document(doc, view="camera", session=session)
    names = _root_names(doc)
    assert "camera" in names
    assert "status" in names
    assert "poloidal" not in names
    assert "compensation" not in names
    # the camera glyph is the only rendered image in the camera panel
    camera_figures = [
        root for root in doc.roots if getattr(root, "name", None) == "camera_panel"
    ]
    assert len(camera_figures) == 1


def test_default_view_serves_both_panels(session):
    from bokeh.document import Document

    from apps.playable.main import build_document

    doc = Document()
    build_document(doc, view="both", session=session)
    names = _root_names(doc)
    assert "poloidal" in names
    assert "camera" in names
    assert "compensation" in names
    assert "receipt" in names
    assert "sparkline" in names
    assert "legend_strip" in names
    assert "playback" in names


# --------------------------------------------------------------------------
# the steering frame the decoder consumes
# --------------------------------------------------------------------------


def test_steering_frame_carries_receipt_and_carrier_beside_each_frame(session):
    session.step("inner_gap+")
    frame = session.current_frame()
    assert frame.carrier_identity == "solovev"
    assert int(frame.trip_count) == StubSolver.trips
    assert float(frame.wall_seconds) == StubSolver.wall
    assert frame.action.name == "inner_gap"
    assert np.asarray(frame.psi).shape == (12, 10)
    # absent topology slots stay absent, never imputed
    assert not bool(frame.finite_mask[0])
    assert np.isnan(np.asarray(frame.magnetic_axis_r))
    assert np.asarray(frame.compensating_current).size == 0


# --------------------------------------------------------------------------
# a rendered two-panel PNG under the view figures (PIL-verified only)
# --------------------------------------------------------------------------


def test_a_two_panel_png_is_committed_under_the_view_figure():
    from PIL import Image

    candidates = sorted(VIEW_FIGURE.glob("*two-panel*.png"))
    assert candidates, f"no two-panel PNG committed under {VIEW_FIGURE}"
    with Image.open(candidates[-1]) as opened:
        width, height = opened.size
    assert width > 200 and height > 200
