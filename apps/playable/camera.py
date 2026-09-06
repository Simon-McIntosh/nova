"""Camera panel seam: the FrameDecoder protocol and the placeholder decoder.

Nova never imports imas-ambix (the coupled-repository rule runs one way), so
the decoded-camera panel of the playable page is fed through a protocol the
app owns: a :class:`FrameDecoder` with ``decode(frame) -> DecodedFrame`` where
the result carries an RGB uint8 image array ``(height, width, 3)``, the decode
wall, and the decoder identity (checkpoint, VQ decoder, corpus digest) that
the steering receipt records beside each frame.  The app takes the decoder as
a dotted path at launch and loads it once per session on the same machine as
the solve; with no decoder given it uses the placeholder, a grey frame
carrying the frame index, so the layout, the keys and the recording work
before the decoder exists.

The placeholder paints the index of the frame being decoded (its decode
count), which equals the session's frame index while the loop decodes once per
keyframe, so the camera panel is readable from the first keyframe and the
recorded per-frame decode records carry the same identity every frame.

The camera panel and its readout strips bind :class:`ColumnDataSource`
channels through the figures declared here beside the channel sources, so a
session that pushes a frame never guesses a shape the renderer does not bind:

- ``camera`` — one ``image_rgba`` glyph on a single cell — ``image`` — (H, W, 4) uint8
- ``sparkline`` — line wall/trips per action — ``index``, ``wall``, ``trips``
  — (n,), (n,), (n,)

``command_legend_text`` renders the key legend with the commanded parameter
values; the record and playback strip over the recorded keyframes is laid out
by the app over :func:`sparkline_figure`.
"""

from __future__ import annotations

import importlib
from time import perf_counter
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import numpy as np
from bokeh.models import ColumnDataSource, LinearAxis, Range1d
from bokeh.plotting import figure

if TYPE_CHECKING:
    from nova.equilibrium.steering_frames import SteeringFrame

#: Placeholder camera image capacity.  The camera figure carries the camera's
#: own aspect, so the figure defaults name the same (height, width) the
#: placeholder paints, matching the placeholder default one-for-one.
PLACEHOLDER_HEIGHT = 480
PLACEHOLDER_WIDTH = 640

#: Grey placeholder face and the painted index ink.
PLACEHOLDER_FACE = (63, 63, 68)
PLACEHOLDER_INK = (235, 235, 235)

#: Sparkline ink: wall on the left axis, trips on the right.
WALL_COLOR = "#3f3f3f"
TRIPS_COLOR = "#a98fd0"


class DecodedFrame(NamedTuple):
    """One decoded camera frame and the latency that produced it."""

    image: object  #: RGB uint8 array of shape (height, width, 3)
    decode_wall: float  #: seconds inside the decode call
    decoder_identity: str  #: checkpoint, VQ decoder or corpus digest


@runtime_checkable
class FrameDecoder(Protocol):
    """Decode one steering frame into an RGB camera image.

    ``frame`` is the steering record the session publishes for the current
    keyframe; imas-ambix implements this protocol against that record and is
    loaded into the app by dotted path on the same machine as the solve.  A
    slow decode runs after the poloidal push, so it delays only the picture,
    and the returned wall and identity are recorded beside the frame.
    """

    decoder_identity: str

    def decode(self, frame: SteeringFrame) -> DecodedFrame: ...


class PlaceholderDecoder:
    """Grey placeholder that paints the frame index on every decoded frame.

    The index is the decoder's own decode count, which equals the session's
    frame index while the loop decodes once per keyframe, so the placeholder
    reads "frame 0001" on the prime, "frame 0002" on the first move and so on.
    """

    decoder_identity = "placeholder:frame-index"

    def __init__(
        self, *, height: int = PLACEHOLDER_HEIGHT, width: int = PLACEHOLDER_WIDTH
    ):
        self.height = int(height)
        self.width = int(width)
        self._count = 0

    def decode(self, frame) -> DecodedFrame:
        del frame
        started = perf_counter()
        image = _painted_index(self._count + 1, height=self.height, width=self.width)
        self._count += 1
        return DecodedFrame(image, perf_counter() - started, self.decoder_identity)


def _painted_index(index: int, *, height: int, width: int) -> np.ndarray:
    """Return the grey RGB frame with ``index`` painted at its centre."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (width, height), color=PLACEHOLDER_FACE)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (width // 2, height // 2),
        f"frame {index:04d}",
        fill=PLACEHOLDER_INK,
        anchor="mm",
    )
    return np.asarray(canvas, dtype=np.uint8)


def load_decoder(dotted_path: str | None) -> FrameDecoder:
    """Return the decoder a dotted path names, or the placeholder when absent.

    The dotted path is the module and attribute of a concrete
    :class:`FrameDecoder`, ``module.path.…:ClassName`` (a bare dotted name
    splits on the final dot), loaded once per session on the machine that runs
    the solve; with no path given the grey placeholder renders before any
    decoder exists.
    """
    if not dotted_path:
        return PlaceholderDecoder()
    dotted = str(dotted_path)
    module_name, separator, attribute = dotted.partition(":")
    if not separator or not attribute:
        module_name, _, attribute = dotted.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        decoder = getattr(module, attribute)()
    except AttributeError as error:
        raise ImportError(
            f"decoder dotted path {dotted!r} names no {attribute!r} in {module_name!r}"
        ) from error
    if not isinstance(decoder, FrameDecoder):
        raise TypeError(
            f"dotted path {dotted!r} names {type(decoder).__name__}, "
            "which does not implement the FrameDecoder protocol"
        )
    return decoder


# ---------------------------------------------------------------------------
# camera image: one image_rgba glyph on a single-cell source, no axes
# ---------------------------------------------------------------------------


def _rgba(rgb: np.ndarray) -> np.ndarray:
    """Return an (H, W, 4) uint8 RGBA view of an (H, W, 3) RGB frame."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"a decoded frame image must be (height, width, 3) RGB, got {rgb.shape}"
        )
    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    return np.dstack([rgb, alpha])


def camera_sources() -> dict[str, ColumnDataSource]:
    """Return the camera panel channel sources with their bound columns set."""
    blank = np.zeros((PLACEHOLDER_HEIGHT, PLACEHOLDER_WIDTH, 4), dtype=np.uint8)
    return {
        "camera": ColumnDataSource(data={"image": [blank]}),
        "sparkline": ColumnDataSource(data={"index": [], "wall": [], "trips": []}),
    }


def camera_figure(
    source: dict[str, ColumnDataSource] | None = None,
    *,
    height: int = PLACEHOLDER_HEIGHT,
    width: int = PLACEHOLDER_WIDTH,
    name: str = "camera",
) -> figure:
    """Return the decoded-camera panel in the camera's own aspect.

    One ``image_rgba`` glyph on a single-cell source, no axes, grid or frame,
    so the panel shows exactly the decoded image the loop pushes per keyframe.
    """
    camera_source = source.get("camera") if source is not None else None
    if camera_source is None:
        camera_source = ColumnDataSource(
            data={
                "image": [
                    np.zeros((PLACEHOLDER_HEIGHT, PLACEHOLDER_WIDTH, 4), dtype=np.uint8)
                ]
            }
        )
    camera = figure(name=name, width=width, height=height, toolbar_location=None)
    camera.axis.visible = False
    camera.grid.visible = False
    camera.outline_line_color = None
    camera.image_rgba("image", x=0, y=0, dw=width, dh=height, source=camera_source)
    return camera


# ---------------------------------------------------------------------------
# readout strip: keyframe wall and trips per action as a sparkline
# ---------------------------------------------------------------------------


def sparkline_figure(
    source: dict[str, ColumnDataSource] | None = None,
    *,
    height: int = 150,
    name: str = "sparkline",
) -> figure:
    """Return the per-action keyframe wall and trips sparkline.

    Wall rides the left axis; trips share the figure on their own right range,
    so a keyframe chain's time and its trip count read together along the same
    index.  The record and playback strip the page lays over it rides the same
    index.
    """
    sparkline_source = source.get("sparkline") if source is not None else None
    if sparkline_source is None:
        sparkline_source = ColumnDataSource(data={"index": [], "wall": [], "trips": []})
    chart = figure(
        name=name,
        height=height,
        toolbar_location=None,
        x_axis_label="keyframe",
        y_axis_label="wall / s",
    )
    chart.extra_y_ranges = {"trips": Range1d(start=0, end=8)}
    chart.add_layout(LinearAxis(y_range_name="trips", axis_label="trips"), "right")
    chart.line("index", "wall", source=sparkline_source, color=WALL_COLOR, width=1.5)
    chart.line(
        "index",
        "trips",
        source=sparkline_source,
        color=TRIPS_COLOR,
        width=1.5,
        y_range_name="trips",
    )
    return chart


# ---------------------------------------------------------------------------
# pushed camera channels and the key legend text
# ---------------------------------------------------------------------------


def camera_push(session, decoded) -> dict[str, dict[str, object]]:
    """Return the camera and sparkline columns for one pushed keyframe.

    ``decoded`` is the :class:`DecodedFrame` the session recorded for the
    frame, or ``None`` when the solve refused and the last good frame is held;
    a held frame pushes the sparkline only and leaves the camera cell showing
    the previous image, never an interpolated one.
    """
    channels: dict[str, dict[str, object]] = {}
    indices = np.arange(len(session.receipts), dtype=float)
    wall = np.asarray([receipt.wall for receipt in session.receipts], dtype=float)
    trips = np.asarray([receipt.trips for receipt in session.receipts], dtype=int)
    channels["sparkline"] = {"index": indices, "wall": wall, "trips": trips}
    if decoded is not None:
        channels["camera"] = {"image": [_rgba(decoded.image)]}
    return channels


def command_legend_text(shape) -> str:
    """Return the key legend with the commanded parameter values as HTML.

    Every control the key map steps is named with its current commanded value,
    so the legend reads the session's commanded set, not the static key help.
    """
    lines = ["<b>commanded</b> — R Z κ δu δl Xr Xz gap-in gap-out"]
    values = (
        f"R {shape.axis_r:.3f} m · Z {shape.axis_z:+.3f} m · "
        f"κ {shape.elongation:.2f} · δu {shape.triangularity_upper:+.2f} · "
        f"δl {shape.triangularity_lower:+.2f} · "
        f"Xr {shape.x_point_r:.3f} m · Xz {shape.x_point_z:+.3f} m · "
        f"gap-in {shape.inner_gap:.3f} m · gap-out {shape.outer_gap:.3f} m"
    )
    lines.append(values)
    return "<br>".join(lines)
