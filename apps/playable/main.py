"""Playable forward solve: keyboard-steered keyframes, poloidal view and camera.

The document holds one :class:`~apps.playable.session.PlayableSession` over
the carrier the ``machine`` session argument selects (the Solov'ev machine by
default, the MAST frozen-six carrier when ``machine=mast``).  Key presses on
the focused poloidal view step one control parameter, warm re-solve, and push
the styled poloidal channels — the clipped plasma cells, unfilled contours,
wall and coil outlines, and the topology markers — plus the compensating
currents and the keyframe receipt row.

Beside the poloidal figure the document carries the decoded-camera panel: one
``image_rgba`` glyph on a single-cell source at the camera's own aspect, fed
through the :class:`~apps.playable.camera.FrameDecoder` protocol.  The decoder
is named by a dotted path in the ``decoder`` session argument and loaded once
per session (the grey placeholder painting the frame index when no decoder is
given).  Per keyframe the decode runs after the poloidal push, so a slow
decode delays only the picture, and the decode wall and decoder identity are
recorded beside the frame in the session.  A status line under the camera
carries the frame index, the decode wall, the keyframe wall and a HELD marker
when the solve refuses, so the last good frame stays shown and is never
interpolated.  Two readout strips sit beneath: the key legend with the
commanded values and the compensating currents per circuit as thin bars, and
the keyframe wall and trips per action as a sparkline with a record and
playback strip.

The same document serves under a ``view`` request argument: the default shows
both panels; ``view=camera`` renders the camera panel and its status line
alone, so a second browser window on another screen shows the picture while
the first carries the controls, both reading the one server-side session.
"""

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, CustomJS, Div, Slider

from apps.playable.camera import (
    camera_figure,
    camera_push,
    camera_sources,
    command_legend_text,
    load_decoder,
    sparkline_figure,
)
from apps.playable.machines import (
    MachineUnavailable,
    build_session,
    machine_argument,
    machine_coil_outlines,
)
from apps.playable.session import frame_push
from apps.playable.shape import key_help
from apps.pulsedesign.poloidal_view import (
    channel_sources,
    close_outline,
    compensation_figure,
    keyframe_receipt,
    poloidal_channels,
    poloidal_figure,
)

#: View layouts the ``view`` request argument may select.
VIEWS = ("both", "camera")

#: Install a window-level keydown listener once the document is ready; every
#: key press bumps the channel source with the released key so the server-side
#: callback re-solves and pushes the next keyframe.
KEY_LISTENER = """
const handler = (event) => {
  if (event.repeat) { return; }
  src.data = {key: [event.key]};
};
document.addEventListener('keydown', handler);
"""


def view_argument(arguments, default: str = "both") -> str:
    """Resolve the ``view`` session argument from Bokeh request arguments.

    The default document carries both panels; ``view=camera`` renders the
    camera panel and its status line alone for a second window.
    """
    values = arguments.get("view")
    if not values:
        return default
    value = values[0].decode() if isinstance(values[0], bytes) else str(values[0])
    if value not in VIEWS:
        raise ValueError(f"unknown view {value!r}; choose from {VIEWS}")
    return value


def decoder_argument(arguments, default: str | None = None) -> str | None:
    """Resolve the ``decoder`` dotted path from Bokeh request arguments.

    A missing argument loads the grey placeholder; a dotted path names a
    concrete :class:`~apps.playable.camera.FrameDecoder` loaded once per
    session on the machine that runs the solve.
    """
    values = arguments.get("decoder")
    if not values:
        return default
    value = values[0].decode() if isinstance(values[0], bytes) else str(values[0])
    return value or default


def make_sources():
    """Return the shared channel sources with their bound columns initialised.

    Pre-binding every column a renderer binds keeps the served document free
    of BAD_COLUMN_NAME warnings before the first keyframe.
    """
    sources = channel_sources()
    sources.update(camera_sources())
    return sources


def wire_keyboard(doc, session, sources, status, on_key):
    """Return the server callback that steps one key and pushes a frame."""
    keys = ColumnDataSource(data={"key": [""]})
    install = CustomJS(args={"src": keys}, code=KEY_LISTENER)
    doc.js_on_event("document_ready", install)

    def key_handler(attr, old, new):
        del attr, old
        key = str(new["key"][0])
        keys.data = {"key": [""]}  # consume the press
        if not key:
            return
        try:
            on_key(key)
        except Exception as error:  # keep the document alive across a bad key
            status.text = f"<b>error:</b> {type(error).__name__}: {error}"

    keys.on_change("data", key_handler)
    return key_handler


def _solved(session) -> bool:
    """Return whether the session's latest solve converged.

    A stub carrier without a fixed-point record reads as converged; a real
    equilibrium's refusal flips the camera to HELD with the last good frame.
    """
    fixed_point = getattr(getattr(session, "equilibrium", None), "fixed_point", None)
    return bool(getattr(fixed_point, "converged", True))


def build_document(
    doc,
    *,
    machine: str = "solovev",
    decoder: str | None = None,
    view: str = "both",
    session=None,
):
    """Populate the document with one session, the poloidal view and camera.

    ``session`` overrides the machine-built session (used by the tests to
    drive a stub solver); otherwise the session is built from the carrier the
    ``machine`` argument selects and the decoder is loaded once, by dotted
    path or as the placeholder.  Returns a small handle naming the pushed
    sources, the status and legend lines and the key handler the tests drive.
    """
    if session is None:
        try:
            session = build_session(machine)
        except MachineUnavailable as error:
            doc.add_root(column(Div(text=f"<pre>{error}</pre>"), name="error"))
            return
    session.decoder = load_decoder(decoder)

    sources = make_sources()
    machine_carrier = getattr(getattr(session, "solver", None), "machine", None)

    if session.wall is not None:
        wall = session.wall
        closed = close_outline(wall)
        sources["wall"].data = {"x": closed[:, 0], "z": closed[:, 1]}
        # The coil channel is the machine's own conductor outlines beside the
        # wall, not a decorative ring; an unreachable outline source yields an
        # empty channel rather than an error document.
        try:
            coil_outlines = machine_coil_outlines(machine, machine_carrier)
        except MachineUnavailable:
            coil_outlines = ()
    else:
        coil_outlines = ()

    poloidal = poloidal_figure(sources)
    camera = camera_figure(sources)
    compensation = compensation_figure(sources)
    receipt = keyframe_receipt(sources)
    sparkline = sparkline_figure(sources)

    status = Div(
        text="<b>ready</b> — press a key to solve the next keyframe", name="status"
    )
    legend = Div(text=command_legend_text(session.shape), name="legend")
    help_text = Div(text=key_help())

    # The record and playback strip over the recorded receipts: the record
    # toggle, the live frame counter, the playback slider positioning over the
    # session's keyframes, and the export button (the netCDF session write of
    # the recorded run arrives with the frame-schema followup).
    record = Button(label="record ●", button_type="primary", width=110)
    counter = Div(text=f"session frame {session.frame_index:04d} / 0000")
    playback_slider = Slider(
        start=0, end=0, step=1, value=0, width=220, title="playback"
    )
    export = Button(label="export", width=90)

    def toggle_record():
        session.recording = not session.recording
        record.label = "recording" if session.recording else "record ●"

    def set_playback(attr, old, new):
        del attr, old
        counter.text = f"session frame {int(new):04d} / {session.frame_index:04d}"

    def export_session():
        # The recorded run's netCDF session write is the frame-schema
        # followup's; the export button lands alongside it.
        status.text = (
            "export writes the recorded session with the frame-schema "
            "followup (§5); the receipts are held in the session until then"
        )

    record.on_click(toggle_record)
    playback_slider.on_change("value", set_playback)
    export.on_click(export_session)

    playback = column(
        sparkline, row(record, counter, playback_slider, export), name="playback"
    )

    def on_key(key):
        receipt_row = session.step(key)
        # Push the poloidal channels first: a moved key resets the picture the
        # physics owns before the camera decode starts.  The session also
        # publishes the filled flux image the styled view does not bind, so a
        # channel with no bound source is skipped rather than raised on.
        for name, data in frame_push(session).items():
            if name in sources:
                sources[name].data = data
        if machine_carrier is not None:
            channels = poloidal_channels(
                session.equilibrium,
                machine_carrier.profile,
                wall=session.wall,
                coils=coil_outlines,
            )
            for name, data in channels.items():
                sources[name].data = data

        # The decode runs after the poloidal push, so a slow decode delays
        # only the picture.  When the solve refuses, the camera keeps the last
        # good frame (nothing is pushed) and the status line says HELD.
        held = not _solved(session)
        decoded = None if held else session.decode_frame()
        for name, data in camera_push(session, decoded).items():
            sources[name].data = data

        moved = (
            f" {receipt_row.parameter} {receipt_row.delta:+.4g}"
            if receipt_row.parameter is not None
            else ""
        )
        decode_ms = "—" if decoded is None else f"{decoded.decode_wall * 1000.0:.0f} ms"
        status.text = (
            f"<b>frame {session.frame_index:04d}</b>{moved}"
            f" · decode {decode_ms}"
            f" · keyframe {receipt_row.wall * 1000.0:.0f} ms"
            f" · trips {receipt_row.trips}"
        )
        if held:
            status.text += " · <b>HELD</b> — last good frame shown"
        legend.text = command_legend_text(session.shape)
        playback_slider.end = max(1, session.frame_index)
        playback_slider.value = session.frame_index
        counter.text = (
            f"session frame {session.frame_index:04d} / {session.frame_index:04d}"
        )

    # The first key press is the prime: the solver warms from the machine's
    # seed, so the document itself never pays a startup solve and the first
    # keyframe is what establishes the initial view.
    sources["points"].data = {
        "x": session.shape.control_points()[0],
        "z": session.shape.control_points()[1],
    }

    camera_panel = column(camera, status, name="camera_panel")
    legend_strip = column(legend, name="legend_strip")
    # The compensating-current bars and the keyframe receipt row stay a root
    # of their own beside the strips, as in the pre-camera layout.
    receipts = column(compensation, receipt, name="receipts")

    if view == "camera":
        # A second window on another screen: the picture and its status line
        # alone, reading the same server-side session holder.
        doc.add_root(camera_panel)
    else:
        panels = row(poloidal, camera_panel, name="panels")
        readouts = row(legend_strip, playback, name="readouts")
        for root in (
            help_text,
            panels,
            readouts,
            receipts,
        ):
            doc.add_root(root)

    wire_keyboard(doc, session, sources, status, on_key)
    return {"sources": sources, "status": status, "legend": legend, "on_key": on_key}


def main():
    """Serve the app with the carrier, decoder and view the arguments select."""
    context = curdoc().session_context
    if context is None:
        # Imported outside a served session (tests import the document
        # builder directly); with no session there is nothing to populate and
        # no carrier to build.
        return
    arguments = context.request.arguments
    machine = machine_argument(arguments)
    view = view_argument(arguments)
    decoder = decoder_argument(arguments)
    build_document(curdoc(), machine=machine, decoder=decoder, view=view)


main()
