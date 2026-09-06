"""Playable forward solve: keyboard-steered keyframes behind the poloidal view.

The document holds one :class:`~apps.playable.session.PlayableSession` over
the carrier the ``machine`` session argument selects (the Solov'ev machine by
default, the MAST frozen-six carrier when ``machine=mast``).  Key presses on
the focused poloidal view step one control parameter, warm re-solve, and push
the styled poloidal channels — the clipped plasma cells, unfilled contours,
wall and coil outlines, and the topology markers — plus the compensating
currents and the keyframe receipt row, to the shared poloidal channels.
"""

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Div

from apps.playable.machines import MachineUnavailable, build_session, machine_argument
from apps.playable.session import frame_push
from apps.playable.shape import key_help
from apps.pulsedesign.poloidal_view import (
    channel_sources,
    close_outline,
    compensation_figure,
    external_coil_rings,
    keyframe_receipt,
    poloidal_channels,
    poloidal_figure,
)

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


def make_sources():
    """Return the shared channel sources with their bound columns initialised.

    Pre-binding every column a renderer binds keeps the served document free
    of BAD_COLUMN_NAME warnings before the first keyframe.
    """
    return channel_sources()


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


def build_document(doc, *, machine: str = "solovev"):
    """Populate the document with one session and the shared poloidal view."""
    try:
        session = build_session(machine)
    except MachineUnavailable as error:
        doc.add_root(column(Div(text=f"<pre>{error}</pre>"), name="error"))
        return

    sources = make_sources()
    machine_carrier = getattr(getattr(session, "solver", None), "machine", None)

    if session.wall is not None:
        wall = session.wall
        closed = close_outline(wall)
        sources["wall"].data = {"x": closed[:, 0], "z": closed[:, 1]}
        circuits = getattr(machine_carrier, "circuit_count", 0)
        coil_rings = external_coil_rings(wall, circuits) if circuits else ()
    else:
        coil_rings = ()

    poloidal = poloidal_figure(sources)
    compensation = compensation_figure(sources)
    receipt = keyframe_receipt(sources)

    status = Div(text="<b>ready</b> — press a key to solve the next keyframe")
    help_text = Div(text=key_help())

    def on_key(key):
        receipt_row = session.step(key)
        for name, data in frame_push(session).items():
            sources[name].data = data
        if machine_carrier is not None:
            channels = poloidal_channels(
                session.equilibrium,
                machine_carrier.profile,
                wall=session.wall,
                coils=coil_rings,
            )
            for name, data in channels.items():
                sources[name].data = data
        moved = (
            f" {receipt_row.parameter} {receipt_row.delta:+.4g}"
            if receipt_row.parameter is not None
            else ""
        )
        status.text = (
            f"<b>{receipt_row.key}</b>{moved}"
            f" — wall {receipt_row.wall:.3f} s, trips {receipt_row.trips}"
        )

    # The first key press is the prime: the solver warms from the machine's
    # seed, so the document itself never pays a startup solve and the first
    # keyframe is what establishes the initial view.
    sources["points"].data = {
        "x": session.shape.control_points()[0],
        "z": session.shape.control_points()[1],
    }

    for root in (
        help_text,
        poloidal,
        column(status, row(compensation, receipt), name="receipts"),
    ):
        doc.add_root(root)

    wire_keyboard(doc, session, sources, status, on_key)


def main():
    """Serve the app with the carrier the session argument selects."""
    context = curdoc().session_context
    arguments = context.request.arguments if context is not None else {}
    machine = machine_argument(arguments)
    build_document(curdoc(), machine=machine)


main()
