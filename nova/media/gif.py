"""Compose a matplotlib figure's frames into a GIF through Pillow.

Nova's generic :class:`nova.graphics.plot.Animate` exporter reaches MoviePy,
which is declared in neither ``pyproject.toml`` nor the lock, so every
``animate`` path in the tree raises on import. Pillow is present and writes
GIF directly, so that is the route here.

One figure is reused for every frame and cleared between them, which keeps
the compositor's memory flat whether a pulse is fifty frames or five hundred.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:
    import matplotlib
    import PIL.Image


def figure_frame(figure: matplotlib.figure.Figure) -> PIL.Image.Image:
    """Rasterise one figure into an image, through the Agg canvas.

    The canvas is attached here rather than by the caller so a figure built by
    :func:`nova.media.layout.three_view` needs no backend of its own; nothing
    in this package touches pyplot or a display.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


def _shared_palette(frames: Sequence[PIL.Image.Image]) -> list[PIL.Image.Image]:
    """Return every frame mapped onto one palette, without dithering.

    Pillow otherwise builds a separate adaptive palette per frame and
    error-diffuses into it, which speckles thin lines and text and makes a
    frame flicker against the next. Figures are drawn in a few flat colours,
    so one palette taken from a strip of evenly spaced frames holds them all,
    and nearest-colour mapping keeps every edge crisp.
    """
    from PIL import Image

    # Pillow maps colours onto a palette at reduced precision, so a pure white
    # page lands on a near-white entry and comes back off-white. Pixels within
    # fifteen levels of white in every channel are set to exact white first,
    # which leaves no near-white entry for the lookup to land on.
    def snapped(frame):
        array = np.asarray(frame).copy()
        array[np.all(array >= 240, axis=2)] = 255
        return Image.fromarray(array)

    frames = [snapped(frame) for frame in frames]
    chosen = np.unique(np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int))
    width, height = frames[0].size
    strip = Image.new("RGB", (width, height * len(chosen)))
    for position, index in enumerate(chosen):
        strip.paste(frames[index], (0, position * height))
    palette = strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    return [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
    ]


def write_gif(
    frames: Sequence[PIL.Image.Image],
    path: str | Path,
    duration: float = 10.0,
    loop: int = 0,
    fps: float | None = None,
    durations: Sequence[int] | None = None,
) -> dict[str, object]:
    """Write ``frames`` as a GIF lasting ``duration`` seconds.

    The per-frame delay is derived from the requested total and the frame
    count, so the length of the animation is the thing the caller controls.
    GIF stores that delay in hundredths of a second, so it is rounded to at
    least one tick and the achieved duration is reported rather than assumed.
    ``fps``, when given, sets the frame rate instead and the duration follows
    from the frame count. ``durations`` gives each frame its own hold in
    milliseconds, which is how a sequence sampled unevenly in time still plays
    at a uniform rate of its own time.
    """
    if not frames:
        raise ValueError("a GIF needs at least one frame")
    if duration <= 0.0:
        raise ValueError("the requested duration must be positive")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fps is not None:
        if fps <= 0.0:
            raise ValueError("the requested frame rate must be positive")
        duration = len(frames) / fps
    milliseconds = max(10, int(round(1000.0 * duration / len(frames) / 10.0) * 10))
    if durations is not None:
        if len(durations) != len(frames):
            raise ValueError("durations must give one hold per frame")
        milliseconds = [max(10, int(round(hold / 10.0)) * 10) for hold in durations]
    indexed = _shared_palette(frames)
    indexed[0].save(
        target,
        save_all=True,
        append_images=indexed[1:],
        duration=milliseconds,
        loop=loop,
        optimize=True,
    )
    achieved = (
        sum(milliseconds) if durations is not None else milliseconds * len(frames)
    ) / 1000.0
    return {
        "path": str(target),
        "frames": len(frames),
        "frame_milliseconds": milliseconds,
        "requested_seconds": float(duration),
        "achieved_seconds": achieved,
        "bytes": target.stat().st_size,
        "size": list(frames[0].size),
    }


def write_contact_sheet(
    frames: Sequence[PIL.Image.Image],
    path: str | Path,
    columns: int = 3,
    count: int = 6,
    gap: int = 8,
    background: tuple[int, int, int] = (255, 255, 255),
) -> dict[str, object]:
    """Tile evenly spaced frames into one still image.

    An animation needs a still companion for two reasons. The figure index
    scans only static raster and vector images, so a GIF alone is published
    but never listed; and a talk needs a frame that can be pointed at. The
    frames are sampled evenly across the sequence, always including the first
    and last, so the sheet reads as the pulse rather than as its middle.
    """
    if not frames:
        raise ValueError("a contact sheet needs at least one frame")
    if columns < 1 or count < 1:
        raise ValueError("a contact sheet needs a positive column and frame count")
    from PIL import Image

    chosen = np.unique(
        np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int)
    )
    tiles = [frames[index] for index in chosen]
    columns = min(columns, len(tiles))
    rows = -(-len(tiles) // columns)
    width, height = tiles[0].size
    sheet = Image.new(
        "RGB",
        (
            columns * width + (columns - 1) * gap,
            rows * height + (rows - 1) * gap,
        ),
        background,
    )
    for position, tile in enumerate(tiles):
        row, column = divmod(position, columns)
        sheet.paste(tile, (column * (width + gap), row * (height + gap)))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)
    return {
        "path": str(target),
        "tiles": len(tiles),
        "tile_indices": chosen.tolist(),
        "columns": columns,
        "rows": rows,
        "size": list(sheet.size),
        "bytes": target.stat().st_size,
    }


def animate(
    figure: matplotlib.figure.Figure,
    steps: Iterable[object],
    render: Callable[[object], None],
    path: str | Path,
    duration: float = 10.0,
    contact_sheet: str | Path | None = None,
    sheet_columns: int = 3,
    sheet_count: int = 6,
    fps: float | None = None,
    dpi: float | None = None,
    durations: Sequence[int] | None = None,
) -> dict[str, object]:
    """Render every step into ``figure`` and write the result as a GIF.

    ``render`` draws one step and is responsible for clearing what it needs;
    :meth:`nova.media.layout.ThreeView.clear` is the usual first call. When
    ``contact_sheet`` is given, a still sheet of evenly spaced frames is
    written beside the animation from the frames already in hand. ``dpi``
    rasterises at that resolution instead of the figure's own, which is how a
    frame stays sharp when it is shown larger than the figure was drawn.
    """
    if dpi is not None:
        figure.set_dpi(dpi)
    frames = []
    for step in steps:
        render(step)
        frames.append(figure_frame(figure))
    receipt = write_gif(frames, path, duration=duration, fps=fps, durations=durations)
    if contact_sheet is not None:
        receipt["contact_sheet"] = write_contact_sheet(
            frames, contact_sheet, columns=sheet_columns, count=sheet_count
        )
    return receipt
