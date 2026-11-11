"""Saved-state geometry roles and checked full-width figure saving."""

from pathlib import Path

from scripts.prototypes.diiid_forward_convergence.render_panels import (
    _check_text_artists,
)

from nova.media.ink import DEFAULT_INK

SOLVED = DEFAULT_INK.flux_color
REFERENCE = "#333333"
WALL = "#929292"
CANVAS = (14, 7.2)


def save(figure, axes, notes, output: Path, preview: Path):
    """Refuse every clipped, colliding or in-panel text label before writing."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    problems = _check_text_artists(figure, output)
    for note in notes:
        if any(
            note.get_window_extent(renderer).overlaps(ax.get_window_extent(renderer))
            for ax in axes
        ):
            problems.append(f"label over data: {note.get_text()}")
    if problems:
        raise ValueError("; ".join(problems))
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg")
    text = output.read_text().replace(
        'width="1008pt" height="518.4pt"', 'width="31.1cm" height="16cm"'
    )
    output.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")
    figure.savefig(preview, dpi=100)
    if output.stat().st_size > 300000:
        raise ValueError(f"SVG exceeds repository artifact ceiling: {output}")
    print(
        f"FIGURE {output.name}: text clipping=0; text overlaps=0; labels over data=0; "
        f"bytes={output.stat().st_size}",
        flush=True,
    )
