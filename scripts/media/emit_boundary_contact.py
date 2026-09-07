"""Emit the boundary-to-limiter contact of one labelled shot, per topology class.

A limited boundary must touch its limiter. The reconstruction these labels
reproduce touches at 1.1 mm, so the gap is a direct read on whether a limited
solve is sitting on the wall or floating off it -- the visible symptom that
opened the wall-anchor investigation.

The statistic comes from the same summary function the figure receipt writes,
because the point of banking it is comparability: a regenerated receipt and
this record must be the same measurement, not two similar ones. The figure
receipts published before the statistic was instrumented carry no gap field,
so without this the comparison would rest on a restated number.

Classes are kept apart and only the limited one is an acceptance measure. A
diverted boundary stands off the wall by construction and its gap is expected
to be large; quoting a pooled median would hide the class that matters behind
the class that does not.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from nova.media.poloidal import boundary_wall_gap
from nova.media.sources.mast_efit import read_pulse
from nova.media.sources.nova_labels import read_labels

#: The figure script owns the summary; it is loaded rather than reimplemented
#: so the banked figure cannot drift from the one a receipt reports.
_FIGURE_SCRIPT = Path(__file__).with_name("nova_thomson_gif.py")


def _summary_function():
    """Return the figure script's own gap summary."""
    spec = importlib.util.spec_from_file_location("thomson_figure", _FIGURE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._gap_summary


def contact(shot: int) -> dict[str, object]:
    """Return one shot's per-class boundary-to-limiter contact in millimetres."""
    summarise = _summary_function()
    frames, provenance = read_labels(shot)
    machine = read_pulse(shot).geometry
    gaps = np.array(
        [boundary_wall_gap(frame.boundary, machine.limiter) for frame in frames]
    )
    limited = np.array([not frame.diverted for frame in frames])
    return {
        "shot": shot,
        "session": provenance.get("session"),
        "frame_count": len(frames),
        "limited_frame_count": int(limited.sum()),
        "diverted_frame_count": int((~limited).sum()),
        "boundary_wall_gap_mm": {
            "limited": summarise(gaps[limited]),
            "diverted": summarise(gaps[~limited]),
        },
        "admission": "branch_guard_ok, not conditioned on the reference centroid",
        "wall_section_count": len(machine.limiter),
        "acceptance": (
            "the limited median must reach the reconstruction's own 1.1 mm "
            "contact; the diverted gap is a standoff and is expected"
        ),
        "statistic_source": (
            f"{_FIGURE_SCRIPT.name} gap summary, the same function the figure "
            "receipt writes"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Print one shot's contact record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=27079)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)
    record = contact(arguments.shot)
    text = json.dumps(record, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
