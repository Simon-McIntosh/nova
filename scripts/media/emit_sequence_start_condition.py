"""Locate the cold frame within each shot's drawn sequence and score it.

``cold_start`` marks the first stored frame of a run or one following a branch
guard failure. A figure draws only guarded frames, so the first frame it draws
is itself cold whenever its stored predecessor failed the guard -- which puts
the cold condition AT the start of every drawn sequence rather than near it.
That distinction decides whether the cold exception is a residual sitting
beside the frames a viewer and a decoder start from, or the very frame they
start on.

Each frame's axis-to-boundary flux span is reported beside its own shot's
median span, because position alone does not say whether the frame is bad.
The two together answer the question a presentation actually poses: what does
the audience see first, and is it representative of the pulse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nova.media.sources.nova_labels import read_labels

#: The shots written into the repair validation root.
DEFAULT_SHOTS = (27079, 21978, 21983, 21985, 21986, 21989, 22086)


def _cold_flags(dataset) -> np.ndarray:
    """Return the cold-start flag per stored frame.

    A run's first frame is cold, and so is any frame whose predecessor failed
    the branch guard: both are solves with no trustworthy state behind them.
    """
    guard = np.asarray(dataset["branch_guard_ok"].values, dtype=bool)
    cold = np.zeros(guard.size, dtype=bool)
    cold[0] = True
    cold[1:] = ~guard[:-1]
    return cold


def shot_record(shot: int, session: Path | None) -> dict[str, object]:
    """Return where the cold frames sit in one shot's drawn sequence."""
    from nova.equilibrium.steering_frames import read_session

    keywords = {} if session is None else {"dirname": session}
    frames, provenance = read_labels(shot, **keywords)
    dataset = read_session(filename=str(shot), dirname=provenance["session"])
    stored_time = np.asarray(dataset["time"].values, dtype=float)
    cold = _cold_flags(dataset)

    drawn_time = np.array([frame.time for frame in frames])
    span = np.array(
        [abs(frame.surface_flux[0] - frame.surface_flux[-1]) for frame in frames]
    )
    stored_position = [
        int(np.argmin(np.abs(stored_time - time))) for time in drawn_time
    ]
    cold_ranks = [
        position
        for position, index in enumerate(stored_position, start=1)
        if cold[index]
    ]
    median = float(np.median(span))
    return {
        "shot": shot,
        "stored_frame_count": int(stored_time.size),
        "drawn_frame_count": len(frames),
        "first_drawn_time_s": float(drawn_time[0]),
        "first_drawn_stored_index": stored_position[0],
        "first_drawn_is_cold": bool(cold[stored_position[0]]),
        "first_drawn_is_limited": bool(not frames[0].diverted),
        "cold_ranks_in_drawn_sequence": cold_ranks,
        "first_drawn_span_wb": float(span[0]),
        "median_span_wb": median,
        "first_drawn_span_as_fraction_of_median": float(span[0] / median),
        "rank_of_first_drawn_by_span": int(np.argsort(np.argsort(span))[0] + 1),
        "session": provenance["session"],
    }


def main(argv: list[str] | None = None) -> int:
    """Print, and optionally persist, the start condition of each sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, nargs="+", default=list(DEFAULT_SHOTS))
    parser.add_argument("--session", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)
    records = [shot_record(shot, arguments.session) for shot in arguments.shots]
    summary = {
        "question": (
            "does the cold condition sit at the start of a drawn sequence or "
            "beside it, and is the first frame drawn representative of its pulse"
        ),
        "shots": records,
        "first_drawn_is_cold_on_every_shot": all(
            record["first_drawn_is_cold"] for record in records
        ),
        "first_drawn_is_weakest_on_every_shot": all(
            record["rank_of_first_drawn_by_span"] == 1 for record in records
        ),
        "worst_first_drawn_fraction_of_median": min(
            record["first_drawn_span_as_fraction_of_median"] for record in records
        ),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
