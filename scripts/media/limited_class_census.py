"""Fill the topology-class by start-condition table across the label corpus.

Reads only session files and the level-1 efm store: no solve, no device. For
every labelled shot it computes, per frame, the branch guard, the topology
class, whether the frame follows a guard failure (a cold start), and the
ratio of nova's axis-to-boundary flux span to EFIT's on the nearest slice.

The session root is an argument because the census is read at least twice:
once over the corpus as written, and again over a corpus relabelled into a
new root after a solver repair. Both readings must come from this one
implementation -- a second implementation reading the same shape would make
its own first divergence look like a repair effect.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.steering_frames import read_session

SESSION = Path("/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29")
STORE = Path("/work/projects/imas_gpu/mast/level1/shots")
DEFAULT_OUTPUT = Path.home() / "nova-media-logs/limited_class_census.json"


def shot_rows(shot: int, session: Path, store: Path) -> list[dict]:
    """Return one row per frame of one shot, or an empty list when unreadable."""
    dataset = read_session(filename=str(shot), dirname=str(session))
    time = np.asarray(dataset["time"].values, dtype=float)
    guard = np.asarray(dataset["branch_guard_ok"].values, dtype=bool)
    diverted = np.asarray(dataset["diverted"].values, dtype=bool)
    surface = np.asarray(dataset["flux_surface_psi"].values, dtype=float)

    group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
    stored = np.asarray(group["time"], dtype=float)
    axis = np.asarray(group["psi_axis"], dtype=float)
    boundary = np.asarray(group["psi_boundary"], dtype=float)

    cold = np.zeros(time.size, dtype=bool)
    cold[0] = True
    cold[1:] = ~guard[:-1]

    rows = []
    for index in range(time.size):
        if not guard[index]:
            continue
        nearest = int(np.argmin(np.abs(stored - time[index])))
        span = abs(float(surface[0, index] - surface[-1, index]))
        reference = abs(TOTAL_FLUX_FACTOR * (axis[nearest] - boundary[nearest]))
        if not np.isfinite(span) or not (reference > 0.0):
            continue
        rows.append(
            {
                "shot": int(shot),
                "time_s": float(time[index]),
                "diverted": bool(diverted[index]),
                "cold": bool(cold[index]),
                "ratio": float(span / reference),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    """Census every labelled shot of one session root and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=SESSION)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    session, store = arguments.session, arguments.store
    shots = sorted(int(path.stem) for path in session.glob("*.nc"))
    print(f"censusing {len(shots)} shots under {session}")
    rows: list[dict] = []
    unreadable = []
    for position, shot in enumerate(shots):
        if not (store / f"{shot}.zarr").is_dir():
            continue
        try:
            rows.extend(shot_rows(shot, session, store))
        except Exception as error:  # a corpus read, not a gate
            unreadable.append((shot, f"{type(error).__name__}: {error}"))
        if position % 100 == 0:
            print(
                f"  ... {position}/{len(shots)} shots, {len(rows)} frames", flush=True
            )

    print(
        f"\nshots attempted {len(shots)}  unreadable {len(unreadable)}  "
        f"guarded frames {len(rows)}"
    )
    ratio = np.asarray([row["ratio"] for row in rows])
    diverted = np.asarray([row["diverted"] for row in rows])
    cold = np.asarray([row["cold"] for row in rows])

    print(
        f"\n{'class / start':26}{'n':>7}{'median':>9}{'p10':>8}{'p90':>8}"
        f"{'frac<0.5':>10}"
    )
    for is_diverted, name in ((True, "diverted"), (False, "limited ")):
        for is_cold, start in ((False, "warm start"), (True, "COLD start")):
            select = (diverted == is_diverted) & (cold == is_cold)
            values = ratio[select]
            if values.size == 0:
                print(f"{name} / {start:14}{0:>7}{'-':>9}")
                continue
            print(
                f"{name} / {start:14}{values.size:>7}{np.median(values):9.3f}"
                f"{np.quantile(values, 0.1):8.3f}{np.quantile(values, 0.9):8.3f}"
                f"{np.mean(values < 0.5):10.3f}"
            )

    print(
        "\nshots contributing a limited COLD start:",
        len({row["shot"] for row in rows if not row["diverted"] and row["cold"]}),
    )
    print("\ndiverted warm ratio against time, to test the control's own structure")
    times = np.asarray([row["time_s"] for row in rows])
    control = (diverted) & (~cold)
    for low, high in (
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 1.0),
    ):
        band = control & (times >= low) & (times < high)
        if band.sum() < 5:
            continue
        print(
            f"  {1e3 * low:4.0f}-{1e3 * high:4.0f} ms  n={band.sum():5d}"
            f"  median {np.median(ratio[band]):.3f}"
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "session": str(session),
                "store": str(store),
                "shots_attempted": len(shots),
                "rows": rows,
                "unreadable": unreadable,
            },
            indent=1,
        )
    )
    print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
