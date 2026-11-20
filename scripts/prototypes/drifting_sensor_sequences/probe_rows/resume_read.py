"""Re-read the remaining slices of one imas-efit arm, tolerating empty boundaries.

The drift arm's read stopped at slice 174 because that slice's output carries no
boundary outline; the solver wrote a time slice there, so the arm's remaining
slices can still be read.  A slice whose boundary is missing is recorded in the
missing list rather than mixed into the series as a fabricated point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STUDY = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261119T065220776269-drifting-sensor-probe-rows/study"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--slices", type=int, default=200)
    args = parser.parse_args()

    from nova.imas.mast_efit_referee import EfitRefereeError, q95_from_flux_map
    from nova.imas.mast_efit_referee import read_equilibrium_state

    reference = json.loads((STUDY / "generating.json").read_text())
    wall = str(STUDY / "conf" / "machine-wall.npz")
    output = STUDY / ("efit-" + args.label)
    path = STUDY / ("efit-%s.jsonl" % args.label)
    missing = []
    with path.open("a") as stream:
        for index in range(args.start, args.slices):
            try:
                state = read_equilibrium_state(
                    output, time_index=index, wall_source=wall
                )
            except EfitRefereeError as error:
                missing.append(index)
                print("SKIP %d %s" % (index, error), flush=True)
                continue
            q95 = float("nan")
            if state.flux_map is not None:
                q95 = float(q95_from_flux_map(state, vacuum_b0=reference["b0"]))
            record = dict(
                slice=index,
                axis=np.asarray(state.magnetic_axis_m, dtype=float).tolist(),
                flux_axis=float(state.flux_axis),
                flux_boundary=float(state.flux_boundary),
                flux_span=float(state.flux_boundary - state.flux_axis),
                plasma_current=float(state.plasma_current_a),
                q95=q95,
                q95_stored=float(state.q95),
            )
            stream.write(json.dumps(record) + "\n")
            stream.flush()
    (STUDY / ("efit-%s.missing" % args.label)).write_text(
        "\n".join(str(index) for index in missing) + "\n"
    )
    print(
        "EFIT_ROWS_APPENDED %d slices, %d without a boundary outline"
        % (args.slices - args.start - len(missing), len(missing)),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
