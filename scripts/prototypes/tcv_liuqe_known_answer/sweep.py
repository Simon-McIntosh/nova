"""Score nova's read on every representative TCV 77549 LIUQE slice.

One process, one allocation: the whole sweep is a single job rather than one
job per slice, and each row is appended to the receipt as it lands, so an
expiry costs one slice rather than the run. The rows carry the boundary-index
certificate beside the labels, which is what lets a refusal be told apart from
an agreement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imas

from scripts.prototypes.tcv_liuqe_known_answer import measure

SLICES = (0, 6, 14, 24, 34, 46, 118, 206, 248, 291, 333, 376, 456, 542, 611, 796, 1257)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--liuqe", required=True)
    parser.add_argument("--wall", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--slices", default=",".join(str(index) for index in SLICES))
    parser.add_argument("--resolutions", default="16,24,32")
    args = parser.parse_args()

    indices = [int(token) for token in args.slices.split(",") if token.strip()]
    resolutions = tuple(int(r) for r in args.resolutions.split(","))
    wall_rings = measure.read_wall(args.wall)
    db = imas.DBEntry(f"imas:hdf5?path={args.liuqe}", "r", dd_version="3.41.0")
    eq = db.get("equilibrium")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for index in indices:
            view = measure.slice_view(eq, index)
            field = measure.TraceSplineField.from_record(
                measure.build_state(
                    view["rgrid"],
                    view["zgrid"],
                    view["psi"],
                    wall_rings,
                    view["axis"][0],
                    view["axis"][1],
                )
            )
            row = dict(index=index)
            row.update(measure.liuqe_reference(view, field))
            try:
                row.update(measure.measure_slice(view, wall_rings, resolutions))
            except Exception as error:  # noqa: BLE001 - one slice must not stop the sweep
                row.update(
                    resolution=None,
                    nova_label="unread",
                    attempts=f"{type(error).__name__}: {error}",
                )
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(
                "done",
                index,
                row["liuqe_time"],
                row["liuqe_label"],
                row["nova_label"],
                row.get("coverage_total"),
                row.get("coverage_index_sum"),
                flush=True,
            )
    db.close()
    print("ROWS", len(indices), flush=True)


if __name__ == "__main__":
    main()
