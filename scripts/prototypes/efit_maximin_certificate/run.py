"""Driver: certify the three escape-comparison rows and write the exports.

Rows come from ``docs/research/efit-escape-level-comparison.html`` (merged at
03a0f1d04).  On all three the two codes start from the same axis, so each row
tests the boundary rule rather than the axis.  Every path and cut is evaluated
on the record's own scipy spline; nova's read and the field-swap adapter are not
evaluated here at all.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from scripts.prototypes.efit_maximin_certificate.certificate import (  # noqa: E402
    REPORT,
    certify,
    write_row,
)

MASTU = REPORT / "traces/mastu-fullPulse/boundary_slice_1_iter_5.txt"
SYNTH_A = REPORT / "crossing/synthetic_crossing_a.txt"
SYNTH_B = REPORT / "crossing/synthetic_crossing_b.txt"

ROWS = [
    dict(
        case="mastu-fullPulse-s1-4-5-B",
        source=str(MASTU),
        axis=(0.8473058496500641, -0.1920736910170366),
        target=(0.3327971398830414, 1.3030896186828613),
        nova_level=0.020530973894263525,
        efit_level=0.019652442525099203,
        nova_label="limited",
        efit_label="diverted",
    ),
    dict(
        case="synthetic_crossing_a",
        source=str(SYNTH_A),
        axis=(1.4333360301757094, 0.199990625074199),
        target=(0.85, 0.2),
        nova_level=0.011338328562728484,
        efit_level=0.014214735553450476,
        nova_label="diverted",
        efit_label="limited",
    ),
    dict(
        case="synthetic_crossing_b",
        source=str(SYNTH_B),
        axis=(1.4416617671318872, 0.19999062507419887),
        target=(0.85, 0.2),
        nova_level=0.011404085542709722,
        efit_level=0.01257716874881473,
        nova_label="diverted",
        efit_label="limited",
    ),
]


def main(out_dir: pathlib.Path, n=800):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in ROWS:
        row = certify(
            case=spec["case"],
            source=spec["source"],
            path=pathlib.Path(spec["source"]),
            axis=spec["axis"],
            target=spec["target"],
            nova_level=spec["nova_level"],
            efit_level=spec["efit_level"],
            nova_label=spec["nova_label"],
            efit_label=spec["efit_label"],
            n=n,
        )
        write_row(row, out_dir)
        rows.append(row)
        print(
            f"done {row.case} maximin={row.maximin_axis_to_target} "
            f"min={row.path_min} cut={row.blocking_cut_exists}",
            flush=True,
        )
    payload = [r.__dict__ for r in rows]
    (out_dir / "certificates.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("ROWS", len(rows))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--n", type=int, default=800)
    args = parser.parse_args()
    main(args.out_dir, n=args.n)