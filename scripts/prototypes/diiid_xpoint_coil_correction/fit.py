"""Fit the nineteen-conductor equivalent error on one row's solved basis state.

The external field the forward solve carries is compared with the field EFIT's
own map implies, and the gauge-free difference is projected onto the external
conductor response columns. The coefficients are an equivalent error of the
forward field, not a measured current correction: the current an arm applies is
the measured one with this error removed, which carries the opposite sign. The
fit is recomputed here rather than copied, so every row's correction comes from
one method on that row's own state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def row_of(bench, frame):
    paths = sorted(bench.DEFAULT_DATA.glob("*.parquet"))
    selected = bench.select_frames(
        paths, bench.EXECUTION_FRAME_COUNT, bench.polarity_population()
    )
    match = [item for item in selected if item.frame == frame]
    if not match:
        raise SystemExit(f"frame {frame} is not in the declared cohort")
    columns = (
        bench._LABEL_COLUMNS
        + bench._GEOMETRY_COLUMNS
        + bench._CURRENT_COLUMNS
        + bench._PLASMA_CURRENT_COLUMNS
    )
    row = bench._read(match[0].path, columns)
    row["_source_path"] = str(match[0].path)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from benchmarks import diiid_forward_gs_match as bench
    from scripts.prototypes.diiid_xpoint_attribution import attribute

    row = row_of(bench, args.frame)
    archive = Path(row["_source_path"])
    result = attribute.analyze(args.frame, args.basis, archive, args.output)
    names = result["conductor_names"]
    measured = result["conductor_current_a_turn"]
    fit = result["conductor_fit"]
    equivalent = fit["equivalent_current_a_turn"]
    correction = {name: float(value) for name, value in zip(names, equivalent)}
    report = {
        "frame": args.frame,
        "basis": str(args.basis),
        "archive": str(archive),
        "rank": int(fit["rank"]),
        "scaled_condition": float(fit["scaled_condition"]),
        "fitted_squared_fraction": float(fit["fitted_squared_fraction"]),
        "conductor_names": list(names),
        "measured_a_turn": [float(item) for item in measured],
        "equivalent_error_a_turn": [float(item) for item in equivalent],
        "applied_correction_a_turn": [correction[name] for name in names],
        "relative_correction": [
            0.0 if current == 0 else float(value) / float(current)
            for current, value in zip(measured, equivalent)
        ],
        "fit_positive_control_residual_fraction": float(
            result["fit_positive_control_residual_fraction"]
        ),
        "measured_offset_cm": [float(item) for item in result["measured_offset_cm"]],
        "physical_image_offset_cm": result.get("physical_image_offset_cm"),
    }
    (args.output / f"correction-{args.frame}.json").write_text(
        json.dumps(correction, indent=2) + "\n"
    )
    (args.output / f"conductor-fit-{args.frame}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("CONDUCTOR_FIT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()