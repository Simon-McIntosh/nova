"""The axis-against-slice figures and the table of both codes' sequences.

Every measure comes from ``analyse.py``'s reduction, so neither code is
summarised here by a second rule.  What this module adds is the figure the
study is read from -- the axis error of one code against slice number, one
trace per noise sequence, one figure per code -- and the slice of largest
disagreement between the two codes, which the contour panel is drawn at.

The trace colour names the code; within a code the control sequence is drawn in
neutral grey, so the sequence the study is about (the 1/f draw) carries colour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyse  # noqa: E402

STUDY = analyse.STUDY

CONTROL = "#929292"
COLOURS = {"nova": "#1f5fa8", "efit": "#2e8b57"}
SEQUENCES = ("white", "drift")
SLICES = 200
FACTOR = 0.01


def arm_rows(code: str, sequence: str) -> list[dict]:
    """The longest 200-slice series of this code and sequence."""
    pattern = "%s-%s-%d-%.4f" % (code, sequence, SLICES, FACTOR)
    paths = sorted(STUDY.glob(pattern + "*.jsonl"))
    runs = analyse.grouped(paths)
    if not runs:
        return []
    # Two row sets can exist for imas-efit; the flux-loop arm is the one the
    # comparison is stated on, and the tag that names it is the arm's own.
    return max(
        runs.items(), key=lambda item: ("-loops" in item[0], len(item[1]))
    )[1]


def axis_error(rows: list[dict], truth: dict) -> np.ndarray:
    """Distance of each slice's reconstructed axis from truth, metres."""
    if not rows:
        return np.zeros(0)
    return analyse.axis_distance(rows, truth["axis"])


def disagreement(left: list[dict], right: list[dict]) -> int:
    """The slice where the two reconstructions' axes stand furthest apart."""
    left_axis = {
        int(row["slice"]): np.asarray(row["axis"], float) for row in left
    }
    right_axis = {
        int(row["slice"]): np.asarray(row["axis"], float) for row in right
    }
    shared = sorted(set(left_axis) & set(right_axis))
    if not shared:
        return 0
    distance = [
        float(np.linalg.norm(left_axis[index] - right_axis[index]))
        for index in shared
    ]
    return int(shared[int(np.argmax(distance))])


def draw_code(code: str, truth: dict, output: Path) -> Path:
    """One code's axis error against slice, one trace per noise sequence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.edge_sheet.figure import save_checked

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.6,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(figsize=(14.0, 7.2), dpi=100)
    for sequence in SEQUENCES:
        error = axis_error(arm_rows(code, sequence), truth) * 1000.0
        axes.plot(
            np.arange(error.size),
            error,
            color=COLOURS[code] if sequence == "drift" else CONTROL,
            linewidth=3.0 if sequence == "drift" else 2.4,
            zorder=3 if sequence == "drift" else 2,
            label="1/f drift" if sequence == "drift" else "white (control)",
        )
    axes.set_xlabel("slice")
    axes.set_ylabel("axis error [mm]")
    axes.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checked(figure, str(output))
    plt.close(figure)
    return output


def html_rows(table: dict) -> str:
    """The table's rows, so the record quotes the reduction rather than a copy."""
    rows = []
    for code in ("nova", "efit"):
        for sequence in SEQUENCES:
            arm = table[code][sequence]
            if arm is None:
                rows.append(
                    "<tr><td>%s</td><td>%s</td><td>none</td></tr>"
                    % (code, sequence)
                )
                continue
            rows.append(
                "<tr><td>%s</td><td>%s</td><td>%d</td><td>%.2f</td><td>%.2f</td>"
                "<td>%.5f</td><td>%.5f</td><td>%.4f</td><td>%.4f</td>"
                "<td>%.3f</td><td>%.3f</td><td>%.3f</td></tr>"
                % (
                    code,
                    sequence,
                    arm["slices"],
                    arm["axis_rms"] * 1000.0,
                    arm["axis_mean_50"] * 1000.0,
                    arm["flux_rms"],
                    arm["flux_mean_50"],
                    arm["q95_rms"],
                    arm["q95_mean_50"],
                    arm["axis_lag1"],
                    arm["flux_lag1"],
                    arm["q95_lag1"],
                )
            )
    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--table", default="")
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args(argv)
    truth = json.loads((STUDY / "generating.json").read_text())
    directory = Path(args.figure_dir)
    table = {}
    for code in ("nova", "efit"):
        table[code] = {}
        for sequence in SEQUENCES:
            rows = arm_rows(code, sequence)
            table[code][sequence] = analyse.reduce_arm(rows, truth) if rows else None
        draw_code(code, truth, directory / ("axis-error-%s.svg" % code))
    worst = disagreement(arm_rows("nova", "drift"), arm_rows("efit", "drift"))
    payload = {
        "slices": SLICES,
        "factor": FACTOR,
        "arms": table,
        "largest_disagreement_slice": worst,
    }
    print(json.dumps(payload, indent=2))
    if args.html:
        print(html_rows(table), flush=True)
    if args.table:
        Path(args.table).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())