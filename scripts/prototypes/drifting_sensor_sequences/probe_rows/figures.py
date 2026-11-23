"""The axis-against-slice figures and the table of both codes' sequences.

Every measure comes from ``analyse.py``'s reduction, so neither code is
summarised here by a second rule.  The arm selection is that module's too: each
figure and the gate read the arm its code declares, so a column cannot be drawn
on one arm and gated on another, and the row set each column was fitted from is
written into the table the record quotes.

What this module adds is the figure the study is read from -- the reconstructed
magnetic axis against slice number, one trace per noise sequence, one figure
per code, radial and vertical position in their own panels -- and the slice of
largest disagreement between the two codes, which the contour panel is drawn
at.
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
COMPONENTS = ((0, "R [m]"), (1, "Z [m]"))


def arm_rows(code: str, sequence: str, arm: str | None = None) -> list[dict]:
    """The slices of this code's declared arm, or the arm a caller names.

    The overlay is drawn on the row-matched pair -- nova from the same 24 flux
    loops imas-efit reads -- so the two states come from one information set.
    """
    if arm is None:
        return analyse.selected_rows(code, sequence, SLICES, FACTOR)
    stem = analyse.declared_stem(code, sequence, SLICES, FACTOR, rows=arm)
    paths = sorted(STUDY.glob("%s.jsonl" % stem)) + sorted(
        STUDY.glob("%s-shard*.jsonl" % stem)
    )
    return analyse.grouped(paths).get(stem, [])


def axis_position(rows: list[dict]) -> np.ndarray:
    """Each slice's reconstructed axis as a (slice, R/Z) array."""
    if not rows:
        return np.zeros((0, 2))
    ordered = sorted(rows, key=lambda row: int(row["slice"]))
    return np.asarray([row["axis"] for row in ordered], dtype=float)


def disagreement(left: list[dict], right: list[dict]) -> int:
    """The slice where the two reconstructions' axes stand furthest apart."""
    left_axis = {int(row["slice"]): np.asarray(row["axis"], float) for row in left}
    right_axis = {int(row["slice"]): np.asarray(row["axis"], float) for row in right}
    shared = sorted(set(left_axis) & set(right_axis))
    if not shared:
        return 0
    distance = [
        float(np.linalg.norm(left_axis[index] - right_axis[index])) for index in shared
    ]
    return int(shared[int(np.argmax(distance))])


def draw_code(code: str, truth: dict, output: Path) -> Path:
    """One code's axis position against slice, one trace per noise sequence."""
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
            "axes.grid": False,
            "lines.linewidth": 2.6,
            "svg.fonttype": "none",
        }
    )
    # The axes are placed by hand so the strip above them carries the legend
    # and every margin around a tick label is explicit; a layout engine that
    # decides both leaves a label flush with the canvas edge.
    figure, panels = plt.subplots(2, 1, figsize=(14.0, 7.2), dpi=100, sharex=True)
    figure.subplots_adjust(
        left=0.085, right=0.945, top=0.875, bottom=0.115, hspace=0.58
    )
    axis_truth = np.asarray(truth["axis"], dtype=float)
    handles = {}
    for panel, (component, label) in zip(panels, COMPONENTS):
        panel.axhline(
            float(axis_truth[component]),
            color=CONTROL,
            linewidth=1.2,
            linestyle=":",
            zorder=1,
        )
        for sequence in SEQUENCES:
            values = axis_position(arm_rows(code, sequence))
            if values.shape[0] == 0:
                continue
            slices = np.arange(values.shape[0])
            drift = sequence == "drift"
            # The control is dashed and drawn over the subject: on nova's two
            # sequences the traces coincide to a few tenths of a millimetre, and
            # a solid control underneath would hide there entirely.
            (line,) = panel.plot(
                slices,
                values[:, component],
                color=COLOURS[code] if drift else CONTROL,
                linewidth=3.0 if drift else 2.4,
                linestyle="solid" if drift else (0, (4, 2)),
                zorder=3 if drift else 4,
                label="1/f drift" if drift else "white (control)",
            )
            handles["1/f drift" if drift else "white (control)"] = line
        panel.set_ylabel(label)
        panel.set_xlim(0, SLICES - 1)
        # Ticks are placed at fixed fractions of each panel's own range: an
        # automatic locator puts one at the boundary, where it meets the
        # neighbouring panel's label across the gap.
        panel.margins(y=0.26)
        low, high = panel.get_ylim()
        panel.set_yticks([low + 0.28 * (high - low), low + 0.72 * (high - low)])
        panel.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    panels[-1].set_xlabel("slice")
    if handles:
        figure.legend(
            handles=[
                handles[key]
                for key in ("1/f drift", "white (control)")
                if key in handles
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            frameon=False,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checked(figure, str(output))
    plt.close(figure)
    return output


def html_rows(table: dict) -> str:
    """The table's body rows, so the record quotes the reduction."""
    rows = []
    for code in ("nova", "efit"):
        for sequence in SEQUENCES:
            arm = table.get(code, {}).get(sequence)
            if arm is None:
                rows.append(
                    "<tr><td>%s</td><td>%s</td><td>none</td></tr>" % (code, sequence)
                )
                continue
            rows.append(
                "<tr><td>%s</td><td>%s</td><td>%d</td>"
                "<td>%.2f</td><td>%.2f</td>"
                "<td>%.5f</td><td>%.5f</td>"
                "<td>%.4f</td><td>%.4f</td>"
                "<td>%.3f</td><td>%.3f</td><td>%.3f</td>"
                "<td>%.3f</td><td>%.3f</td></tr>"
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
                    arm["axis_lag1_r"],
                    arm["axis_lag1_z"],
                    arm["flux_lag1"],
                    arm["q95_lag1"],
                )
            )
    return "\n".join(rows)


def main(argv=None) -> int:
    """Reduce both codes, draw each figure, and write the table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--table", default="")
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--nova-arm", choices=("all", "loops"), default="all")
    args = parser.parse_args(argv)
    truth = json.loads((STUDY / "generating.json").read_text())
    directory = Path(args.figure_dir)
    table = {}
    for code in ("nova", "efit"):
        table[code] = {}
        for sequence in SEQUENCES:
            rows = arm_rows(code, sequence)
            table[code][sequence] = analyse.reduce_arm(rows, truth) if rows else None
        draw_code(code, truth, directory / ("axis-position-%s.svg" % code))
    worst = disagreement(
        arm_rows("nova", "drift", args.nova_arm), arm_rows("efit", "drift")
    )
    payload = dict(
        table=table,
        disagreement_slice=worst,
        row_set=dict(analyse.DECLARED_ROWS),
    )
    if args.table:
        Path(args.table).write_text(json.dumps(payload, indent=2))
    if args.html:
        print(html_rows(table))
    print("DISAGREEMENT slice=%d" % worst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
