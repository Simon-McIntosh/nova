"""Render measured membership arms and tabulate persisted Bell row receipts."""

# ruff: noqa: E501
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def measured_rows(directory):
    rows = {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        if "rows" not in data:
            continue
        for row in data["rows"]:
            rows[(data["kind"], row["cells"], row["representation"])] = row
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("figure", type=Path)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rows = measured_rows(args.reference)
    if args.prefix:
        for path in args.prefix.glob("*.log"):
            kind = path.stem.split("-")[0]
            for line in path.read_text().splitlines():
                if line.startswith("READ_ROW "):
                    row = json.loads(line.removeprefix("READ_ROW "))
                    rows[(kind, row["cells"], row["representation"])] = row
    current = measured_rows(args.rows)
    rows.update(current)
    plt.style.use("data-ink")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained")
    colours = {"limited": "#427e7b", "diverted": "#856895"}
    styles = {
        "closed_form": ":",
        "kernel": "-",
        "nonconforming_jet": "--",
        "bell_jet": "-.",
    }
    labels = {
        "closed_form": "closed form",
        "kernel": "kernel",
        "nonconforming_jet": "nonconforming",
        "bell_jet": "Bell",
    }
    normalized = {
        (kind, cells, "nonconforming_jet" if arm == "jet" else arm): row
        for (kind, cells, arm), row in rows.items()
    }
    for axis, metric in zip(
        axes, ("smooth_membership_error", "saddle_membership_error"), strict=True
    ):
        for kind, colour in colours.items():
            for arm, style in styles.items():
                values = sorted(
                    (cells, row[metric])
                    for (case, cells, name), row in normalized.items()
                    if case == kind and name == arm and row[metric] is not None
                )
                if values:
                    axis.plot(
                        *np.asarray(values).T,
                        color=colour,
                        linestyle=style,
                        marker="o",
                        linewidth=3 if arm == "bell_jet" else 2.6,
                        label=f"{kind} {labels[arm]}",
                    )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="cells",
            ylabel=("smooth" if metric.startswith("smooth") else "saddle")
            + " error [median cells]",
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
        axis.legend(frameon=False, fontsize=13)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure)
    plt.close(figure)
    if args.report:
        lines = [
            "# Conforming cached-jet read",
            "",
            "**Decision to lock: pending the completed ladder and Picard reference qualification.**",
            "",
            "The element is the reduced quintic Bell triangle: vertex value, gradient and symmetric Hessian, with a cubic normal derivative on each edge. Each uncut sampling hex is split into four triangles, fanned from its first existing vertex. Wall-clipped portions use that same field. No interpolation nodes are added.",
            "",
            "The read contracts the runtime coupling matrix with the booked moments, adds the fixed exterior jets, and assembles the element inside the compiled read. Dense matrices are checkpointed under the GPFS evidence store by node/coupling/kernel digest.",
            "",
            "|Case|Cells|Arm|Axis [pitch]|X [pitch]|Smooth|Saddle|Budget|Valid|Cold [s]|Warm [s]|Peak RSS [KiB]|",
            "|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
        ]

        def show(value):
            return "—" if value is None else f"{value:.6g}"

        for (kind, cells, arm), row in sorted(current.items()):
            lines.append(
                f"|{kind}|{cells}|{arm}|"
                + "|".join(
                    show(row[key])
                    for key in (
                        "axis_error_pitches",
                        "x_error_pitches",
                        "smooth_membership_error",
                        "saddle_membership_error",
                        "membership_bound",
                    )
                )
                + f"|{row['valid']}/{row['qualified']}|"
                + "|".join(
                    show(row[key])
                    for key in ("cold_compile_seconds", "warm_seconds", "peak_rss_kib")
                )
                + "|"
            )
        lines += [
            "",
            "|Case|Cells|Matrix bytes|Build [s]|Quartic value / gradient / Hessian|Edge value / normal jump|",
            "|---|---:|---:|---:|---|---|",
        ]
        receipts = []
        for path in sorted(args.rows.glob("*.json")):
            data = json.loads(path.read_text())
            if "element_controls" not in data:
                continue
            receipts.append(data)
            control = data["element_controls"]
            lines.append(
                f"|{data['kind']}|{data['cells']}|{data.get('jet_bytes', 'pending')}|{data.get('jet_build_seconds', 'pending')}|"
                + " / ".join(
                    show(control["quartic"][key])
                    for key in ("value", "gradient", "hessian")
                )
                + "|"
                + " / ".join(
                    show(control["edges"][key]) for key in ("value_jump", "normal_jump")
                )
                + "|"
            )
        lines += ["", "|Case|Error|Order from 2,000 to 5,000 cells|", "|---|---|---:|"]
        for kind in colours:
            pair = {data["cells"]: data for data in receipts if data["kind"] == kind}
            if not {2000, 5000} <= pair.keys():
                continue
            coarse, fine = pair[2000], pair[5000]
            pitch_ratio = coarse["rows"][0]["pitch_m"] / fine["rows"][0]["pitch_m"]
            for quantity in ("value", "gradient", "hessian"):
                errors = [
                    data["element_controls"]["approximation"][quantity]
                    for data in (coarse, fine)
                ]
                order = np.log(errors[0] / errors[1]) / np.log(pitch_ratio)
                lines.append(f"|{kind}|{quantity}|{order:.6g}|")
        lines += [
            "",
            "Orders are raw measured maximum-error slopes; an error at floating-point roundoff cannot establish truncation order. Picard membership columns use the analytic fractions as the score reference; the stored live-kernel scalar row alone cannot establish pointwise membership agreement.",
            "",
            "[Bell element definition](https://defelement.org/elements/bell.html).",
            "",
            "The figure includes measured rows only. A missing rung is unmeasured, never a zero. Kernel points are reused solely at 2,000 cells.",
        ]
        args.report.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
