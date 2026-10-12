"""Render measured membership arms and tabulate persisted Bell row receipts."""

# ruff: noqa: E501
import argparse
import json
import math
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
    parser.add_argument("--additional-rows", type=Path, action="append", default=[])
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--curvature", choices=("polynomial", "finite-difference"), default="polynomial"
    )
    args = parser.parse_args()
    reference_rows = measured_rows(args.reference)
    rows = dict(reference_rows)
    if args.prefix:
        for path in args.prefix.glob("*.log"):
            kind = path.stem.split("-")[0]
            for line in path.read_text().splitlines():
                if line.startswith("READ_ROW "):
                    row = json.loads(line.removeprefix("READ_ROW "))
                    rows[(kind, row["cells"], row["representation"])] = row
    current = {}
    directories = [args.rows, *args.additional_rows]
    for directory in directories:
        current.update(measured_rows(directory))
    rows.update(current)
    plt.style.use("data-ink")
    plt.rcParams["svg.hashsalt"] = "bell-membership"
    figure, axes = plt.subplots(1, 2, figsize=(14, 9), layout="constrained")
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
        for kind in colours:
            if metric.startswith("saddle") and kind == "limited":
                continue
            limits = sorted(
                (cells, row["membership_bound"])
                for (case, cells, arm), row in current.items()
                if case == kind and arm == "closed_form"
            )
            if limits:
                axis.plot(
                    *np.asarray(limits).T, color="0.45", linestyle=":", linewidth=1.2
                )
                axis.annotate(
                    f"{kind} budget",
                    limits[-1],
                    xytext=(6, 0),
                    textcoords="offset points",
                    color="0.35",
                    fontsize=20,
                    va="center",
                    annotation_clip=False,
                )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="cells",
            ylabel=("smooth" if metric.startswith("smooth") else "saddle")
            + " error [median cells]",
        )
        axis.margins(x=0.45)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
        axis.legend(frameon=False, fontsize=20)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, metadata={"Date": None})
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
            "|Case|Cells|Arm|Axis vs analytic [pitch]|X vs analytic [pitch]|Smooth|Saddle|Budget|Valid/qualified|Membership pass|Cold [s]|Warm [s]|Peak RSS [KiB]|",
            "|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|",
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
                + f"|{row['valid']}/{row['qualified']}|{row['passes_membership']}|"
                + "|".join(
                    show(row[key])
                    for key in ("cold_compile_seconds", "warm_seconds", "peak_rss_kib")
                )
                + "|"
            )
        lines += [
            "",
            "|Case|Cells|Matrix bytes|Build counter [s]|Persisted nodes|Quartic value / gradient / Hessian|Edge value / normal jump|",
            "|---|---:|---:|---:|---|---|---|",
        ]
        by_case = {}
        for directory in directories:
            for path in sorted(directory.glob("*.json")):
                data = json.loads(path.read_text())
                if "element_controls" in data:
                    data["receipt_path"] = path
                    by_case[data["kind"], data["cells"]] = data
        receipts = [by_case[key] for key in sorted(by_case)]
        stores = []
        for data in receipts:
            control = data["element_controls"]
            log = data["receipt_path"].with_suffix(".log").read_text()
            paths = [
                Path(line.split("path=", 1)[1].split()[0])
                for line in log.splitlines()
                if line.startswith("JET_STORE ")
            ]
            progress = {}
            if paths:
                progress = json.loads(paths[-1].with_name("progress.json").read_text())
                stores.append((data["kind"], data["cells"], paths[-1], progress))
            size = data.get("jet_bytes", progress.get("bytes", "pending"))
            wall = data.get(
                "jet_build_seconds", progress.get("build_seconds", "pending")
            )
            persisted = progress.get("nodes_complete", "pending")
            lines.append(
                f"|{data['kind']}|{data['cells']}|{size}|{wall}|{persisted}|"
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
        lines += [
            "",
            "Matrix files and construction checkpoints:",
            "",
            *[
                f"- {kind}, {cells} cells: `{path}`; "
                f"checkpoint construction wall {progress['build_seconds']:.9g} s."
                for kind, cells, path, progress in stores
            ],
            "",
            "A persisted matrix is not evidence that its device load or read succeeded. "
            "When a read receipt is absent, the build counter above is the "
            "checkpointed construction time only; persisted-node count identifies "
            "a partial build. Matrix bytes exclude the small NumPy header.",
        ]
        lines += ["", "## Capacity and scaling", ""]
        for kind in colours:
            sizes = {
                cells: progress["bytes"]
                for case, cells, _, progress in stores
                if case == kind
            }
            if {2000, 10000} <= sizes.keys():
                ratio = sizes[10000] / sizes[2000]
                exponent = math.log(ratio) / math.log(5)
                lines.append(
                    f"{kind.capitalize()} dense coupling storage grows by {ratio:.6g} "
                    f"times from 2,000 to 10,000 cells (measured exponent {exponent:.6g} "
                    "against cell count). The matrix shape is nodes × cells × 3 × 6, "
                    "with eight bytes per entry."
                )
                lines.append("")
        for directory in directories:
            for path in sorted(directory.glob("*-10000.log")):
                errors = [
                    line
                    for line in path.read_text().splitlines()
                    if line.startswith("jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED")
                ]
                if errors:
                    lines += [
                        f"Device-load refusal in `{path}`:",
                        "",
                        f"> {errors[-1]}",
                        "",
                    ]
        lines += [
            "The 10,000-cell capacity exception is a device-allocation failure, "
            "not a measured read. No Bell compile or warm-read wall is inferred for "
            "a row that stopped before the read. A fresh-process cache retry preserves "
            "the first failure log and avoids repeating completed matrix construction.",
            "",
        ]
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
        candidate = current.get(("diverted", 2000, "picard_bell"))
        reference = reference_rows.get(("diverted", 2000, "picard_kernel"))
        if candidate is not None and reference is not None:
            assert candidate["cells"] == reference["cells"]
            assert candidate["pitch_m"] == reference["pitch_m"]
            lower = abs(
                candidate["saddle_membership_error"]
                - reference["saddle_membership_error"]
            )
            upper = (
                candidate["saddle_membership_error"]
                + reference["saddle_membership_error"]
            )
            axis_difference = (
                np.linalg.norm(np.asarray(candidate["axis"]) - reference["axis"])
                / candidate["pitch_m"]
            )
            x_difference = (
                np.linalg.norm(
                    np.asarray(candidate["x_points"]) - reference["x_points"]
                )
                / candidate["pitch_m"]
            )
            bound = candidate["membership_bound"]
            if lower > bound:
                lines[2] = (
                    f"**Decision to lock: reject the measured {args.curvature} Bell "
                    "curvature configuration as the production representation. "
                    "At 2,000 cells its Picard saddle disagreement with the live-kernel "
                    f"reference is at least {lower:.9g}, or {lower / bound:.4g} times the budget.**"
                )
            else:
                lines[2] = (
                    "**Decision to lock: keep production qualification open. "
                    "The Picard scalar comparison does not establish the required "
                    "pointwise membership bound, and the full ladder is not complete.**"
                )
            lines += [
                "",
                "## Picard qualification",
                "",
                "For the same 2,000-cell carrier, saddle set, area normalization and analytic "
                "reference A, the reverse triangle inequality gives "
                "‖Bell − kernel‖∞ ≥ |‖Bell − A‖∞ − ‖kernel − A‖∞|. "
                "A lower bound above the budget establishes a failure, even though "
                "they do not determine the exact pointwise disagreement.",
                "",
                f"Bell saddle error against A: {candidate['saddle_membership_error']:.12g}; "
                f"kernel: {reference['saddle_membership_error']:.12g}; "
                f"disagreement lower bound: {lower:.12g}; budget: {bound:.12g}.",
                "",
                f"The kernel-relative axis and X-point errors are {axis_difference:.12g} "
                f"and {x_difference:.12g} pitches. The same norms also give an upper bound "
                f"of {upper:.12g} on saddle disagreement; its exact value is not stored.",
                "",
                "The X-point is restored and both reads are valid, but that does not qualify "
                "their saddle membership. Roundoff continuity and quartic completeness "
                "are established; the production solve-state bound requires its own evidence.",
            ]
        lines += [
            "",
            "The 550-cell smoke predates coefficient conditioning and times a "
            "preassembled field. The larger ladder includes runtime moment contraction "
            "and coefficient assembly in its read wall. Limited-field approximation "
            "errors are at roundoff because that analytic field is quartic; its raw "
            "slopes above are not truncation orders.",
        ]
        lines += ["", f"Curvature treatment for these rows: **{args.curvature}**."]
        if args.comparison:
            lines += [
                "",
                "## Retained finite-difference comparison",
                "",
                "These earlier rows remain in their original receipt directory. "
                "They use finite differences of Hessians for curvature, while the main "
                "table differentiates the selected Bell polynomial. The same Picard "
                "failure in both configurations rules out that adapter as its remedy.",
                "",
                "|Case|Cells|Arm|Smooth|Saddle|Warm [s]|",
                "|---|---:|---|---:|---:|---:|",
            ]
            for (kind, cells, arm), row in sorted(
                measured_rows(args.comparison).items()
            ):
                if arm not in ("bell_jet", "picard_bell"):
                    continue
                lines.append(
                    f"|{kind}|{cells}|{arm}|"
                    + "|".join(
                        show(row[key])
                        for key in (
                            "smooth_membership_error",
                            "saddle_membership_error",
                            "warm_seconds",
                        )
                    )
                    + "|"
                )
            lines += ["", f"Retained receipts: `{args.comparison}`."]
        lines += [
            "",
            "## Receipt provenance",
            "",
            *[f"- Native rows: `{directory}`." for directory in directories],
            f"- Reused kernel rows: `{args.reference}`.",
            "",
            "Matrix build counters include historical build time plus the current "
            "resume or cache-load overhead when the load completed. Peak RSS is the "
            "process high-water mark, not device memory. Per-rung progress files retain the "
            "checkpointed construction wall separately. A reused Picard executable "
            "can have near-zero compile time; its warm wall still includes the full read.",
        ]
        args.report.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
