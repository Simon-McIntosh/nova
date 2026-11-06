"""Table saved perturbation receipts and draw their stage sensitivity."""

import argparse
from decimal import Decimal, localcontext
from html import escape
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import InkStyle
from scripts.prototypes.bank_solve_read.plot import save_checked


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    receipts = {
        arm: json.loads((args.run / f"{arm}.json").read_text())
        for arm in ("gpu", "cpu")
    }
    rows, summaries = [], {}
    for arm, receipt in receipts.items():
        norm = np.asarray([p["norm"] for p in receipt["perturbations"]])
        assert norm.shape == (64, len(receipt["lanes"]))
        stats = dict(
            median=np.median(norm, axis=0),
            minimum=norm.min(axis=0),
            maximum=norm.max(axis=0),
            spread=np.ptp(norm, axis=0),
            above=np.sum(norm > 1e-10, axis=0),
        )
        summaries[arm] = {k: v.tolist() for k, v in stats.items()}
        for i, lane in enumerate(receipt["lanes"]):
            rows.append(
                [
                    arm,
                    lane,
                    f"{receipt['baseline_norm'][i]:.4e}",
                    f"{stats['median'][i]:.4e}",
                    f"{stats['minimum'][i]:.4e}",
                    f"{stats['maximum'][i]:.4e}",
                    f"{stats['spread'][i]:.4e}",
                    f"{stats['median'][i] / 1e-10:.3f}",
                    int(stats["above"][i]),
                ]
            )
    (args.run / "lane-table.html").write_text(
        table(
            [
                "Device",
                "Lane",
                "At saved state",
                "Median",
                "Minimum",
                "Maximum",
                "Max − min",
                "Median / certificate",
                "Above / 64",
            ],
            rows,
        )
    )
    samples = np.load(args.run / "gpu-residuals.npz")
    input_spread = np.max(
        np.ptp(np.stack([samples[f"perturbation_{i}"] for i in range(64)]), axis=0),
        axis=-1,
    )
    rows = []
    chart = [("moments (1–4 ulp)", np.median(input_spread), input_spread.max())]
    core = {
        "field_coefficients",
        "nodal_jets",
        "stationary_fluxes",
        "whole_cells",
        "conic_cells",
        "normal_cells",
        "normalization",
        "subtraction",
    }
    for directory, receipt_name, arrays_name in (
        (args.run, "gpu.json", "gpu-residuals.npz"),
        (args.run, "geometry.json", "geometry-residuals.npz"),
        (args.run / "field", "geometry.json", "geometry-residuals.npz"),
    ):
        receipt = json.loads((directory / receipt_name).read_text())
        arrays = np.load(directory / arrays_name)
        for name in receipt["stages"]:
            spread = np.max(np.ptp(arrays[name], axis=0), axis=-1)
            delta = np.max(np.abs(arrays[name] - arrays["baseline"]), axis=-1)
            summary = dict(
                median_change=float(np.median(delta)),
                maximum_change=float(delta.max()),
                median_spread=float(np.median(spread)),
                maximum_spread=float(spread.max()),
            )
            summaries[name] = summary
            rows.append(
                [name.replace("_", " "), *[f"{summary[k]:.4e}" for k in summary]]
            )
            if spread.max() > 0 and (name in core or directory.name == "field"):
                chart.append((name.replace("_", " "), np.median(spread), spread.max()))
    captured = np.load(args.run / "propagation.npz")
    transform, residual = captured["field_0_transform"], captured["field_0_residual"]
    coefficients = captured["field_0_coefficients"]
    conditioning = []
    with localcontext() as context:
        context.prec = 80
        for index in range(6, len(coefficients)):
            products = [
                Decimal.from_float(float(a)) * Decimal.from_float(float(b))
                for a, b in zip(transform[index], residual)
            ]
            total = sum(products)
            conditioning.append(
                dict(
                    coefficient=index,
                    reference=str(total),
                    absolute_error=float(
                        abs(total - Decimal.from_float(float(coefficients[index])))
                    ),
                    cancellation=float(sum(abs(v) for v in products) / abs(total))
                    if total
                    else 0.0,
                )
            )
    summaries["conditioning"] = conditioning
    summaries["inverse_scale_fourth"] = float(captured["field_0_scale"] ** -4)
    (args.run / "stage-table.html").write_text(
        table(
            [
                "Injected output",
                "Median |Δresidual|∞",
                "Maximum |Δresidual|∞",
                "Median lane spread",
                "Maximum lane spread",
            ],
            rows,
        )
    )
    (args.run / "summary.json").write_text(json.dumps(summaries, indent=2))
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    for y, (name, median, maximum) in enumerate(chart):
        if median > 0:
            ax.hlines(y, median, maximum, linewidth=2.6, color="#929292", zorder=2)
        ax.plot(
            median if median > 0 else np.nan,
            y,
            "o",
            color="#929292",
            markersize=7,
            zorder=3,
            label="median lane" if y == 1 else None,
        )
        ax.plot(
            maximum,
            y,
            "o",
            color="#cc0000" if maximum > 1e-10 else InkStyle().flux_color,
            markersize=7,
            zorder=4,
            label="above certificate"
            if y == 0
            else ("largest lane" if y == 1 else None),
        )
    ax.axvline(
        1e-10,
        color="#666666",
        linestyle=":",
        linewidth=1.2,
        label="certificate",
        zorder=1,
    )
    ax.set_xscale("log")
    ax.set_xlim(1e-28, 1e-9)
    ax.set_xticks([1e-26, 1e-22, 1e-18, 1e-14, 1e-10])
    ax.set_yticks(range(len(chart)), [c[0] for c in chart])
    ax.invert_yaxis()
    ax.set_xlabel("Residual spread across 64 perturbations")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    fig.legend(loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.14, top=0.88)
    save_checked(fig, args.figure)
    args.figure.write_text(
        "\n".join(x.rstrip() for x in args.figure.read_text().splitlines()) + "\n"
    )
    fig.savefig(args.run / "stage-spread.png", dpi=100)
    plt.close(fig)
    print("SUMMARY", json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
