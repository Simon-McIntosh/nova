"""Draw certificate latency and analytic-flux error from completed receipts."""

# ruff: noqa: E402

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "equilibrium-comparison"
import matplotlib.pyplot as plt
import nova
from nova.media.ink import DEFAULT_INK


CASES = ("limited", "diverted")
REQUESTS = (2000, 5000, 10000)
CASE_STYLE = {
    "limited": dict(color=DEFAULT_INK.flux_color, linestyle="-"),
    "diverted": dict(color="#353535", linestyle="--"),
}


def _receipt(roots, kind, cells):
    filename = f"{kind}-{cells}.json"
    found = [root / filename for root in roots if (root / filename).is_file()]
    if len(found) != 1:
        raise ValueError(f"expected one receipt for {filename}, found {found}")
    return json.loads(found[0].read_text())


def _style(axis):
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.2)
    axis.spines["bottom"].set_linewidth(1.2)
    axis.tick_params(labelsize=20, width=1.2)
    axis.set_xlabel("realised cells", fontsize=22)


def _save_wall(booking, frozen, local, warm, output):
    measures = (
        ("exact cut-cell booking", lambda key: booking[key]["exact_booking_ms"]),
        ("hard centroid booking", lambda key: booking[key]["hard_ms"]),
        ("one-pitch smooth booking", lambda key: booking[key]["smooth_ms"]),
        (
            "held-support exact step",
            lambda key: frozen[key]["rows"]["exact_analytic_perturbed"][
                "iteration_wall_ms"
            ][-1],
        ),
        ("local derivative", lambda key: local[key]["assembly_wall_ms"]),
        ("cached-factor warm slice", None),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 10.5), sharex=True, sharey=True)
    for axis, (name, measure) in zip(axes.flat, measures, strict=True):
        for kind in CASES:
            if measure is None:
                row = warm[kind]
                slice_row = row["slices"][0]
                x = [row["realised_cells"]]
                y = [slice_row["warm_wall_ms"]]
                axis.plot(
                    x,
                    y,
                    linewidth=3.0,
                    marker="o" if slice_row["converged"] else "x",
                    markersize=11,
                    **CASE_STYLE[kind],
                )
            else:
                points = [(kind, cells) for cells in REQUESTS]
                receipts = [booking[key] for key in points]
                x = [row["realised_cells"] for row in receipts]
                y = [measure(key) for key in points]
                axis.plot(
                    x, y, linewidth=3.0, marker="o", markersize=8, **CASE_STYLE[kind]
                )
            offsets = {"limited": (5, 6), "diverted": (5, -16)}
            if name == "one-pitch smooth booking":
                offsets = {"limited": (5, -16), "diverted": (5, 6)}
            elif name == "cached-factor warm slice":
                offsets = {"limited": (10, 0), "diverted": (10, 0)}
            label = kind
            if measure is None:
                outcome = (
                    "qualified" if warm[kind]["slices"][0]["converged"] else "failed"
                )
                label = f"{kind} {outcome}"
            axis.annotate(
                label,
                (x[-1], y[-1]),
                xytext=offsets[kind],
                textcoords="offset points",
                fontsize=20,
                color=CASE_STYLE[kind]["color"],
            )
        axis.axhline(1.0, color="#777777", linewidth=1.2, linestyle=":")
        axis.set_yscale("log")
        axis.set_ylim(0.05, 10000)
        axis.set_xlim(1700, 15000)
        title_height = 0.08 if measure is None else 0.94
        axis.text(
            0.04,
            title_height,
            name,
            transform=axis.transAxes,
            fontsize=20,
            va="bottom" if measure is None else "top",
        )
        _style(axis)
        axis.set_xticks((2000, 5000, 10000), labels=("2k", "5k", "10k"))
    axes[0, 0].annotate(
        "1 ms",
        (2200, 1.0),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=18,
        color="#666666",
    )
    axes[0, 0].set_ylabel("H200 wall [ms]", fontsize=22)
    axes[1, 0].set_ylabel("H200 wall [ms]", fontsize=22)
    figure.subplots_adjust(
        left=0.09, right=0.96, bottom=0.10, top=0.98, wspace=0.10, hspace=0.22
    )
    figure.savefig(output, format="svg", facecolor="white", metadata={"Date": None})
    plt.close(figure)


def _save_flux(rows, output):
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    arms = (
        ("exact_flux_error", "exact", "#353535", "--"),
        ("hard", "hard", DEFAULT_INK.flux_color, "-"),
        ("smooth", "smooth", "#8b5a8f", "-"),
    )
    for axis, kind in zip(axes, CASES, strict=True):
        for key, label, color, linestyle in arms:
            points = [rows[(kind, cells)] for cells in REQUESTS]
            usable = [
                (
                    row["realised_cells"],
                    row[key]
                    if key == "exact_flux_error"
                    else row["candidates"][key]["analytic_flux_error"],
                )
                for row in points
                if (
                    row[key]
                    if key == "exact_flux_error"
                    else row["candidates"][key]["analytic_flux_error"]
                )
                is not None
            ]
            if not usable:
                continue
            x, y = zip(*usable, strict=True)
            axis.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=3.0,
                marker="o",
                markersize=8,
            )
            axis.annotate(
                label,
                (x[-1], y[-1]),
                xytext=(5, -14 if kind == "diverted" and key == "hard" else 7),
                textcoords="offset points",
                fontsize=20,
                color=color,
            )
        axis.set_yscale("log")
        axis.set_xlim(1700, 15000)
        axis.set_xticks((2000, 5000, 10000))
        axis.set_title(kind, fontsize=20)
        _style(axis)
    axes[0].set_ylabel("analytic flux error [relative sup]", fontsize=22)
    figure.subplots_adjust(left=0.10, right=0.96, bottom=0.16, top=0.88, wspace=0.08)
    figure.savefig(output, format="svg", facecolor="white", metadata={"Date": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--bias", type=Path)
    parser.add_argument("--frozen", type=Path, action="append", required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(f"NOVA_MODULE={nova.__file__} CWD={Path.cwd().resolve()}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    wall = {
        (kind, cells): _receipt(args.profile, kind, cells)
        for kind in CASES
        for cells in REQUESTS
    }
    frozen = {
        (kind, cells): _receipt(args.frozen, kind, cells)
        for kind in CASES
        for cells in REQUESTS
    }
    local = {
        (kind, cells): _receipt((args.local,), kind, cells)
        for kind in CASES
        for cells in REQUESTS
    }
    warm = {kind: _receipt((args.warm,), kind, 2000) for kind in CASES}
    _save_wall(wall, frozen, local, warm, args.out / "wall-by-cells.svg")
    if args.bias is not None:
        bias = {
            (kind, cells): _receipt((args.bias,), kind, cells)
            for kind in CASES
            for cells in REQUESTS
        }
        _save_flux(bias, args.out / "flux-error-by-cells.svg")


if __name__ == "__main__":
    main()
