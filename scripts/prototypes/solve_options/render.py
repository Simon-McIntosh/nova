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
ARMS = (
    ("exact_booking_ms", "exact booking"),
    ("hard_ms", "hard centroid"),
    ("smooth_ms", "smoothed centroid"),
)


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


def _save_wall(rows, output):
    figure, axes = plt.subplots(1, len(ARMS), figsize=(14, 5.5), sharey=True)
    for axis, (key, name) in zip(axes, ARMS, strict=True):
        for kind in CASES:
            points = [rows[(kind, cells)] for cells in REQUESTS]
            x = [row["realised_cells"] for row in points]
            y = [row[key] for row in points]
            axis.plot(x, y, linewidth=3.0, marker="o", markersize=8, **CASE_STYLE[kind])
            axis.annotate(
                kind,
                (x[-1], y[-1]),
                xytext=(5, 5 if kind == "limited" else -12),
                textcoords="offset points",
                fontsize=20,
                color=CASE_STYLE[kind]["color"],
            )
        axis.axhline(1.0, color="#777777", linewidth=1.2, linestyle=":")
        axis.annotate(
            "1 ms",
            (2100, 1.0),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=18,
            color="#666666",
        )
        axis.set_yscale("log")
        axis.set_ylim(0.05, 500)
        axis.set_xlim(1700, 15000)
        axis.set_title(name, fontsize=20)
        _style(axis)
    axes[0].set_ylabel("warm booking wall [ms]", fontsize=22)
    figure.subplots_adjust(left=0.08, right=0.96, bottom=0.19, top=0.88, wspace=0.08)
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(f"NOVA_MODULE={nova.__file__} CWD={Path.cwd().resolve()}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    wall = {
        (kind, cells): _receipt(args.profile, kind, cells)
        for kind in CASES
        for cells in REQUESTS
    }
    _save_wall(wall, args.out / "wall-by-cells.svg")
    if args.bias is not None:
        bias = {
            (kind, cells): _receipt((args.bias,), kind, cells)
            for kind in CASES
            for cells in REQUESTS
        }
        _save_flux(bias, args.out / "flux-error-by-cells.svg")


if __name__ == "__main__":
    main()
