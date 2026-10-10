"""Render the resolution-pivot assessment figures from run receipts.

Every plotted number is read from a receipt written by the measurement run
that produced it; nothing is typed into this module. The report tables are
read as well, purely as a cross-check, so that a receipt that has drifted
from the figure it is supposed to support is refused rather than drawn.

Provenance of the plotted series:

* Certificate-map relative sup, and the exact arm's booking-image and
  exterior-closure decomposition -- the row receipts under
  ``r-20261009T181553786165-proto-support``.
* Topology-read saddle membership error against realised cells, the
  per-rung acceptance bound, and the lost-X-point row -- the row receipts
  under ``r-20261009T160304793011-proto-ladder`` (current read and the
  quadratic-only arm) and that run's ``proto-ladder.md`` acceptance table.
* Expanded equations per packed-kernel call site and the cold compile
  walls, native against batched -- the attribution and row receipts under
  ``r-20261009T160337032033-proto-order2/shared-kernel-1283105``.

Run with the shared interpreter and the worktree on the path::

    /home/ITER/mcintos/Code/nova/.venv/bin/python \
        scripts/prototypes/pivot_figures.py

Negative control: the script refuses to draw when a cited receipt file is
missing, or when a plotted receipt value disagrees with the value stated in
the run's report table.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("data-ink")

RUNS = Path.home() / ".config/reckon/crew/runs"
REPORTS = Path.home() / ".config/reckon/crew/reports/nova/resolution-pivot"

SUPPORT_ROWS = RUNS / "r-20261009T181553786165-proto-support/ordered/rows"
LADDER = RUNS / "r-20261009T160304793011-proto-ladder"
ORDER2 = RUNS / "r-20261009T160337032033-proto-order2/shared-kernel-1283105"

OUT = Path(__file__).resolve().parents[2] / "docs/figures/resolution-pivot"

MAP_BOUND = 0.01  # plan acceptance bound for certificate-map relative sup

# Series colours: one hue per case, a lightness step per support arm.
DIVERTED_EXACT = "#1f6f9c"
DIVERTED_LEGACY = "#4a94c0"
LIMITED_EXACT = "#a8500f"
LIMITED_LEGACY = "#c87a2c"
BOOKING = "#123a52"
EXTERIOR = "#7a3a08"
GREY = "#6b6b6b"

LABEL_SIZE = 20
TICK_SIZE = 20
AXIS_LABEL_SIZE = 22
HEADLINE_LW = 3.0
OTHER_LW = 2.6
DASH_LW = 2.4
BOUND_LW = 1.2


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"REFUSING: cited receipt is missing: {path}")
    with path.open() as handle:
        return json.load(handle)


def _read_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"REFUSING: cited report is missing: {path}")
    return path.read_text()


def _assert_matches(
    receipt: float, stated: float, what: str, tol: float = 1e-8
) -> None:
    scale = max(abs(receipt), abs(stated), 1.0)
    if abs(receipt - stated) > tol * scale:
        raise SystemExit(
            f"REFUSING: {what}: receipt {receipt!r} disagrees with report {stated!r}"
        )


def support_rows() -> dict[str, dict]:
    wanted = (
        "diverted-550-exact",
        "diverted-2000-exact",
        "diverted-5000-exact",
        "diverted-550-legacy",
        "diverted-2000-legacy",
        "diverted-5000-legacy",
        "limited-550-exact",
        "limited-2000-exact",
        "limited-5000-exact",
        "limited-550-legacy",
        "limited-2000-legacy",
        "limited-5000-legacy",
    )
    return {name: _load(SUPPORT_ROWS / f"{name}.json") for name in wanted}


def ladder_rows() -> dict[str, dict]:
    wanted = ("diverted-550-A", "diverted-2000-A", "diverted-5000-A")
    rows = {name: _load(LADDER / "rows" / f"{name}.json") for name in wanted}
    quad = ("diverted-550", "diverted-2000", "diverted-5000")
    rows.update(
        {
            f"quad-{name}": _load(LADDER / "quadratic-rows" / f"{name}.json")
            for name in quad
        }
    )
    return rows


def ladder_bounds() -> dict[tuple[str, int], float]:
    """Per-rung acceptance bound, read from the run's acceptance table."""
    text = _read_text(REPORTS / "proto-ladder.md")
    bounds: dict[tuple[str, int], float] = {}
    for match in re.finditer(
        r"^\| (limited|diverted) \| (\d+) \| ([0-9.eE+-]+) \|", text, re.MULTILINE
    ):
        bounds[(match.group(1), int(match.group(2)))] = float(match.group(3))
    expected = {
        ("limited", 550): 0.000343398,
        ("limited", 2000): 9.00177e-05,
        ("limited", 5000): 3.52633e-05,
        ("diverted", 550): 0.0038486,
        ("diverted", 2000): 0.00100109,
        ("diverted", 5000): 0.00039275,
    }
    for key, value in expected.items():
        if key not in bounds:
            raise SystemExit(
                f"REFUSING: bound for {key} absent from the acceptance table"
            )
        _assert_matches(bounds[key], value, f"bound {key}")
    return bounds


def order2_series() -> dict:
    native = _load(ORDER2 / "native-132-attribution.json")
    batched = _load(ORDER2 / "batched-132-attribution.json")
    walls = {}
    for variant in ("native", "batched"):
        for cells in (132, 550):
            receipt = _load(ORDER2 / f"{variant}-{cells}.json")
            walls[(variant, cells)] = receipt["cold_compile_wall_seconds"]
    return {"native": native, "batched": batched, "walls": walls}


def direct_label(ax, x, y, text, colour, dx=0.0, dy=0.0, size=LABEL_SIZE):
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(6 + dx, dy),
        ha="left",
        va="center",
        fontsize=size,
        color=colour,
    )


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_certificate_map(rows: dict[str, dict], report_text: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 9))

    def series(case: str, arm: str, field: str):
        keys = [f"{case}-{rung}-{arm}" for rung in (550, 2000, 5000)]
        xs = [rows[key]["realised_cells"] for key in keys]
        ys = [rows[key][field] for key in keys]
        return xs, ys

    specs = [
        ("diverted", "exact", "map_relative_sup", DIVERTED_EXACT, "diverted exact"),
        ("diverted", "legacy", "map_relative_sup", DIVERTED_LEGACY, "diverted legacy"),
        ("limited", "exact", "map_relative_sup", LIMITED_EXACT, "limited exact"),
        ("limited", "legacy", "map_relative_sup", LIMITED_LEGACY, "limited legacy"),
    ]
    for case, arm, field, colour, label in specs:
        xs, ys = series(case, arm, field)
        ax.plot(xs, ys, "-o", color=colour, lw=HEADLINE_LW, ms=11, label=label)
        direct_label(ax, xs[-1], ys[-1], label, colour)

    dash_specs = [
        (
            "diverted",
            "booking_image_relative_sup",
            BOOKING,
            (0, (6, 3)),
            "booking (exact)",
        ),
        (
            "diverted",
            "exterior_closure_relative_sup",
            BOOKING,
            (0, (2, 3)),
            "exterior (exact)",
        ),
        (
            "limited",
            "booking_image_relative_sup",
            EXTERIOR,
            (0, (6, 3)),
            "booking (exact)",
        ),
        (
            "limited",
            "exterior_closure_relative_sup",
            EXTERIOR,
            (0, (2, 3)),
            "exterior (exact)",
        ),
    ]
    for case, field, colour, dash, label in dash_specs:
        xs, ys = series(case, "exact", field)
        ax.plot(xs, ys, color=colour, lw=DASH_LW, ls=dash, label=f"{case} {label}")

    ax.axhline(MAP_BOUND, color=GREY, lw=BOUND_LW, ls=(0, (1, 2)), label="bound 0.01")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(460.0, 9000.0)
    ax.set_ylim(6e-8, 1.1)
    ax.set_xlabel("realised cells", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("relative sup", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE, width=1.2)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    despine(ax)
    handles, labels = ax.get_legend_handles_labels()
    order = [
        i for i, label in enumerate(labels) if "(" in label or label.startswith("bound")
    ]
    legend = ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        fontsize=LABEL_SIZE - 6,
        frameon=False,
        loc="lower right",
        ncol=2,
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(2.4)
    fig.tight_layout()
    fig.savefig(OUT / "certificate-map-support.svg")

    _assert_matches(
        rows["diverted-2000-exact"]["map_relative_sup"],
        0.0194733185172,
        "diverted exact map at 2098 cells",
    )
    _assert_matches(
        rows["diverted-5000-legacy"]["map_relative_sup"],
        0.0861086001,
        "diverted legacy map at 5158 cells",
    )
    _assert_matches(
        rows["diverted-550-exact"]["booking_image_relative_sup"],
        0.0249260493055,
        "diverted exact booking at 550 cells",
    )
    _assert_matches(
        rows["diverted-5000-exact"]["exterior_closure_relative_sup"],
        0.0111257410628,
        "diverted exact exterior at 5158 cells",
    )
    for needle in ("0.0194733185172", "0.197326979", "0.0861086001"):
        if needle not in report_text:
            raise SystemExit(f"REFUSING: report table no longer states {needle}")
    plt.close(fig)


def figure_topology_read(
    rows: dict[str, dict], bounds: dict[tuple[str, int], float]
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))

    def poly(prefix, field, cells):
        xs = [rows[f"{prefix}-{cell}"]["cells"] for cell in cells]
        ys = [rows[f"{prefix}-{cell}"][field] for cell in cells]
        return xs, ys

    rungs = (550, 2000, 5000)
    ax_x, ax_y = poly("diverted", "saddle_membership_error", [f"{c}-A" for c in rungs])
    qx, qy = poly("quad-diverted", "saddle_membership_error", rungs)
    bx = np.array([rows[f"diverted-{cell}-A"]["cells"] for cell in rungs], dtype=float)
    by = np.array([bounds[("diverted", cell)] for cell in rungs], dtype=float)

    ax.plot(bx, by, color=GREY, lw=BOUND_LW, ls=(0, (1, 2)))
    ax.annotate(
        "saddle membership bound",
        (bx[-1], by[-1]),
        fontsize=LABEL_SIZE,
        color=GREY,
        va="top",
        ha="right",
        xytext=(-4, -6),
        textcoords="offset points",
    )
    ax.plot(ax_x, ax_y, "-o", color=DIVERTED_EXACT, lw=HEADLINE_LW, ms=11)
    direct_label(ax, ax_x[-1], ax_y[-1], "current read", DIVERTED_EXACT)
    ax.plot(qx, qy, "-s", color=LIMITED_EXACT, lw=OTHER_LW, ms=11)
    direct_label(ax, qx[0], qy[0], "quadratic-only arm", LIMITED_EXACT, dx=8, dy=-20)

    lost_x, lost_y = qx[1], qy[1]
    if rows["quad-diverted-2000"]["x_error_pitches"] is not None:
        raise SystemExit(
            "REFUSING: the 2000-cell quadratic row is expected to lose its X-point"
        )
    ax.plot(
        [lost_x],
        [lost_y],
        "o",
        ms=20,
        mfc="none",
        mec=LIMITED_EXACT,
        mew=2.4,
    )
    ax.annotate(
        "X-point lost at 2000 cells",
        (lost_x, lost_y),
        fontsize=LABEL_SIZE,
        color=LIMITED_EXACT,
        xytext=(-10, 22),
        textcoords="offset points",
        ha="right",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(480.0, 7200.0)
    ax.set_xlabel("cells", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("saddle membership error", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE, width=1.2)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    despine(ax)
    fig.tight_layout()
    fig.savefig(OUT / "topology-read-membership.svg")

    _assert_matches(
        rows["diverted-5000-A"]["saddle_membership_error"],
        2.4920558962140258e-05,
        "current read saddle error at 5000 cells",
    )
    _assert_matches(
        rows["quad-diverted-2000"]["saddle_membership_error"],
        0.7280493010799182,
        "quadratic-only saddle error at 2000 cells",
    )
    plt.close(fig)


def figure_kernel_equations(data: dict) -> None:
    stages = ["_sampled_flux", "_null_census", "_curvature_derivatives", "_wall_events"]
    fig, ax = plt.subplots(figsize=(14, 9))

    width = 0.8
    x = 0.0
    centres, labels = [], []
    top = 0.0
    for stage in stages:
        sites_n = [
            s for s in data["native"]["packed_kernel_call_sites"] if s["stage"] == stage
        ]
        sites_b = [
            s
            for s in data["batched"]["packed_kernel_call_sites"]
            if s["stage"] == stage
        ]
        group_start = x
        for site in sites_n:
            count = site["body_expanded_equations"]
            ax.bar(x, count, width=width, color=BOOKING, linewidth=0)
            top = max(top, count)
            x += width + 0.15
        x += 0.5
        for site in sites_b:
            count = site["body_expanded_equations"]
            ax.bar(x, count, width=width, color=LIMITED_EXACT, hatch="//", linewidth=0)
            top = max(top, count)
            x += width + 0.15
        centres.append(0.5 * (group_start + x - width - 0.15))
        labels.append(stage.lstrip("_").replace("_", "\n"))
        x += 1.0

    for variant in ("native", "batched"):
        sites = data[variant]["packed_kernel_call_sites"]
        partition = sum(site["body_expanded_equations"] for site in sites) + len(sites)
        if partition != data[variant]["kernel_call_partition_equations"]:
            raise SystemExit(
                f"REFUSING: {variant} site bodies plus call equations do not "
                "reproduce the recorded kernel-call partition"
            )

    native_wall = data["walls"][("native", 132)]
    batched_wall = data["walls"][("batched", 132)]
    ax.set_ylim(0.0, top * 1.75)
    ax.text(
        0.5,
        0.97,
        f"native: 10 sites, {data['native']['expanded_equations']:,} "
        f"expanded equations, cold compile {native_wall:.1f} s",
        transform=ax.transAxes,
        fontsize=LABEL_SIZE,
        color=BOOKING,
        ha="center",
        va="top",
    )
    ax.text(
        0.5,
        0.90,
        f"batched: 5 sites, {data['batched']['expanded_equations']:,} "
        f"expanded equations, cold compile {batched_wall:.1f} s",
        transform=ax.transAxes,
        fontsize=LABEL_SIZE,
        color=LIMITED_EXACT,
        ha="center",
        va="top",
    )

    ax.set_xticks(centres)
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax.set_xlim(-1.0, x)
    ax.set_ylabel("expanded equations per call site", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE, width=1.2)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    despine(ax)

    first_native = data["native"]["packed_kernel_call_sites"][0][
        "body_expanded_equations"
    ]
    first_batched = data["batched"]["packed_kernel_call_sites"][0][
        "body_expanded_equations"
    ]
    ax.annotate(
        "native",
        (0.0, first_native),
        fontsize=LABEL_SIZE,
        color=BOOKING,
        ha="center",
        va="bottom",
        xytext=(0, 4),
        textcoords="offset points",
    )
    ax.annotate(
        "batched",
        (1.9, first_batched),
        fontsize=LABEL_SIZE,
        color=LIMITED_EXACT,
        ha="center",
        va="bottom",
        xytext=(0, 4),
        textcoords="offset points",
    )
    fig.tight_layout()
    fig.savefig(OUT / "kernel-expanded-equations.svg")

    _assert_matches(native_wall, 932.8877603709698, "native 132-cell cold compile")
    _assert_matches(batched_wall, 477.65302151814103, "batched 132-cell cold compile")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = support_rows()
    figure_certificate_map(rows, _read_text(REPORTS / "proto-support.md"))
    figure_topology_read(ladder_rows(), ladder_bounds())
    figure_kernel_equations(order2_series())
    print("wrote three figures to", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
