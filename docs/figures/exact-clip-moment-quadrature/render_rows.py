"""Row-by-row convergence of the closed-form cut moments against their fences.

Every swept row carries a receipt under the closed-form sweep's report
directory: ``parts/<case>-<cells>.json`` holds, per arc vertex count, the three
moment series' relative L2 against the retained fan. Each row's fence is one
tenth of the smallest other error term that row carries; on the four rows the
coupling study reached that term is the row's own frozen-image error, and on the
five it did not the term is the fan's refinement floor, whose tenth sits below
any polyline's own round-off floor.

Panel (a) draws one row's three series against the counts, so the order of
convergence and the floor it lands on are both visible. Panel (b) draws every
swept row normalised by its own fence, and carries the five fallback rows that
the earlier figure left out: their fence is replaced by the closed-form
evaluation's own round-off envelope, measured independently of the fan by
``closed_form_roundoff_floor.py``, so a finite fence exists for them and their
position against it is a reading rather than an absent column. A row meets its
fence at the first count whose worst moment sits at or below one.
"""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
REPORTS = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local/exact-closed-form"
)
FLOOR = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local/"
    "ecq-fallback-floor/closed-form-roundoff-floor.json"
)
READ = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local/"
    "ecq-fallback-floor/fallback-floor-read.json"
)
MOMENTS = ("current", "radial", "vertical")
MOMENT_COLOUR = {"current": "#1f77b4", "radial": "#d95f02", "vertical": "#2e7d32"}
CASE_LABEL = {
    "weak-rotation-reactor-static": "weak",
    "moderate-rotation-conventional-static": "moderate",
    "strong-rotation-compact-static": "strong",
}
CASE_COLOUR = {
    "weak-rotation-reactor-static": "#1f77b4",
    "moderate-rotation-conventional-static": "#d95f02",
    "strong-rotation-compact-static": "#2e7d32",
}
CELL_STYLE = {-110: "-", -300: "--", -1000: ":"}
#: The floor receipt names the current moment's quantity ``area``.
FLOOR_KEY = {"area": "current", "radial": "radial", "vertical": "vertical"}
#: The row the two panels' series are drawn from: the only row the sweep meets
#: below the top of its range, so inflections inside the range are visible.
EXAMPLE = ("weak-rotation-reactor-static", -110)


def receipts() -> dict[tuple[str, int], dict]:
    measured = {}
    for path in sorted((REPORTS / "parts").glob("*.json")):
        row = json.loads(path.read_text())
        measured[(row["case"], row["requested_cells"])] = row
    return measured


def counts_of(row: dict) -> list[int]:
    return sorted(int(key) for key in row["counts"])


def series(row: dict, count: int) -> dict[str, float]:
    return row["counts"][str(count)]["moment_relative_l2_against_fan"]


def fence(row: dict, floor: dict[str, float]) -> dict[str, float]:
    """The row's own fence: its coupling tenth, or the measured floor."""
    if len(set(row["counts"]["8"]["budget"].values())) == 1:
        stored = row["counts"]["8"]["budget"]
        return {name: stored[name] / 10.0 for name in MOMENTS}
    return {name: floor[name] for name in MOMENTS}


def smallest_met(row: dict, fence_values: dict[str, float]) -> int | None:
    for count in counts_of(row):
        observed = series(row, count)
        if all(observed[name] <= fence_values[name] for name in MOMENTS):
            return count
    return None


def main() -> None:
    measured = receipts()
    floor_receipt = json.loads(FLOOR.read_text())
    floor = {
        FLOOR_KEY[name]: value
        for name, value in floor_receipt["measured_envelope"].items()
    }
    read = json.loads(READ.read_text())
    fallback = {(row["case"], row["requested_cells"]) for row in read["rows"]}
    example = measured[EXAMPLE]
    counts = counts_of(example)

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=150)
    for name in MOMENTS:
        left.plot(
            counts,
            [series(example, count)[name] for count in counts],
            marker="o",
            color=MOMENT_COLOUR[name],
            label=name,
        )
    example_fence = fence(example, floor)
    left.axhline(
        example_fence["current"],
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="fence: one tenth of the row's other error term",
    )
    first = smallest_met(example, example_fence)
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xticks(counts)
    left.set_xticklabels([str(count) for count in counts])
    left.set_xlabel("arc vertex count")
    left.set_ylabel("relative L2 against the retained fan")
    left.grid(alpha=0.25, which="both")
    left.legend(fontsize=8, frameon=False, loc="lower left")
    left.set_title(
        "(a) %s %d: the three moment series against the fence\n"
        "smallest count met: %s"
        % (
            CASE_LABEL[EXAMPLE[0]],
            abs(EXAMPLE[1]),
            "none" if first is None else str(first),
        ),
        fontsize=10,
    )

    for (case, cells), row in sorted(measured.items()):
        row_fence = fence(row, floor)
        counts_row = counts_of(row)
        ratio = [
            max(series(row, count)[name] / row_fence[name] for name in MOMENTS)
            for count in counts_row
        ]
        is_fallback = (case, cells) in fallback
        right.plot(
            counts_row,
            ratio,
            marker="o" if not is_fallback else "D",
            markersize=4,
            color=CASE_COLOUR[case],
            linestyle=CELL_STYLE[cells],
            label="%s %d" % (CASE_LABEL[case], abs(cells)),
        )
    right.axhline(1.0, color="#333333", linewidth=1.4)
    right.annotate(
        "the fence: coupling tenth where the study reached the row,\n"
        "closed form's own round-off envelope on the five fallback rows",
        xy=(8.2, 1.0),
        xytext=(0, -44),
        textcoords="offset points",
        fontsize=7.0,
        color="#333333",
    )
    right.set_xscale("log", base=2)
    right.set_yscale("log")
    right.set_xticks([8, 16, 32, 64, 128])
    right.set_xticklabels(["8", "16", "32", "64", "128"])
    right.set_xlabel("arc vertex count")
    right.set_ylabel("worst moment, as a multiple of the row's fence")
    right.set_title(
        "(b) all nine swept rows, the five fallback rows (diamonds) included;\n"
        "the four coupling rows cross the fence, the five fallback rows do not",
        fontsize=10,
    )
    right.grid(alpha=0.25, which="both")
    right.legend(fontsize=8, frameon=False, loc="lower left", ncols=3)

    caption = (
        "Fences are per-row and per-moment: on the four rows the coupling study "
        "reached, one tenth of that row's frozen-image error; on the five it did "
        "not, the closed-form evaluation's own round-off envelope (%.2e current, "
        "%.2e radial, %.2e vertical), measured on an analytic level set as the "
        "largest relative difference two independent arc parametrisations of it "
        "produce. The five fallback rows are drawn here for the first time: their "
        "worst moment at 128 vertices (1.59e-15 to 2.50e-15) sits above that "
        "envelope, because at 128 vertices the residual against the retained fan "
        "is the fan's own round-off rather than the closed form's, so no swept "
        "count meets either fence and the honest column stays 'none'."
        % (floor["current"], floor["radial"], floor["vertical"])
    )
    figure.text(
        0.5,
        0.015,
        "\n".join(textwrap.wrap(caption, 132)),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#333333",
    )
    figure.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.26)
    for suffix in (".svg", ".png"):
        figure.savefig(OUT / f"rows{suffix}")
    print("wrote", OUT / "rows.png")
    print("floor", json.dumps(floor, sort_keys=True))
    for key, row in sorted(measured.items()):
        print(
            "row %s %d fallback=%s smallest_met=%s"
            % (
                CASE_LABEL[key[0]],
                abs(key[1]),
                key in fallback,
                smallest_met(row, fence(row, floor)),
            )
        )


if __name__ == "__main__":
    main()
