"""Summarise what a lane run's cache lookups were served by, and plot it.

Reads a pinned pre-warm document, the lane log whose lookups are attributed,
the earlier lane log whose writes that run reads back, and the served directory
listing, then writes an attribution receipt and a three-panel figure. Every
column is drawn from a measured receipt, never from an assumption about which
key set contains which.
"""

import argparse
import json
import os
import re
from pathlib import Path

MARKER_SUFFIX = "-cache"
WALL = re.compile(r"^PYTEST_WALL_SECONDS=(\d+)")


def pin_programs(path):
    document = json.loads(Path(path).read_text())
    programs = {}
    for row in document["rows"]:
        for program in row["programs"]:
            programs.setdefault(program["cache_key"], program)
    return programs


def prewarm_programs(path):
    programs = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("{"):
            for program in json.loads(line).get("programs", []):
                key = program["cache_key"]
                previous = programs.get(key, {})
                programs[key] = {
                    "program": program.get("program") or previous.get("program", ""),
                    "compile_seconds": max(
                        program.get("compile_seconds", 0.0),
                        previous.get("compile_seconds", 0.0),
                    ),
                }
    return programs


def lane_receipt(path):
    receipt = None
    rows = {}
    wall = None
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("CACHE_GUARD_RECEIPT="):
            receipt = json.loads(line.split("=", 1)[1])
            for row in receipt["rows"]:
                rows.setdefault(row["cache_key"], row)
        found = WALL.match(line)
        if found:
            wall = int(found.group(1))
    if receipt is None:
        raise SystemExit("no receipt in %s" % path)
    return receipt, rows, wall


def served_keys(digest):
    keys = set()
    for name in os.listdir(digest):
        keys.add(name[: -len(MARKER_SUFFIX)] if name.endswith(MARKER_SUFFIX) else name)
    return keys


def short_name(program):
    return (program.rsplit("-", 1)[0] or program)[:30]


def costed(rows):
    items = [
        (row.get("compile_seconds", 0.0), key, row.get("program", "").lstrip("-"))
        for key, row in rows.items()
    ]
    items = [item for item in items if item[0] > 0.0]
    items.sort(key=lambda item: item[0], reverse=True)
    return items


def build(arguments):
    pin = pin_programs(arguments.pin)
    prewarm = prewarm_programs(arguments.prewarm_receipt)
    receipt, lane, lane_wall = lane_receipt(arguments.lane)
    earlier_receipt, earlier, earlier_wall = lane_receipt(arguments.earlier_lane)
    disk = served_keys(arguments.served_directory)
    outcome = {
        key: ("hit" if row.get("hits") and not row.get("misses") else "miss")
        for key, row in lane.items()
    }
    requested = set(lane)
    earlier_cost = costed(earlier)
    prewarm_cost = costed(prewarm)
    summary = {
        "lane_wall_seconds": lane_wall,
        "earlier_wall_seconds": earlier_wall,
        "lane_hits": sum(row.get("hits", 0) for row in lane.values()),
        "lane_programs": len(lane),
        "lane_misses": sum(row.get("misses", 0) for row in lane.values()),
        "lane_compile_seconds": round(
            sum(row.get("compile_seconds", 0.0) for row in lane.values()), 3
        ),
        "earlier_compile_seconds": round(
            sum(row.get("compile_seconds", 0.0) for row in earlier.values()), 3
        ),
    }
    summary.update(
        {
            "earlier_programs_with_compile": len(earlier_cost),
            "pin_programs": len(pin),
            "pin_shared_with_lane": len(set(pin) & requested),
            "served_entries": len(disk),
            "lane_keys_absent_from_disk": len(requested - disk),
            "pin_keys_absent_from_disk": len(set(pin) - disk),
            "costliest_lane_programs": [
                {
                    "program": program,
                    "earlier_compile_seconds": round(seconds, 3),
                    "later_compile_seconds": round(
                        lane.get(key, {}).get("compile_seconds", 0.0), 3
                    ),
                    "later_outcome": outcome.get(key, "not-requested"),
                }
                for seconds, key, program in earlier_cost[: arguments.top]
            ],
            "costliest_prewarm_programs": [
                {
                    "program": program,
                    "prewarm_compile_seconds": round(seconds, 3),
                    "requested_by_lane": key in requested,
                    "later_outcome": outcome.get(key, "not-requested"),
                }
                for seconds, key, program in prewarm_cost[: arguments.top]
            ],
        }
    )
    return summary, (earlier_cost, prewarm_cost, requested, outcome, earlier_receipt)


def render(summary, series, path_stem, top):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    earlier_cost, prewarm_cost, requested, outcome, _earlier_receipt = series
    colour = {"hit": "#2e7d32", "miss": "#c62828", "not-requested": "#9e9e9e"}

    def bars(axis, rows, colours, xlabel, title):
        axis.barh(
            range(len(rows)),
            [value for value, _k, _p in rows][::-1],
            color=colours[::-1],
        )
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels(
            [short_name(program) for _v, _k, program in rows][::-1], fontsize=7
        )
        axis.set_xlabel(xlabel)
        axis.set_title(title, fontsize=10)
        axis.grid(False)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    figure, (first, second, third) = plt.subplots(1, 3, figsize=(16.4, 6.2))
    rows = earlier_cost[:top]
    bars(
        first,
        rows,
        [colour[outcome.get(key, "not-requested")] for _v, key, _p in rows],
        "compile seconds, earlier lane run",
        "earlier lane run\nbars marked by the attributed run's outcome",
    )
    rows = prewarm_cost[:top]
    bars(
        second,
        rows,
        [
            colour["hit"] if key in requested else colour["not-requested"]
            for _v, key, _p in rows
        ],
        "compile seconds, pre-warm",
        "pre-warm\ngrey bars: the attributed run never requests them",
    )
    third.bar(
        ["earlier run", "attributed run"],
        [summary["earlier_wall_seconds"] or 0, summary["lane_wall_seconds"] or 0],
        color=["#1565c0", "#2e7d32"],
    )
    third.set_ylabel("wall seconds, lane job")
    third.set_title("wall clock, same target and node: one compile", fontsize=10)
    third.annotate(
        "%d s, of which %d s compiling"
        % (summary["earlier_wall_seconds"], round(summary["earlier_compile_seconds"])),
        (0, summary["earlier_wall_seconds"]),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    third.grid(False)
    for side in ("top", "right"):
        third.spines[side].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colour[name])
        for name in ("hit", "miss", "not-requested")
    ]
    labels = [
        "hit in the attributed run",
        "miss in the attributed run (%d)" % summary["lane_misses"],
        "program the attributed run never requests",
    ]
    figure.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path_stem + ".png", dpi=140)
    figure.savefig(path_stem + ".svg")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--prewarm-receipt", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--earlier-lane", required=True)
    parser.add_argument("--served-directory", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--top", type=int, default=20)
    arguments = parser.parse_args()
    summary, series = build(arguments)
    Path(arguments.receipt).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    render(summary, series, arguments.figure, arguments.top)
    for key in sorted(summary):
        if not isinstance(summary[key], list):
            print("%-34s %s" % (key, summary[key]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
