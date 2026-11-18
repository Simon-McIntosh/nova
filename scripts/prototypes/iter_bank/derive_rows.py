"""Attribute the rebuilt bank's first-derivative rows against the two banks.

Reads the two published bank npz archives, takes the two first-derivative
components (d psi / d R and d psi / d Z), and reports:

* the unfiltered maximum relative departure and where it sits, with its
  absolute size and the magnitude it was divided by;
* the same over the entries whose magnitude exceeds a filter window, which is
  the statistic the fragment quotes beside the 1e-10 done-when;
* one line per entry whose relative departure exceeds a threshold, carrying
  its node, cell, moment, component, absolute departure and magnitude;
* a MATERIAL verdict on whether any entry of magnitude above the window
  departs by more than 1e-7 in absolute terms, the boundary the lead named for
  withdrawing the accepted exception.

Every figure is computed per entry: the relative departure is
``|head - current| / |current|``, so a near-zero magnitude can carry a large
relative departure while its absolute departure stays at the archive's noise.
The node-aggregated norm the certification script reports is printed beside it
so the two statistics cannot be mistaken for one another.

Run it from the repository root with ``PYTHONPATH=.``, so the shared helper
(and this script's own module path) resolves:
``PYTHONPATH=. python scripts/prototypes/iter_bank/derive_rows.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.prototypes.iter_bank.recertify import relative

DERIVATIVE = slice(1, 3)
COMPONENT = {0: "dpsi/dR", 1: "dpsi/dZ"}
DEFAULT_THRESHOLD = 1e-10
DEFAULT_WINDOW = 1e-3
DEFAULT_BOUNDARY = None


def load(path):
    """Read a bank npz and return its rows, active-node count and path."""
    with np.load(path, allow_pickle=False) as archive:
        return dict(
            rows=np.asarray(archive["rows"]),
            active_nodes=int(archive["active_nodes"]),
            bucket=int(archive["bucket"]),
            path=str(path),
        )


def compare(
    head_rows,
    current_rows,
    *,
    threshold=DEFAULT_THRESHOLD,
    window=DEFAULT_WINDOW,
    boundary=DEFAULT_BOUNDARY,
    components=DERIVATIVE,
):
    """Compare two stored row arrays and return the report the script prints.

    ``head_rows`` and ``current_rows`` are the archives' ``rows`` arrays, shaped
    ``(nodes, cells, moments, 6)``; ``components`` selects the derivative slice
    under comparison, so a different slice is a different measurement and every
    statistic here is computed from that slice alone.
    """
    head = np.asarray(head_rows)[:, :, :, components].astype(np.float64)
    current = np.asarray(current_rows)[:, :, :, components].astype(np.float64)
    delta = head - current
    magnitude = np.abs(current)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(magnitude > 0.0, np.abs(delta) / magnitude, np.inf)
    zero_magnitude = magnitude == 0.0
    finite = np.isfinite(rel)
    keep = magnitude > window

    def locate(flat_index):
        node, cell, moment, component = np.unravel_index(flat_index, rel.shape)
        return dict(
            node=int(node),
            cell=int(cell),
            moment=int(moment),
            component=COMPONENT[int(component)],
            absolute_departure=float(delta[node, cell, moment, component]),
            magnitude=float(current[node, cell, moment, component]),
            relative=float(rel[node, cell, moment, component]),
        )

    unfiltered = dict(
        count=int(finite.sum()),
        max_relative=float(np.max(rel[finite])),
        median_relative=float(np.median(rel[finite])),
        location=locate(int(np.argmax(np.where(finite, rel, -np.inf)))),
        above_threshold=int((rel > threshold).sum()),
    )
    filtered = dict(
        window=window,
        count=int(keep.sum()),
        excluded=int((~keep & finite).sum()),
        max_relative=float(np.max(np.where(keep, rel, -np.inf))),
        median_relative=float(np.median(rel[keep])) if keep.any() else None,
        above_threshold=int(((rel > threshold) & keep).sum()),
        max_absolute=float(np.max(np.abs(delta)[keep])) if keep.any() else None,
        max_relative_location=locate(int(np.argmax(np.where(keep, rel, -np.inf)))),
        max_absolute_location=locate(
            int(np.argmax(np.where(keep, np.abs(delta), -np.inf)))
        ),
    )
    report = dict(
        entries=int(rel.size),
        components=[COMPONENT[index] for index in COMPONENT],
        zero_magnitude_entries=int(zero_magnitude.sum()),
        threshold=threshold,
        window=window,
        terminal_boundary=boundary,
        unfiltered=unfiltered,
        filtered=filtered,
    )
    if zero_magnitude.any():
        report["zero_magnitude_max_abs_delta"] = float(
            np.max(np.abs(delta)[zero_magnitude])
        )
    for index, name in COMPONENT.items():
        part = rel[..., index]
        part_finite = np.isfinite(part)
        part_keep = magnitude[..., index] > window
        entry = dict(
            count=int(part_finite.sum()),
            above_threshold=int((part > threshold).sum()),
            filtered_count=int(part_keep.sum()),
            filtered_above_threshold=int(((part > threshold) & part_keep).sum()),
            max_relative=float(np.max(part[part_finite])),
            median_relative=float(np.median(part[part_finite])),
            filtered_max_relative=None,
            filtered_median_relative=None,
            filtered_max_absolute=None,
            filtered_max_relative_location=None,
        )
        if part_keep.any():
            entry["filtered_max_relative"] = float(
                np.max(np.where(part_keep, part, -np.inf))
            )
            entry["filtered_median_relative"] = float(np.median(part[part_keep]))
            entry["filtered_max_absolute"] = float(
                np.max(np.abs(delta[..., index])[part_keep])
            )
            selected = int(np.argmax(np.where(part_keep, part, -np.inf)))
            node, cell, moment = np.unravel_index(selected, part.shape)
            entry["filtered_max_relative_location"] = dict(
                node=int(node),
                cell=int(cell),
                moment=int(moment),
                component=name,
                absolute_departure=float(delta[..., index][node, cell, moment]),
                magnitude=float(current[..., index][node, cell, moment]),
                relative=float(part[node, cell, moment]),
            )
        report[name] = entry
    departed = np.argwhere(rel > threshold)
    report["departing_count"] = int(len(departed))
    report["departures"] = [
        locate(int(np.ravel_multi_index(tuple(row), rel.shape))) for row in departed
    ]
    aggregated = relative(delta, current, axis=0)
    report["node_aggregated"] = dict(
        shape=list(aggregated.shape),
        max_relative=float(np.max(aggregated)),
        median_relative=float(np.median(aggregated)),
    )
    if boundary is None:
        report["material"] = dict(boundary=None, count=0, present=None)
    else:
        material = keep & (np.abs(delta) > boundary)
        report["material"] = dict(
            boundary=boundary, count=int(material.sum()), present=bool(material.any())
        )
        if material.any():
            report["material"]["largest"] = locate(
                int(np.argmax(np.where(material, np.abs(delta), -np.inf)))
            )
    return report


def exit_status(report):
    """The status the script leaves: 2 only when a named boundary is crossed."""
    return 2 if report["material"]["present"] else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-bank", type=Path, required=True)
    parser.add_argument("--current-bank", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW)
    parser.add_argument(
        "--material",
        type=float,
        default=DEFAULT_BOUNDARY,
        help="absolute departure that fails the run; none by default",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    head = load(args.head_bank)
    current = load(args.current_bank)
    if head["rows"].shape != current["rows"].shape:
        raise SystemExit("the two banks have different row shapes")
    active = min(head["active_nodes"], current["active_nodes"])
    if head["active_nodes"] != current["active_nodes"]:
        print("ACTIVE_NODES_DIFFER", head["active_nodes"], current["active_nodes"])
    print(
        "BANKS "
        + json.dumps(
            dict(
                head=head["path"],
                current=current["path"],
                head_shape=list(head["rows"].shape),
                current_shape=list(current["rows"].shape),
                head_active_nodes=head["active_nodes"],
                current_active_nodes=current["active_nodes"],
                active=active,
            ),
            sort_keys=True,
        ),
        flush=True,
    )
    report = compare(
        head["rows"][:active],
        current["rows"][:active],
        threshold=args.threshold,
        window=args.window,
        boundary=args.material,
    )
    report["head_bank"] = head["path"]
    report["current_bank"] = current["path"]
    for key in ("unfiltered", "filtered", "node_aggregated", "dpsi/dR", "dpsi/dZ"):
        print(key.upper() + " " + json.dumps(report[key], sort_keys=True), flush=True)
    for departure in report["departures"]:
        print("DEPARTURE " + json.dumps(departure, sort_keys=True), flush=True)
    print("MATERIAL " + json.dumps(report["material"], sort_keys=True), flush=True)
    report["completed"] = True
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print("DERIVE_COMPLETE " + str(args.output), flush=True)
    raise SystemExit(exit_status(report))


if __name__ == "__main__":
    main()
