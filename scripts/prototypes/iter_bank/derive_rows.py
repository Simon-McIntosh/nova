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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DERIVATIVE = slice(1, 3)
COMPONENT = {0: "dpsi/dR", 1: "dpsi/dZ"}


def load(path):
    """Read a bank npz and return its rows, active-node count and path."""
    with np.load(path, allow_pickle=False) as archive:
        return dict(
            rows=np.asarray(archive["rows"]),
            active_nodes=int(archive["active_nodes"]),
            bucket=int(archive["bucket"]),
            path=str(path),
        )


def relative(delta, scale, axis=0):
    """The certification script's node-aggregated relative norm."""
    numerator = np.sum(delta**2, axis=axis)
    denominator = np.sum(np.asarray(scale) ** 2, axis=axis)
    return np.asarray(np.sqrt(numerator / denominator), dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-bank", type=Path, required=True)
    parser.add_argument("--current-bank", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1e-10)
    parser.add_argument("--window", type=float, default=1e-3)
    parser.add_argument("--material", type=float, default=1e-7)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    head = load(args.head_bank)
    current = load(args.current_bank)
    report = dict(
        head_bank=head["path"],
        current_bank=current["path"],
        head_shape=list(head["rows"].shape),
        current_shape=list(current["rows"].shape),
        head_active_nodes=head["active_nodes"],
        current_active_nodes=current["active_nodes"],
        threshold=args.threshold,
        window=args.window,
        material_boundary=args.material,
    )
    if head["rows"].shape != current["rows"].shape:
        raise SystemExit("the two banks have different row shapes")
    active = min(head["active_nodes"], current["active_nodes"])
    if head["active_nodes"] != current["active_nodes"]:
        print("ACTIVE_NODES_DIFFER", head["active_nodes"], current["active_nodes"])

    head_rows = head["rows"][:active, :, :, DERIVATIVE].astype(np.float64)
    current_rows = current["rows"][:active, :, :, DERIVATIVE].astype(np.float64)
    delta = head_rows - current_rows
    magnitude = np.abs(current_rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(magnitude > 0.0, np.abs(delta) / magnitude, np.inf)
    zero_magnitude = magnitude == 0.0

    def locate(flat_index):
        node, cell, moment, component = np.unravel_index(flat_index, rel.shape)
        return dict(
            node=int(node),
            cell=int(cell),
            moment=int(moment),
            component=COMPONENT[int(component)],
            absolute_departure=float(delta[node, cell, moment, component]),
            magnitude=float(current_rows[node, cell, moment, component]),
            relative=float(rel[node, cell, moment, component]),
        )

    report["entries"] = int(rel.size)
    report["zero_magnitude_entries"] = int(zero_magnitude.sum())
    if zero_magnitude.any():
        report["zero_magnitude_max_abs_delta"] = float(
            np.max(np.abs(delta)[zero_magnitude])
        )

    finite = np.isfinite(rel)
    unfiltered = dict(
        count=int(finite.sum()),
        max_relative=float(np.max(rel[finite])),
        median_relative=float(np.median(rel[finite])),
        location=locate(int(np.argmax(np.where(finite, rel, -np.inf)))),
        above_threshold=int((rel > args.threshold).sum()),
    )
    report["unfiltered"] = unfiltered
    print("UNFILTERED " + json.dumps(unfiltered, sort_keys=True), flush=True)

    keep = magnitude > args.window
    filtered = dict(
        window=args.window,
        count=int(keep.sum()),
        excluded=int((~keep & finite).sum()),
        max_relative=float(np.max(np.where(keep, rel, -np.inf))),
        median_relative=float(np.median(rel[keep])) if keep.any() else None,
        above_threshold=int(((rel > args.threshold) & keep).sum()),
        max_absolute=float(np.max(np.abs(delta)[keep])) if keep.any() else None,
        max_relative_location=locate(int(np.argmax(np.where(keep, rel, -np.inf)))),
        max_absolute_location=locate(
            int(np.argmax(np.where(keep, np.abs(delta), -np.inf)))
        ),
    )
    report["filtered"] = filtered
    print("FILTERED " + json.dumps(filtered, sort_keys=True), flush=True)

    for name, index in (("dpsi/dR", 0), ("dpsi/dZ", 1)):
        part = rel[..., index]
        part_finite = np.isfinite(part)
        magnitude_part = magnitude[..., index]
        report[name] = dict(
            count=int(part_finite.sum()),
            above_threshold=int((part > args.threshold).sum()),
            filtered_count=int((magnitude_part > args.window).sum()),
            filtered_above_threshold=int(
                ((part > args.threshold) & (magnitude_part > args.window)).sum()
            ),
            max_relative=float(np.max(part[part_finite])),
            median_relative=float(np.median(part[part_finite])),
        )
        print(
            name.replace("/", "_") + " " + json.dumps(report[name], sort_keys=True),
            flush=True,
        )

    rows = np.argwhere(rel > args.threshold)
    report["departing_count"] = int(len(rows))
    for node, cell, moment, component in rows:
        payload = dict(
            node=int(node),
            cell=int(cell),
            moment=int(moment),
            component=COMPONENT[int(component)],
            absolute_departure=float(delta[node, cell, moment, component]),
            magnitude=float(current_rows[node, cell, moment, component]),
            relative=float(rel[node, cell, moment, component]),
        )
        print("DEPARTURE " + json.dumps(payload, sort_keys=True), flush=True)

    aggregated = relative(delta, current_rows, axis=0)
    report["node_aggregated"] = dict(
        shape=list(aggregated.shape),
        max_relative=float(np.max(aggregated)),
        median_relative=float(np.median(aggregated)),
    )
    print(
        "NODE_AGGREGATED " + json.dumps(report["node_aggregated"], sort_keys=True),
        flush=True,
    )

    material = keep & (np.abs(delta) > args.material)
    report["material"] = dict(
        boundary=args.material, count=int(material.sum()), present=bool(material.any())
    )
    if material.any():
        node, cell, moment, component = np.unravel_index(
            int(np.argmax(np.where(material, np.abs(delta), -np.inf))), delta.shape
        )
        report["material"]["largest"] = dict(
            node=int(node),
            cell=int(cell),
            moment=int(moment),
            component=COMPONENT[int(component)],
            absolute_departure=float(delta[node, cell, moment, component]),
            magnitude=float(current_rows[node, cell, moment, component]),
        )
    print("MATERIAL " + json.dumps(report["material"], sort_keys=True), flush=True)

    report["completed"] = True
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print("DERIVE_COMPLETE " + str(args.output), flush=True)
    raise SystemExit(2 if material.any() else 0)


if __name__ == "__main__":
    main()
