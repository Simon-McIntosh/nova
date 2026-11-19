"""Compare the probe's kernel rows against a psi central difference.

Three questions, one measurement each:

* whether psi is even or odd across the cell's mirror plane, which decides
  whether the vertical derivative of that moment must vanish there;
* the central-difference derivative of psi per step, with its convergence
  envelope.  This is the only independent reference: psi is the kernel's own
  value, which the two kernels share, so a difference of values is not a
  difference of formulas;
* how the repaired derivative behaves across the corner-height band, where the
  kernel switches from its analytic tangent to the stencil.

Run with ``/home/ITER/mcintos/Code/nova/.venv/bin/python
scripts/prototypes/polygon_symmetry_violation/scrutiny.py --input <probe.npz>``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def by_key(data):
    """One record per (node, cell, moment), with its three arms merged."""
    records = {}
    for index in range(len(data["arm"])):
        key = (
            int(data["node"][index]),
            int(data["cell"][index]),
            int(data["moment"][index]),
        )
        record = records.setdefault(
            key,
            dict(
                node=key[0],
                cell=key[1],
                moment=key[2],
                repaired=None,
                base=None,
                point=None,
                psi=[],
                sweep=[],
            ),
        )
        arm = str(data["arm"][index])
        if arm == "direct":
            record["repaired"] = float(data["derivative"][index])
            record["point"] = [float(value) for value in data["point"][index]]
        elif arm == "sweep":
            record["sweep"].append(
                (float(data["offset"][index]), float(data["derivative"][index]))
            )
        else:
            record["psi"].append(
                (
                    float(data["sign"][index]) * float(data["step"][index]),
                    float(data["value"][index]),
                )
            )
    for record in records.values():
        record["sweep"].sort()
    return list(records.values())


def central(record, step):
    """The psi central difference at one step magnitude, or None."""
    forward = dict(record["psi"]).get(step)
    backward = dict(record["psi"]).get(-step)
    if forward is None or backward is None:
        return None
    return (forward - backward) / (2.0 * step)


def parity(record, step):
    """psi(+step) +/- psi(-step), which says whether psi is even or odd."""
    forward = dict(record["psi"])[step]
    backward = dict(record["psi"])[-step]
    return dict(even=0.5 * (forward + backward), odd=0.5 * (forward - backward))


def envelope(values):
    """Successive differences of the step ladder, which bound the limit."""
    steps = sorted(values, key=lambda item: -item[0])
    return [dict(step=step, derivative=derivative) for step, derivative in steps]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    with np.load(args.input) as archive:
        data = {key: np.asarray(archive[key]) for key in archive}
    with open(args.split) as stream:
        split = json.load(stream)
    base = {}
    for entry in split["violations_A"] + split["worst_B"]:
        base[(entry["node"], entry["cell"], entry["moment"])] = entry["base_value"]

    rows = []
    for record in by_key(data):
        steps = sorted({abs(offset) for offset, _ in record["psi"]}, reverse=True)
        ladder = [(step, central(record, step)) for step in steps]
        ladder = [(step, value) for step, value in ladder if value is not None]
        discrepancies = [abs(a[1] - b[1]) for a, b in zip(ladder, ladder[1:])]
        rows.append(
            dict(
                node=record["node"],
                cell=record["cell"],
                moment=record["moment"],
                point=record["point"],
                repaired=record["repaired"],
                base=base.get((record["node"], record["cell"], record["moment"])),
                ladder=envelope(ladder),
                finest=ladder[-1][1] if ladder else None,
                largest_gap=max(discrepancies) if discrepancies else None,
                parity=parity(record, max(abs(offset) for offset, _ in record["psi"])),
                sweep=record["sweep"],
            )
        )
    print(json.dumps(rows, indent=1))
    if args.json:
        with open(args.json, "w") as stream:
            json.dump(rows, stream, indent=1)


if __name__ == "__main__":
    main()
