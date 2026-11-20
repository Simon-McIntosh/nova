"""Compare the delivered, targeted and global twelve-node volume rules.

For each shell this assembles the linear Galerkin interaction under three
section rules and reports, for each, the metric-weighted quotient through the
delivered seam, the operator's distance from the global twelve-node operator,
and the assembly wall.  The rules are installed by replacing the pair selector
that the volume block reads, so the comparison exercises the assembled
operator rather than a hand-edited copy of the block.

    python scripts/iron_operator_bound/compare.py --output <dir> --tag final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot import tiledassembly  # noqa: E402

assert jax.config.jax_enable_x64

LABELS = (
    ("uniform-200", 0.2, 0.2),
    ("uniform-160", 0.16, 0.16),
    ("uniform-40", 0.04, 0.04),
    ("graded-16-80", 0.016, 0.08),
)


def distance(actual, reference):
    """Return the Frobenius and entrywise distances relative to the reference."""
    difference = np.asarray(actual) - np.asarray(reference)
    return dict(
        frobenius=float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        entrywise=float(np.max(np.abs(difference)) / np.max(np.abs(reference))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="final")
    parser.add_argument("--labels", nargs="+", default=[row[0] for row in LABELS])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / f"compare-{args.tag}.jsonl"
    region = iron.gate_shell()
    shipped = tiledassembly._volume_near_mask
    for label, fine, coarse in LABELS:
        if label not in args.labels:
            continue
        field = iron.graded_cell_size(
            [0.28, 0.4], [-0.4, 0.4], fine=fine, coarse=coarse, cells_across=20
        )
        cells = iron._mesh(region, field)
        count = len(cells.polygons)

        def delivered(polygons):
            return np.zeros((len(polygons), len(polygons)), dtype=bool)

        def global12(polygons):
            return np.ones((len(polygons), len(polygons)), dtype=bool)

        rows = dict(label=label, cells=count)
        operators = {}
        for rule, selector in (
            ("delivered", delivered),
            ("targeted", shipped),
            ("global12", global12),
        ):
            tiledassembly._volume_near_mask = selector
            clock = time.perf_counter()
            operator = np.asarray(iron._galerkin_interaction(cells, basis="linear"))
            wall = time.perf_counter() - clock
            operators[rule] = operator
            rows[rule] = dict(
                quotient=float(
                    iron.galerkin_spectrum(cells, operator, basis="linear")["maximum"]
                ),
                wall=wall,
                refined=int(selector(cells.polygons).sum()),
            )
            print(json.dumps(dict(label=label, rule=rule, **rows[rule])), flush=True)
        tiledassembly._volume_near_mask = shipped
        rows["distance"] = {
            rule: distance(operators[rule], operators["global12"])
            for rule in ("delivered", "targeted")
        }
        rows["targeted_matches_global"] = (
            max(rows["distance"]["targeted"].values()) <= 1e-10
        )
        with receipt.open("a") as stream:
            stream.write(json.dumps(rows) + "\n")
        print(json.dumps(rows), flush=True)


main()
