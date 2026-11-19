"""Measure the demagnetisation-bound quotient of the small octagonal shells.

The quotient is the reciprocal extreme of the metric-weighted energy against
the local polynomial Gram blocks, read through the delivered seam
(:func:`nova.biot.magnetisation._galerkin_interaction` and
:func:`nova.biot.magnetisation.galerkin_spectrum`) rather than through a
hand-assembled block, so the row reports the operator the model returns.  One
receipt row lands per label as soon as that label completes.

    python scripts/iron_operator_bound/shells.py --output <dir> --tag before
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import shapely

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot import sectionaverage  # noqa: E402

assert jax.config.jax_enable_x64

LABELS = (
    ("uniform-200", 0.2, 0.2),
    ("uniform-160", 0.16, 0.16),
    ("uniform-40", 0.04, 0.04),
    ("graded-16-80", 0.016, 0.08),
    ("graded-8-40", 0.008, 0.04),
)


def gate_shell():
    """Return the closed octagonal shell of the linear spectral gate."""
    return (
        shapely.Point(0, 0)
        .buffer(0.4, 2)
        .difference(shapely.Point(0, 0).buffer(0.28, 2))
        .intersection(shapely.box(0, -0.4, 0.4, 0.4))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="shell")
    parser.add_argument("--labels", nargs="+", default=[row[0] for row in LABELS])
    parser.add_argument(
        "--volume-rule",
        choices=("delivered", "repaired"),
        default="delivered",
        help="section rule of the volume potential's target points",
    )
    args = parser.parse_args()
    if args.volume_rule == "repaired":
        delivery = sectionaverage.section_nodes

        def repaired(polygon, order):
            return delivery(polygon, max(12, order * 2))

        sectionaverage.section_nodes = repaired
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / f"shells-{args.tag}.jsonl"
    for label, fine, coarse in LABELS:
        if label not in args.labels:
            continue
        clock = time.perf_counter()
        field = iron.graded_cell_size(
            [0.28, 0.4],
            [-0.4, 0.4],
            fine=fine,
            coarse=coarse,
            cells_across=20,
        )
        cells = iron._mesh(gate_shell(), field)
        interaction = iron._galerkin_interaction(cells, basis="linear")
        spectrum = iron.galerkin_spectrum(cells, interaction, basis="linear")
        row = dict(
            label=label,
            tag=args.tag,
            fine=fine,
            coarse=coarse,
            cells=int(spectrum["cells"]),
            maximum=spectrum["maximum"],
            minimum=spectrum["minimum"],
            volume_symmetry=spectrum["volume_symmetry"],
            volume_rule=args.volume_rule,
            wall=time.perf_counter() - clock,
        )
        with receipt.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


main()
