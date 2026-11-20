"""Profile one bank pair's jet against radius under each arm.

The second derivative in R is unbounded at an offending pair. An unbounded
second derivative has two possible sources: a jump in the first derivative, or
a divergent term in the differentiated expression. This script separates them
by evaluating the shared jet program at points straddling the target radius and
recording the booked rows there, so the step in the gradient and the step in the
value can both be followed as the offset shrinks.

A gradient step that holds its size as the offset shrinks is a discontinuity in
the first derivative. A step that falls with the offset is a smooth row
program.

Every number comes from the repository's own jet program,
``nova.equilibrium.fixed_points._source_jets``; an arm is that program with its
row function swapped, which is what ``kernel.row_program`` supplies.
"""

import argparse
import json

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax.numpy as jnp  # noqa: E402

from nova.equilibrium.fixed_points import _source_jets  # noqa: E402

from scripts.prototypes.bank_hessian_blowup.kernel import (  # noqa: E402
    ARMS,
    coupling_from_geometry,
    row_program,
)

CERTIFIED = (
    "/home/ITER/mcintos/.config/reckon/crew/archive/runs/"
    "r-20261117T110237355670-iter-bank-recertify/h200-head/store/sha256/"
    "81ee8fdf8056855f0846f0e00b7213434451be8e9557369b74ef35a23a62fd4f/bank.npz"
)
OFFSETS = (1e-3, 1e-6, 1e-9, 1e-12)
VALUE_ENTRY = 0
RADIAL_ENTRY = 1


def booked(coupling, arm, target, cell):
    with row_program(arm):
        jets = _source_jets(jnp.asarray(target), coupling)
    return [[float(value) for value in moment] for moment in np.asarray(jets[cell])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--cell", type=int, required=True)
    parser.add_argument("--moment", type=int, default=1)
    arguments = parser.parse_args()

    coupling = coupling_from_geometry()
    with np.load(CERTIFIED, allow_pickle=False) as archive:
        rows = np.asarray(archive["rows"])
        nodes = np.asarray(archive["nodes"])

    origin = np.asarray(nodes[arguments.node], dtype=np.float64)
    certificate = np.asarray(rows[arguments.node, arguments.cell], dtype=np.float64)

    record = dict(
        node=arguments.node,
        cell=arguments.cell,
        moment=arguments.moment,
        point=[float(value) for value in origin],
        certified=[float(value) for value in certificate.reshape(-1)],
        offsets=list(OFFSETS),
        value_entry=VALUE_ENTRY,
        radial_entry=RADIAL_ENTRY,
        arms={},
    )
    for arm in ARMS:
        arm_rows = {}
        for offset in (0.0, *OFFSETS, *(-value for value in OFFSETS)):
            target = jnp.asarray([origin[0] + offset, origin[1]])
            arm_rows[f"{offset:+.0e}"] = booked(coupling, arm, target, arguments.cell)
            print(
                "ARM", arguments.node, arguments.cell, arm, "%+.0e" % offset, flush=True
            )
        record["arms"][arm] = arm_rows

    with open(arguments.output, "w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print("JUMP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
