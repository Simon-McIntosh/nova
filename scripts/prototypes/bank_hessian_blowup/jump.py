"""Profile one bank pair's radial gradient against radius under each arm.

The second derivative in R is unbounded at an offending pair. An unbounded
second derivative has exactly two possible sources: a jump in the first
derivative, or a divergent term in the differentiated expression. This script
separates them by evaluating the booked radial gradient at points straddling
the target radius and reporting the step-to-step change at shrinking offsets.

A jump that holds its size as the booked gradient shrinks is a discontinuity in
the first derivative. A difference that falls with the offset is a smooth row
value.

The booked radial gradient is the first component of the gradient the shared jet
program ``nova.equilibrium.fixed_points._source_jets`` returns, so this script
reads ``2 pi R rows[6:9]`` from that program rather than assembling the rows
itself. Arms are the row programs the jet comparison swaps in.
"""

import argparse
import json

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
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
    "81ee8fdf8056855f0846e00b7213434451be8e9557369b74ef35a23a62fd4f/bank.npz"
)
OFFSETS = (1e-3, 1e-6, 1e-9, 1e-12)


def radial_gradient(coupling, arm):
    def radial(point):
        with row_program(arm):
            return _source_jets(jnp.asarray(point), coupling)[..., 0, :]

    return jax.jit(radial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--cell", type=int, required=True)
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
        point=[float(value) for value in origin],
        certified=[float(value) for value in certificate.reshape(-1)],
        offsets=list(OFFSETS),
        arms={},
    )
    for arm in ARMS:
        gradient = radial_gradient(coupling, arm)
        pair = {}
        for offset in (0.0, *OFFSETS, *(-value for value in OFFSETS)):
            point = jnp.asarray([origin[0] + offset, origin[1]])
            pair[f"{offset:+.0e}"] = [
                float(value) for value in np.asarray(gradient(point))[arguments.cell]
            ]
        record["arms"][arm] = pair
        centre = pair[f"{0.0:+.0e}"]
        print(
            "ARM",
            arm,
            "dR(m0,m1,m2)",
            " ".join(f"{value:.9e}" for value in centre),
            flush=True,
        )
        for offset in OFFSETS:
            up = pair[f"{offset:+.0e}"]
            down = pair[f"{-offset:+.0e}"]
            jumps = [a - b for a, b in zip(up, down, strict=True)]
            print(
                "  offset %.0e jump" % offset,
                " ".join(f"{value:+.6e}" for value in jumps),
                flush=True,
            )

    with open(arguments.output, "w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print("JUMP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
