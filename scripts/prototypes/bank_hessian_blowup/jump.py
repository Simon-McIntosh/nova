"""Profile one bank pair's radial gradient against radius under each arm.

The second derivative in R is unbounded at an offending pair. An unbounded
second derivative has exactly two possible sources: a jump in the first
derivative, or a divergent term in the differentiated expression. This script
separates them by evaluating the booked radial gradient at points straddling
the target radius and reporting the step-to-step change at shrinking offsets.

A jump that holds its size as the offset shrinks is a discontinuity in the
first derivative. A jump that grows as the offset shrinks is a kink whose
quadratic term diverges. A jump that shrinks with the offset is smooth.

Arms are the same three the jet comparison uses: the shipped stable-height
custom JVP, the stable-pole evaluation under plain ``jacfwd``, and the
pre-repair primal path.
"""

import argparse
import json

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nova.biot.polygonanalytic import (  # noqa: E402
    _packed_analytic_moments_primal,
    packed_analytic_moments,
)
from nova.equilibrium.topology import BiotMomentCoupling  # noqa: E402

GEOMETRY = "/home/ITER/mcintos/.cache/nova-iter-bank/dina-rerun/bank/geometry.npz"
CERTIFIED = (
    "/home/ITER/mcintos/.config/reckon/crew/archive/runs/"
    "r-20261117T110237355670-iter-bank-recertify/h200-head/store/sha256/"
    "81ee8fdf8056855f0846f0e00b7213434451be8e9557369b74ef35a23a62fd4f/bank.npz"
)
OFFSETS = (1e-3, 1e-6, 1e-9, 1e-12)


def coupling_from_geometry():
    with np.load(GEOMETRY, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices"])
        counts = np.asarray(archive["vertex_count"])
    polygons = tuple(
        vertices[index, : int(count)] for index, count in enumerate(counts)
    )
    return BiotMomentCoupling.from_polygons(polygons)


def radial_gradient(coupling, arm):
    def radial(point):
        arguments = (
            jnp.broadcast_to(point[0], coupling.norm.shape),
            jnp.broadcast_to(point[1], coupling.norm.shape),
            coupling.edge,
            coupling.weight,
            coupling.norm,
            coupling.centre.T,
            coupling.centre.T,
            coupling.reflection_axis,
            coupling.reflection_partner,
        )
        if arm == "shipped":
            rows = packed_analytic_moments(jnp, *arguments)
        elif arm == "stable_poles":
            rows = _packed_analytic_moments_primal(jnp, *arguments, stable_poles=True)
        else:
            rows = _packed_analytic_moments_primal(jnp, *arguments)
        # The booked radial gradient is 2*pi*R times the vertical field rows.
        return 2 * jnp.pi * point[0] * jnp.stack(rows[6:9], axis=-1)

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
    for arm in ("shipped", "stable_poles", "plain"):
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
